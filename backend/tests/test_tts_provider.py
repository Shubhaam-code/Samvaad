"""Unit tests for the real OpenAITTS provider adapter.

All tests use dependency injection with mock/stub SDK clients:
- ZERO network/API calls made during tests
- Tests provider request mapping (input, model, voice, response_format, speed, timeout)
- Tests provider response mapping (English, Hindi, audio bytes, content type, latency)
- Tests model and provider attribute reporting
- Tests timeout, rate-limit, connection, and auth error wrapping in TTSError
- Tests API key redaction security guarantee in error messages and repr
- Tests dependency injection and unconfigured provider state
- Tests custom TTS_BASE_URL behavior (OpenAI-compatible endpoints)
- Tests get_tts() dependency resolution, caching, and FakeTTS exclusion guarantee
- Tests batch synthesis and order preservation
"""

from __future__ import annotations

import types
from typing import Optional
from unittest.mock import MagicMock

import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from app.api.dependencies import get_tts
from app.settings import settings
from app.tts import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SPEED,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_VOICE,
    OpenAITTS,
    TTSError,
    TTSRequest,
    TTSResponse,
    create_openai_tts,
    is_openai_tts_configured,
)

VALID_MP3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x00synth-audio-mp3-bytes"
VALID_WAV_BYTES = (
    b"RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    b"\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00"
    b"\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00"
)
SECRET_KEY = "sk-proj-tts-secret-key-1234567890abcdef"


class MockOpenAISpeechClient:
    """Mock OpenAI SDK client injected into OpenAITTS for offline tests."""

    def __init__(
        self,
        *,
        return_audio: bytes = VALID_MP3_BYTES,
        should_raise: Optional[Exception] = None,
    ) -> None:
        self.audio = MagicMock()
        self.should_raise = should_raise
        self.return_audio = return_audio
        self.audio.speech.create.side_effect = self._create
        self.call_count = 0
        self.last_kwargs: dict[str, object] = {}

    def _create(self, **kwargs: object) -> MagicMock:
        self.call_count += 1
        self.last_kwargs = kwargs
        if self.should_raise:
            raise self.should_raise

        mock_resp = MagicMock()
        mock_resp.content = self.return_audio
        mock_resp.read.return_value = self.return_audio
        return mock_resp


def _make_sdk_error(
    exc_cls: type[BaseException],
    message: str,
    status_code: int = 500,
) -> BaseException:
    """Build a real openai SDK exception with a minimal stub response context."""
    response = types.SimpleNamespace(
        status_code=status_code,
        request=types.SimpleNamespace(headers={}, method="POST", url="https://stub"),
        headers={},
    )
    return exc_cls(message, response=response, body=None)


# ===========================================================================
# 1. OpenAITTS Construction & Defaults
# ===========================================================================

class TestOpenAITTSConstruction:
    """Test 1: Construction, configuration validation, and repr safety."""

    def test_construction_with_injected_client(self) -> None:
        mock_client = MockOpenAISpeechClient()
        tts = OpenAITTS(client=mock_client)

        assert tts.provider == "openai_tts"
        assert tts.model_name == DEFAULT_MODEL_NAME

    def test_construction_missing_key_default_url_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key is required when using the default OpenAI base URL"):
            OpenAITTS(api_key=None, base_url=None)

    def test_construction_custom_base_url_allows_no_key(self) -> None:
        # Non-default base URL (e.g. local TTS server) does not require a key
        mock_client = MockOpenAISpeechClient()
        tts = OpenAITTS(
            base_url="http://localhost:8000/v1",
            client=mock_client,
        )
        assert tts.provider == "openai_tts"

    def test_construction_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            OpenAITTS(model="   ", client=MockOpenAISpeechClient())

        with pytest.raises(ValueError, match="timeout_seconds must be a positive number"):
            OpenAITTS(timeout_seconds=-5.0, client=MockOpenAISpeechClient())

        with pytest.raises(ValueError, match="max_text_length must be a positive integer"):
            OpenAITTS(max_text_length=0, client=MockOpenAISpeechClient())

        with pytest.raises(ValueError, match="max_audio_size_mb must be a positive number"):
            OpenAITTS(max_audio_size_mb=-1.0, client=MockOpenAISpeechClient())

    def test_repr_never_leaks_api_key(self) -> None:
        mock_client = MockOpenAISpeechClient()
        tts = OpenAITTS(
            api_key=SECRET_KEY,
            client=mock_client,
        )
        repr_str = repr(tts)
        assert SECRET_KEY not in repr_str
        assert "OpenAITTS" in repr_str


# ===========================================================================
# 2. OpenAITTS Request & Kwargs Forwarding
# ===========================================================================

class TestOpenAITTSSynthesis:
    """Test 2: Request forwarding to client.audio.speech.create."""

    def test_synthesis_kwargs_forwarding(self) -> None:
        mock_client = MockOpenAISpeechClient(return_audio=VALID_MP3_BYTES)
        tts = OpenAITTS(
            model="tts-1",
            voice="alloy",
            output_format="mp3",
            speed=1.0,
            timeout_seconds=25.0,
            client=mock_client,
        )

        req = TTSRequest(
            text="Testing speech synthesis in Goa",
            voice="nova",
            model="tts-1-hd",
            output_format="mp3",
            speed=1.25,
        )
        resp = tts.synthesize(req)

        assert mock_client.call_count == 1
        assert mock_client.last_kwargs["input"] == "Testing speech synthesis in Goa"
        assert mock_client.last_kwargs["model"] == "tts-1-hd"
        assert mock_client.last_kwargs["voice"] == "nova"
        assert mock_client.last_kwargs["response_format"] == "mp3"
        assert mock_client.last_kwargs["speed"] == 1.25
        assert mock_client.last_kwargs["timeout"] == 25.0

        assert resp.audio == VALID_MP3_BYTES
        assert resp.format == "mp3"
        assert resp.content_type == "audio/mpeg"
        assert resp.model == "tts-1-hd"
        assert resp.provider == "openai_tts"
        assert resp.character_count == len("Testing speech synthesis in Goa")
        assert resp.latency_ms is not None
        assert resp.latency_ms >= 0.0
        assert resp.metadata == {"voice": "nova", "speed": 1.25}

    def test_synthesis_wav_format(self) -> None:
        mock_client = MockOpenAISpeechClient(return_audio=VALID_WAV_BYTES)
        tts = OpenAITTS(client=mock_client)

        req = TTSRequest(text="WAV audio test", output_format="wav")
        resp = tts.synthesize(req)

        assert mock_client.last_kwargs["response_format"] == "wav"
        assert resp.format == "wav"
        assert resp.content_type == "audio/wav"
        assert resp.audio == VALID_WAV_BYTES

    def test_synthesis_hindi_text(self) -> None:
        mock_client = MockOpenAISpeechClient(return_audio=VALID_MP3_BYTES)
        tts = OpenAITTS(client=mock_client)

        req = TTSRequest(text="गोवा में आपका स्वागत है।", voice="shimmer", language="hi")
        resp = tts.synthesize(req)

        assert mock_client.last_kwargs["input"] == "गोवा में आपका स्वागत है।"
        assert mock_client.last_kwargs["voice"] == "shimmer"
        assert resp.character_count == len("गोवा में आपका स्वागत है।")

    def test_synthesis_empty_audio_raises_tts_error(self) -> None:
        mock_client = MockOpenAISpeechClient(return_audio=b"")
        tts = OpenAITTS(client=mock_client)

        req = TTSRequest(text="Empty audio check")
        with pytest.raises(TTSError, match="returned empty audio content"):
            tts.synthesize(req)

    def test_synthesis_batch_preserves_order(self) -> None:
        mock_client = MockOpenAISpeechClient(return_audio=VALID_MP3_BYTES)
        tts = OpenAITTS(client=mock_client)

        reqs = [
            TTSRequest(text="Segment 1"),
            TTSRequest(text="Segment 2"),
            TTSRequest(text="Segment 3"),
        ]
        responses = tts.synthesize_batch(reqs)

        assert len(responses) == 3
        assert mock_client.call_count == 3
        assert responses[0].character_count == len("Segment 1")
        assert responses[1].character_count == len("Segment 2")
        assert responses[2].character_count == len("Segment 3")

    def test_synthesis_batch_invalid_argument(self) -> None:
        mock_client = MockOpenAISpeechClient()
        tts = OpenAITTS(client=mock_client)

        with pytest.raises(ValueError, match="requests must be a list"):
            tts.synthesize_batch("not-a-list")  # type: ignore[arg-type]


# ===========================================================================
# 3. Error Wrapping and Credential Redaction
# ===========================================================================

class TestOpenAITTSErrorHandling:
    """Test 3: Provider error wrapping and API key sanitization."""

    def test_authentication_error_wrapped(self) -> None:
        err = _make_sdk_error(AuthenticationError, f"Invalid key: {SECRET_KEY}", status_code=401)
        mock_client = MockOpenAISpeechClient(should_raise=err)
        tts = OpenAITTS(api_key=SECRET_KEY, client=mock_client)

        with pytest.raises(TTSError) as exc_info:
            tts.synthesize(TTSRequest(text="Auth failure check"))

        msg = str(exc_info.value)
        assert SECRET_KEY not in msg
        assert "[REDACTED]" in msg
        assert "AuthenticationError" in msg

    def test_rate_limit_error_wrapped(self) -> None:
        err = _make_sdk_error(RateLimitError, "Quota exceeded for TTS", status_code=429)
        mock_client = MockOpenAISpeechClient(should_raise=err)
        tts = OpenAITTS(api_key=SECRET_KEY, client=mock_client)

        with pytest.raises(TTSError, match="RateLimitError"):
            tts.synthesize(TTSRequest(text="Rate limit test"))

    def test_timeout_error_wrapped(self) -> None:
        err = APITimeoutError(request=MagicMock())
        mock_client = MockOpenAISpeechClient(should_raise=err)
        tts = OpenAITTS(api_key=SECRET_KEY, client=mock_client)

        with pytest.raises(TTSError, match="APITimeoutError"):
            tts.synthesize(TTSRequest(text="Timeout test"))

    def test_connection_error_wrapped(self) -> None:
        err = APIConnectionError(request=MagicMock())
        mock_client = MockOpenAISpeechClient(should_raise=err)
        tts = OpenAITTS(api_key=SECRET_KEY, client=mock_client)

        with pytest.raises(TTSError, match="APIConnectionError"):
            tts.synthesize(TTSRequest(text="Connection test"))


# ===========================================================================
# 4. Configuration Helpers & Dependency Injection
# ===========================================================================

class TestTTSConfigurationAndDI:
    """Test 4: is_openai_tts_configured and get_tts() dependency."""

    def test_is_openai_tts_configured_truth_table(self) -> None:
        # Key present -> configured
        assert is_openai_tts_configured(api_key="sk-test", base_url=DEFAULT_BASE_URL) is True
        # Custom base URL -> configured (even without key)
        assert is_openai_tts_configured(api_key=None, base_url="http://localhost:8000/v1") is True
        assert is_openai_tts_configured(api_key="", base_url="http://localhost:8000/v1") is True
        # Default base URL + no key -> NOT configured
        assert is_openai_tts_configured(api_key=None, base_url=DEFAULT_BASE_URL) is False
        assert is_openai_tts_configured(api_key="", base_url=DEFAULT_BASE_URL) is False

    def test_get_tts_unconfigured_returns_none(self) -> None:
        # Under autouse test fixtures, settings are unconfigured
        tts = get_tts()
        assert tts is None

    def test_get_tts_never_returns_fake_tts(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "tts_provider", "fake")
        # get_tts only resolves openai_tts in production, never fake
        assert get_tts() is None

    def test_get_tts_configured_returns_openai_tts(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "tts_provider", "openai_tts")
        monkeypatch.setattr(settings, "tts_api_key", "sk-proj-tts-key")
        monkeypatch.setattr(settings, "tts_base_url", None)
        monkeypatch.setattr(settings, "tts_model", "tts-1")

        tts1 = get_tts()
        assert tts1 is not None
        assert isinstance(tts1, OpenAITTS)
        assert tts1.model_name == "tts-1"
        assert tts1.provider == "openai_tts"

        # Cached instance returned
        tts2 = get_tts()
        assert tts2 is tts1

    def test_get_tts_falls_back_to_llm_credentials(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "tts_provider", "openai_tts")
        monkeypatch.setattr(settings, "tts_api_key", None)
        monkeypatch.setattr(settings, "llm_api_key", "sk-shared-openai-key")

        tts = get_tts()
        assert tts is not None
        assert isinstance(tts, OpenAITTS)

    def test_create_openai_tts_helper(self) -> None:
        mock_client = MockOpenAISpeechClient()
        tts = create_openai_tts(
            model="tts-1-hd",
            voice="fable",
            output_format="wav",
            speed=1.5,
            client=mock_client,
        )
        assert isinstance(tts, OpenAITTS)
        assert tts.model_name == "tts-1-hd"

    def test_synthesis_all_supported_formats(self) -> None:
        format_samples = {
            "opus": b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00opus-test-bytes",
            "aac": b"\xff\xf1\x50\x80\x00\x1f\xfcaac-test-bytes",
            "flac": b"fLaC\x00\x00\x00\x22flac-test-bytes",
            "pcm": b"\x00\x00\x01\x00pcm-test-bytes",
        }
        for fmt, audio in format_samples.items():
            mock_client = MockOpenAISpeechClient(return_audio=audio)
            tts = OpenAITTS(client=mock_client)
            resp = tts.synthesize(TTSRequest(text=f"Testing {fmt}", output_format=fmt))
            assert resp.format == fmt
            assert resp.audio == audio

    def test_synthesis_dict_and_bytes_response_mapping(self) -> None:
        # Test duck-typed dict response
        class DictClient:
            def __init__(self):
                self.audio = MagicMock()
                self.audio.speech.create.return_value = {"audio": VALID_MP3_BYTES}

        tts_dict = OpenAITTS(client=DictClient())
        resp1 = tts_dict.synthesize(TTSRequest(text="Dict response"))
        assert resp1.audio == VALID_MP3_BYTES

        # Test raw bytes response
        class RawBytesClient:
            def __init__(self):
                self.audio = MagicMock()
                self.audio.speech.create.return_value = VALID_MP3_BYTES

        tts_bytes = OpenAITTS(client=RawBytesClient())
        resp2 = tts_bytes.synthesize(TTSRequest(text="Raw bytes response"))
        assert resp2.audio == VALID_MP3_BYTES

    def test_synthesis_mismatched_audio_raises_tts_error(self) -> None:
        # Return WAV bytes when MP3 was requested
        mock_client = MockOpenAISpeechClient(return_audio=VALID_WAV_BYTES)
        tts = OpenAITTS(client=mock_client)

        with pytest.raises(TTSError, match="TTS audio validation failure"):
            tts.synthesize(TTSRequest(text="Format mismatch test", output_format="mp3"))

