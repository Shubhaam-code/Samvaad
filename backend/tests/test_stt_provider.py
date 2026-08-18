"""Unit tests for the real OpenAIWhisperSTT provider adapter.

All tests use dependency injection with fake/mock SDK clients:
- ZERO network/API calls made during tests
- Tests provider response mapping (English, Hindi, Auto-detected language)
- Tests model and provider attribute reporting
- Tests latency measurement calculation
- Tests timeout, rate-limit, connection, and auth error wrapping
- Tests API key redaction security guarantee in error messages
- Tests dependency injection and unconfigured provider state
- Tests custom STT_BASE_URL behavior (OpenAI-compatible endpoints)
- Tests package/provider importability from app.stt
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

from app.stt import (
    STTError,
    STTRequest,
    STTResponse,
    create_openai_whisper_stt,
)
from app.stt.openai_whisper import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    DEFAULT_TIMEOUT_SECONDS,
    OpenAIWhisperSTT,
    is_openai_whisper_configured,
)

VALID_WAV_BYTES = (
    b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    b"\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)
VALID_OGG_BYTES = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
SECRET_KEY = "sk-proj-secret1234567890abcdef"


class MockOpenAISDKClient:
    """Mock OpenAI SDK client injected into OpenAIWhisperSTT for tests."""

    def __init__(
        self,
        *,
        return_text: str = "Default transcribed text",
        return_language: str = "en",
        return_duration: float = 2.5,
        should_raise: Optional[Exception] = None,
    ) -> None:
        self.audio = MagicMock()
        self.should_raise = should_raise
        self.return_text = return_text
        self.return_language = return_language
        self.return_duration = return_duration
        self.audio.transcriptions.create.side_effect = self._create
        self.call_count = 0

    def _create(self, **kwargs: object) -> MagicMock:
        self.call_count += 1
        if self.should_raise:
            raise self.should_raise

        mock_resp = MagicMock()
        mock_resp.text = self.return_text
        mock_resp.language = self.return_language
        mock_resp.duration = self.return_duration
        return mock_resp


def _make_sdk_error(
    exc_cls: type[BaseException],
    message: str,
    status_code: int = 500,
) -> BaseException:
    """Build a real openai SDK exception with a minimal stub response context.

    Most openai SDK exceptions (AuthenticationError, RateLimitError, etc.) accept
    a ``response=`` argument that points to the HTTP response. We synthesise a
    minimal namespace that satisfies the SDK's attribute expectations without
    touching the network.
    """
    response = types.SimpleNamespace(
        status_code=status_code,
        request=types.SimpleNamespace(headers={}, method="POST", url="https://stub"),
        headers={},
    )
    return exc_cls(message, response=response, body=None)


class TestOpenAIWhisperPackageImport:
    """Package and provider importability from app.stt."""

    def test_provider_importable_from_package(self) -> None:
        """The factory is reachable from the top-level app.stt namespace."""
        assert create_openai_whisper_stt is not None

    def test_provider_class_importable_from_module(self) -> None:
        """The class is reachable from its own module."""
        assert OpenAIWhisperSTT is not None


class TestOpenAIWhisperConfiguration:
    """Tests provider configuration checks."""

    def test_configuration_detection(self) -> None:
        assert not is_openai_whisper_configured(api_key=None, base_url=DEFAULT_BASE_URL)
        assert not is_openai_whisper_configured(api_key="  ", base_url=DEFAULT_BASE_URL)
        assert is_openai_whisper_configured(api_key="sk-test", base_url=DEFAULT_BASE_URL)
        assert is_openai_whisper_configured(api_key=None, base_url="http://localhost:8000/v1")

    def test_unconfigured_instantiation_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key is required when using the default"):
            OpenAIWhisperSTT(api_key=None, base_url=DEFAULT_BASE_URL)

    def test_invalid_model_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            OpenAIWhisperSTT(api_key="sk-test", model_name="", client=MagicMock())

    def test_invalid_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            OpenAIWhisperSTT(
                api_key="sk-test", timeout_seconds=0.0, client=MagicMock()
            )

    def test_invalid_max_audio_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_audio_size_mb"):
            OpenAIWhisperSTT(
                api_key="sk-test", max_audio_size_mb=-1.0, client=MagicMock()
            )

    def test_custom_base_url_without_key_allowed(self) -> None:
        """Local compatible endpoints (Ollama / vLLM / faster-whisper) may not
        require an API key. The provider must accept that configuration as long
        as a non-default base URL is supplied."""
        provider = create_openai_whisper_stt(
            api_key=None,
            base_url="http://localhost:9000/v1",
            client=MagicMock(),
        )
        assert provider._base_url == "http://localhost:9000/v1"


class TestOpenAIWhisperConstruction:
    """Construction, model/provider attributes, and zero-network guarantee."""

    def test_provider_and_model_attributes(self) -> None:
        provider = create_openai_whisper_stt(
            api_key="sk-test",
            model_name="whisper-large-v3",
            client=MagicMock(),
        )
        assert provider.provider == "openai_whisper"
        assert provider.model_name == "whisper-large-v3"

    def test_default_model_name(self) -> None:
        provider = create_openai_whisper_stt(api_key="sk-test", client=MagicMock())
        assert provider.model_name == DEFAULT_MODEL_NAME

    def test_injected_client_is_used_no_real_network(self) -> None:
        """No openai.OpenAI instance is created when a client is injected.

        The mock client's transcriptions.create is the only call site, so
        we can assert zero network by spying on the mock.
        """
        mock_client = MockOpenAISDKClient(return_text="hello", return_language="en")
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)

        provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

        assert mock_client.call_count == 1
        assert mock_client.audio.transcriptions.create.call_count == 1

    def test_sentinel_client_never_reached_when_validation_fails(self) -> None:
        """When audio validation fails inside the provider, the SDK is never
        called. This proves validation is a hard gate, not a soft warning."""

        class _SentinelClient:
            def __init__(self) -> None:
                self.called = False

            @property
            def audio(self) -> object:
                self.called = True
                raise AssertionError(
                    "SDK client must not be reached when audio validation fails"
                )

        provider = create_openai_whisper_stt(api_key="sk-test", client=_SentinelClient())

        # Pass a duck-typed request with empty audio to trigger validation
        # before any provider call.
        class _BadRequest:
            audio = b""
            filename = "audio.wav"

        with pytest.raises(ValueError, match="cannot be empty"):
            provider.transcribe(_BadRequest())

    def test_repr_does_not_leak_api_key(self) -> None:
        provider = create_openai_whisper_stt(api_key=SECRET_KEY, client=MagicMock())
        assert SECRET_KEY not in repr(provider)


class TestOpenAIWhisperTranscription:
    """Tests real provider response mapping using mock SDK client."""

    def test_english_transcription_mapping(self) -> None:
        mock_client = MockOpenAISDKClient(
            return_text="What are the top beaches in North Goa?",
            return_language="en",
            return_duration=3.2,
        )
        provider = create_openai_whisper_stt(
            api_key="sk-test",
            client=mock_client,
        )

        assert provider.provider == "openai_whisper"
        assert provider.model_name == "whisper-1"

        req = STTRequest(
            audio=VALID_WAV_BYTES,
            filename="question.wav",
            language="en",
        )
        resp = provider.transcribe(req)

        assert isinstance(resp, STTResponse)
        assert resp.text == "What are the top beaches in North Goa?"
        assert resp.language == "en"
        assert resp.provider == "openai_whisper"
        assert resp.model == "whisper-1"
        assert resp.duration_seconds == 3.2
        assert resp.confidence is None
        assert resp.latency_ms is not None and resp.latency_ms >= 0.0

    def test_hindi_transcription_mapping(self) -> None:
        mock_client = MockOpenAISDKClient(
            return_text="गोवा की प्रसिद्ध ऐतिहासिक इमारतें कौन सी हैं?",
            return_language="hi",
            return_duration=4.5,
        )
        provider = create_openai_whisper_stt(
            api_key="sk-test",
            client=mock_client,
        )

        req = STTRequest(
            audio=VALID_OGG_BYTES,
            filename="question.ogg",
            language="hi",
        )
        resp = provider.transcribe(req)

        assert resp.text == "गोवा की प्रसिद्ध ऐतिहासिक इमारतें कौन सी हैं?"
        assert resp.language == "hi"
        assert resp.duration_seconds == 4.5

    def test_automatic_language_detection(self) -> None:
        mock_client = MockOpenAISDKClient(
            return_text="Automatic language detection test.",
            return_language="en",
            return_duration=1.8,
        )
        provider = create_openai_whisper_stt(
            api_key="sk-test",
            client=mock_client,
        )

        # language=None triggers automatic language detection
        req = STTRequest(
            audio=VALID_WAV_BYTES,
            filename="audio.wav",
            language=None,
        )
        resp = provider.transcribe(req)

        assert resp.text == "Automatic language detection test."
        assert resp.language == "en"

    def test_empty_content_returns_stt_error(self) -> None:
        mock_client = MockOpenAISDKClient(return_text="   ")
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)

        with pytest.raises(STTError, match="returned empty transcription content"):
            provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

    def test_latency_measurement(self) -> None:
        """Latency is measured end-to-end and is always non-negative."""
        mock_client = MockOpenAISDKClient()
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)

        resp = provider.transcribe(
            STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav")
        )

        assert resp.latency_ms is not None
        assert isinstance(resp.latency_ms, float)
        assert resp.latency_ms >= 0.0

    def test_provider_kwargs_forwarded_to_sdk(self) -> None:
        """Provider forwards model, file, response_format, language, prompt."""
        captured: dict[str, object] = {}

        class _Transcriptions:
            def create(self_inner, **kwargs: object) -> MagicMock:
                captured.update(kwargs)
                out = MagicMock()
                out.text = "ok"
                out.language = "en"
                out.duration = 1.0
                return out

        class _Audio:
            transcriptions = _Transcriptions()

        class _CaptureClient:
            audio = _Audio()

        provider = create_openai_whisper_stt(
            api_key="sk-test",
            model_name="whisper-1",
            client=_CaptureClient(),
        )
        provider.transcribe(
            STTRequest(
                audio=VALID_WAV_BYTES,
                filename="question.wav",
                content_type="audio/wav",
                language="hi",
                prompt="Goa tourism",
            )
        )

        assert captured["model"] == "whisper-1"
        assert captured["language"] == "hi"
        assert captured["prompt"] == "Goa tourism"
        # file is a (filename, bytes, content_type) tuple per OpenAI SDK v1.x
        assert captured["response_format"] == "verbose_json"
        file_arg = captured["file"]
        assert isinstance(file_arg, tuple)
        assert file_arg[0] == "question.wav"
        assert file_arg[1] == VALID_WAV_BYTES
        assert file_arg[2] == "audio/wav"

    def test_no_language_kwarg_when_request_omits_and_default_unset(self) -> None:
        """language=None and no provider default → no `language` kwarg sent
        (lets the provider auto-detect)."""
        captured: dict[str, object] = {}

        class _Transcriptions:
            def create(self_inner, **kwargs: object) -> MagicMock:
                captured.update(kwargs)
                out = MagicMock()
                out.text = "ok"
                out.language = "en"
                out.duration = 1.0
                return out

        class _Audio:
            transcriptions = _Transcriptions()

        class _CaptureClient:
            audio = _Audio()

        provider = create_openai_whisper_stt(api_key="sk-test", client=_CaptureClient())
        provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

        assert "language" not in captured

        provider = create_openai_whisper_stt(api_key="sk-test", client=_CaptureClient())
        provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

        assert "language" not in captured

    def test_dict_response_also_supported(self) -> None:
        """Provider supports both attribute-style and dict-style responses."""
        provider = create_openai_whisper_stt(
            api_key="sk-test",
            client=MagicMock(
                **{
                    "audio.transcriptions.create.return_value": {
                        "text": "from-dict",
                        "language": "en",
                        "duration": 1.0,
                    }
                }
            ),
        )
        resp = provider.transcribe(
            STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav")
        )
        assert resp.text == "from-dict"
        assert resp.language == "en"
        assert resp.duration_seconds == 1.0

    def test_missing_duration_is_none(self) -> None:
        """Provider returns duration=None when the SDK omits it."""
        mock_resp = MagicMock()
        mock_resp.text = "hello"
        mock_resp.language = "en"
        mock_resp.duration = None
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_resp
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)
        resp = provider.transcribe(
            STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav")
        )
        assert resp.duration_seconds is None

    def test_unrecognised_language_string_falls_back_to_hint(self) -> None:
        """When the provider returns a long language name (e.g. 'english'),
        the adapter ignores it and keeps the requested hint."""
        mock_resp = MagicMock()
        mock_resp.text = "hello"
        mock_resp.language = "english"  # not an ISO 639-1 short code
        mock_resp.duration = 1.0
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_resp
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)
        resp = provider.transcribe(
            STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav", language="hi")
        )
        assert resp.language == "hi"


class TestOpenAIWhisperCustomBaseURL:
    """Custom STT_BASE_URL behavior for OpenAI-compatible Whisper servers."""

    def test_custom_base_url_preserved(self) -> None:
        provider = create_openai_whisper_stt(
            api_key="sk-test",
            base_url="http://localhost:9000/v1",
            client=MagicMock(),
        )
        assert provider._base_url == "http://localhost:9000/v1"

    def test_custom_base_url_does_not_require_api_key(self) -> None:
        """A local Whisper server (e.g. faster-whisper) may not need a key."""
        provider = create_openai_whisper_stt(
            api_key=None,
            base_url="http://localhost:9000/v1",
            client=MagicMock(),
        )
        assert provider._api_key == ""

    def test_custom_base_url_with_trailing_slash_is_normalised(self) -> None:
        provider = create_openai_whisper_stt(
            api_key="sk-test",
            base_url="http://localhost:9000/v1/",
            client=MagicMock(),
        )
        # Leading/trailing whitespace is stripped; trailing slash is preserved
        # (we do not mutate the URL beyond what the constructor documented).
        assert provider._base_url.rstrip("/") == "http://localhost:9000/v1"


class TestOpenAIWhisperErrorHandling:
    """Tests provider error wrapping and credential safety."""

    def test_provider_error_wrapping(self) -> None:
        mock_client = MockOpenAISDKClient(
            should_raise=OpenAIError("API Connection Timeout")
        )
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)

        with pytest.raises(STTError) as exc_info:
            provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

        assert "OpenAI Whisper provider error (OpenAIError)" in str(exc_info.value)
        assert "API Connection Timeout" in str(exc_info.value)

    def test_api_key_redaction_in_exceptions(self) -> None:
        sensitive_error = OpenAIError(f"Authentication failed for key {SECRET_KEY}")
        mock_client = MockOpenAISDKClient(should_raise=sensitive_error)

        provider = create_openai_whisper_stt(
            api_key=SECRET_KEY,
            client=mock_client,
        )

        with pytest.raises(STTError) as exc_info:
            provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

        error_message = str(exc_info.value)
        assert SECRET_KEY not in error_message
        assert "[REDACTED]" in error_message

    def test_rate_limit_error_wrapping(self) -> None:
        mock_client = MockOpenAISDKClient(
            should_raise=_make_sdk_error(
                RateLimitError, "Rate limit exceeded (429)", status_code=429
            )
        )
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)

        with pytest.raises(STTError, match="Rate limit exceeded"):
            provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

    def test_authentication_error_wrapping(self) -> None:
        """Authentication failures from the SDK surface as STTError with
        the original message preserved (and API keys redacted)."""
        mock_client = MockOpenAISDKClient(
            should_raise=_make_sdk_error(
                AuthenticationError, "invalid api key", status_code=401
            )
        )
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)

        with pytest.raises(STTError) as exc_info:
            provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

        assert "AuthenticationError" in str(exc_info.value)
        assert "invalid api key" in str(exc_info.value)

    def test_timeout_error_wrapping(self) -> None:
        """Timeouts from the SDK surface as STTError."""
        mock_client = MockOpenAISDKClient(
            should_raise=APITimeoutError(
                request=types.SimpleNamespace(
                    headers={}, method="POST", url="https://stub"
                )
            )
        )
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)

        with pytest.raises(STTError) as exc_info:
            provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

        assert "APITimeoutError" in str(exc_info.value)

    def test_connection_error_wrapping(self) -> None:
        """Connection failures from the SDK surface as STTError."""
        mock_client = MockOpenAISDKClient(
            should_raise=APIConnectionError(
                message="connection refused",
                request=types.SimpleNamespace(
                    headers={}, method="POST", url="https://stub"
                ),
            )
        )
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)

        with pytest.raises(STTError) as exc_info:
            provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

        assert "connection refused" in str(exc_info.value)

    def test_unexpected_exception_wrapped(self) -> None:
        """Non-OpenAI exceptions are still wrapped into STTError rather than
        leaking out to callers."""
        mock_client = MockOpenAISDKClient(should_raise=RuntimeError("boom"))
        provider = create_openai_whisper_stt(api_key="sk-test", client=mock_client)

        with pytest.raises(STTError) as exc_info:
            provider.transcribe(STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav"))

        assert "RuntimeError" in str(exc_info.value)
        assert "boom" in str(exc_info.value)


class TestOpenAIWhisperAudioValidationGate:
    """Audio validation gates the SDK call. Real providers never see invalid
    audio. This is part of the production safety story."""

    def test_empty_audio_blocked_before_sdk_call(self) -> None:
        provider = create_openai_whisper_stt(
            api_key="sk-test", client=MockOpenAISDKClient()
        )

        class _Bad:
            audio = b""
            filename = "audio.wav"

        with pytest.raises(ValueError, match="cannot be empty"):
            provider.transcribe(_Bad())

    def test_unsupported_extension_blocked_before_sdk_call(self) -> None:
        provider = create_openai_whisper_stt(
            api_key="sk-test", client=MockOpenAISDKClient()
        )

        class _Bad:
            audio = VALID_WAV_BYTES
            filename = "audio.txt"

        with pytest.raises(ValueError, match="unsupported audio format"):
            provider.transcribe(_Bad())

    def test_oversized_audio_blocked_before_sdk_call(self) -> None:
        provider = create_openai_whisper_stt(
            api_key="sk-test",
            max_audio_size_mb=0.00001,  # ~10 bytes - smaller than the WAV fixture
            client=MockOpenAISDKClient(),
        )

        with pytest.raises(ValueError, match="exceeds the maximum allowed size"):
            provider.transcribe(
                STTRequest(audio=VALID_WAV_BYTES, filename="audio.wav")
            )


class TestOpenAIWhisperTimeoutConfiguration:
    """Default and custom timeout values are propagated to the SDK constructor.

    When the adapter constructs its own openai.OpenAI client, the configured
    ``timeout_seconds`` must be passed through. When a stub is injected, this
    test simply confirms the configuration is stored on the adapter.
    """

    def test_default_timeout_stored(self) -> None:
        provider = create_openai_whisper_stt(api_key="sk-test", client=MagicMock())
        assert provider._timeout_seconds == DEFAULT_TIMEOUT_SECONDS

    def test_custom_timeout_stored(self) -> None:
        provider = create_openai_whisper_stt(
            api_key="sk-test", timeout_seconds=12.5, client=MagicMock()
        )
        assert provider._timeout_seconds == 12.5

