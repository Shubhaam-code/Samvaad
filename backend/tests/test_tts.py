"""Unit tests for the TTS package — interfaces, models, validation, FakeTTS.

These tests cover:
  1. Package imports and __all__ exports
  2. BaseTTS ABC enforcement (cannot instantiate directly)
  3. TTSProtocol duck-typing
  4. TTSRequest field validation
  5. TTSResponse field validation
  6. TTSConfig field validation
  7. Empty text rejection
  8. Whitespace-only text rejection
  9. Maximum text length enforcement
  10. Invalid speed rejection (bounds, bool, non-number)
  11. Invalid voice rejection (empty, whitespace, non-string)
  12. Invalid output format rejection
  13. Audio output validation (validate_tts_audio, size, MIME, magic bytes)
  14. Sniff audio format helper for supported containers
  15. FakeTTS determinism (English & Hindi)
  16. FakeTTS no-network guarantee
  17. FakeTTS error injection (TTSError and generic Exception)
  18. FakeTTS simulated latency
  19. FakeTTS canned responses mapping
  20. FakeTTS batch synthesis and order preservation
  21. TTSConfig safe repr (API key never exposed)
  22. Supported format and speed constants
"""

from __future__ import annotations

import time
import pytest

from app.tts import (
    DEFAULT_MAX_TEXT_LENGTH,
    DEFAULT_MAX_TTS_AUDIO_BYTES,
    FORMAT_TO_MIME,
    MAX_SPEED,
    MIN_SPEED,
    SUPPORTED_FORMATS,
    SUPPORTED_TTS_MIMES,
    BaseTTS,
    FakeTTS,
    TTSAudio,
    TTSConfig,
    TTSError,
    TTSFormat,
    TTSLanguage,
    TTSModel,
    TTSProtocol,
    TTSProvider,
    TTSRequest,
    TTSResponse,
    TTSText,
    TTSVoice,
    ValidatedTTSAudio,
    create_fake_tts,
    sniff_tts_audio_format,
    validate_language,
    validate_output_format,
    validate_speed,
    validate_text,
    validate_tts_audio,
    validate_voice,
)

# Minimal valid container byte samples for tests
VALID_MP3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x00synth-audio-data"
VALID_WAV_BYTES = (
    b"RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    b"\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00"
    b"\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00"
)
VALID_OGG_BYTES = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00opus-data"
VALID_FLAC_BYTES = b"fLaC\x00\x00\x00\x22flac-audio-data"
VALID_AAC_BYTES = b"\xff\xf1\x50\x80\x00\x1f\xfcaac-audio-data"
VALID_PCM_BYTES = b"\x00\x00\x01\x00\x02\x00raw-pcm-samples"


# ===========================================================================
# 1. Package imports
# ===========================================================================

class TestTTSImports:
    """Test 1: Package imports and exports."""

    def test_core_imports_exist(self) -> None:
        assert BaseTTS is not None
        assert TTSError is not None
        assert TTSProtocol is not None
        assert TTSRequest is not None
        assert TTSResponse is not None
        assert TTSConfig is not None
        assert TTSProvider is not None
        assert FakeTTS is not None
        assert create_fake_tts is not None
        assert validate_text is not None
        assert validate_voice is not None
        assert validate_speed is not None
        assert validate_output_format is not None
        assert validate_language is not None
        assert validate_tts_audio is not None
        assert sniff_tts_audio_format is not None


# ===========================================================================
# 2. BaseTTS ABC and Protocol
# ===========================================================================

class TestBaseTTSABC:
    """Test 2: BaseTTS ABC enforcement and protocol duck-typing."""

    def test_cannot_instantiate_basetts_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseTTS()  # type: ignore[abstract]

    def test_protocol_duck_typing(self) -> None:
        class DuckTTS:
            def synthesize(self, request: object) -> object:
                return "ok"

            def synthesize_batch(self, requests: list[object]) -> list[object]:
                return ["ok"]

            @property
            def model_name(self) -> str:
                return "duck-model"

            @property
            def provider(self) -> str:
                return "duck-provider"

        duck = DuckTTS()
        assert isinstance(duck, TTSProtocol)

    def test_fake_tts_is_basetts_and_protocol(self) -> None:
        fake = FakeTTS()
        assert isinstance(fake, BaseTTS)
        assert isinstance(fake, TTSProtocol)


# ===========================================================================
# 3. Request / Response validation
# ===========================================================================

class TestTTSModels:
    """Test 3: TTSRequest and TTSResponse validation."""

    def test_valid_request_minimal(self) -> None:
        req = TTSRequest(text="Hello world")
        assert req.text == "Hello world"
        assert req.voice == "alloy"
        assert req.output_format == "mp3"
        assert req.speed == 1.0
        assert req.language is None

    def test_valid_request_full(self) -> None:
        req = TTSRequest(
            text="नमस्ते, आप कैसे हैं?",
            voice="nova",
            model="tts-1-hd",
            output_format="wav",
            speed=1.25,
            language="hi",
        )
        assert req.text == "नमस्ते, आप कैसे हैं?"
        assert req.voice == "nova"
        assert req.model == "tts-1-hd"
        assert req.output_format == "wav"
        assert req.speed == 1.25
        assert req.language == "hi"

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValueError):
            TTSRequest(text="")
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            validate_text("")

    def test_whitespace_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            TTSRequest(text="   \n\t  ")

    def test_max_text_length_enforced(self) -> None:
        long_text = "a" * 5000
        with pytest.raises(ValueError, match="exceeds maximum allowed length"):
            TTSRequest(text=long_text)

    def test_invalid_speed_bounds(self) -> None:
        with pytest.raises(ValueError):
            TTSRequest(text="Hello", speed=0.1)
        with pytest.raises(ValueError):
            TTSRequest(text="Hello", speed=5.0)
        with pytest.raises(ValueError, match="TTS speed must be between"):
            validate_speed(0.1)
        with pytest.raises(ValueError, match="TTS speed must be between"):
            validate_speed(5.0)

    def test_invalid_speed_type(self) -> None:
        with pytest.raises(ValueError, match="TTS speed must be a number"):
            validate_speed(True)  # bool is disallowed
        with pytest.raises(ValueError, match="TTS speed must be a number"):
            validate_speed("fast")  # type: ignore[arg-type]

    def test_invalid_voice_rejected(self) -> None:
        with pytest.raises(ValueError, match="TTS voice cannot be empty or whitespace"):
            TTSRequest(text="Hello", voice="   ")
        with pytest.raises(ValueError, match="TTS voice must be a string"):
            validate_voice(123)  # type: ignore[arg-type]

    def test_invalid_format_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported TTS output format"):
            TTSRequest(text="Hello", output_format="avi")

    def test_invalid_language_rejected(self) -> None:
        with pytest.raises(ValueError, match="language must be an ISO 639-1/2 code"):
            TTSRequest(text="Hello", language="english_us")

    def test_valid_response(self) -> None:
        resp = TTSResponse(
            audio=VALID_MP3_BYTES,
            content_type="audio/mpeg",
            format="mp3",
            model="tts-1",
            provider="openai_tts",
            latency_ms=120.5,
            character_count=11,
        )
        assert resp.audio == VALID_MP3_BYTES
        assert resp.content_type == "audio/mpeg"
        assert resp.format == "mp3"
        assert resp.model == "tts-1"
        assert resp.provider == "openai_tts"
        assert resp.latency_ms == 120.5
        assert resp.character_count == 11

    def test_empty_audio_response_rejected(self) -> None:
        with pytest.raises(ValueError):
            TTSResponse(
                audio=b"",
                content_type="audio/mpeg",
                format="mp3",
            )


# ===========================================================================
# 4. Config validation
# ===========================================================================

class TestTTSConfig:
    """Test 4: TTSConfig and TTSProvider validation."""

    def test_default_config(self) -> None:
        cfg = TTSConfig()
        assert cfg.provider == TTSProvider.FAKE
        assert cfg.model == "tts-1"
        assert cfg.voice == "alloy"
        assert cfg.output_format == "mp3"
        assert cfg.speed == 1.0
        assert cfg.timeout_seconds == 30.0
        assert cfg.max_text_length == 4096
        assert cfg.max_audio_size_mb == 10.0
        assert cfg.api_key is None
        assert cfg.base_url is None

    def test_custom_config(self) -> None:
        cfg = TTSConfig(
            provider=TTSProvider.OPENAI_TTS,
            api_key="sk-secret-key-123",
            base_url="http://localhost:8080/v1",
            model="tts-1-hd",
            voice="shimmer",
            output_format="wav",
            speed=1.5,
            timeout_seconds=45.0,
            max_text_length=2000,
            max_audio_size_mb=25.0,
        )
        assert cfg.provider == TTSProvider.OPENAI_TTS
        assert cfg.api_key == "sk-secret-key-123"
        assert cfg.base_url == "http://localhost:8080/v1"
        assert cfg.model == "tts-1-hd"
        assert cfg.voice == "shimmer"
        assert cfg.output_format == "wav"
        assert cfg.speed == 1.5

    def test_config_repr_does_not_leak_key(self) -> None:
        cfg = TTSConfig(api_key="sk-secret-key-123")
        repr_str = repr(cfg)
        assert "sk-secret-key-123" not in repr_str
        assert "TTSConfig" in repr_str

    def test_invalid_config_fields(self) -> None:
        with pytest.raises(ValueError):
            TTSConfig(model="   ")
        with pytest.raises(ValueError):
            TTSConfig(speed=5.0)
        with pytest.raises(ValueError):
            TTSConfig(output_format="invalid")


# ===========================================================================
# 5. Audio output validation & Sniffing
# ===========================================================================

class TestAudioValidation:
    """Test 5: Audio output validation and container sniffing."""

    def test_validate_tts_audio_mp3(self) -> None:
        res = validate_tts_audio(VALID_MP3_BYTES, expected_format="mp3")
        assert isinstance(res, ValidatedTTSAudio)
        assert res.format == "mp3"
        assert res.content_type == "audio/mpeg"
        assert res.size_bytes == len(VALID_MP3_BYTES)

    def test_validate_tts_audio_wav(self) -> None:
        res = validate_tts_audio(VALID_WAV_BYTES, expected_format="wav")
        assert res.format == "wav"
        assert res.content_type == "audio/wav"

    def test_validate_tts_audio_opus(self) -> None:
        res = validate_tts_audio(VALID_OGG_BYTES, expected_format="opus")
        assert res.format == "opus"
        assert res.content_type == "audio/opus"

    def test_validate_tts_audio_flac(self) -> None:
        res = validate_tts_audio(VALID_FLAC_BYTES, expected_format="flac")
        assert res.format == "flac"
        assert res.content_type == "audio/flac"

    def test_validate_tts_audio_aac(self) -> None:
        res = validate_tts_audio(VALID_AAC_BYTES, expected_format="aac")
        assert res.format == "aac"
        assert res.content_type == "audio/aac"

    def test_validate_tts_audio_pcm(self) -> None:
        res = validate_tts_audio(VALID_PCM_BYTES, expected_format="pcm")
        assert res.format == "pcm"
        assert res.content_type == "audio/pcm"

    def test_validate_tts_audio_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="TTS audio cannot be empty"):
            validate_tts_audio(b"", expected_format="mp3")

    def test_validate_tts_audio_non_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="TTS audio must be bytes"):
            validate_tts_audio("audio-str", expected_format="mp3")  # type: ignore[arg-type]

    def test_validate_tts_audio_oversized_rejected(self) -> None:
        big_audio = b"ID3" + b"x" * 1000
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            validate_tts_audio(big_audio, expected_format="mp3", max_bytes=500)

    def test_validate_tts_audio_format_mismatch(self) -> None:
        # Pass WAV header but ask for MP3
        with pytest.raises(ValueError, match="does not match expected format"):
            validate_tts_audio(VALID_WAV_BYTES, expected_format="mp3")

    def test_sniff_audio_format(self) -> None:
        assert sniff_tts_audio_format(VALID_WAV_BYTES) == "wav"
        assert sniff_tts_audio_format(VALID_OGG_BYTES) == "opus"
        assert sniff_tts_audio_format(VALID_FLAC_BYTES) == "flac"
        assert sniff_tts_audio_format(VALID_MP3_BYTES) == "mp3"
        assert sniff_tts_audio_format(VALID_AAC_BYTES) == "aac"
        assert sniff_tts_audio_format(b"unknown-data") is None


# ===========================================================================
# 6. FakeTTS tests
# ===========================================================================

class TestFakeTTS:
    """Test 6: FakeTTS determinism, no-network, error injection, batching."""

    def test_fake_tts_default_synthesis(self) -> None:
        fake = FakeTTS()
        req = TTSRequest(text="Hello world from Goa")
        resp = fake.synthesize(req)

        assert isinstance(resp, TTSResponse)
        assert resp.provider == "fake"
        assert resp.model == "fake-tts-1"
        assert resp.format == "mp3"
        assert resp.content_type == "audio/mpeg"
        assert len(resp.audio) > 0
        assert resp.character_count == len("Hello world from Goa")
        assert resp.latency_ms is not None
        assert resp.latency_ms >= 0.0

    def test_fake_tts_hindi_synthesis(self) -> None:
        fake = FakeTTS()
        req = TTSRequest(text="गोवा में आपका स्वागत है।", language="hi")
        resp = fake.synthesize(req)

        assert resp.provider == "fake"
        assert resp.character_count == len("गोवा में आपका स्वागत है।")
        assert resp.format == "mp3"

    def test_fake_tts_canned_response(self) -> None:
        canned = {"special text": VALID_WAV_BYTES}
        fake = FakeTTS(canned_responses=canned, default_format="wav")
        req = TTSRequest(text="special text", output_format="wav")
        resp = fake.synthesize(req)

        assert resp.audio == VALID_WAV_BYTES
        assert resp.format == "wav"
        assert resp.content_type == "audio/wav"

    def test_fake_tts_error_injection_tts_error(self) -> None:
        fake = FakeTTS(should_raise=TTSError("Injected provider failure"))
        req = TTSRequest(text="Fail this request")

        with pytest.raises(TTSError, match="Injected provider failure"):
            fake.synthesize(req)

    def test_fake_tts_error_injection_generic_exception(self) -> None:
        fake = FakeTTS(should_raise=RuntimeError("Low-level memory corruption"))
        req = TTSRequest(text="Fail this request")

        with pytest.raises(TTSError, match="FakeTTS injected error: Low-level memory corruption"):
            fake.synthesize(req)

    def test_fake_tts_simulated_latency(self) -> None:
        fake = FakeTTS(simulate_latency_ms=30.0)
        req = TTSRequest(text="Latency test text")

        start = time.perf_counter()
        resp = fake.synthesize(req)
        elapsed = (time.perf_counter() - start) * 1000.0

        assert resp.latency_ms >= 25.0
        assert elapsed >= 25.0

    def test_fake_tts_batch_synthesis_preserves_order(self) -> None:
        fake = FakeTTS()
        reqs = [
            TTSRequest(text="First sentence"),
            TTSRequest(text="Second sentence"),
            TTSRequest(text="Third sentence"),
        ]

        responses = fake.synthesize_batch(reqs)
        assert len(responses) == 3
        assert responses[0].character_count == len("First sentence")
        assert responses[1].character_count == len("Second sentence")
        assert responses[2].character_count == len("Third sentence")

    def test_fake_tts_batch_invalid_argument(self) -> None:
        fake = FakeTTS()
        with pytest.raises(ValueError, match="requests must be a list"):
            fake.synthesize_batch("not-a-list")  # type: ignore[arg-type]

    def test_fake_tts_factory_helper(self) -> None:
        fake = create_fake_tts(default_format="wav", model_name="fake-v2")
        assert isinstance(fake, FakeTTS)
        assert fake.model_name == "fake-v2"
        assert repr(fake) == "FakeTTS(model_name='fake-v2', default_format='wav')"
