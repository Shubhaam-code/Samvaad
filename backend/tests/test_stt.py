"""Unit tests for the STT package — interfaces, models, validation, FakeSTT.

These tests cover:
  1.  Package imports and __all__ exports
  2.  BaseSTT ABC enforcement (cannot instantiate directly)
  3.  STTProtocol duck-typing
  4.  STTRequest field validation
  5.  STTResponse field validation
  6.  STTConfig field validation
  7.  Empty audio rejection (validate_audio)
  8.  Unsupported MIME type rejection
  9.  Unsupported file extension rejection
  10. Maximum audio size enforcement
  11. Corrupt / unrecognised magic-bytes rejection
  12. Extension / declared-MIME mismatch rejection
  13. All supported formats pass sniffing
  14. MIME alias normalisation
  15. Language validation (base.validate_language)
  16. Context prompt validation (base.validate_context_prompt)
  17. Transcription text validation (base.validate_transcription_text)
  18. FakeSTT defaults (English)
  19. FakeSTT explicit Hindi response
  20. FakeSTT automatic language default (no language hint → "en")
  21. FakeSTT canned-response mapping by audio bytes
  22. FakeSTT STTError injection
  23. FakeSTT general Exception injection (wrapped in STTError)
  24. FakeSTT simulated latency
  25. FakeSTT respects request language hint
  26. STTConfig repr round-trips
  27. validate_audio_bytes helper
  28. sniff_audio_format helper
  29. canonicalize_mime helper
  30. Audio bytes not persisted (FakeSTT returns immediately)

Guarantees:
- Zero network calls.
- No real model or API key required.
- All audio is synthetic minimal bytes (no files committed to the repo).
"""

from __future__ import annotations

import pytest

from app.stt import (
    BaseSTT,
    FakeSTT,
    STTAudio,
    STTConfig,
    STTError,
    STTLanguage,
    STTProtocol,
    STTProvider,
    STTRequest,
    STTResponse,
    STTText,
    ValidatedAudio,
    canonicalize_mime,
    create_fake_stt,
    sniff_audio_format,
    validate_audio,
    validate_audio_bytes,
    validate_context_prompt,
    validate_language,
    validate_transcription_text,
)

# ---------------------------------------------------------------------------
# Minimal valid audio byte fixtures
# ---------------------------------------------------------------------------
# Each is the smallest valid container header for its format.
# They are NOT real audio files — just enough bytes to satisfy the
# magic-byte sniff used by validate_audio().

VALID_WAV = (
    b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    b"\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00"
    b"\x02\x00\x10\x00data\x00\x00\x00\x00"
)
VALID_OGG = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
VALID_WEBM = b"\x1a\x45\xdf\xa3\x99\x42\x86\x81\x01\x42\xf7\x81\x01"
VALID_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00"
VALID_MP4 = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2"
VALID_AAC = b"\xff\xf1\x50\x80\x00\x1f\xfc"  # ADTS sync word (0xFFF1) + AAC header


# ===========================================================================
# 1. Package imports
# ===========================================================================

class TestSTTImports:
    """Test 1: Package imports and __all__ exports."""

    def test_core_imports_exist(self) -> None:
        assert BaseSTT is not None
        assert STTError is not None
        assert STTProtocol is not None
        assert STTRequest is not None
        assert STTResponse is not None
        assert STTConfig is not None
        assert FakeSTT is not None

    def test_type_aliases_importable(self) -> None:
        assert STTAudio is not None
        assert STTText is not None
        assert STTLanguage is not None

    def test_validation_helpers_importable(self) -> None:
        assert validate_audio is not None
        assert validate_audio_bytes is not None
        assert validate_language is not None
        assert validate_context_prompt is not None
        assert validate_transcription_text is not None
        assert canonicalize_mime is not None
        assert sniff_audio_format is not None


# ===========================================================================
# 2. BaseSTT ABC enforcement
# ===========================================================================

class TestBaseSTTAbstractEnforcement:
    """Test 2: BaseSTT cannot be instantiated directly."""

    def test_cannot_instantiate_base_stt(self) -> None:
        with pytest.raises(TypeError):
            BaseSTT()  # type: ignore[abstract]


# ===========================================================================
# 3. STTProtocol duck-typing
# ===========================================================================

class TestSTTProtocol:
    """Test 3: Duck-typed providers satisfy STTProtocol without inheriting."""

    def test_duck_typed_provider_satisfies_protocol(self) -> None:
        class MinimalProvider:
            def transcribe(self, request: object) -> object:
                return STTResponse(text="ok", provider="custom")

            @property
            def model_name(self) -> str:
                return "custom-model"

            @property
            def provider(self) -> str:
                return "custom"

        p = MinimalProvider()
        assert isinstance(p, STTProtocol)
        resp = p.transcribe(object())
        assert resp.text == "ok"  # type: ignore[union-attr]


# ===========================================================================
# 4. STTRequest validation
# ===========================================================================

class TestSTTRequestValidation:
    """Test 4: STTRequest field constraints."""

    def test_valid_minimal_request(self) -> None:
        req = STTRequest(audio=VALID_WAV, filename="audio.wav")
        assert req.audio == VALID_WAV
        assert req.filename == "audio.wav"
        assert req.language is None
        assert req.content_type is None
        assert req.prompt is None

    def test_valid_full_request(self) -> None:
        req = STTRequest(
            audio=VALID_WAV,
            filename="audio.wav",
            content_type="audio/wav",
            language="en",
            prompt="Goa beaches",
        )
        assert req.language == "en"
        assert req.prompt == "Goa beaches"

    def test_language_normalised_to_lowercase(self) -> None:
        req = STTRequest(audio=VALID_WAV, filename="audio.wav", language="HI")
        assert req.language == "hi"

    def test_language_region_code_accepted(self) -> None:
        req = STTRequest(audio=VALID_WAV, filename="audio.wav", language="en-US")
        assert req.language == "en-us"

    def test_invalid_language_raises(self) -> None:
        with pytest.raises(ValueError, match="ISO 639-1"):
            STTRequest(audio=VALID_WAV, filename="audio.wav", language="not_a_lang_code!")

    def test_empty_audio_rejected(self) -> None:
        with pytest.raises(ValueError):
            STTRequest(audio=b"", filename="audio.wav")

    def test_whitespace_filename_rejected(self) -> None:
        with pytest.raises(ValueError):
            STTRequest(audio=VALID_WAV, filename="   ")

    def test_empty_prompt_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            STTRequest(audio=VALID_WAV, filename="audio.wav", prompt="   ")


# ===========================================================================
# 5. STTResponse validation
# ===========================================================================

class TestSTTResponseValidation:
    """Test 5: STTResponse field constraints."""

    def test_valid_response(self) -> None:
        resp = STTResponse(
            text="Hello world",
            language="en",
            provider="openai_whisper",
            model="whisper-1",
            latency_ms=45.3,
            duration_seconds=2.1,
        )
        assert resp.text == "Hello world"
        assert resp.confidence is None  # never invented

    def test_whitespace_only_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="Transcription text cannot be empty"):
            STTResponse(text="   ")

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValueError):
            STTResponse(text="")

    def test_optional_fields_are_none_by_default(self) -> None:
        resp = STTResponse(text="ok")
        assert resp.language is None
        assert resp.provider is None
        assert resp.model is None
        assert resp.latency_ms is None
        assert resp.duration_seconds is None
        assert resp.confidence is None

    def test_latency_ms_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError):
            STTResponse(text="ok", latency_ms=-1.0)

    def test_confidence_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            STTResponse(text="ok", confidence=1.5)
        with pytest.raises(ValueError):
            STTResponse(text="ok", confidence=-0.1)


# ===========================================================================
# 6. STTConfig validation
# ===========================================================================

class TestSTTConfigValidation:
    """Test 6: STTConfig field constraints."""

    def test_default_config(self) -> None:
        config = STTConfig()
        assert config.provider == STTProvider.FAKE
        assert config.model_name is None
        assert config.language is None
        assert config.timeout_seconds == 30.0
        assert config.max_audio_size_mb == 10.0

    def test_custom_config(self) -> None:
        config = STTConfig(
            provider=STTProvider.OPENAI_COMPATIBLE,
            model_name="whisper-1",
            language="hi",
            timeout_seconds=15.0,
            max_audio_size_mb=5.0,
        )
        assert config.provider == STTProvider.OPENAI_COMPATIBLE
        assert config.model_name == "whisper-1"
        assert config.language == "hi"

    def test_invalid_language_rejected(self) -> None:
        with pytest.raises(ValueError, match="ISO 639-1"):
            STTConfig(language="not-valid!")

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValueError):
            STTConfig(timeout_seconds=0.0)

    def test_repr_contains_provider(self) -> None:
        config = STTConfig(provider=STTProvider.FAKE)
        assert "fake" in repr(config)


# ===========================================================================
# 7–14. Audio validation
# ===========================================================================

class TestAudioValidation:
    """Tests 7–14: validate_audio() strict upload validation."""

    # 7. Empty audio rejection
    def test_empty_audio_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_audio(b"", filename="audio.wav")

    # 8. Unsupported MIME type
    def test_unsupported_mime_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported audio MIME type"):
            validate_audio(VALID_WAV, filename="audio.wav", content_type="video/mp4")

    # 9. Unsupported file extension
    def test_unsupported_extension_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported audio format"):
            validate_audio(VALID_WAV, filename="audio.txt")

    # 10. Maximum file size enforcement
    def test_maximum_size_enforced(self) -> None:
        oversized = VALID_WAV + b"\x00" * 100
        with pytest.raises(ValueError, match="exceeds the maximum allowed size"):
            validate_audio(oversized, filename="audio.wav", max_bytes=len(VALID_WAV))

    # 11. Corrupt / unrecognised magic bytes
    def test_corrupt_audio_rejected(self) -> None:
        corrupt = b"THIS_IS_DEFINITELY_NOT_AUDIO_DATA_AT_ALL!!!"
        with pytest.raises(ValueError, match="unrecognized or corrupt"):
            validate_audio(corrupt, filename="audio.wav")

    # 12. Extension / sniffed format mismatch
    def test_extension_magic_mismatch_rejected(self) -> None:
        # WAV header bytes but extension claims .mp3
        with pytest.raises(ValueError, match="does not match the declared format"):
            validate_audio(VALID_WAV, filename="audio.mp3")

    # 13. All supported formats pass sniffing
    def test_all_supported_formats_accepted(self) -> None:
        assert validate_audio(VALID_WAV, "audio.wav").content_type == "audio/wav"
        assert validate_audio(VALID_OGG, "audio.ogg").content_type == "audio/ogg"
        assert validate_audio(VALID_WEBM, "audio.webm").content_type == "audio/webm"
        assert validate_audio(VALID_MP3, "audio.mp3").content_type == "audio/mpeg"
        assert validate_audio(VALID_MP4, "audio.m4a").content_type == "audio/mp4"
        assert validate_audio(VALID_AAC, "audio.aac").content_type == "audio/aac"

    # 14. MIME alias normalisation
    def test_mime_aliases_normalised(self) -> None:
        assert canonicalize_mime("audio/x-wav") == "audio/wav"
        assert canonicalize_mime("audio/wave") == "audio/wav"
        assert canonicalize_mime("audio/mp3") == "audio/mpeg"
        assert canonicalize_mime("audio/x-mpeg") == "audio/mpeg"
        assert canonicalize_mime("audio/x-m4a") == "audio/mp4"
        assert canonicalize_mime("audio/m4a") == "audio/mp4"
        assert canonicalize_mime("audio/opus") == "audio/ogg"
        # Parameters stripped
        assert canonicalize_mime("audio/webm;codecs=opus") == "audio/webm"
        # Unknown type returns None
        assert canonicalize_mime("video/mp4") is None
        assert canonicalize_mime("") is None


# ===========================================================================
# 15–17. Shared validation helpers
# ===========================================================================

class TestValidationHelpers:
    """Tests 15–17: base.py module-level validators."""

    # 15. validate_language
    def test_validate_language_none(self) -> None:
        assert validate_language(None) is None

    def test_validate_language_valid_codes(self) -> None:
        assert validate_language("en") == "en"
        assert validate_language("HI") == "hi"
        assert validate_language("zh") == "zh"

    def test_validate_language_invalid_codes(self) -> None:
        with pytest.raises(ValueError):
            validate_language("invalid")
        with pytest.raises(ValueError):
            validate_language("12")

    # 16. validate_context_prompt
    def test_validate_context_prompt_none(self) -> None:
        assert validate_context_prompt(None) is None

    def test_validate_context_prompt_valid(self) -> None:
        assert validate_context_prompt("Goa tourism") == "Goa tourism"

    def test_validate_context_prompt_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_context_prompt("  ")

    # 17. validate_transcription_text
    def test_validate_transcription_text_valid(self) -> None:
        assert validate_transcription_text("Hello") == "Hello"

    def test_validate_transcription_text_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_transcription_text("  ")

    def test_validate_transcription_text_non_str_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_transcription_text(42)  # type: ignore[arg-type]


# ===========================================================================
# 18–26. validate_audio_bytes and sniff_audio_format helpers
# ===========================================================================

class TestAudioBytesAndSniffHelpers:
    """Tests 27–29: validate_audio_bytes and sniff_audio_format standalone."""

    def test_validate_audio_bytes_valid(self) -> None:
        assert validate_audio_bytes(b"\x00\x01") == b"\x00\x01"

    def test_validate_audio_bytes_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_audio_bytes(b"")

    def test_validate_audio_bytes_non_bytes_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_audio_bytes("not bytes")  # type: ignore[arg-type]

    def test_sniff_returns_wav(self) -> None:
        assert sniff_audio_format(VALID_WAV) == "audio/wav"

    def test_sniff_returns_ogg(self) -> None:
        assert sniff_audio_format(VALID_OGG) == "audio/ogg"

    def test_sniff_returns_webm(self) -> None:
        assert sniff_audio_format(VALID_WEBM) == "audio/webm"

    def test_sniff_returns_mp3(self) -> None:
        assert sniff_audio_format(VALID_MP3) == "audio/mpeg"

    def test_sniff_returns_mp4(self) -> None:
        assert sniff_audio_format(VALID_MP4) == "audio/mp4"

    def test_sniff_returns_aac(self) -> None:
        assert sniff_audio_format(VALID_AAC) == "audio/aac"

    def test_sniff_returns_none_for_unknown(self) -> None:
        assert sniff_audio_format(b"UNKNOWNJUNK") is None

    def test_sniff_returns_none_for_empty(self) -> None:
        assert sniff_audio_format(b"") is None


# ===========================================================================
# FakeSTT provider
# ===========================================================================

class TestFakeSTTProvider:
    """Tests 18–26: FakeSTT deterministic provider behaviour."""

    # 18. Default English transcription
    def test_fake_stt_default_transcription(self) -> None:
        provider = create_fake_stt()
        assert provider.provider == "fake"
        assert provider.model_name == "fake-whisper"

        req = STTRequest(audio=VALID_WAV, filename="audio.wav")
        resp = provider.transcribe(req)

        assert isinstance(resp, STTResponse)
        assert resp.text == "This is a fake transcription of the provided audio."
        assert resp.language == "en"
        assert resp.provider == "fake"
        assert resp.model == "fake-whisper"
        assert resp.latency_ms is not None
        assert resp.latency_ms >= 0.0

    # 19. Explicit Hindi transcription
    def test_fake_stt_hindi(self) -> None:
        provider = create_fake_stt(default_text="नमस्ते, गोवा में आपका स्वागत है।")
        req = STTRequest(audio=VALID_WAV, filename="audio.wav", language="hi")
        resp = provider.transcribe(req)
        assert resp.text == "नमस्ते, गोवा में आपका स्वागत है।"
        assert resp.language == "hi"

    # 20. Automatic language default when no hint supplied
    def test_fake_stt_default_language_is_en(self) -> None:
        provider = create_fake_stt()
        resp = provider.transcribe(STTRequest(audio=VALID_WAV, filename="audio.wav"))
        assert resp.language == "en"

    # 21. Canned responses by audio bytes
    def test_fake_stt_canned_responses(self) -> None:
        canned = {VALID_WAV: "Specific WAV text", VALID_OGG: "Specific OGG text"}
        provider = create_fake_stt(canned_responses=canned)

        resp_wav = provider.transcribe(STTRequest(audio=VALID_WAV, filename="audio.wav"))
        assert resp_wav.text == "Specific WAV text"

        resp_ogg = provider.transcribe(STTRequest(audio=VALID_OGG, filename="audio.ogg"))
        assert resp_ogg.text == "Specific OGG text"

        # Unmatched bytes → default
        resp_mp3 = provider.transcribe(STTRequest(audio=VALID_MP3, filename="audio.mp3"))
        assert resp_mp3.text == "This is a fake transcription of the provided audio."

    # 22. STTError injection
    def test_fake_stt_stt_error_injection(self) -> None:
        err = STTError("Simulated provider failure")
        provider = create_fake_stt(should_raise=err)
        with pytest.raises(STTError, match="Simulated provider failure"):
            provider.transcribe(STTRequest(audio=VALID_WAV, filename="audio.wav"))

    # 23. Generic Exception injection (wrapped in STTError)
    def test_fake_stt_generic_exception_wrapped(self) -> None:
        provider = create_fake_stt(should_raise=RuntimeError("disk error"))
        with pytest.raises(STTError, match="disk error"):
            provider.transcribe(STTRequest(audio=VALID_WAV, filename="audio.wav"))

    # 24. Simulated latency
    def test_fake_stt_simulated_latency_nonzero(self) -> None:
        provider = create_fake_stt(simulate_latency_ms=50.0)
        req = STTRequest(audio=VALID_WAV, filename="audio.wav")
        resp = provider.transcribe(req)
        assert resp.latency_ms is not None
        assert resp.latency_ms >= 50.0

    # 25. Request language hint respected
    def test_fake_stt_request_language_overrides_default(self) -> None:
        provider = create_fake_stt(language="en")
        req = STTRequest(audio=VALID_WAV, filename="audio.wav", language="hi")
        resp = provider.transcribe(req)
        assert resp.language == "hi"

    # 26. Audio validation runs inside FakeSTT (same rules as real providers)
    def test_fake_stt_rejects_empty_audio(self) -> None:
        provider = create_fake_stt()
        # Cannot use STTRequest(audio=b"") — model already rejects it
        # So pass via duck-typed object to reach validate_audio inside transcribe
        class BadRequest:
            audio = b""
            filename = "audio.wav"
            content_type = None
            language = None

        with pytest.raises(ValueError, match="cannot be empty"):
            provider.transcribe(BadRequest())

    def test_fake_stt_rejects_unsupported_format(self) -> None:
        provider = create_fake_stt()

        class BadRequest:
            audio = VALID_WAV
            filename = "audio.txt"
            content_type = None
            language = None

        with pytest.raises(ValueError, match="unsupported audio format"):
            provider.transcribe(BadRequest())

    # 30. Audio bytes not persisted (no external side-effects)
    def test_fake_stt_no_persistence(self) -> None:
        """FakeSTT must not write to disk or retain audio beyond the call."""
        import os
        import tempfile
        tmp_dir = tempfile.gettempdir()
        files_before = set(os.listdir(tmp_dir))
        provider = create_fake_stt()
        provider.transcribe(STTRequest(audio=VALID_WAV, filename="audio.wav"))
        files_after = set(os.listdir(tmp_dir))
        assert files_before == files_after, "FakeSTT must not write temporary files"
