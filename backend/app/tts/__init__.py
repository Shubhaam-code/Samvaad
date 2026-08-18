"""TTS (Text-to-Speech) package for the HH Goa 2026 Voice-enabled RAG system.

Production-quality TTS component supporting multiple output formats ('mp3', 'opus',
'aac', 'flac', 'wav', 'pcm'), multiple voices, speed control, and OpenAI-compatible
TTS endpoints.

Sub-modules
-----------
- types      : type aliases (TTSText, TTSAudio, TTSVoice, TTSModel, TTSFormat, TTSLanguage)
- base       : BaseTTS ABC, TTSProtocol, TTSError, shared validators
- models     : TTSRequest, TTSResponse (Pydantic v2)
- config     : TTSConfig, TTSProvider enum
- validation : strict audio output validation (size, MIME, magic bytes)
- fake       : FakeTTS — deterministic offline provider for tests ONLY
- openai_tts : OpenAITTS — production provider via OpenAI SDK v1.x

Production wiring
-----------------
get_tts() in app/api/dependencies.py returns an OpenAITTS instance when
the provider is configured, or None when unconfigured. FakeTTS is NEVER
returned by get_tts().
"""

from __future__ import annotations

from .base import (
    DEFAULT_MAX_TEXT_LENGTH,
    MAX_SPEED,
    MIN_SPEED,
    SUPPORTED_FORMATS,
    BaseTTS,
    TTSError,
    TTSProtocol,
    validate_language,
    validate_output_format,
    validate_speed,
    validate_text,
    validate_voice,
)
from .config import TTSConfig, TTSProvider
from .fake import FakeTTS, create_fake_tts
from .models import TTSRequest, TTSResponse
from .openai_tts import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_AUDIO_SIZE_MB,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SPEED,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_VOICE,
    OpenAITTS,
    create_openai_tts,
    is_openai_tts_configured,
)
from .sarvam_tts import (
    DEFAULT_SARVAM_SPEAKER,
    DEFAULT_SARVAM_TTS_ENDPOINT,
    DEFAULT_SARVAM_TTS_MODEL,
    SarvamTTS,
    is_sarvam_tts_configured,
)
from .types import (
    TTSAudio,
    TTSFormat,
    TTSLanguage,
    TTSModel,
    TTSText,
    TTSVoice,
)
from .validation import (
    DEFAULT_MAX_TTS_AUDIO_BYTES,
    FORMAT_TO_MIME,
    SUPPORTED_TTS_MIMES,
    ValidatedTTSAudio,
    sniff_tts_audio_format,
    validate_tts_audio,
)

__all__ = [
    # Type aliases
    "TTSText",
    "TTSAudio",
    "TTSVoice",
    "TTSModel",
    "TTSFormat",
    "TTSLanguage",
    # Base interface & validation
    "BaseTTS",
    "TTSProtocol",
    "TTSError",
    "SUPPORTED_FORMATS",
    "MIN_SPEED",
    "MAX_SPEED",
    "DEFAULT_MAX_TEXT_LENGTH",
    "validate_text",
    "validate_voice",
    "validate_speed",
    "validate_output_format",
    "validate_language",
    # Data models
    "TTSRequest",
    "TTSResponse",
    # Configuration
    "TTSConfig",
    "TTSProvider",
    # Audio Validation
    "FORMAT_TO_MIME",
    "SUPPORTED_TTS_MIMES",
    "DEFAULT_MAX_TTS_AUDIO_BYTES",
    "ValidatedTTSAudio",
    "sniff_tts_audio_format",
    "validate_tts_audio",
    # Fake provider (tests ONLY — never used in production)
    "FakeTTS",
    "create_fake_tts",
    # OpenAI TTS provider
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_VOICE",
    "DEFAULT_OUTPUT_FORMAT",
    "DEFAULT_SPEED",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_AUDIO_SIZE_MB",
    "OpenAITTS",
    "create_openai_tts",
    "is_openai_tts_configured",
    # Sarvam TTS provider
    "DEFAULT_SARVAM_SPEAKER",
    "DEFAULT_SARVAM_TTS_ENDPOINT",
    "DEFAULT_SARVAM_TTS_MODEL",
    "SarvamTTS",
    "is_sarvam_tts_configured",
]
