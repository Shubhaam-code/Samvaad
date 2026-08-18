"""STT (Speech-to-Text) package for the HH Goa 2026 Voice-enabled RAG system.

Production-quality STT component supporting English, Hindi, and automatic
language detection via OpenAI Whisper (hosted or OpenAI-compatible endpoints).

Sub-modules
-----------
- types      : type aliases  (STTAudio, STTText, STTLanguage)
- base       : BaseSTT ABC, STTProtocol, STTError, shared validators
- models     : STTRequest, STTResponse (Pydantic v2)
- config     : STTConfig, STTProvider enum
- validation : strict audio upload validation (size, extension, MIME, magic)
- fake       : FakeSTT — deterministic offline provider for tests ONLY
- openai_whisper : OpenAIWhisperSTT — production provider via OpenAI SDK v1.x

Production wiring
-----------------
get_stt() in app/api/dependencies.py returns an OpenAIWhisperSTT instance when
the provider is configured, or None when unconfigured.  FakeSTT is NEVER
returned by get_stt().
"""

from __future__ import annotations

from .base import (
    BaseSTT,
    STTError,
    STTProtocol,
    validate_audio_bytes,
    validate_context_prompt,
    validate_language,
    validate_transcription_text,
)
from .config import STTConfig, STTProvider
from .fake import FakeSTT, create_fake_stt
from .models import STTRequest, STTResponse
from .openai_whisper import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_AUDIO_SIZE_MB,
    DEFAULT_MODEL_NAME,
    DEFAULT_TIMEOUT_SECONDS,
    OpenAIWhisperSTT,
    create_openai_whisper_stt,
    is_openai_whisper_configured,
)
from .types import STTAudio, STTLanguage, STTText
from .validation import (
    DEFAULT_MAX_AUDIO_BYTES,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MIME_TYPES,
    ValidatedAudio,
    canonicalize_mime,
    sniff_audio_format,
    validate_audio,
)

__all__ = [
    # Type aliases
    "STTAudio",
    "STTText",
    "STTLanguage",
    # Base interface
    "BaseSTT",
    "STTProtocol",
    "STTError",
    "validate_audio_bytes",
    "validate_language",
    "validate_context_prompt",
    "validate_transcription_text",
    # Data models
    "STTRequest",
    "STTResponse",
    # Configuration
    "STTConfig",
    "STTProvider",
    # Audio Validation
    "ValidatedAudio",
    "SUPPORTED_MIME_TYPES",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_MAX_AUDIO_BYTES",
    "canonicalize_mime",
    "sniff_audio_format",
    "validate_audio",
    # Fake provider (tests/offline dev only — never used in production)
    "FakeSTT",
    "create_fake_stt",
    # OpenAI Whisper provider (production)
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_AUDIO_SIZE_MB",
    "OpenAIWhisperSTT",
    "create_openai_whisper_stt",
    "is_openai_whisper_configured",
]
