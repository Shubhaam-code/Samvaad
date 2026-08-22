"""Base STT interface and shared validation rules.

Defines the provider-agnostic speech-to-text contract that all concrete
implementations (fake, OpenAI-compatible API, local model) must follow:

- ``transcribe(request)``: one audio request -> one transcription
- ``model_name``: optional model identifier (None until a model is known)
- ``provider``: provider name reported on responses

Shared validation helpers are module-level functions so future
providers and callers reuse the exact same rules.

Phase 7.1: Interface definition + validation only (no production provider).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

from .types import STTAudio, STTText

# ISO 639-1/2 language codes, optionally with a region suffix (e.g. "en-US").
_LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}(-[a-z]{2})?$")


class STTError(Exception):
    """Custom exception raised for STT provider/harness failures.

    Mirrors ``LLMError`` in the LLM layer: concrete providers should
    wrap provider-specific failures in this exception so callers never
    depend on a specific SDK's error types.
    """
    pass


class NoSpeechDetectedError(STTError):
    """Raised when a provider transcribes audio but finds no speech in it.

    This is a property of the caller's audio, not a provider failure, so
    endpoints map it to HTTP 400 rather than 500. It subclasses
    ``STTError`` so existing broad handlers keep working.
    """
    pass


def validate_audio_bytes(audio: object) -> bytes:
    """Validate a single audio input blob.

    Rules:
    - Must be bytes
    - Must not be empty

    Args:
        audio: Raw audio bytes

    Returns:
        The validated audio (unchanged)

    Raises:
        ValueError: If audio is not bytes or is empty
    """
    if not isinstance(audio, bytes):
        raise ValueError(f"STT audio must be bytes, got {type(audio).__name__}")
    if not audio:
        raise ValueError("STT audio cannot be empty")
    return audio


def validate_language(language: Optional[str]) -> Optional[str]:
    """Validate an optional language code.

    Rules:
    - Must be None or a string
    - A provided string must match an ISO 639-1/2 code pattern
      (``en``, ``hi``, ``en-US``, ...), lowercased

    Args:
        language: Optional language hint (None = automatic detection)

    Returns:
        The normalized (lowercased) language, or None

    Raises:
        ValueError: If language is provided but invalid
    """
    if language is None:
        return None
    if not isinstance(language, str):
        raise ValueError(
            f"language must be a string or None, got {type(language).__name__}"
        )
    normalized = language.strip().lower()
    if not _LANGUAGE_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"language must be an ISO 639-1/2 code (e.g. 'en', 'hi', 'en-US'), "
            f"got {language!r}"
        )
    return normalized


def validate_context_prompt(prompt: Optional[str]) -> Optional[str]:
    """Validate an optional transcription context prompt.

    Rules:
    - Must be None or a string
    - A provided string must not be empty or whitespace-only

    Args:
        prompt: Optional context/boosting prompt (provider dependent)

    Returns:
        The validated prompt (unchanged), or None

    Raises:
        ValueError: If prompt is provided but invalid
    """
    if prompt is None:
        return None
    if not isinstance(prompt, str):
        raise ValueError(f"prompt must be a string or None, got {type(prompt).__name__}")
    if not prompt or not prompt.strip():
        raise ValueError("prompt cannot be empty or whitespace-only")
    return prompt


def validate_transcription_text(text: object) -> str:
    """Validate a produced transcription output text.

    Rules:
    - Must be a string
    - Must not be empty
    - Must not be whitespace-only

    Args:
        text: Transcribed text to validate

    Returns:
        The validated text (unchanged)

    Raises:
        ValueError: If text is not a string, empty, or whitespace-only
    """
    if not isinstance(text, str):
        raise ValueError(
            f"Transcription text must be a string, got {type(text).__name__}"
        )
    if not text or not text.strip():
        raise ValueError("Transcription text cannot be empty or whitespace-only")
    return text


class BaseSTT(ABC):
    """Abstract base class for all STT providers.

    Concrete implementations will include:
    - FakeSTT: deterministic offline provider for tests (Phase 7.1)
    - OpenAI-compatible API provider (Phase 7.2)
    - Local model provider (planned)

    All providers must implement transcribe(), must accept an
    STTRequest-like object with ``audio`` bytes, and must report model
    and provider identifiers.

    Phase 7.1: Base interface only (no production provider).
    """

    @abstractmethod
    def transcribe(self, request: object) -> object:
        """Transcribe a single audio request.

        Args:
            request: STTRequest (or duck-typed request-like object) with
                a non-empty ``audio`` bytes attribute

        Returns:
            STTResponse containing the transcription text and metadata

        Raises:
            ValueError: If the request is missing or has invalid audio
            STTError: If the underlying provider fails
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> Optional[str]:
        """Name/identifier of the model used by this provider."""
        pass

    @property
    @abstractmethod
    def provider(self) -> Optional[str]:
        """Provider name reported on transcriptions (e.g. 'fake')."""
        pass


@runtime_checkable
class STTProtocol(Protocol):
    """Protocol defining the STT interface for type checking.

    Allows duck-typed provider implementations that don't explicitly
    inherit from BaseSTT but still follow the contract.
    """

    def transcribe(self, request: object) -> object:
        """Transcribe a single audio request."""
        ...

    @property
    def model_name(self) -> Optional[str]:
        """Name/identifier of the model used by this provider."""
        ...

    @property
    def provider(self) -> Optional[str]:
        """Provider name reported on transcriptions."""
        ...


__all__ = [
    "BaseSTT",
    "STTError",
    "STTProtocol",
    "validate_audio_bytes",
    "validate_context_prompt",
    "validate_language",
    "validate_transcription_text",
]