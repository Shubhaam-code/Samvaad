"""Base TTS interface and shared validation rules.

Defines the provider-agnostic text-to-speech contract that all concrete
implementations (fake, OpenAI-compatible API, local model) must follow:

- ``synthesize(request)``: one text request -> one audio response
- ``synthesize_batch(requests)``: multiple text requests -> multiple audio responses
- ``model_name``: optional model identifier
- ``provider``: provider name reported on responses

Shared validation helpers are module-level functions so future
providers and callers reuse the exact same rules.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

# ISO 639-1/2 language codes, optionally with a region suffix (e.g. "en-US").
_LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}(-[a-z]{2})?$")

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {"mp3", "opus", "aac", "flac", "wav", "pcm"}
)

MIN_SPEED: float = 0.25
MAX_SPEED: float = 4.0
DEFAULT_MAX_TEXT_LENGTH: int = 4096


class TTSError(Exception):
    """Custom exception raised for TTS provider/harness failures.

    Mirrors ``STTError`` and ``LLMError``: concrete providers wrap
    provider-specific failures in this exception so callers never
    depend on a specific SDK's error types.
    """
    pass


def validate_text(text: object, max_length: int = DEFAULT_MAX_TEXT_LENGTH) -> str:
    """Validate a text input for synthesis.

    Rules:
    - Must be a string
    - Must not be empty or whitespace-only
    - Must not exceed max_length characters

    Args:
        text: Input text string
        max_length: Maximum allowed character length

    Returns:
        The validated text string (stripped of leading/trailing whitespace)

    Raises:
        ValueError: If text is invalid or exceeds bounds
    """
    if not isinstance(text, str):
        raise ValueError(f"TTS text must be a string, got {type(text).__name__}")
    stripped = text.strip()
    if not stripped:
        raise ValueError("TTS text cannot be empty or whitespace-only")
    if len(text) > max_length:
        raise ValueError(
            f"TTS text exceeds maximum allowed length of {max_length} characters "
            f"(got {len(text)} characters)"
        )
    return stripped


def validate_voice(voice: object) -> str:
    """Validate a voice identifier.

    Rules:
    - Must be a non-empty string

    Args:
        voice: Voice identifier

    Returns:
        The validated voice string (stripped)

    Raises:
        ValueError: If voice is not a non-empty string
    """
    if not isinstance(voice, str):
        raise ValueError(f"TTS voice must be a string, got {type(voice).__name__}")
    stripped = voice.strip()
    if not stripped:
        raise ValueError("TTS voice cannot be empty or whitespace-only")
    return stripped


def validate_speed(speed: object) -> float:
    """Validate a speech playback speed multiplier.

    Rules:
    - Must be a number (float or int, not bool)
    - Must be in [0.25, 4.0]

    Args:
        speed: Speed multiplier

    Returns:
        The validated speed as float

    Raises:
        ValueError: If speed is not a valid number or out of bounds
    """
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ValueError(f"TTS speed must be a number, got {type(speed).__name__}")
    val = float(speed)
    if val < MIN_SPEED or val > MAX_SPEED:
        raise ValueError(
            f"TTS speed must be between {MIN_SPEED} and {MAX_SPEED}, got {val}"
        )
    return val


def validate_output_format(output_format: object) -> str:
    """Validate an audio output format identifier.

    Rules:
    - Must be a string
    - Must be one of the supported formats ('mp3', 'opus', 'aac', 'flac', 'wav', 'pcm')

    Args:
        output_format: Format identifier

    Returns:
        The canonical lowercase format string

    Raises:
        ValueError: If output_format is unsupported or invalid
    """
    if not isinstance(output_format, str):
        raise ValueError(
            f"TTS output format must be a string, got {type(output_format).__name__}"
        )
    normalized = output_format.strip().lower()
    if normalized not in SUPPORTED_FORMATS:
        supported_str = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"unsupported TTS output format: {output_format!r} (supported: {supported_str})"
        )
    return normalized


def validate_language(language: Optional[str]) -> Optional[str]:
    """Validate an optional language code.

    Rules:
    - Must be None or a string
    - A provided string must match an ISO 639-1/2 code pattern
      (``en``, ``hi``, ``en-US``, ...), lowercased

    Args:
        language: Optional language hint (None = provider default)

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


class BaseTTS(ABC):
    """Abstract base class for all TTS providers.

    Concrete implementations include:
    - FakeTTS: deterministic offline provider for tests
    - OpenAITTS: OpenAI-compatible API provider (tts-1 / tts-1-hd)
    - Local TTS: planned local model provider

    All providers must implement synthesize(), must accept a
    TTSRequest-like object with ``text``, and must report model
    and provider identifiers.
    """

    @abstractmethod
    def synthesize(self, request: object) -> object:
        """Synthesize a single text request into audio.

        Args:
            request: TTSRequest (or duck-typed request-like object) with
                a non-empty ``text`` attribute.

        Returns:
            TTSResponse containing the audio bytes and metadata.

        Raises:
            ValueError: If the request is missing or has invalid attributes.
            TTSError: If the underlying provider fails.
        """
        pass

    def synthesize_batch(self, requests: list[object]) -> list[object]:
        """Synthesize multiple text requests in batch.

        Default implementation executes requests sequentially preserving order.
        Subclasses can override to implement batching / concurrency if supported.

        Args:
            requests: List of TTSRequest-like objects.

        Returns:
            List of TTSResponse-like objects in the same order.

        Raises:
            ValueError: If requests is not a list or contains invalid requests.
            TTSError: If any provider call fails.
        """
        if not isinstance(requests, list):
            raise ValueError(f"requests must be a list, got {type(requests).__name__}")
        return [self.synthesize(req) for req in requests]

    @property
    @abstractmethod
    def model_name(self) -> Optional[str]:
        """Name/identifier of the model used by this provider."""
        pass

    @property
    @abstractmethod
    def provider(self) -> Optional[str]:
        """Provider name reported on responses (e.g. 'openai_tts', 'fake')."""
        pass


@runtime_checkable
class TTSProtocol(Protocol):
    """Protocol defining the TTS interface for type checking.

    Allows duck-typed provider implementations that don't explicitly
    inherit from BaseTTS but still follow the contract.
    """

    def synthesize(self, request: object) -> object:
        """Synthesize a single text request into audio."""
        ...

    def synthesize_batch(self, requests: list[object]) -> list[object]:
        """Synthesize multiple text requests in batch."""
        ...

    @property
    def model_name(self) -> Optional[str]:
        """Name/identifier of the model used by this provider."""
        ...

    @property
    def provider(self) -> Optional[str]:
        """Provider name reported on responses."""
        ...


__all__ = [
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
]
