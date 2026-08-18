"""Data models for the TTS harness.

Defines the canonical request/response structures exchanged between
callers and TTS providers. Models are provider-agnostic:
any future provider (hosted API, local model) can map its native
request/response shapes onto these models.

Audio is handled as transient in-memory ``bytes`` and is never permanently
persisted.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import (
    DEFAULT_MAX_TEXT_LENGTH,
    MAX_SPEED,
    MIN_SPEED,
    SUPPORTED_FORMATS,
    validate_language,
    validate_output_format,
    validate_speed,
    validate_text,
    validate_voice,
)
from .types import TTSAudio, TTSFormat, TTSLanguage, TTSModel, TTSText, TTSVoice


class TTSRequest(BaseModel):
    """A single text-to-speech synthesis request.

    Attributes:
        text: Text to synthesize (non-empty, length bounded)
        voice: Voice identifier (default: 'alloy')
        model: Optional model identifier override
        output_format: Audio format (e.g. 'mp3', 'wav', 'opus')
        speed: Speech playback speed multiplier in [0.25, 4.0]
        language: Optional language hint (ISO 639-1/2 code)
    """

    model_config = ConfigDict(protected_namespaces=())

    text: TTSText = Field(
        ...,
        min_length=1,
        description="Text to synthesize into speech (non-empty, length bounded)",
    )
    voice: TTSVoice = Field(
        "alloy",
        min_length=1,
        description="Voice identifier (e.g. 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')",
    )
    model: Optional[TTSModel] = Field(
        None,
        description="Optional model identifier override (e.g. 'tts-1', 'tts-1-hd')",
    )
    output_format: TTSFormat = Field(
        "mp3",
        description="Audio output format (e.g. 'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm')",
    )
    speed: float = Field(
        1.0,
        ge=MIN_SPEED,
        le=MAX_SPEED,
        description="Speech playback speed multiplier (0.25 to 4.0)",
    )
    language: Optional[TTSLanguage] = Field(
        None,
        description="Optional language hint (ISO 639-1/2 code)",
    )

    @field_validator("text")
    @classmethod
    def check_text(cls, v: str) -> str:
        """Ensure text is a non-empty string within length bounds."""
        return validate_text(v, max_length=DEFAULT_MAX_TEXT_LENGTH)

    @field_validator("voice")
    @classmethod
    def check_voice(cls, v: str) -> str:
        """Ensure voice is a non-empty string."""
        return validate_voice(v)

    @field_validator("speed")
    @classmethod
    def check_speed(cls, v: float) -> float:
        """Ensure speed is within valid bounds."""
        return validate_speed(v)

    @field_validator("output_format")
    @classmethod
    def check_output_format(cls, v: str) -> str:
        """Ensure output format is supported."""
        return validate_output_format(v)

    @field_validator("language")
    @classmethod
    def check_language(cls, v: Optional[str]) -> Optional[str]:
        """Ensure language is an ISO 639-1/2 code when provided."""
        return validate_language(v)


class TTSResponse(BaseModel):
    """A single text-to-speech synthesis response.

    Attributes:
        audio: Raw synthesized audio bytes (in-memory, transient)
        content_type: Canonical MIME type (e.g. 'audio/mpeg', 'audio/wav')
        format: Audio format identifier (e.g. 'mp3', 'wav')
        model: Model identifier that produced the audio
        provider: Provider name that produced the audio
        latency_ms: End-to-end provider latency in milliseconds
        character_count: Count of input characters synthesized
        metadata: Optional additional metadata dictionary
    """

    model_config = ConfigDict(protected_namespaces=())

    audio: TTSAudio = Field(
        ...,
        min_length=1,
        description="Raw synthesized audio bytes (transient, in-memory)",
    )
    content_type: str = Field(
        ...,
        min_length=1,
        description="Canonical MIME type of the audio",
    )
    format: TTSFormat = Field(
        ...,
        min_length=1,
        description="Audio format identifier (e.g. 'mp3', 'opus', 'wav')",
    )
    model: Optional[TTSModel] = Field(
        None,
        description="Model identifier that produced the audio",
    )
    provider: Optional[str] = Field(
        None,
        description="Provider name that produced the audio",
    )
    latency_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="End-to-end latency in milliseconds",
    )
    character_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of characters in the synthesized text",
    )
    metadata: Optional[dict[str, object]] = Field(
        default_factory=dict,
        description="Optional provider metadata",
    )

    @field_validator("audio")
    @classmethod
    def check_audio(cls, v: bytes) -> bytes:
        """Ensure audio is non-empty bytes."""
        if not isinstance(v, bytes) or not v:
            raise ValueError("TTS audio must be non-empty bytes")
        return v

    @field_validator("content_type")
    @classmethod
    def check_content_type(cls, v: str) -> str:
        """Ensure content_type is non-empty."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("content_type must be a non-empty string")
        return v.strip()


__all__ = [
    "TTSRequest",
    "TTSResponse",
]
