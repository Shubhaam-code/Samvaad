"""Data models for the STT harness.

Defines the canonical request/response structures exchanged between
callers and STT providers. Models are deliberately provider-agnostic:
any future provider (hosted API, local model) can map its native
request/response shapes onto these models.

Audio is passed as transient in-memory ``bytes`` and is never stored
permanently by this package.

Phase 7.1: Data models only (no production provider).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import STTAudio, STTText


class STTRequest(BaseModel):
    """A single speech-to-text request.

    Attributes:
        audio: Raw audio bytes (transient, never persisted)
        filename: Original audio filename (extension validated)
        content_type: Optional MIME type (inferred from filename when
            absent)
        language: Optional language hint (None = automatic detection)
        prompt: Optional context/boosting prompt (provider dependent)
    """

    model_config = ConfigDict(protected_namespaces=())

    audio: STTAudio = Field(
        ...,
        min_length=1,
        description="Raw audio bytes (transient, never persisted)",
    )
    filename: str = Field(
        ...,
        min_length=1,
        description="Original audio filename (extension validated)",
    )
    content_type: Optional[str] = Field(
        None,
        description="Optional MIME type (inferred from filename when absent)",
    )
    language: Optional[str] = Field(
        None,
        description="Optional language hint (None = automatic detection)",
    )
    prompt: Optional[str] = Field(
        None,
        description="Optional context/boosting prompt (provider dependent)",
    )

    @field_validator("audio")
    @classmethod
    def validate_audio(cls, v: bytes) -> bytes:
        """Ensure audio is non-empty bytes."""
        if not isinstance(v, bytes) or not v:
            raise ValueError("audio must be non-empty bytes")
        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Ensure filename is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("filename must be a non-empty string")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Ensure language is an ISO 639-1/2 code when provided."""
        if v is None:
            return None
        normalized = v.strip().lower()
        if len(normalized) not in (2, 3) and not (
            len(normalized) == 5 and normalized[2] == "-"
        ):
            raise ValueError(
                f"language must be an ISO 639-1/2 code (e.g. 'en', 'hi', 'en-US'), "
                f"got {v!r}"
            )
        if "-" in normalized:
            head, _, tail = normalized.partition("-")
            if not (len(head) in (2, 3) and len(tail) == 2 and head.isalpha() and tail.isalpha()):
                raise ValueError(
                    f"language must be an ISO 639-1/2 code (e.g. 'en', 'hi', 'en-US'), "
                    f"got {v!r}"
                )
        elif not normalized.isalpha():
            raise ValueError(
                f"language must be an ISO 639-1/2 code (e.g. 'en', 'hi', 'en-US'), "
                f"got {v!r}"
            )
        return normalized

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Ensure prompt is not empty/whitespace when provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("prompt cannot be empty or whitespace-only")
        return v


class STTResponse(BaseModel):
    """A single speech-to-text response.

    Attributes:
        text: Transcribed text output
        language: Language of the transcription (provider detected when
            no hint was given)
        provider: Provider name that produced the transcription
        model: Model identifier that produced the transcription
        latency_ms: End-to-end latency in milliseconds
        duration_seconds: Audio duration in seconds (when the provider
            exposes it)
        confidence: Transcription confidence in [0.0, 1.0] (only when
            the provider exposes it - never invented)
    """

    model_config = ConfigDict(protected_namespaces=())

    text: STTText = Field(
        ...,
        min_length=1,
        description="Transcribed text output",
    )
    language: Optional[str] = Field(
        None,
        description="Language of the transcription (provider detected when no hint)",
    )
    provider: Optional[str] = Field(
        None,
        description="Provider name that produced the transcription",
    )
    model: Optional[str] = Field(
        None,
        description="Model identifier that produced the transcription",
    )
    latency_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="End-to-end latency in milliseconds",
    )
    duration_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="Audio duration in seconds (when the provider exposes it)",
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Transcription confidence in [0.0, 1.0] (provider-exposed only)",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure transcribed text is not empty after stripping."""
        if not isinstance(v, str) or not v or not v.strip():
            raise ValueError("Transcription text cannot be empty or whitespace-only")
        return v


__all__ = [
    "STTRequest",
    "STTResponse",
]