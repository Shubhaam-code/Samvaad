"""TTS provider configuration model.

Represents configuration for a TTS provider. The production
provider/model are configured via environment variables / settings;
this model describes the shape of the configuration and is used for
validation and documentation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import (
    DEFAULT_MAX_TEXT_LENGTH,
    MAX_SPEED,
    MIN_SPEED,
    SUPPORTED_FORMATS,
    validate_output_format,
    validate_speed,
    validate_voice,
)


class TTSProvider(str, Enum):
    """Enumeration of supported TTS providers.

    - FAKE: Deterministic offline provider for tests ONLY
    - OPENAI_TTS: OpenAI-compatible TTS provider (hosted or compatible server)
    - LOCAL: Local TTS model provider (planned)
    """
    FAKE = "fake"
    OPENAI_TTS = "openai_tts"
    LOCAL = "local"


class TTSConfig(BaseModel):
    """Configuration for a TTS provider.

    Attributes:
        provider: Which provider implementation to use
        api_key: Optional API key for hosted providers
        base_url: Optional base URL for OpenAI-compatible endpoints
        model: Model identifier (default: 'tts-1')
        voice: Default voice identifier (default: 'alloy')
        output_format: Default output format (default: 'mp3')
        speed: Default playback speed multiplier (default: 1.0)
        timeout_seconds: Provider API call timeout in seconds
        max_text_length: Maximum allowed text character length
        max_audio_size_mb: Maximum allowed generated audio size in MB
    """

    model_config = ConfigDict(protected_namespaces=())

    provider: TTSProvider = Field(
        TTSProvider.FAKE,
        description="TTS provider implementation",
    )
    api_key: Optional[str] = Field(
        None,
        description="Optional API key (never hardcoded, kept secret)",
    )
    base_url: Optional[str] = Field(
        None,
        description="Optional endpoint base URL override",
    )
    model: str = Field(
        "tts-1",
        min_length=1,
        description="TTS model identifier",
    )
    voice: str = Field(
        "alloy",
        min_length=1,
        description="Default voice identifier",
    )
    output_format: str = Field(
        "mp3",
        description="Default audio output format",
    )
    speed: float = Field(
        1.0,
        ge=MIN_SPEED,
        le=MAX_SPEED,
        description="Default playback speed multiplier",
    )
    timeout_seconds: float = Field(
        30.0,
        gt=0.0,
        description="Provider API timeout in seconds",
    )
    max_text_length: int = Field(
        DEFAULT_MAX_TEXT_LENGTH,
        gt=0,
        description="Maximum accepted text length for synthesis",
    )
    max_audio_size_mb: float = Field(
        10.0,
        gt=0.0,
        description="Maximum accepted generated audio size in MB",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Ensure model is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("model must be a non-empty string")
        return v.strip()

    @field_validator("voice")
    @classmethod
    def check_voice(cls, v: str) -> str:
        """Ensure voice is non-empty."""
        return validate_voice(v)

    @field_validator("speed")
    @classmethod
    def check_speed(cls, v: float) -> float:
        """Ensure speed is in bounds."""
        return validate_speed(v)

    @field_validator("output_format")
    @classmethod
    def check_output_format(cls, v: str) -> str:
        """Ensure output format is supported."""
        return validate_output_format(v)

    def __repr__(self) -> str:
        # Safe repr that NEVER displays or leaks the api_key
        return (
            f"TTSConfig(provider={self.provider.value!r}, "
            f"model={self.model!r}, "
            f"voice={self.voice!r}, "
            f"output_format={self.output_format!r}, "
            f"speed={self.speed}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"max_text_length={self.max_text_length}, "
            f"max_audio_size_mb={self.max_audio_size_mb})"
        )


__all__ = [
    "TTSProvider",
    "TTSConfig",
]
