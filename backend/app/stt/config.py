"""STT provider configuration model.

Represents the configuration for an STT provider. The production
provider/model are chosen via environment configuration (Phase 7.2);
this model simply describes the shape of the final configuration and is
used for validation and documentation.

Phase 7.1: Configuration model only (no production provider selection).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

_LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}(-[a-z]{2})?$")


class STTProvider(str, Enum):
    """Enumeration of supported STT providers.

    - FAKE: Deterministic offline provider for tests (Phase 7.1)
    - OPENAI_COMPATIBLE: OpenAI-compatible Whisper API provider
      (Phase 7.2 - the selected production provider)
    - GEMINI: Google Gemini API provider (planned)
    - LOCAL: Local model provider (planned)
    """
    FAKE = "fake"
    OPENAI_COMPATIBLE = "openai_compatible"
    GEMINI = "gemini"
    LOCAL = "local"


class STTConfig(BaseModel):
    """Configuration for an STT provider.

    Attributes:
        provider: Which provider implementation to use
        model_name: Name or local path of the model (None until chosen)
        language: Default language hint (None = automatic detection)
        timeout_seconds: Provider call timeout in seconds
        max_audio_size_mb: Maximum accepted audio upload size in MB
    """

    model_config = ConfigDict(protected_namespaces=())

    provider: STTProvider = Field(
        STTProvider.FAKE,
        description="STT provider implementation",
    )

    model_name: Optional[str] = Field(
        None,
        description="Model name or local path (unset until model selection)",
    )

    language: Optional[str] = Field(
        None,
        description="Default language hint (None = automatic detection)",
    )

    timeout_seconds: float = Field(
        30.0,
        gt=0.0,
        description="Provider call timeout in seconds",
    )

    max_audio_size_mb: float = Field(
        10.0,
        gt=0.0,
        description="Maximum accepted audio upload size in MB",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: Optional[str]) -> Optional[str]:
        """Ensure model_name is not empty/whitespace when provided."""
        if v is not None and not v.strip():
            raise ValueError("model_name cannot be empty or whitespace-only")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Ensure language is an ISO 639-1/2 code when provided."""
        if v is None:
            return None
        normalized = v.strip().lower()
        if not _LANGUAGE_PATTERN.fullmatch(normalized):
            raise ValueError(
                f"language must be an ISO 639-1/2 code (e.g. 'en', 'hi'), got {v!r}"
            )
        return normalized

    def __repr__(self) -> str:
        return (
            f"STTConfig(provider={self.provider.value}, "
            f"model_name={self.model_name!r}, "
            f"language={self.language!r}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"max_audio_size_mb={self.max_audio_size_mb})"
        )


__all__ = [
    "STTProvider",
    "STTConfig",
]