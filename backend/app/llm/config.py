"""LLM provider configuration model.

Represents the configuration for an LLM provider. The production
provider/model are intentionally NOT chosen here (Phase 6.2); this model
simply describes the shape of the final configuration.

Phase 6.1: Configuration model only (no production provider selection).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers.

    - FAKE: Deterministic offline provider for tests (Phase 6.1)
    - OPENAI_COMPATIBLE: OpenAI-compatible API provider (planned Phase 6.2)
    - GEMINI: Google Gemini API provider (planned Phase 6.2)
    - LOCAL: Local model provider (planned Phase 6.2)
    """
    FAKE = "fake"
    OPENAI_COMPATIBLE = "openai_compatible"
    GEMINI = "gemini"
    LOCAL = "local"


class LLMConfig(BaseModel):
    """Configuration for an LLM provider.

    Attributes:
        provider: Which provider implementation to use
        model_name: Name or local path of the model (None until chosen)
        max_tokens: Default maximum tokens to generate
        temperature: Default sampling temperature in [0.0, 2.0]
        top_p: Default nucleus sampling probability mass in [0.0, 1.0]
        timeout_seconds: Provider call timeout in seconds
    """

    model_config = ConfigDict(protected_namespaces=())

    provider: LLMProvider = Field(
        LLMProvider.FAKE,
        description="LLM provider implementation",
    )

    model_name: Optional[str] = Field(
        None,
        description="Model name or local path (unset until Phase 6.2 model selection)",
    )

    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="Default maximum tokens to generate",
    )

    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature in [0.0, 2.0]",
    )

    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Default nucleus sampling probability mass in [0.0, 1.0]",
    )

    timeout_seconds: float = Field(
        30.0,
        gt=0.0,
        description="Provider call timeout in seconds",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: Optional[str]) -> Optional[str]:
        """Ensure model_name is not empty/whitespace when provided."""
        if v is not None and not v.strip():
            raise ValueError("model_name cannot be empty or whitespace-only")
        return v

    def __repr__(self) -> str:
        return (
            f"LLMConfig(provider={self.provider.value}, "
            f"model_name={self.model_name!r}, "
            f"max_tokens={self.max_tokens}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"timeout_seconds={self.timeout_seconds})"
        )


__all__ = [
    "LLMProvider",
    "LLMConfig",
]
