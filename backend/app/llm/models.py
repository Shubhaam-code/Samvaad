"""Data models for the LLM harness.

Defines the canonical request/response structures exchanged between
callers and LLM providers. Models are deliberately provider-agnostic:
any future provider (OpenAI-compatible, Gemini, local) can map its
native request/response shapes onto these models.

Phase 6.1: Data models only (no production provider).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from .types import LLMPrompt, LLMText


class FinishReason(str, Enum):
    """Possible reasons a generation finished."""

    STOP = "stop"
    LENGTH = "length"


class LLMUsage(BaseModel):
    """Token usage reported for a single generation.

    Attributes:
        prompt_tokens: Tokens consumed by the input prompt
        completion_tokens: Tokens produced in the generated text
        total_tokens: Computed sum of prompt + completion tokens
    """

    prompt_tokens: int = Field(0, ge=0, description="Tokens consumed by the input prompt")
    completion_tokens: int = Field(0, ge=0, description="Tokens produced in the generated text")

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Total tokens used for the generation."""
        return self.prompt_tokens + self.completion_tokens


class LLMRequest(BaseModel):
    """A single text generation request.

    Attributes:
        prompt: User prompt to generate a response for (required)
        system_prompt: Optional system prompt / instructions
        max_tokens: Optional maximum tokens to generate
        temperature: Optional sampling temperature in [0.0, 2.0]
        top_p: Optional nucleus sampling probability mass in [0.0, 1.0]
    """

    model_config = ConfigDict(protected_namespaces=())

    prompt: LLMPrompt = Field(
        ...,
        min_length=1,
        description="User prompt to generate a response for",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Optional system prompt / instructions",
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="Optional maximum tokens to generate",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Optional sampling temperature in [0.0, 2.0]",
    )
    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional nucleus sampling probability mass in [0.0, 1.0]",
    )

    @field_validator("prompt", "system_prompt")
    @classmethod
    def validate_non_empty_text(cls, v: Optional[str]) -> Optional[str]:
        """Ensure prompt/system_prompt are not empty after stripping."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("Prompt text cannot be empty or whitespace-only")
        return v


class LLMResponse(BaseModel):
    """A single text generation response.

    Attributes:
        text: Generated text output
        model: Model identifier that produced the text
        provider: Provider name that produced the text
        finish_reason: Why generation finished (stop or length)
        usage: Token usage for the generation
        latency_ms: Optional end-to-end latency in milliseconds
    """

    model_config = ConfigDict(protected_namespaces=())

    text: LLMText = Field(
        ...,
        min_length=1,
        description="Generated text output",
    )
    model: Optional[str] = Field(
        None,
        description="Model identifier that produced the text",
    )
    provider: Optional[str] = Field(
        None,
        description="Provider name that produced the text",
    )
    finish_reason: FinishReason = Field(
        FinishReason.STOP,
        description="Why generation finished (stop or length)",
    )
    usage: LLMUsage = Field(
        default_factory=LLMUsage,
        description="Token usage for the generation",
    )
    latency_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Optional end-to-end latency in milliseconds",
    )

    @field_validator("text")
    @classmethod
    def validate_generated_text(cls, v: str) -> str:
        """Ensure generated text is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError("Generated text cannot be empty or whitespace-only")
        return v


__all__ = [
    "FinishReason",
    "LLMUsage",
    "LLMRequest",
    "LLMResponse",
]
