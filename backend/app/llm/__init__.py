"""LLM package for text generation.

Phase 6.1: Provider-agnostic LLM harness interface, models, and
configuration only.

- types: predictable type aliases (str-based prompt / text)
- base:  BaseLLM ABC, LLMProtocol, LLMError, shared validation rules
- models: LLMRequest / LLMResponse / LLMUsage / FinishReason
- config: LLMConfig / LLMProvider
- fake:  FakeLLM - deterministic, offline, hash-based (tests only)

Phase 6.2 (planned): Production LLM provider integration
(OpenAI-compatible / Gemini / local), following the same interface.

No provider SDK is required: the fake provider and all validation are
implemented with the standard library plus pydantic.
"""

from .base import (
    BaseLLM,
    LLMError,
    LLMProtocol,
    validate_batch,
    validate_generated_text,
    validate_max_tokens,
    validate_prompt,
    validate_system_prompt,
    validate_temperature,
    validate_top_p,
)
from .config import LLMConfig, LLMProvider
from .fake import FakeLLM, create_fake_llm
from .models import FinishReason, LLMRequest, LLMResponse, LLMUsage
from .types import LLMPrompt, LLMText

__all__ = [
    # Type aliases
    "LLMPrompt",
    "LLMText",
    # Base interface
    "BaseLLM",
    "LLMProtocol",
    "LLMError",
    # Validation rules
    "validate_prompt",
    "validate_system_prompt",
    "validate_max_tokens",
    "validate_temperature",
    "validate_top_p",
    "validate_generated_text",
    "validate_batch",
    # Data models
    "FinishReason",
    "LLMUsage",
    "LLMRequest",
    "LLMResponse",
    # Configuration
    "LLMConfig",
    "LLMProvider",
    # Fake provider (tests/offline dev)
    "FakeLLM",
    "create_fake_llm",
]
