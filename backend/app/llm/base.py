"""Base LLM interface and shared validation rules.

This module defines the provider-agnostic text generation contract that all
concrete implementations (fake, OpenAI-compatible API, Gemini, local model)
must follow.

The interface is intentionally small:

- ``generate(request)``: one request -> one response
- ``generate_batch(requests)``: many requests -> many responses (order preserved)
- ``model_name``: optional model identifier (None until a model is known)
- ``provider``: provider name reported on responses

Shared validation helpers are provided as module-level functions so that
future providers and callers can reuse the exact same rules.

Phase 6.1: Interface definition + validation only (no production provider).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol

from .types import LLMText


class LLMError(Exception):
    """Custom exception raised for LLM provider/harness failures.

    Mirrors ``VectorStoreError`` in the vectorstore layer: concrete
    providers should wrap provider-specific failures in this exception
    so callers never depend on a specific SDK's error types.
    """
    pass


def validate_prompt(prompt: str) -> str:
    """Validate a single generation prompt.

    Rules:
    - Must be a string
    - Must not be empty
    - Must not be whitespace-only

    Args:
        prompt: Prompt text to validate

    Returns:
        The validated prompt (unchanged)

    Raises:
        ValueError: If prompt is not a string, empty, or whitespace-only
    """
    if not isinstance(prompt, str):
        raise ValueError(f"LLM prompt must be a string, got {type(prompt).__name__}")
    if not prompt:
        raise ValueError("LLM prompt cannot be empty")
    if not prompt.strip():
        raise ValueError("LLM prompt cannot be whitespace-only")
    return prompt


def validate_system_prompt(system_prompt: Optional[str]) -> Optional[str]:
    """Validate an optional system prompt.

    Rules:
    - Must be None or a string
    - A provided string must not be empty or whitespace-only

    Args:
        system_prompt: Optional system prompt to validate

    Returns:
        The validated system prompt (unchanged)

    Raises:
        ValueError: If system_prompt is provided but not a string, empty,
                    or whitespace-only
    """
    if system_prompt is None:
        return None
    if not isinstance(system_prompt, str):
        raise ValueError(
            f"system_prompt must be a string or None, got {type(system_prompt).__name__}"
        )
    if not system_prompt or not system_prompt.strip():
        raise ValueError("system_prompt cannot be empty or whitespace-only")
    return system_prompt


def validate_max_tokens(max_tokens: Optional[int]) -> Optional[int]:
    """Validate an optional max_tokens generation parameter.

    Rules:
    - Must be None or a positive integer
    - Must not be a bool

    Args:
        max_tokens: Maximum tokens to generate (None = provider default)

    Returns:
        The validated max_tokens (unchanged)

    Raises:
        ValueError: If max_tokens is not a positive integer
    """
    if max_tokens is None:
        return None
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ValueError(
            f"max_tokens must be a positive integer or None, got {type(max_tokens).__name__}"
        )
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    return max_tokens


def validate_temperature(temperature: Optional[float]) -> Optional[float]:
    """Validate an optional temperature generation parameter.

    Rules:
    - Must be None or a finite number
    - Must be within [0.0, 2.0]

    Args:
        temperature: Sampling temperature (None = provider default)

    Returns:
        The validated temperature (unchanged)

    Raises:
        ValueError: If temperature is not a number or outside [0.0, 2.0]
    """
    if temperature is None:
        return None
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError(
            f"temperature must be a number or None, got {type(temperature).__name__}"
        )
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"temperature must be within [0.0, 2.0], got {temperature}")
    return temperature


def validate_top_p(top_p: Optional[float]) -> Optional[float]:
    """Validate an optional top_p generation parameter.

    Rules:
    - Must be None or a finite number
    - Must be within [0.0, 1.0]

    Args:
        top_p: Nucleus sampling probability mass (None = provider default)

    Returns:
        The validated top_p (unchanged)

    Raises:
        ValueError: If top_p is not a number or outside [0.0, 1.0]
    """
    if top_p is None:
        return None
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise ValueError(
            f"top_p must be a number or None, got {type(top_p).__name__}"
        )
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"top_p must be within [0.0, 1.0], got {top_p}")
    return top_p


def validate_generated_text(text: str) -> str:
    """Validate a produced generation output text.

    Rules:
    - Must be a string
    - Must not be empty
    - Must not be whitespace-only

    Args:
        text: Generated text to validate

    Returns:
        The validated text (unchanged)

    Raises:
        ValueError: If text is not a string, empty, or whitespace-only
    """
    if not isinstance(text, str):
        raise ValueError(f"Generated text must be a string, got {type(text).__name__}")
    if not text or not text.strip():
        raise ValueError("Generated text cannot be empty or whitespace-only")
    return text


def validate_batch(requests: list[object]) -> list[object]:
    """Validate a batch of request objects for generate_batch().

    Rules:
    - Must be a list
    - Must not be empty (at least one request required)
    - Every item must provide a non-empty ``prompt`` attribute
      (LLMRequest or duck-typed request-like object)

    Args:
        requests: List of LLMRequest (or duck-typed request-like) objects

    Returns:
        The validated list (unchanged)

    Raises:
        ValueError: If the list is not a list, is empty, or contains an
                    item without a non-empty prompt attribute
    """
    if not isinstance(requests, list):
        raise ValueError(f"LLM batch must be a list, got {type(requests).__name__}")
    if not requests:
        raise ValueError("LLM batch cannot be empty")
    for index, request in enumerate(requests):
        if not hasattr(request, "prompt"):
            raise ValueError(
                f"Item at index {index} is not a request-like object: "
                f"expected a prompt attribute, got {type(request).__name__}"
            )
        prompt = request.prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Item at index {index} has a missing or empty prompt"
            )
    return requests


class BaseLLM(ABC):
    """Abstract base class for all LLM text generation providers.

    Concrete implementations will include:
    - FakeLLM: deterministic offline provider for tests (Phase 6.1)
    - OpenAI-compatible API provider (planned Phase 6.2)
    - Gemini / other API provider (planned Phase 6.2)
    - Local model provider (planned Phase 6.2)

    All providers must implement generate() and generate_batch(), must
    preserve input ordering in generate_batch(), and must report model
    and provider identifiers.

    Phase 6.1: Base interface only (no production provider).
    """

    @abstractmethod
    def generate(self, request: object) -> object:
        """Generate a text response for a single request.

        Args:
            request: LLMRequest (or duck-typed request-like object) with
                at least a non-empty ``prompt`` attribute

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            ValueError: If the request is missing or has an empty prompt
            LLMError: If the underlying provider fails
        """
        pass

    @abstractmethod
    def generate_batch(self, requests: list[object]) -> list[object]:
        """Generate responses for a batch of requests, preserving order.

        ``[r1, r2, r3]`` must produce ``[response(r1), response(r2), response(r3)]``.

        Args:
            requests: Non-empty list of LLMRequest (or duck-typed) objects

        Returns:
            List of LLMResponse in exactly the same order as the input

        Raises:
            ValueError: If the batch is empty or contains invalid requests
            LLMError: If the underlying provider fails
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> Optional[str]:
        """Name/identifier of the model used by this provider.

        May be None until a production model is selected.
        """
        pass

    @property
    @abstractmethod
    def provider(self) -> Optional[str]:
        """Provider name reported on generated responses (e.g. 'fake')."""
        pass


class LLMProtocol(Protocol):
    """Protocol defining the LLM interface for type checking.

    Allows duck-typed provider implementations that don't explicitly
    inherit from BaseLLM but still follow the contract.
    """

    def generate(self, request: object) -> object:
        """Generate a text response for a single request."""
        ...

    def generate_batch(self, requests: list[object]) -> list[object]:
        """Generate responses for a batch of requests (order preserved)."""
        ...

    @property
    def model_name(self) -> Optional[str]:
        """Name/identifier of the model used by this provider."""
        ...

    @property
    def provider(self) -> Optional[str]:
        """Provider name reported on generated responses."""
        ...


__all__ = [
    "BaseLLM",
    "LLMError",
    "LLMProtocol",
    "validate_prompt",
    "validate_system_prompt",
    "validate_max_tokens",
    "validate_temperature",
    "validate_top_p",
    "validate_generated_text",
    "validate_batch",
]
