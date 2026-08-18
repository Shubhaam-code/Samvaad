"""Deterministic fake LLM provider for offline testing.

This provider NEVER contacts the network and NEVER loads a model.
Generated text is derived purely from the SHA-256 hash of the input
prompt, so it is:

- Deterministic: same prompt always produces the same text
  (across calls, instances, and process runs)
- Stable: responses are reproducible on any machine
- Offline: no model download, no API calls, no external dependencies

Token usage is estimated deterministically from word counts, and
``finish_reason`` defaults to ``stop``. The fake provider is a
realistic stand-in for downstream logic (wiring, latency tracking,
metadata propagation) without needing a production model.

The production LLM provider is deliberately NOT chosen here;
selection happens in Phase 6.2.

Phase 6.1: Fake provider only.
"""

from __future__ import annotations

import hashlib
import random
from typing import Final, Optional

from .base import BaseLLM, validate_batch, validate_max_tokens, validate_prompt
from .models import FinishReason, LLMRequest, LLMResponse, LLMUsage

_DEFAULT_MODEL_NAME = "fake-llm"
_DEFAULT_MAX_TOKENS = 256

# Deterministic answer-like word pool (English + Hindi) used to build
# fake responses. Content is illustrative only and never grounded.
_ANSWER_WORD_POOL: Final[list[str]] = [
    "goa", "beach", "temple", "tourism", "fort", "church", "market",
    "answer", "grounded", "retrieval", "evidence", "citation",
    "गोवा", "समुद्र", "मंदिर", "पर्यटन", "किला", "बाजार", "उत्तर", "साक्ष्य",
]

_MIN_WORDS = 4
_MAX_WORDS = 10


class FakeLLM(BaseLLM):
    """Deterministic, offline, hash-based LLM provider for testing.

    Args:
        model_name: Identifier reported on generated responses
        max_tokens: Upper bound on the number of words generated
            (also reported on generated responses)
        latency_ms: Simulated end-to-end latency reported on responses

    Raises:
        ValueError: If model_name is empty or max_tokens is invalid
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        latency_ms: float = 0.0,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name must be a non-empty string, got {model_name!r}")
        validate_max_tokens(max_tokens)
        if not isinstance(latency_ms, (int, float)) or isinstance(latency_ms, bool):
            raise ValueError(f"latency_ms must be a number, got {type(latency_ms).__name__}")
        if latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {latency_ms}")
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._latency_ms = float(latency_ms)

    @property
    def model_name(self) -> str:
        """Identifier reported on generated responses."""
        return self._model_name

    @property
    def provider(self) -> str:
        """Provider name reported on generated responses."""
        return "fake"

    @property
    def max_tokens(self) -> int:
        """Upper bound on the number of words generated."""
        return self._max_tokens

    def generate(self, request: object) -> LLMResponse:
        """Generate a deterministic response for a single request.

        Args:
            request: LLMRequest (or duck-typed request-like object) with
                a non-empty ``prompt`` attribute

        Returns:
            LLMResponse with deterministic text, usage estimates, and
            provider/model metadata

        Raises:
            ValueError: If the request is missing or has an empty prompt
        """
        prompt = self._validate_request(request)
        text = self._deterministic_text(prompt)
        return LLMResponse(
            text=text,
            model=self._model_name,
            provider=self.provider,
            finish_reason=FinishReason.STOP,
            usage=LLMUsage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(text.split()),
            ),
            latency_ms=self._latency_ms,
        )

    def generate_batch(self, requests: list[object]) -> list[LLMResponse]:
        """Generate deterministic responses for a batch, preserving order.

        ``["A", "B", "C"]`` always produces ``[response(A), response(B), response(C)]``.

        Args:
            requests: Non-empty list of LLMRequest (or duck-typed) objects

        Returns:
            List of LLMResponse in exactly the same order as the input

        Raises:
            ValueError: If the batch is empty or contains invalid requests
        """
        validate_batch(requests)
        return [self.generate(request) for request in requests]

    def _validate_request(self, request: object) -> str:
        """Validate a request-like object and return its prompt.

        Args:
            request: LLMRequest or duck-typed request-like object

        Returns:
            The validated prompt string

        Raises:
            ValueError: If the request is missing or has an empty prompt
        """
        if request is None or not hasattr(request, "prompt"):
            raise ValueError(
                f"request must be an LLMRequest or provide a prompt attribute, "
                f"got {type(request).__name__}"
            )
        return validate_prompt(request.prompt)

    def _deterministic_text(self, prompt: str) -> str:
        """Generate deterministic word-pool text seeded by the prompt.

        Uses a SHA-256 digest of the UTF-8 encoded prompt as the seed for
        Python's Mersenne Twister PRNG, which is specified to be
        reproducible across runs and platforms. Word count is capped by
        the configured max_tokens.
        """
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:16], byteorder="big")
        rng = random.Random(seed)

        count = rng.randint(_MIN_WORDS, _MAX_WORDS)
        count = max(1, min(count, self._max_tokens))
        words = [rng.choice(_ANSWER_WORD_POOL) for _ in range(count)]
        return " ".join(words)

    def __repr__(self) -> str:
        return (
            f"FakeLLM(model_name={self._model_name!r}, "
            f"max_tokens={self._max_tokens}, "
            f"latency_ms={self._latency_ms})"
        )


def create_fake_llm(
    model_name: str = _DEFAULT_MODEL_NAME,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    latency_ms: float = 0.0,
) -> FakeLLM:
    """Create a FakeLLM for testing and offline development."""
    return FakeLLM(model_name=model_name, max_tokens=max_tokens, latency_ms=latency_ms)


__all__ = [
    "FakeLLM",
    "create_fake_llm",
]
