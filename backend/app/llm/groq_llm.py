"""Groq Cloud Llama 3.1 8B Instant LLM provider adapter (Phase 5.4).

Provides ultra-fast (<80ms TTFT, 500-800 tokens/sec) grounded answer generation
using Groq Cloud's production inference engine.

Guarantees:
- Complies with ``BaseLLM`` interface.
- Default model: ``llama-3.1-8b-instant``.
- Zero-leak credential security: API keys are redacted from all exception messages.
- Testability: supports injected ``client`` for 100% offline unit tests.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from openai import OpenAI, OpenAIError

from .base import (
    BaseLLM,
    LLMError,
    validate_batch,
    validate_generated_text,
    validate_max_tokens,
    validate_prompt,
    validate_system_prompt,
    validate_temperature,
    validate_top_p,
)
from .models import FinishReason, LLMRequest, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GROQ_TIMEOUT_SECONDS = 30.0

_MS_ROUND = 4
_FINISH_REASON_LENGTH = "length"


def _redact_key(text: str, key: Optional[str]) -> str:
    """Redact sensitive API keys from exception text."""
    if not key or not text:
        return text
    return text.replace(key, "[REDACTED]")


def is_groq_configured(api_key: Optional[str]) -> bool:
    """Check whether a Groq API key is present and configured."""
    if not api_key:
        return False
    return bool(api_key.strip())


class GroqLLM(BaseLLM):
    """Groq Cloud LLM provider adapter.

    Args:
        api_key: Groq API key (from environment)
        model_name: Model identifier (default: "llama-3.1-8b-instant")
        base_url: Groq API base URL (default: "https://api.groq.com/openai/v1")
        timeout_seconds: Request timeout in seconds
        client: Optional injected OpenAI client for offline testing
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GROQ_MODEL,
        base_url: str = DEFAULT_GROQ_BASE_URL,
        timeout_seconds: float = DEFAULT_GROQ_TIMEOUT_SECONDS,
        client: Optional[object] = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("api_key is required when no custom client is injected")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._api_key = api_key.strip() if api_key else ""
        self._model_name = model_name.strip() or DEFAULT_GROQ_MODEL
        self._base_url = base_url.strip() or DEFAULT_GROQ_BASE_URL
        self._timeout_seconds = timeout_seconds

        if client is not None:
            self._client = client
        else:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout_seconds,
            )

    @property
    def model_name(self) -> str:
        """The active Groq model identifier."""
        return self._model_name

    @property
    def provider(self) -> str:
        """Provider name reported on responses."""
        return "groq"

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate an answer using Groq Cloud inference.

        Args:
            request: Validated LLMRequest

        Returns:
            LLMResponse with generated text, finish reason, and latency

        Raises:
            LLMError: On generation or network failures
        """
        validate_prompt(request.prompt)
        if request.system_prompt is not None:
            validate_system_prompt(request.system_prompt)
        if request.max_tokens is not None:
            validate_max_tokens(request.max_tokens)
        if request.temperature is not None:
            validate_temperature(request.temperature)
        if request.top_p is not None:
            validate_top_p(request.top_p)

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        kwargs = {
            "model": self._model_name,
            "messages": messages,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        start_time = time.perf_counter()

        try:
            raw_response = self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            safe_msg = _redact_key(str(exc), self._api_key)
            raise LLMError(f"Groq API error: {safe_msg}") from exc
        except Exception as exc:
            safe_msg = _redact_key(str(exc), self._api_key)
            raise LLMError(f"Groq generation failed: {safe_msg}") from exc

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        if not raw_response.choices:
            raise LLMError("Groq returned empty choices list")

        choice = raw_response.choices[0]
        raw_text = choice.message.content or ""
        clean_text = validate_generated_text(raw_text)

        finish_reason = (
            FinishReason.LENGTH
            if choice.finish_reason == _FINISH_REASON_LENGTH
            else FinishReason.STOP
        )

        usage = None
        if raw_response.usage:
            usage = LLMUsage(
                prompt_tokens=getattr(raw_response.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(raw_response.usage, "completion_tokens", 0) or 0,
            )

        return LLMResponse(
            text=clean_text,
            finish_reason=finish_reason,
            model=getattr(raw_response, "model", self._model_name) or self._model_name,
            provider=self.provider,
            latency_ms=round(duration_ms, _MS_ROUND),
            usage=usage,
        )

    def generate_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Generate responses sequentially for a batch of requests."""
        validate_batch(requests)
        return [self.generate(req) for req in requests]
