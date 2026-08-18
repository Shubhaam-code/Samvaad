"""OpenAI-compatible LLM provider adapter (Phase 6.4).

Implements the existing ``BaseLLM`` contract (``generate`` /
``generate_batch`` / ``model_name`` / ``provider``) on top of the
official OpenAI Python SDK. It works against the OpenAI API and any
OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, Groq, ...) via a
configurable ``base_url``.

Key properties:

- Provider-agnostic contract preserved: callers only ever see
  ``LLMRequest`` / ``LLMResponse`` and the ``LLMError`` exception.
- Credentials come from configuration (environment variables), never
  from code, and are never included in errors, logs, or responses.
- The SDK client is injectable for offline tests: when a ``client``
  object is provided, no ``openai.OpenAI`` client is constructed and no
  network connection is possible.
- All provider SDK failures are wrapped into ``LLMError`` with a
  sanitized message (no keys, no headers, no request payloads).

The provider is considered configured when:
- an ``api_key`` is present, or
- a non-default ``base_url`` is present (local compatible servers need
  no key).

Phase 6.2 (planned): production provider selection; Phase 6.4: first
real provider implemented behind the same interface.
"""

from __future__ import annotations

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

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 60.0

_MS_ROUND = 4

# Provider finish_reason strings mapped onto the harness FinishReason enum.
# Unknown or missing values conservatively map to STOP.
_FINISH_REASON_LENGTH = "length"


def is_openai_compatible_configured(*, api_key: Optional[str], base_url: str) -> bool:
    """Decide whether the OpenAI-compatible provider is configured.

    Rules:
    - An API key always counts as configured (hosted OpenAI or any
      key-protected compatible endpoint).
    - A non-default base URL counts as configured even without a key
      (local OpenAI-compatible servers need no authentication).
    - No key and the default OpenAI base URL is NOT configured: the
      provider would be unusable, so /api/chat must keep returning 501.

    Args:
        api_key: Provider API key (or None / empty)
        base_url: Resolved base URL (defaults to OpenAI when None)

    Returns:
        True when a usable provider configuration is present
    """
    if api_key and api_key.strip():
        return True
    return base_url != DEFAULT_BASE_URL


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI-compatible chat completions provider.

    Args:
        api_key: Provider API key (required for the default OpenAI URL;
            optional for local compatible endpoints)
        base_url: Endpoint base URL (defaults to the OpenAI API)
        model_name: Model identifier served at the endpoint
        timeout_seconds: Provider call timeout in seconds
        client: Optional injected client for tests; when provided, no
            SDK client is constructed and no network access is possible

    Raises:
        ValueError: If model_name/timeout/base_url are invalid, or if
            neither an API key nor a non-default base URL is configured
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: Optional[object] = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name must be a non-empty string, got {model_name!r}")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(f"base_url must be a non-empty string, got {base_url!r}")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                f"timeout_seconds must be a positive number, got {timeout_seconds!r}"
            )
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError(
                f"api_key must be a string or None, got {type(api_key).__name__}"
            )

        self._api_key = (api_key or "").strip()
        self._base_url = base_url
        self._model_name = model_name
        self._timeout_seconds = float(timeout_seconds)

        if client is not None:
            self._client = client
        else:
            if not self._api_key and base_url == DEFAULT_BASE_URL:
                raise ValueError(
                    "api_key is required when using the default OpenAI base URL"
                )
            # An empty api_key with a local base URL produces no
            # Authorization header (the SDK only rejects None, which
            # would fall back to the OPENAI_API_KEY environment).
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout_seconds,
            )

    @property
    def model_name(self) -> str:
        """Identifier reported on generated responses."""
        return self._model_name

    @property
    def provider(self) -> str:
        """Provider name reported on generated responses."""
        return "openai_compatible"

    def generate(self, request: object) -> LLMResponse:
        """Generate a response for a single request.

        Args:
            request: LLMRequest (or duck-typed request-like object) with
                a non-empty ``prompt`` attribute

        Returns:
            LLMResponse with the provider text, model, usage,
            finish_reason, and measured latency

        Raises:
            ValueError: If the request is invalid
            LLMError: If the underlying provider fails
        """
        prompt = validate_prompt(request.prompt)
        system_prompt = validate_system_prompt(getattr(request, "system_prompt", None))
        max_tokens = validate_max_tokens(getattr(request, "max_tokens", None))
        temperature = validate_temperature(getattr(request, "temperature", None))
        top_p = validate_top_p(getattr(request, "top_p", None))

        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, object] = {"model": self._model_name, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

        start = time.perf_counter()
        try:
            raw = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - wrap every provider failure
            raise self._wrap_provider_error(exc) from exc
        latency_ms = round((time.perf_counter() - start) * 1000.0, _MS_ROUND)

        return self._map_response(raw, latency_ms)

    def generate_batch(self, requests: list[object]) -> list[LLMResponse]:
        """Generate responses for a batch, preserving input order.

        ``[r1, r2, r3]`` produces ``[response(r1), response(r2), response(r3)]``.

        Args:
            requests: Non-empty list of LLMRequest (or duck-typed) objects

        Returns:
            List of LLMResponse in exactly the same order as the input

        Raises:
            ValueError: If the batch is empty or contains invalid requests
            LLMError: If the underlying provider fails
        """
        validate_batch(requests)
        return [self.generate(request) for request in requests]

    def _map_response(self, raw: object, latency_ms: float) -> LLMResponse:
        """Map a provider completion object onto the harness LLMResponse.

        Args:
            raw: Provider completion object (duck-typed for tests)
            latency_ms: Measured end-to-end latency

        Returns:
            An LLMResponse with provider text, model, usage, and finish_reason

        Raises:
            LLMError: If the provider returned no choices or empty content
        """
        if not getattr(raw, "choices", None):
            raise LLMError("OpenAI-compatible provider returned no choices")

        choice = raw.choices[0]
        content = getattr(getattr(choice, "message", None), "content", None)
        if content is None or not str(content).strip():
            raise LLMError("OpenAI-compatible provider returned empty content")

        model = getattr(raw, "model", None) or self._model_name

        usage = LLMUsage()
        raw_usage = getattr(raw, "usage", None)
        if raw_usage is not None:
            usage = LLMUsage(
                prompt_tokens=int(getattr(raw_usage, "prompt_tokens", None) or 0),
                completion_tokens=int(getattr(raw_usage, "completion_tokens", None) or 0),
            )

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == _FINISH_REASON_LENGTH:
            mapped_reason = FinishReason.LENGTH
        else:
            mapped_reason = FinishReason.STOP

        return LLMResponse(
            text=validate_generated_text(str(content)),
            model=model,
            provider=self.provider,
            finish_reason=mapped_reason,
            usage=usage,
            latency_ms=latency_ms,
        )

    def _wrap_provider_error(self, exc: Exception) -> LLMError:
        """Wrap a provider failure into a sanitized LLMError.

        The message never contains the API key (redacted if the SDK ever
        echoes it), authorization headers, or request payloads - only the
        exception type and the SDK's public message text.

        Args:
            exc: The underlying provider exception

        Returns:
            An LLMError safe to surface in API responses and logs
        """
        text = str(exc)
        if self._api_key and self._api_key in text:
            text = text.replace(self._api_key, "[REDACTED]")
        return LLMError(
            f"OpenAI-compatible provider error ({type(exc).__name__}): {text}"
        )

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleLLM(model_name={self._model_name!r}, "
            f"base_url={self._base_url!r}, "
            f"timeout_seconds={self._timeout_seconds})"
        )


def create_openai_compatible_llm(
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    client: Optional[object] = None,
) -> OpenAICompatibleLLM:
    """Create an OpenAI-compatible LLM provider.

    Args:
        api_key: Provider API key (required for the default OpenAI URL)
        base_url: Endpoint base URL (defaults to the OpenAI API)
        model_name: Model identifier served at the endpoint
        timeout_seconds: Provider call timeout in seconds
        client: Optional injected client for tests (no network)

    Returns:
        A configured OpenAICompatibleLLM instance
    """
    return OpenAICompatibleLLM(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        client=client,
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TIMEOUT_SECONDS",
    "OpenAICompatibleLLM",
    "create_openai_compatible_llm",
    "is_openai_compatible_configured",
]
