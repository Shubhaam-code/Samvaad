"""Structured Model Orchestration Harness (Phase 5.5).

Wraps LLM providers with:
1. Exponential backoff retries for transient failures (429, timeouts, 503).
2. Circuit Breaker protection to prevent thundering herds during outages.
3. Graceful degradation and fallback responses guaranteeing zero unhandled 500 crashes.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from .base import BaseLLM, LLMError
from .circuit_breaker import CircuitBreaker, CircuitBreakerError
from .models import FinishReason, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_MESSAGE = (
    "I apologize, but the response generation service is currently experiencing high load. "
    "Please try again in a few moments."
)


class ModelOrchestrationHarness(BaseLLM):
    """Resilience wrapper for LLM providers.

    Args:
        provider_llm: The underlying BaseLLM provider instance.
        max_retries: Maximum retry attempts for transient failures.
        base_backoff_seconds: Initial backoff delay in seconds.
        max_backoff_seconds: Maximum backoff delay cap.
        circuit_breaker: Optional injected CircuitBreaker instance.
        fallback_message: Fallback text returned on unrecoverable outages.
        sleep_fn: Injected sleep function for offline/mock test timing.
    """

    def __init__(
        self,
        provider_llm: BaseLLM,
        max_retries: int = 3,
        base_backoff_seconds: float = 0.2,
        max_backoff_seconds: float = 2.0,
        circuit_breaker: Optional[CircuitBreaker] = None,
        fallback_message: str = DEFAULT_FALLBACK_MESSAGE,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds must be non-negative")

        self._llm = provider_llm
        self._max_retries = max_retries
        self._base_backoff = base_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._circuit_breaker = circuit_breaker or CircuitBreaker(failure_threshold=3, recovery_time_seconds=30.0)
        self._fallback_message = fallback_message
        self._sleep_fn = sleep_fn or time.sleep

    @property
    def model_name(self) -> str:
        """The underlying model identifier."""
        return self._llm.model_name

    @property
    def provider(self) -> str:
        """The underlying provider identifier."""
        return self._llm.provider

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """The active circuit breaker instance."""
        return self._circuit_breaker

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response with circuit breaker guards, retries, and fallback recovery.

        Args:
            request: Validated LLMRequest

        Returns:
            LLMResponse from provider or structured fallback on provider outage
        """
        if not self._circuit_breaker.can_execute():
            logger.warning("Circuit breaker is OPEN. Returning graceful fallback response.")
            return LLMResponse(
                text=self._fallback_message,
                finish_reason=FinishReason.STOP,
                model=self.model_name,
                provider=self.provider,
                latency_ms=0.0,
            )

        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._llm.generate(request)
                self._circuit_breaker.record_success()
                return response
            except LLMError as exc:
                last_error = exc
                logger.warning(
                    "LLM generation attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                if attempt < self._max_retries:
                    backoff = min(self._max_backoff, self._base_backoff * (2 ** attempt))
                    self._sleep_fn(backoff)
            except Exception as exc:
                last_error = exc
                logger.error("Unexpected error during LLM generation: %s", exc)
                break

        # Record failure with circuit breaker
        self._circuit_breaker.record_failure()

        logger.error(
            "All %d generation attempts failed. Returning fallback. Error: %s",
            self._max_retries + 1,
            last_error,
        )

        return LLMResponse(
            text=self._fallback_message,
            finish_reason=FinishReason.STOP,
            model=self.model_name,
            provider=self.provider,
            latency_ms=0.0,
        )

    def generate_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Execute a batch of requests through the harness."""
        return [self.generate(req) for req in requests]
