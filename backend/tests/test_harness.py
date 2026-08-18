"""Unit tests for the Model Orchestration Harness & Circuit Breaker (Phase 5.5).

All tests are 100% offline with zero real network delays.
"""

from unittest.mock import MagicMock
import pytest

from app.llm.base import BaseLLM, LLMError
from app.llm.circuit_breaker import CircuitBreaker, CircuitBreakerState
from app.llm.harness import DEFAULT_FALLBACK_MESSAGE, ModelOrchestrationHarness
from app.llm.models import FinishReason, LLMRequest, LLMResponse


class StubLLM(BaseLLM):
    """Test stub for simulating model successes and failures."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return "stub-model"

    @property
    def provider(self) -> str:
        return "stub-provider"

    @property
    def call_count(self) -> int:
        return self._call_count

    def generate(self, request: LLMRequest) -> LLMResponse:
        self._call_count += 1
        if not self._responses:
            raise LLMError("Out of stubbed responses")
        res = self._responses.pop(0)
        if isinstance(res, Exception):
            raise res
        return LLMResponse(
            text=str(res),
            finish_reason=FinishReason.STOP,
            model=self.model_name,
            provider=self.provider,
            latency_ms=10.0,
        )

    def generate_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        return [self.generate(req) for req in requests]


def test_harness_invalid_args():
    stub = StubLLM([])
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        ModelOrchestrationHarness(stub, max_retries=-1)
    with pytest.raises(ValueError, match="base_backoff_seconds must be non-negative"):
        ModelOrchestrationHarness(stub, base_backoff_seconds=-0.5)


def test_harness_successful_single_attempt():
    stub = StubLLM(["Capital is Panaji."])
    sleep_mock = MagicMock()
    harness = ModelOrchestrationHarness(stub, max_retries=3, sleep_fn=sleep_mock)

    req = LLMRequest(prompt="What is Goa capital?")
    res = harness.generate(req)

    assert res.text == "Capital is Panaji."
    assert stub.call_count == 1
    assert sleep_mock.call_count == 0
    assert harness.circuit_breaker.state == CircuitBreakerState.CLOSED


def test_harness_retry_on_transient_failure_then_succeed():
    # Attempt 1: fails with 429 rate limit / network error
    # Attempt 2: succeeds
    stub = StubLLM([LLMError("429 Rate Limit Exceeded"), "Successful answer"])
    sleep_mock = MagicMock()
    harness = ModelOrchestrationHarness(
        stub,
        max_retries=2,
        base_backoff_seconds=0.1,
        sleep_fn=sleep_mock,
    )

    req = LLMRequest(prompt="Hello")
    res = harness.generate(req)

    assert res.text == "Successful answer"
    assert stub.call_count == 2
    assert sleep_mock.call_count == 1
    # Check backoff delay was 0.1s
    sleep_mock.assert_called_once_with(0.1)
    assert harness.circuit_breaker.state == CircuitBreakerState.CLOSED


def test_harness_exhausted_retries_returns_fallback():
    stub = StubLLM([
        LLMError("Timeout 1"),
        LLMError("Timeout 2"),
        LLMError("Timeout 3"),
    ])
    sleep_mock = MagicMock()
    harness = ModelOrchestrationHarness(
        stub,
        max_retries=2,
        base_backoff_seconds=0.1,
        sleep_fn=sleep_mock,
    )

    req = LLMRequest(prompt="Hello")
    res = harness.generate(req)

    assert res.text == DEFAULT_FALLBACK_MESSAGE
    assert stub.call_count == 3
    assert sleep_mock.call_count == 2
    # 1 failure recorded with breaker (1 batch of exhausted attempts)
    assert harness.circuit_breaker.consecutive_failures == 1


def test_circuit_breaker_trips_open_and_short_circuits():
    stub = StubLLM([
        LLMError("Fatal error 1"),
        LLMError("Fatal error 2"),
        LLMError("Fatal error 3"),
    ])
    breaker = CircuitBreaker(failure_threshold=2, recovery_time_seconds=60.0)
    harness = ModelOrchestrationHarness(
        stub,
        max_retries=0,
        circuit_breaker=breaker,
    )

    req = LLMRequest(prompt="Query")

    # Call 1 -> Fails -> failure_count = 1, state = CLOSED
    res1 = harness.generate(req)
    assert res1.text == DEFAULT_FALLBACK_MESSAGE
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.consecutive_failures == 1

    # Call 2 -> Fails -> failure_count = 2 -> Trips OPEN
    res2 = harness.generate(req)
    assert res2.text == DEFAULT_FALLBACK_MESSAGE
    assert breaker.state == CircuitBreakerState.OPEN

    # Call 3 -> Breaker is OPEN -> provider is NEVER called
    initial_calls = stub.call_count
    res3 = harness.generate(req)
    assert res3.text == DEFAULT_FALLBACK_MESSAGE
    assert stub.call_count == initial_calls  # Short-circuited!


def test_circuit_breaker_manual_reset():
    breaker = CircuitBreaker(failure_threshold=1, recovery_time_seconds=60.0)
    breaker.record_failure()
    assert breaker.state == CircuitBreakerState.OPEN

    breaker.reset()
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.can_execute() is True
