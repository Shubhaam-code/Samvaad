"""Circuit Breaker for external Model & API calls (Phase 5.5).

Prevents cascading failures, thundering herds, and hanging requests by
monitoring failure rates and short-circuiting downstream calls when a provider
becomes unresponsive or degraded.
"""

from __future__ import annotations

from enum import Enum
import threading
import time


class CircuitBreakerState(str, Enum):
    """Possible states of a circuit breaker."""

    CLOSED = "closed"          # Healthy: requests pass through normally
    OPEN = "open"              # Tripped: requests are short-circuited immediately
    HALF_OPEN = "half_open"    # Probing: allowing a canary request to test recovery


class CircuitBreakerError(Exception):
    """Raised when an operation is rejected by an OPEN circuit breaker."""


class CircuitBreaker:
    """Thread-safe circuit breaker implementation.

    Args:
        failure_threshold: Number of consecutive failures before tripping OPEN.
        recovery_time_seconds: Time to remain in OPEN state before testing with HALF_OPEN.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_time_seconds: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_time_seconds <= 0:
            raise ValueError("recovery_time_seconds must be positive")

        self._failure_threshold = failure_threshold
        self._recovery_time_seconds = recovery_time_seconds

        self._state = CircuitBreakerState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Current state of the circuit breaker."""
        with self._lock:
            self._evaluate_state()
            return self._state

    @property
    def consecutive_failures(self) -> int:
        """Current count of consecutive failures."""
        with self._lock:
            return self._consecutive_failures

    def _evaluate_state(self) -> None:
        """Evaluate whether OPEN state has expired and should transition to HALF_OPEN."""
        if self._state == CircuitBreakerState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_time_seconds:
                self._state = CircuitBreakerState.HALF_OPEN

    def can_execute(self) -> bool:
        """Check whether a request is allowed to proceed."""
        with self._lock:
            self._evaluate_state()
            return self._state in (CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN)

    def record_success(self) -> None:
        """Record a successful execution, resetting failure counters."""
        with self._lock:
            self._consecutive_failures = 0
            self._state = CircuitBreakerState.CLOSED

    def record_failure(self) -> None:
        """Record a failed execution, incrementing counters and tripping if threshold met."""
        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()
            if self._consecutive_failures >= self._failure_threshold:
                self._state = CircuitBreakerState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._consecutive_failures = 0
            self._last_failure_time = 0.0
            self._state = CircuitBreakerState.CLOSED
