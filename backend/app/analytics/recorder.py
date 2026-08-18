"""Thread-safe, in-memory latency aggregation for the chat pipeline.

Records ONLY real measurements produced by the POST /api/chat endpoint:

- ``record_success``: called for 200 completions; updates the per-stage
  latency aggregates (guardrail, retrieval, LLM, grounding, total) and
  the successful request count.
- ``record_rejected``: called for 400 guardrail rejections; increments
  rejected_count ONLY - latency aggregates are never touched.
- ``record_error``: called for 501/503/500 outcomes; increments
  error_count ONLY.

No database, no external monitoring service, no fabricated metrics:
every number stored comes from an actual endpoint measurement.

All state is guarded by a single threading.Lock so the recorder is safe
under FastAPI's threadpool concurrency.
"""

from __future__ import annotations

import threading
from typing import Final, Optional

# Stage keys recorded per successful request - must match the
# LatencyBreakdown field names produced by POST /api/chat.
STAGE_KEYS: Final[tuple[str, ...]] = (
    "guardrail_ms",
    "retrieval_ms",
    "llm_ms",
    "grounding_ms",
    "total_ms",
)


class LatencyRecorder:
    """Accumulates real chat pipeline latency statistics in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._sums: dict[str, float] = {}
        self._mins: dict[str, float] = {}
        self._maxs: dict[str, float] = {}
        for key in STAGE_KEYS:
            self._counts[key] = 0
            self._sums[key] = 0.0
            self._mins[key] = 0.0
            self._maxs[key] = 0.0
        self._request_count = 0
        self._rejected_count = 0
        self._error_count = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_success(self, latencies: dict[str, float]) -> None:
        """Record a successful pipeline completion with its real latencies.

        Args:
            latencies: Mapping with exactly the STAGE_KEYS keys, values in ms

        Raises:
            ValueError: If latencies is missing keys or contains invalid values
        """
        missing = [key for key in STAGE_KEYS if key not in latencies]
        if missing:
            raise ValueError(
                f"record_success requires exactly {list(STAGE_KEYS)} "
                f"keys, missing: {missing}"
            )
        extra = [key for key in latencies if key not in STAGE_KEYS]
        if extra:
            raise ValueError(f"Unknown stage keys in latencies: {extra}")

        with self._lock:
            self._request_count += 1
            for key in STAGE_KEYS:
                value = latencies[key]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(
                        f"Latency value for '{key}' must be a number, got {type(value).__name__}"
                    )
                value = float(value)
                if value < 0.0:
                    raise ValueError(f"Latency value for '{key}' cannot be negative: {value}")
                self._counts[key] += 1
                self._sums[key] += value
                if self._counts[key] == 1:
                    self._mins[key] = value
                    self._maxs[key] = value
                else:
                    self._mins[key] = min(self._mins[key], value)
                    self._maxs[key] = max(self._maxs[key], value)

    def record_rejected(self) -> None:
        """Record an input guardrail rejection (400). Never touches latency aggregates."""
        with self._lock:
            self._rejected_count += 1

    def record_error(self) -> None:
        """Record a non-successful outcome (501/503/500). Never touches latency aggregates."""
        with self._lock:
            self._error_count += 1

    def reset(self) -> None:
        """Reset all accumulated state (used by tests)."""
        with self._lock:
            for key in STAGE_KEYS:
                self._counts[key] = 0
                self._sums[key] = 0.0
                self._mins[key] = 0.0
                self._maxs[key] = 0.0
            self._request_count = 0
            self._rejected_count = 0
            self._error_count = 0

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a consistent deep copy of the current statistics.

        Returns:
            Dict with request_count, rejected_count, error_count, and a
            per-stage stats dict {key: {request_count, sum_ms, mean_ms,
            min_ms, max_ms}}. Zeroed when nothing has been recorded.
        """
        with self._lock:
            stages: dict = {}
            for key in STAGE_KEYS:
                count = self._counts[key]
                total = self._sums[key]
                stages[key] = {
                    "request_count": count,
                    "sum_ms": round(total, 4),
                    "mean_ms": round(total / count, 4) if count else 0.0,
                    "min_ms": self._mins[key] if count else 0.0,
                    "max_ms": self._maxs[key] if count else 0.0,
                }
            return {
                "request_count": self._request_count,
                "rejected_count": self._rejected_count,
                "error_count": self._error_count,
                "latency": stages,
            }


latency_recorder = LatencyRecorder()
"""Process-wide singleton recorder for the chat pipeline."""


def record_success(latencies: dict[str, float]) -> None:
    """Record a successful chat completion (module-level convenience)."""
    latency_recorder.record_success(latencies)


def record_rejected() -> None:
    """Record a guardrail rejection (module-level convenience)."""
    latency_recorder.record_rejected()


def record_error() -> None:
    """Record a non-successful chat outcome (module-level convenience)."""
    latency_recorder.record_error()


def reset() -> None:
    """Reset the singleton recorder (module-level convenience)."""
    latency_recorder.reset()


__all__ = [
    "STAGE_KEYS",
    "LatencyRecorder",
    "latency_recorder",
    "record_success",
    "record_rejected",
    "record_error",
    "reset",
]
