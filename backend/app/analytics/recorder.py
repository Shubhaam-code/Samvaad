"""Thread-safe, in-memory latency aggregation for the chat and voice pipelines.

Records ONLY real measurements produced by POST /api/chat and
POST /api/voice-query:

- ``record_success``: called for 200 completions; updates per-stage
  latency aggregates and percentile sample windows.
- ``record_rejected``: called for 400 guardrail rejections; increments
  rejected_count ONLY - latency aggregates are never touched.
- ``record_error``: called for 501/503/500 outcomes; increments
  error_count ONLY.

No database, no external monitoring service, no fabricated metrics:
every number stored comes from an actual endpoint measurement.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Deque, Final, Optional

# Stage keys recorded per successful request.
STAGE_KEYS: Final[tuple[str, ...]] = (
    "stt_ms",
    "retrieval_ms",
    "llm_ms",
    "tts_ms",
    "guardrail_ms",
    "grounding_ms",
    "total_ms",
)

# Dashboard-facing stages (STT -> Retrieval -> LLM -> TTS).
DASHBOARD_STAGE_KEYS: Final[tuple[str, ...]] = (
    "stt_ms",
    "retrieval_ms",
    "llm_ms",
    "tts_ms",
)

MAX_SAMPLES: Final[int] = 500
SUB_200MS_TARGET: Final[float] = 200.0


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute a percentile using linear interpolation."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 4)

    k = (len(sorted_vals) - 1) * (p / 100.0)
    floor_idx = math.floor(k)
    ceil_idx = math.ceil(k)
    if floor_idx == ceil_idx:
        return round(sorted_vals[int(k)], 4)

    lower = sorted_vals[int(floor_idx)]
    upper = sorted_vals[int(ceil_idx)]
    return round(lower + (upper - lower) * (k - floor_idx), 4)


def compute_stage_percentiles(values: list[float]) -> dict[str, float]:
    """Return P50, P70, and P100 for a list of latency samples."""
    if not values:
        return {"p50_ms": 0.0, "p70_ms": 0.0, "p100_ms": 0.0}

    sorted_vals = sorted(values)
    return {
        "p50_ms": _percentile(sorted_vals, 50.0),
        "p70_ms": _percentile(sorted_vals, 70.0),
        "p100_ms": round(sorted_vals[-1], 4),
    }


class LatencyRecorder:
    """Accumulates real pipeline latency statistics in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._sums: dict[str, float] = {}
        self._mins: dict[str, float] = {}
        self._maxs: dict[str, float] = {}
        self._samples: dict[str, Deque[float]] = {}
        for key in STAGE_KEYS:
            self._counts[key] = 0
            self._sums[key] = 0.0
            self._mins[key] = 0.0
            self._maxs[key] = 0.0
            self._samples[key] = deque(maxlen=MAX_SAMPLES)
        self._request_count = 0
        self._rejected_count = 0
        self._error_count = 0

    def record_success(self, latencies: dict[str, float]) -> None:
        """Record a successful pipeline completion with its real latencies."""
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
                self._samples[key].append(value)
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
                self._samples[key].clear()
            self._request_count = 0
            self._rejected_count = 0
            self._error_count = 0

    def snapshot(self) -> dict:
        """Return a consistent deep copy of the current statistics."""
        with self._lock:
            stages: dict = {}
            for key in STAGE_KEYS:
                count = self._counts[key]
                total = self._sums[key]
                sample_list = list(self._samples[key])
                percentiles = compute_stage_percentiles(sample_list)
                stages[key] = {
                    "request_count": count,
                    "sum_ms": round(total, 4),
                    "mean_ms": round(total / count, 4) if count else 0.0,
                    "min_ms": self._mins[key] if count else 0.0,
                    "max_ms": self._maxs[key] if count else 0.0,
                    **percentiles,
                }
            total_p50 = stages["total_ms"]["p50_ms"]
            return {
                "request_count": self._request_count,
                "rejected_count": self._rejected_count,
                "error_count": self._error_count,
                "sub_200ms_achieved": total_p50 > 0.0 and total_p50 < SUB_200MS_TARGET,
                "latency": stages,
            }


latency_recorder = LatencyRecorder()
"""Process-wide singleton recorder for the chat and voice pipelines."""


def record_success(latencies: dict[str, float]) -> None:
    """Record a successful chat or voice completion (module-level convenience)."""
    latency_recorder.record_success(latencies)


def record_rejected() -> None:
    """Record a guardrail rejection (module-level convenience)."""
    latency_recorder.record_rejected()


def record_error() -> None:
    """Record a non-successful chat or voice outcome (module-level convenience)."""
    latency_recorder.record_error()


def reset() -> None:
    """Reset the singleton recorder (module-level convenience)."""
    latency_recorder.reset()


__all__ = [
    "STAGE_KEYS",
    "DASHBOARD_STAGE_KEYS",
    "MAX_SAMPLES",
    "SUB_200MS_TARGET",
    "LatencyRecorder",
    "compute_stage_percentiles",
    "latency_recorder",
    "record_success",
    "record_rejected",
    "record_error",
    "reset",
]
