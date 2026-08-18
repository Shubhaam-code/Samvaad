"""Analytics package for chat pipeline latency instrumentation.

Phase 6.3: Lightweight, thread-safe, in-memory aggregation of the REAL
latency measurements produced by POST /api/chat. No database, no
external monitoring service, no fabricated metrics.

- recorder: LatencyRecorder singleton + record_success/record_rejected/
  record_error wrappers.
"""

from .recorder import (
    STAGE_KEYS,
    LatencyRecorder,
    latency_recorder,
    record_error,
    record_rejected,
    record_success,
    reset,
)

__all__ = [
    "STAGE_KEYS",
    "LatencyRecorder",
    "latency_recorder",
    "record_success",
    "record_rejected",
    "record_error",
    "reset",
]
