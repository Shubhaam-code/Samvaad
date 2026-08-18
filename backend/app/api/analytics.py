"""GET /api/analytics/latency endpoint.

Exposes the real, thread-safe, in-memory latency aggregates collected
from POST /api/chat:

- Successful (200) completions feed the per-stage latency statistics.
- Guardrail rejections (400) increment rejected_count only.
- Other non-successful outcomes (501/503/500) increment error_count only.

When no requests have been recorded, the endpoint returns HTTP 200 with
zeroed statistics (request_count == 0 and all-zero LatencyStats) - no
fabricated samples, no nulls, no errors.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.analytics.recorder import STAGE_KEYS, latency_recorder
from app.api.schemas import AnalyticsResponse, LatencyStats

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/latency", response_model=AnalyticsResponse)
def get_latency_analytics() -> AnalyticsResponse:
    """Return aggregated chat pipeline latency statistics.

    Empty state returns 200 with zeroed statistics.
    """
    snapshot = latency_recorder.snapshot()

    return AnalyticsResponse(
        request_count=snapshot["request_count"],
        rejected_count=snapshot["rejected_count"],
        error_count=snapshot["error_count"],
        guardrail_ms=LatencyStats(**snapshot["latency"]["guardrail_ms"]),
        retrieval_ms=LatencyStats(**snapshot["latency"]["retrieval_ms"]),
        llm_ms=LatencyStats(**snapshot["latency"]["llm_ms"]),
        grounding_ms=LatencyStats(**snapshot["latency"]["grounding_ms"]),
        total_ms=LatencyStats(**snapshot["latency"]["total_ms"]),
    )


__all__ = [
    "router",
    "STAGE_KEYS",
]
