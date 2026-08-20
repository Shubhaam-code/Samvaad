"""GET /api/analytics/latency endpoint.

Exposes the real, thread-safe, in-memory latency aggregates collected
from POST /api/chat and POST /api/voice-query:

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


def _build_stats(stage_key: str, snapshot: dict) -> LatencyStats:
    stage = snapshot["latency"][stage_key]
    return LatencyStats(**stage)


@router.get("/latency", response_model=AnalyticsResponse)
def get_latency_analytics() -> AnalyticsResponse:
    """Return aggregated pipeline latency statistics with percentile bands."""
    snapshot = latency_recorder.snapshot()

    return AnalyticsResponse(
        request_count=snapshot["request_count"],
        rejected_count=snapshot["rejected_count"],
        error_count=snapshot["error_count"],
        sub_200ms_achieved=snapshot["sub_200ms_achieved"],
        stt_ms=_build_stats("stt_ms", snapshot),
        retrieval_ms=_build_stats("retrieval_ms", snapshot),
        llm_ms=_build_stats("llm_ms", snapshot),
        tts_ms=_build_stats("tts_ms", snapshot),
        guardrail_ms=_build_stats("guardrail_ms", snapshot),
        grounding_ms=_build_stats("grounding_ms", snapshot),
        total_ms=_build_stats("total_ms", snapshot),
    )


__all__ = [
    "router",
    "STAGE_KEYS",
]
