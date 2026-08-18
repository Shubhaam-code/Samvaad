"""API package for HTTP-facing endpoints.

Phase 6.2: Chat endpoint integration over the existing real components
(guardrails, retrieval, LLM harness). No STT/TTS, no voice endpoint.
"""

from .chat import router as chat_router
from .schemas import ChatRequest, ChatResponse, Citation, LatencyBreakdown

__all__ = [
    "chat_router",
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "LatencyBreakdown",
]
