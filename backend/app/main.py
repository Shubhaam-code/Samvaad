"""
FastAPI application entrypoint.

Phase 1: Project Foundation
- Health check endpoint
- CORS for local development
- Configuration via environment variables

Phase 6.2: Chat API integration
- POST /api/chat router
- Global JSON exception handlers for RetrievalError / LLMError / unexpected

Phase 6.3: Latency analytics
- GET /api/analytics/latency router
- Error outcomes recorded into the analytics recorder

Phase 6.4: Real LLM provider
- Settings extracted to app/settings.py (LLM provider configuration)
- get_llm() in app/api/dependencies.py wires the OpenAI-compatible
  provider when configured (API key or custom base URL)
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.analytics import record_error
from app.api.analytics import router as analytics_router
from app.api.chat import router as chat_router
from app.api.voice import router as voice_router
from app.llm.base import LLMError
from app.retrieval.orchestrator import RetrievalError
from app.settings import settings
from app.stt.base import STTError
from app.tts.base import TTSError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HH Goa RAG Backend",
    version="0.1.0",
    description="Voice-Enabled RAG system backend (Phase 1 - foundation).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(voice_router)
app.include_router(analytics_router)


@app.exception_handler(STTError)
async def stt_error_handler(request: Request, exc: STTError) -> JSONResponse:
    """Handle STT provider failures with a structured JSON response."""
    logger.error("STT failure: %s", exc)
    record_error()
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "STT_FAILED", "message": str(exc)}},
    )


@app.exception_handler(TTSError)
async def tts_error_handler(request: Request, exc: TTSError) -> JSONResponse:
    """Handle TTS provider failures with a structured JSON response."""
    logger.error("TTS failure: %s", exc)
    record_error()
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "TTS_FAILED", "message": str(exc)}},
    )


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    """Handle retrieval pipeline failures with a structured JSON response."""
    logger.error("Retrieval failure: %s", exc)
    record_error()
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "RETRIEVAL_FAILED", "message": str(exc)}},
    )


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    """Handle LLM provider failures with a structured JSON response."""
    logger.error("LLM failure: %s", exc)
    record_error()
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "LLM_FAILED", "message": str(exc)}},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected failures with a structured JSON response."""
    logger.exception("Unexpected error on %s %s", request.method, request.url.path)
    record_error()
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "INTERNAL_ERROR", "message": str(exc)}},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Lightweight health check used for readiness/liveness probes."""
    return {"status": "ok", "service": settings.app_name}
