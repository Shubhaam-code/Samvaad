"""
FastAPI application entrypoint.

Phase 1: Project Foundation
- Health check endpoint
- CORS for local development
- Configuration via environment variables
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    app_name: str = "rag-backend"
    cors_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


settings = Settings()

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


@app.get("/health")
async def health() -> dict[str, str]:
    """Lightweight health check used for readiness/liveness probes."""
    return {"status": "ok", "service": settings.app_name}
