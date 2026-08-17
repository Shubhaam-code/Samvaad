"""Application settings loaded from environment variables / .env file.

Extracted from ``app.main`` (Phase 6.4) so that dependency wiring can
read configuration without importing the FastAPI app (avoiding a
circular import between ``app.main`` and ``app.api.dependencies``).

Phase 6.4 adds the LLM provider configuration:
- ``LLM_PROVIDER``: which provider implementation to wire (only
  ``openai_compatible`` is supported so far; anything else leaves the
  provider unconfigured and /api/chat returns 501)
- ``LLM_API_KEY``: provider API key (never hardcoded, never committed)
- ``LLM_BASE_URL``: optional override for any OpenAI-compatible endpoint
  (local servers such as Ollama / LM Studio / vLLM need no API key)
- ``LLM_MODEL``: model identifier served at the endpoint
- ``LLM_TIMEOUT_SECONDS``: provider call timeout

Values are read from the process environment first, then from
``backend/.env`` if present. See .env.example for the full template.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_rag_index_dir() -> str:
    """Default RAG index directory, resolved relative to the backend root.

    Repo-derived (not machine-specific): ``backend/data/index``. This
    keeps the builder and the runtime consistent regardless of the
    current working directory.
    """
    backend_root = Path(__file__).resolve().parents[1]
    return str(backend_root / "data" / "index")


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "rag-backend"
    cors_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # LLM provider configuration (Phase 6.4)
    llm_provider: str = "openai_compatible"
    llm_api_key: Optional[str] = Field(
        None,
        description="OpenAI-compatible provider API key (from environment only)",
    )
    llm_base_url: Optional[str] = Field(
        None,
        description="OpenAI-compatible base URL (None = default OpenAI endpoint)",
    )
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = Field(
        60.0,
        gt=0.0,
        description="Provider call timeout in seconds",
    )

    # STT provider configuration
    stt_provider: str = "openai_whisper"
    stt_api_key: Optional[str] = Field(
        None,
        description="STT provider API key (from environment only)",
    )
    stt_base_url: Optional[str] = Field(
        None,
        description="STT base URL (None = default OpenAI endpoint)",
    )
    stt_model: str = "whisper-1"
    stt_language: Optional[str] = Field(
        None,
        description="Default STT language hint (None = automatic detection)",
    )
    stt_timeout_seconds: float = Field(
        30.0,
        gt=0.0,
        description="STT provider call timeout in seconds",
    )
    stt_max_audio_size_mb: float = Field(
        10.0,
        gt=0.0,
        description="Maximum accepted audio upload size in MB",
    )

    # TTS provider configuration
    tts_provider: str = "openai_tts"
    tts_api_key: Optional[str] = Field(
        None,
        description="TTS provider API key (from environment only)",
    )
    tts_base_url: Optional[str] = Field(
        None,
        description="TTS base URL (None = default OpenAI endpoint)",
    )
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    tts_output_format: str = "mp3"
    tts_speed: float = Field(
        1.0,
        ge=0.25,
        le=4.0,
        description="Default speech speed multiplier",
    )
    tts_timeout_seconds: float = Field(
        30.0,
        gt=0.0,
        description="TTS provider call timeout in seconds",
    )
    tts_max_text_length: int = Field(
        4096,
        gt=0,
        description="Maximum accepted text length for synthesis",
    )
    tts_max_audio_size_mb: float = Field(
        10.0,
        gt=0.0,
        description="Maximum accepted audio response size in MB",
    )

    # RAG index configuration (Phase 5.3)
    rag_index_dir: str = Field(
        default_factory=_default_rag_index_dir,
        description="Directory of the persisted RAG index (built by scripts.build_index)",
    )
    rag_vector_store: str = Field(
        "faiss",
        description="Vector store backend for the index ('faiss' or 'numpy')",
    )
    rag_embedding_model: str = Field(
        "intfloat/multilingual-e5-small",
        description="Embedding model the index must have been built with",
    )
    rag_embedding_device: str = Field(
        "auto",
        description="Embedding inference device ('auto', 'cpu', or 'cuda'/'cuda:N')",
    )
    rag_top_k: int = Field(
        5,
        ge=1,
        description="Default number of nearest neighbors to retrieve",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


settings = Settings()

__all__ = [
    "Settings",
    "settings",
]
