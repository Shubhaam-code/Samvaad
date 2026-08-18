"""FastAPI dependency wiring for the chat API.

Central place where production components are resolved per-request.
Dependencies are intentionally provider-agnostic and overridable:

- ``get_llm``: Returns the configured real LLM provider, or None if no
  real provider is configured. A provider is configured when the
  OpenAI-compatible settings are present (API key, or a custom base
  URL for a local compatible server); otherwise the endpoint returns
  501. FakeLLM must NEVER be returned here in production - tests
  inject it through dependency_overrides instead.
- ``get_stt``: Returns the configured real STT provider (OpenAIWhisperSTT),
  or None if no real provider is configured. FakeSTT is NEVER returned
  here - tests inject it through dependency_overrides instead. When
  get_stt() returns None a future /api/voice-query endpoint should return
  501.
- ``get_orchestrator``: Returns a RetrievalOrchestrator wired to the
  persisted vector index, or None if no index is configured/built.
  The endpoint returns 503 when this is None.
- ``get_guardrail_pipeline`` / ``get_grounding_verifier``: Always
  available - both are deterministic and have no external dependencies,
  so the input guardrail can run even without an LLM or index.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from app.guardrails.grounding_verifier import GroundingVerifier
from app.guardrails.pipeline import GuardrailPipeline
from app.embedding.huggingface import HuggingFaceEmbedder
from app.indexing.loader import index_exists, load_index, resolve_index_dir
from app.llm.openai_compatible import (
    DEFAULT_BASE_URL as LLM_DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    OpenAICompatibleLLM,
    is_openai_compatible_configured,
)
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.settings import settings
from app.stt.base import BaseSTT
from app.stt.openai_whisper import (
    OpenAIWhisperSTT,
    is_openai_whisper_configured,
)
from app.tts.base import BaseTTS
from app.tts.openai_tts import (
    OpenAITTS,
    is_openai_tts_configured,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider caches
# ---------------------------------------------------------------------------
# Settings are loaded once at startup and are effectively immutable at
# runtime, so each distinct configuration key gets exactly one shared
# client instance (connection pooling).  The locks make concurrent first
# requests safe (FastAPI sync endpoints run in a threadpool).

_llm_cache: dict[tuple[object, ...], OpenAICompatibleLLM] = {}
_llm_cache_lock = threading.Lock()

_stt_cache: dict[tuple[object, ...], OpenAIWhisperSTT] = {}
_stt_cache_lock = threading.Lock()

_tts_cache: dict[tuple[object, ...], OpenAITTS] = {}
_tts_cache_lock = threading.Lock()

_orchestrator_cache: dict[tuple[object, ...], RetrievalOrchestrator] = {}
_orchestrator_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# LLM dependency
# ---------------------------------------------------------------------------


def get_llm() -> Optional[OpenAICompatibleLLM]:
    """Resolve the configured real LLM provider.

    FakeLLM is NEVER returned here. Tests inject it via dependency_overrides.

    Returns:
        A cached OpenAICompatibleLLM when the provider is configured,
        or None when no real provider is configured (endpoint returns
        501 in that case).
    """
    if settings.llm_provider != "openai_compatible":
        return None

    base_url = settings.llm_base_url or LLM_DEFAULT_BASE_URL
    if not is_openai_compatible_configured(
        api_key=settings.llm_api_key,
        base_url=base_url,
    ):
        return None

    key: tuple[object, ...] = (
        settings.llm_api_key,
        base_url,
        settings.llm_model or DEFAULT_MODEL_NAME,
        settings.llm_timeout_seconds,
    )
    with _llm_cache_lock:
        if key not in _llm_cache:
            _llm_cache[key] = OpenAICompatibleLLM(
                api_key=settings.llm_api_key,
                base_url=base_url,
                model_name=settings.llm_model or DEFAULT_MODEL_NAME,
                timeout_seconds=settings.llm_timeout_seconds,
            )
        return _llm_cache[key]


# ---------------------------------------------------------------------------
# STT dependency
# ---------------------------------------------------------------------------


def get_stt() -> Optional[BaseSTT]:
    """Resolve the configured real STT provider (Sarvam AI or OpenAI Whisper).

    FakeSTT is NEVER returned here. Tests inject it via dependency_overrides.

    Returns:
        A cached BaseSTT (SarvamSTT or OpenAIWhisperSTT) when the provider is configured,
        or None when no real provider is configured.
    """
    if settings.stt_provider == "sarvam":
        from app.stt.sarvam_stt import SarvamSTT, is_sarvam_stt_configured  # noqa: PLC0415
        api_key = settings.sarvam_api_key or settings.stt_api_key
        if not is_sarvam_stt_configured(api_key):
            return None
        key: tuple[object, ...] = (
            "sarvam",
            api_key,
            settings.sarvam_stt_model,
            settings.stt_timeout_seconds,
            settings.stt_max_audio_size_mb,
        )
        with _stt_cache_lock:
            if key not in _stt_cache:
                _stt_cache[key] = SarvamSTT(
                    api_key=api_key,
                    model=settings.sarvam_stt_model,
                    timeout_seconds=settings.stt_timeout_seconds,
                    max_audio_size_mb=int(settings.stt_max_audio_size_mb * 1024 * 1024),
                )
            return _stt_cache[key]

    if settings.stt_provider != "openai_whisper":
        return None

    # Resolve credentials: STT-specific key takes priority, then LLM key as
    # fallback (allows sharing one OpenAI key for both chat and transcription).
    api_key = settings.stt_api_key or settings.llm_api_key

    # Resolve base URL: STT-specific URL takes priority.
    # None / empty → official OpenAI API (handled inside OpenAIWhisperSTT).
    base_url_override: Optional[str] = settings.stt_base_url or settings.llm_base_url or None

    # is_openai_whisper_configured needs the resolved URL string, not None
    from app.stt.openai_whisper import _OPENAI_DEFAULT_BASE_URL  # noqa: PLC0415
    resolved_url = base_url_override or _OPENAI_DEFAULT_BASE_URL

    if not is_openai_whisper_configured(api_key=api_key, base_url=resolved_url):
        return None

    key: tuple[object, ...] = (
        "openai_whisper",
        api_key,
        base_url_override,
        settings.stt_model or "whisper-1",
        settings.stt_language,
        settings.stt_timeout_seconds,
        settings.stt_max_audio_size_mb,
    )
    with _stt_cache_lock:
        if key not in _stt_cache:
            _stt_cache[key] = OpenAIWhisperSTT(
                api_key=api_key,
                base_url=base_url_override,   # None → OpenAI default (handled by provider)
                model_name=settings.stt_model or "whisper-1",
                language=settings.stt_language,
                timeout_seconds=settings.stt_timeout_seconds,
                max_audio_size_mb=settings.stt_max_audio_size_mb,
            )
        return _stt_cache[key]


# ---------------------------------------------------------------------------
# TTS dependency
# ---------------------------------------------------------------------------


def get_tts() -> Optional[BaseTTS]:
    """Resolve the configured real TTS provider (Sarvam AI or OpenAI TTS).

    FakeTTS is NEVER returned here. Tests inject it via dependency_overrides.

    Returns:
        A cached BaseTTS (SarvamTTS or OpenAITTS) when the provider is configured,
        or None when no real provider is configured.
    """
    if settings.tts_provider == "sarvam":
        from app.tts.sarvam_tts import SarvamTTS, is_sarvam_tts_configured  # noqa: PLC0415
        api_key = settings.sarvam_api_key or settings.tts_api_key
        if not is_sarvam_tts_configured(api_key):
            return None
        key: tuple[object, ...] = (
            "sarvam",
            api_key,
            settings.sarvam_tts_model,
            settings.sarvam_speaker,
            settings.tts_timeout_seconds,
            settings.tts_max_text_length,
            settings.tts_max_audio_size_mb,
        )
        with _tts_cache_lock:
            if key not in _tts_cache:
                _tts_cache[key] = SarvamTTS(
                    api_key=api_key,
                    model=settings.sarvam_tts_model,
                    speaker=settings.sarvam_speaker,
                    timeout_seconds=settings.tts_timeout_seconds,
                    max_text_length=settings.tts_max_text_length,
                    max_audio_size_mb=settings.tts_max_audio_size_mb,
                )
            return _tts_cache[key]

    if settings.tts_provider != "openai_tts":
        return None

    # Resolve credentials: TTS-specific key takes priority, then LLM key as fallback
    api_key = settings.tts_api_key or settings.llm_api_key

    # Resolve base URL: TTS-specific URL takes priority
    base_url_override: Optional[str] = settings.tts_base_url or settings.llm_base_url or None

    from app.tts.openai_tts import _OPENAI_DEFAULT_BASE_URL  # noqa: PLC0415
    resolved_url = base_url_override or _OPENAI_DEFAULT_BASE_URL

    if not is_openai_tts_configured(api_key=api_key, base_url=resolved_url):
        return None

    key: tuple[object, ...] = (
        "openai_tts",
        api_key,
        base_url_override,
        settings.tts_model or "tts-1",
        settings.tts_voice or "alloy",
        settings.tts_output_format or "mp3",
        settings.tts_speed,
        settings.tts_timeout_seconds,
        settings.tts_max_text_length,
        settings.tts_max_audio_size_mb,
    )
    with _tts_cache_lock:
        if key not in _tts_cache:
            _tts_cache[key] = OpenAITTS(
                api_key=api_key,
                base_url=base_url_override,
                model_name=settings.tts_model or "tts-1",
                voice=settings.tts_voice or "alloy",
                output_format=settings.tts_output_format or "mp3",
                speed=settings.tts_speed,
                timeout_seconds=settings.tts_timeout_seconds,
                max_text_length=settings.tts_max_text_length,
                max_audio_size_mb=settings.tts_max_audio_size_mb,
            )
        return _tts_cache[key]


# ---------------------------------------------------------------------------
# Other dependencies
# ---------------------------------------------------------------------------


def get_orchestrator() -> Optional[RetrievalOrchestrator]:
    """Resolve the retrieval orchestrator wired to the persisted vector index.

    Loads the persisted production index (vector store + chunk resolver +
    manifest) built by ``scripts.build_index`` and wires it into a real
    RetrievalOrchestrator with the real HuggingFace embedder. Fake
    components are never used here.

    Returns:
        A RetrievalOrchestrator over the persisted index, or None when
        no index is configured/built yet (endpoint returns 503 in that
        case).

    Raises:
        IndexCompatibilityError: If the configured index exists but is
            incompatible (wrong embedding model/dimension/backend) or
            corrupt - a clear error instead of bad retrieval results.
    """
    index_dir = resolve_index_dir(settings.rag_index_dir)
    if index_dir is None or not index_exists(index_dir):
        return None

    key: tuple[object, ...] = (
        str(index_dir),
        settings.rag_embedding_model,
        settings.rag_embedding_device,
        settings.rag_vector_store,
        settings.rag_top_k,
    )
    with _orchestrator_cache_lock:
        cached = _orchestrator_cache.get(key)
        if cached is not None:
            return cached

        vector_store, resolver, manifest = load_index(
            index_dir,
            expected_model_name=settings.rag_embedding_model,
            expected_backend=settings.rag_vector_store,
        )
        embedder = HuggingFaceEmbedder(
            model_name=settings.rag_embedding_model,
            device=settings.rag_embedding_device,
            local_files_only=True,
        )
        orchestrator = RetrievalOrchestrator(
            embedder=embedder,
            vector_store=vector_store,
            resolver=resolver,
            top_k=settings.rag_top_k,
        )
        _orchestrator_cache[key] = orchestrator
        logger.info(
            "Wired orchestrator to index '%s' (backend=%s, dimension=%d, "
            "vectors=%d, top_k=%d)",
            index_dir,
            manifest.vector_store.backend,
            manifest.embedding.dimension,
            vector_store.count,
            settings.rag_top_k,
        )
        return orchestrator


def get_guardrail_pipeline() -> GuardrailPipeline:
    """Resolve the pre-retrieval guardrail pipeline (always available)."""
    return GuardrailPipeline()


def get_grounding_verifier() -> GroundingVerifier:
    """Resolve the post-generation grounding verifier (always available)."""
    return GroundingVerifier()


__all__ = [
    "get_llm",
    "get_stt",
    "get_tts",
    "get_orchestrator",
    "get_guardrail_pipeline",
    "get_grounding_verifier",
]
