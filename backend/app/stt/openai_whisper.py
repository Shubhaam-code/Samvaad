"""OpenAI-compatible Whisper STT provider adapter (Phase 7.2).

Implements the ``BaseSTT`` contract on top of the official OpenAI Python
SDK v1.x (``client.audio.transcriptions.create``).

Endpoint behaviour
------------------
- ``STT_BASE_URL`` absent / empty → official OpenAI API
  (https://api.openai.com/v1).  Requires ``STT_API_KEY``.
- ``STT_BASE_URL`` set to any other URL → OpenAI-compatible Whisper
  endpoint (self-hosted faster-whisper, vLLM Whisper, etc.).  Key is
  optional when the server does not require authentication.

Audio handling
--------------
Audio is validated strictly (size, extension, MIME, magic bytes) before
being sent to the provider.  The bytes are passed as an in-memory file
tuple ``(filename, bytes, content_type)``; no temporary files are written.

Credential safety
-----------------
- API keys come from the environment / settings only — never from code.
- API keys are never logged, echoed in exception text, or included in
  any response field.
- If the SDK ever echoes the key in an error message, it is redacted
  to ``[REDACTED]`` before the exception is surfaced.

Testability
-----------
Pass ``client=<stub>`` to bypass real SDK client construction entirely.
The stub must expose ``audio.transcriptions.create(**kwargs)`` returning
an object with ``.text``, ``.language`` and ``.duration`` attributes (or
``None`` for fields the provider omits).  Zero network calls are made
during test collection or execution.

Language support
----------------
- ``language="en"``  → English (explicit hint to the model)
- ``language="hi"``  → Hindi (explicit hint to the model)
- ``language=None``  → automatic detection (provider-side, no hint sent)
- Any ISO 639-1 two-letter code is accepted.

Confidence
----------
OpenAI Whisper does not expose a per-transcription confidence score at
the top-level ``/audio/transcriptions`` endpoint (segment-level
``avg_logprob`` is intentionally omitted).  ``confidence`` is always
``None`` in STTResponse — it is never invented.
"""

from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI, OpenAIError

from .base import (
    BaseSTT,
    STTError,
    validate_context_prompt,
    validate_language,
    validate_transcription_text,
)
from .models import STTResponse
from .validation import DEFAULT_MAX_AUDIO_BYTES, validate_audio

# The default OpenAI REST base URL.  Any deviation from this value is treated
# as an OpenAI-compatible endpoint and does not require an API key.
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

DEFAULT_BASE_URL: str = _OPENAI_DEFAULT_BASE_URL
DEFAULT_MODEL_NAME: str = "whisper-1"
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_AUDIO_SIZE_MB: float = 10.0

_MS_ROUND = 4


def is_openai_whisper_configured(*, api_key: Optional[str], base_url: str) -> bool:
    """Return True when the provider has enough configuration to function.

    Rules (mirrors is_openai_compatible_configured in the LLM layer):
    - A non-empty API key always counts as configured.
    - A non-default base URL (any OpenAI-compatible Whisper server) counts
      as configured even without a key — many local servers need no auth.
    - Missing key AND the default OpenAI base URL → NOT configured.

    Args:
        api_key: Resolved API key string, or None / empty string.
        base_url: Resolved base URL string (never empty after resolution).

    Returns:
        True when a usable configuration exists.
    """
    if api_key and api_key.strip():
        return True
    return base_url.rstrip("/") != _OPENAI_DEFAULT_BASE_URL.rstrip("/")


class OpenAIWhisperSTT(BaseSTT):
    """OpenAI Whisper API speech-to-text provider.

    Supports both the official OpenAI API and any OpenAI-compatible
    Whisper endpoint via ``base_url``.  Language hints (``en``, ``hi``)
    and automatic detection (``language=None``) are supported.

    Args:
        api_key: Provider API key.  Required for the default OpenAI base
            URL; optional for compatible local endpoints.
        base_url: Endpoint base URL.  Absent / None → official OpenAI API.
            Any other value → OpenAI-compatible endpoint.
        model_name: Whisper model served at the endpoint (e.g. ``whisper-1``
            for hosted OpenAI; ``Systran/faster-whisper-large-v3`` or similar
            for local servers).
        language: Default language hint applied when the STTRequest carries
            no language.  None = automatic detection.
        timeout_seconds: Hard timeout for each provider API call.
        max_audio_size_mb: Maximum accepted upload size in MB.  Validated
            before bytes are sent to the provider.
        client: Injected SDK-compatible object for offline tests.  When
            provided, no ``openai.OpenAI`` instance is created and no
            network connection is possible.  Must expose
            ``audio.transcriptions.create(**kwargs)``.

    Raises:
        ValueError: If constructor arguments are invalid.
        ValueError: If no API key and the default base URL are both set
            (provider would be unusable without an injected test client).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        language: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_audio_size_mb: float = DEFAULT_MAX_AUDIO_SIZE_MB,
        client: Optional[object] = None,
    ) -> None:
        # --- argument validation ---
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name must be a non-empty string, got {model_name!r}")
        if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
            raise ValueError(f"base_url must be a non-empty string or None, got {base_url!r}")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                f"timeout_seconds must be a positive number, got {timeout_seconds!r}"
            )
        if (
            not isinstance(max_audio_size_mb, (int, float))
            or isinstance(max_audio_size_mb, bool)
            or max_audio_size_mb <= 0
        ):
            raise ValueError(
                f"max_audio_size_mb must be a positive number, got {max_audio_size_mb!r}"
            )
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError(
                f"api_key must be a string or None, got {type(api_key).__name__}"
            )

        self._api_key: str = (api_key or "").strip()
        # Resolve base URL: None or empty → canonical default
        self._base_url: str = (
            base_url.strip() if (base_url and base_url.strip()) else _OPENAI_DEFAULT_BASE_URL
        )
        self._model_name = model_name.strip()
        self._language = validate_language(language)
        self._timeout_seconds = float(timeout_seconds)
        self._max_audio_size_mb = float(max_audio_size_mb)

        if client is not None:
            # Injected client — used exclusively in tests.
            # No network connection is possible via this path.
            self._client = client
        else:
            # Production path — require either a key or a non-default URL.
            if not self._api_key and self._base_url == _OPENAI_DEFAULT_BASE_URL:
                raise ValueError(
                    "api_key is required when using the default OpenAI base URL "
                    "(or set STT_BASE_URL to a local OpenAI-compatible endpoint)"
                )
            self._client = OpenAI(
                api_key=self._api_key or "local",
                base_url=self._base_url,
                timeout=self._timeout_seconds,
            )

    # ------------------------------------------------------------------
    # BaseSTT properties
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        """Provider name reported on responses."""
        return "openai_whisper"

    @property
    def model_name(self) -> str:
        """Whisper model identifier reported on responses."""
        return self._model_name

    # ------------------------------------------------------------------
    # Public transcription method
    # ------------------------------------------------------------------

    def transcribe(self, request: object) -> STTResponse:
        """Transcribe an audio request using the OpenAI Whisper API.

        Audio is validated strictly before any provider call:
        1. Maximum size check (configurable, default 10 MB).
        2. Supported extension check (.wav, .mp3, .m4a, .aac, .webm, .ogg).
        3. MIME type check when content_type is declared.
        4. Container magic-byte sniffing (detects corrupt / mismatched files).

        The validated bytes are sent to the provider as an in-memory file
        tuple — no temporary files are written and no audio is persisted.

        Args:
            request: Object with ``audio`` (bytes) and ``filename`` (str).
                Optional: ``content_type`` (str), ``language`` (str),
                ``prompt`` (str).

        Returns:
            STTResponse with text, language, provider, model, latency_ms,
            and duration_seconds (when the provider returns it).
            confidence is always None — Whisper does not expose it.

        Raises:
            ValueError: If request attributes are missing or audio is invalid.
            STTError: If the provider API call fails or times out.
        """
        if not hasattr(request, "audio") or not hasattr(request, "filename"):
            raise ValueError("request must have 'audio' and 'filename' attributes")

        audio_bytes: bytes = getattr(request, "audio")
        filename: str = getattr(request, "filename")
        content_type: Optional[str] = getattr(request, "content_type", None)
        req_language: Optional[str] = getattr(request, "language", None)
        prompt: Optional[str] = getattr(request, "prompt", None)

        # Layer 1–4: strict upload validation (size, extension, MIME, magic)
        max_bytes = int(self._max_audio_size_mb * 1024 * 1024)
        validated = validate_audio(
            audio_bytes,
            filename=filename,
            content_type=content_type,
            max_bytes=max_bytes,
        )

        # Resolve effective language: request hint > provider default > None (auto)
        effective_language = validate_language(req_language) or self._language
        valid_prompt = validate_context_prompt(prompt)

        # Build provider call kwargs
        # file parameter: tuple (filename, bytes, content_type) is the
        # correct form for openai SDK v1.x FileTypes.
        kwargs: dict[str, object] = {
            "model": self._model_name,
            "file": (filename, audio_bytes, validated.content_type),
            "response_format": "verbose_json",
        }
        if effective_language is not None:
            kwargs["language"] = effective_language
        if valid_prompt is not None:
            kwargs["prompt"] = valid_prompt

        # Provider call with latency measurement
        start = time.perf_counter()
        try:
            raw = self._client.audio.transcriptions.create(**kwargs)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — wrap every SDK failure
            raise self._wrap_provider_error(exc) from exc
        latency_ms = round((time.perf_counter() - start) * 1000.0, _MS_ROUND)

        return self._map_response(raw, latency_ms, default_language=effective_language)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_response(
        self,
        raw: object,
        latency_ms: float,
        default_language: Optional[str],
    ) -> STTResponse:
        """Map the provider response onto the canonical STTResponse.

        Handles both real SDK response objects and duck-typed test stubs.
        confidence is always None — Whisper does not return it at the
        transcription level.

        Args:
            raw: Provider response (TranscriptionVerbose or duck-typed stub).
            latency_ms: Measured provider call latency.
            default_language: Language hint to fall back to when the provider
                does not report a detected language.

        Returns:
            STTResponse.

        Raises:
            STTError: If the provider returned empty or missing text.
        """
        # Extract text — attribute first (SDK object), then dict fallback
        text_val: Optional[str] = None
        if hasattr(raw, "text"):
            text_val = str(raw.text) if raw.text is not None else None  # type: ignore[union-attr]
        elif isinstance(raw, dict):
            text_val = raw.get("text")

        if not text_val or not text_val.strip():
            raise STTError("OpenAI Whisper provider returned empty transcription content")

        clean_text = validate_transcription_text(text_val)

        # Extract detected language (provider may return a full name like "english")
        detected_lang: Optional[str] = None
        if hasattr(raw, "language"):
            detected_lang = raw.language  # type: ignore[union-attr]
        elif isinstance(raw, dict):
            detected_lang = raw.get("language")

        # Normalise: use validate_language for short codes; ignore long names
        try:
            lang_code = validate_language(detected_lang) or default_language
        except ValueError:
            # Provider returned a full language name (e.g. "english") — ignore,
            # fall back to the hint.
            lang_code = default_language

        # Extract audio duration (optional — provider may not return it)
        duration_sec: Optional[float] = None
        if hasattr(raw, "duration") and raw.duration is not None:  # type: ignore[union-attr]
            try:
                duration_sec = float(raw.duration)  # type: ignore[union-attr]
            except (TypeError, ValueError):
                duration_sec = None
        elif isinstance(raw, dict) and raw.get("duration") is not None:
            try:
                duration_sec = float(raw["duration"])
            except (TypeError, ValueError):
                duration_sec = None

        return STTResponse(
            text=clean_text,
            language=lang_code,
            provider=self.provider,
            model=self._model_name,
            latency_ms=latency_ms,
            duration_seconds=duration_sec,
            confidence=None,  # Whisper API does not expose transcription confidence
        )

    def _wrap_provider_error(self, exc: Exception) -> STTError:
        """Wrap a provider SDK exception into a sanitized STTError.

        The API key is redacted if it somehow appears in the exception text.
        No raw audio, headers, or request payloads are included.

        Args:
            exc: The underlying provider exception.

        Returns:
            An STTError safe to surface in API responses and logs.
        """
        text = str(exc)
        if self._api_key and self._api_key in text:
            text = text.replace(self._api_key, "[REDACTED]")
        return STTError(
            f"OpenAI Whisper provider error ({type(exc).__name__}): {text}"
        )

    def __repr__(self) -> str:
        return (
            f"OpenAIWhisperSTT(model_name={self._model_name!r}, "
            f"base_url={self._base_url!r}, "
            f"timeout_seconds={self._timeout_seconds})"
        )


def create_openai_whisper_stt(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    language: Optional[str] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_audio_size_mb: float = DEFAULT_MAX_AUDIO_SIZE_MB,
    client: Optional[object] = None,
) -> "OpenAIWhisperSTT":
    """Factory helper to create an OpenAIWhisperSTT provider instance.

    Args:
        api_key: Provider API key.
        base_url: Endpoint base URL (None → official OpenAI API).
        model_name: Whisper model identifier.
        language: Default language hint (None → automatic detection).
        timeout_seconds: Provider call timeout.
        max_audio_size_mb: Maximum accepted audio size in MB.
        client: Injected stub client for offline tests.

    Returns:
        Configured OpenAIWhisperSTT instance.
    """
    return OpenAIWhisperSTT(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        language=language,
        timeout_seconds=timeout_seconds,
        max_audio_size_mb=max_audio_size_mb,
        client=client,
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_AUDIO_SIZE_MB",
    "OpenAIWhisperSTT",
    "create_openai_whisper_stt",
    "is_openai_whisper_configured",
]
