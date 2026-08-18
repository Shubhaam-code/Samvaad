"""Sarvam AI Speech-to-Text (STT) provider adapter (Phase 5.3).

Provides ultra-fast (<100ms) Indic Speech-to-Text inference via Sarvam AI's
production API endpoint (``https://api.sarvam.ai/speech-to-text``).

Guarantees:
- Provider-agnostic ``BaseSTT`` interface compliance.
- Strict audio validation (format, size, magic bytes) before dispatch.
- Multi-dialect Indic language mapping (Hindi, English, Bengali, Tamil, Telugu, etc.)
  with automatic language detection ("unknown").
- Credential safety: API keys are never echoed in error messages or logs.
- Testability: accepts injected ``http_client`` for 100% offline unit tests.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from .base import (
    BaseSTT,
    STTError,
    validate_language,
    validate_transcription_text,
)
from .models import STTRequest, STTResponse
from .validation import DEFAULT_MAX_AUDIO_BYTES, validate_audio

logger = logging.getLogger(__name__)

DEFAULT_SARVAM_STT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"
DEFAULT_SARVAM_STT_MODEL = "saaras:v2"

# Mapping ISO 639-1 / BCP-47 codes to Sarvam language_code
_SARVAM_LANG_MAP: dict[str, str] = {
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "en": "en-IN",
    "en-in": "en-IN",
    "bn": "bn-IN",
    "bn-in": "bn-IN",
    "gu": "gu-IN",
    "gu-in": "gu-IN",
    "kn": "kn-IN",
    "kn-in": "kn-IN",
    "ml": "ml-IN",
    "ml-in": "ml-IN",
    "mr": "mr-IN",
    "mr-in": "mr-IN",
    "or": "od-IN",
    "od": "od-IN",
    "pa": "pa-IN",
    "pa-in": "pa-IN",
    "ta": "ta-IN",
    "ta-in": "ta-IN",
    "te": "te-IN",
    "te-in": "te-IN",
}


def _map_language_to_sarvam(lang: Optional[str]) -> str:
    """Map standard language code to Sarvam API language_code format."""
    if not lang:
        return "unknown"
    normalized = lang.strip().lower()
    return _SARVAM_LANG_MAP.get(normalized, "unknown")


def _redact_key(text: str, key: Optional[str]) -> str:
    """Redact sensitive API keys from exception strings."""
    if not key or not text:
        return text
    return text.replace(key, "[REDACTED]")


def is_sarvam_stt_configured(api_key: Optional[str]) -> bool:
    """Check if Sarvam STT has an active API key configured."""
    if not api_key:
        return False
    return bool(api_key.strip())


class SarvamSTT(BaseSTT):
    """Sarvam AI Speech-to-Text adapter.

    Args:
        api_key: Sarvam AI subscription key
        model: Model identifier (default: "saaras:v2")
        endpoint: API endpoint URL (default: "https://api.sarvam.ai/speech-to-text")
        timeout_seconds: Request timeout in seconds
        max_audio_bytes: Maximum allowed audio upload size
        http_client: Optional injected httpx.Client for testing
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_SARVAM_STT_MODEL,
        endpoint: str = DEFAULT_SARVAM_STT_ENDPOINT,
        timeout_seconds: float = 30.0,
        max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        if not api_key and http_client is None:
            raise ValueError("api_key is required when no custom http_client is injected")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_audio_bytes <= 0:
            raise ValueError("max_audio_bytes must be positive")

        self._api_key = api_key.strip() if api_key else ""
        self._model = model.strip() or DEFAULT_SARVAM_STT_MODEL
        self._endpoint = endpoint.strip() or DEFAULT_SARVAM_STT_ENDPOINT
        self._timeout_seconds = timeout_seconds
        self._max_audio_bytes = max_audio_bytes
        self._http_client = http_client

    @property
    def model_name(self) -> str:
        """The active Sarvam STT model identifier."""
        return self._model

    @property
    def provider(self) -> str:
        """Provider name reported on responses."""
        return "sarvam"

    def transcribe(self, request: STTRequest) -> STTResponse:
        """Transcribe an audio file using Sarvam AI STT.

        Args:
            request: Validated STTRequest containing audio bytes and metadata

        Returns:
            STTResponse with transcript, language code, and latency

        Raises:
            STTError: On network, authorization, or transcription errors
        """
        # Validate audio bytes, format, and magic numbers
        validated_audio = validate_audio(
            request.audio,
            filename=request.filename,
            content_type=request.content_type,
            max_bytes=self._max_audio_bytes,
        )

        sarvam_lang = _map_language_to_sarvam(request.language)
        headers = {
            "api-subscription-key": self._api_key,
        }

        # Prepare multipart/form-data upload
        files = {
            "file": (request.filename, request.audio, validated_audio.content_type),
        }
        data = {
            "model": self._model,
            "language_code": sarvam_lang,
        }
        if request.prompt:
            data["prompt"] = request.prompt

        start_time = time.perf_counter()

        try:
            if self._http_client is not None:
                resp = self._http_client.post(
                    self._endpoint,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=self._timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    resp = client.post(
                        self._endpoint,
                        headers=headers,
                        files=files,
                        data=data,
                    )
        except httpx.TimeoutException as exc:
            raise STTError(f"Sarvam STT request timed out after {self._timeout_seconds}s") from exc
        except Exception as exc:
            safe_msg = _redact_key(str(exc), self._api_key)
            raise STTError(f"Sarvam STT network error: {safe_msg}") from exc

        duration_sec = time.perf_counter() - start_time

        if resp.status_code != 200:
            safe_body = _redact_key(resp.text, self._api_key)
            if resp.status_code in (401, 403):
                raise STTError(f"Sarvam STT authentication failed (status {resp.status_code}): {safe_body}")
            raise STTError(f"Sarvam STT failed with status {resp.status_code}: {safe_body}")

        try:
            payload = resp.json()
        except Exception as exc:
            raise STTError(f"Failed to parse Sarvam STT JSON response: {exc}") from exc

        transcript = payload.get("transcript", "")
        detected_lang = payload.get("language_code", sarvam_lang)

        # Validate that transcribed text is clean
        transcript_clean = validate_transcription_text(transcript)

        return STTResponse(
            text=transcript_clean,
            language=detected_lang,
            confidence=None,
            duration_seconds=round(duration_sec, 3),
            latency_ms=round(duration_sec * 1000.0, 2),
            model=self._model,
            provider=self.provider,
        )
