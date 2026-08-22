"""Sarvam AI Text-to-Speech (TTS) provider adapter (Phase 5.3).

Provides high-fidelity Indic speech synthesis via Sarvam AI's production
endpoint (``https://api.sarvam.ai/text-to-speech``).

Guarantees:
- Complies with ``BaseTTS`` interface.
- Synthesizes natural Indic voices (Hindi, Indian English, Tamil, Bengali, Telugu, etc.).
- Converts base64 audio response into validated raw in-memory WAV bytes.
- Credential safety: API keys are redacted and never logged or exposed.
- Testability: supports injected ``http_client`` for offline unit tests.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Optional

import httpx

from .base import (
    DEFAULT_MAX_TEXT_LENGTH,
    BaseTTS,
    TTSError,
    validate_speed,
    validate_text,
)
from .models import TTSRequest, TTSResponse
from .validation import validate_tts_audio

logger = logging.getLogger(__name__)

DEFAULT_SARVAM_TTS_ENDPOINT = "https://api.sarvam.ai/text-to-speech"
DEFAULT_SARVAM_TTS_MODEL = "bulbul:v2"
DEFAULT_SARVAM_SPEAKER = "anushka"

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
    """Map standard language code to Sarvam TTS target_language_code."""
    if not lang:
        return "hi-IN"
    normalized = lang.strip().lower()
    return _SARVAM_LANG_MAP.get(normalized, "hi-IN")


def _redact_key(text: str, key: Optional[str]) -> str:
    """Redact sensitive API keys from exception text."""
    if not key or not text:
        return text
    return text.replace(key, "[REDACTED]")


def is_sarvam_tts_configured(api_key: Optional[str]) -> bool:
    """Check if Sarvam TTS has an active API key configured."""
    if not api_key:
        return False
    return bool(api_key.strip())


# Speakers accepted by bulbul:v2, per the Sarvam API's own rejection message.
# Keep this list exact: any name present here but unknown to the model is
# forwarded verbatim and the request fails with HTTP 400. Names outside this
# set fall back to the configured default speaker instead.
SARVAM_SPEAKERS = {
    "anushka",
    "abhilash",
    "manisha",
    "vidya",
    "arya",
    "karun",
    "hitesh",
}


class SarvamTTS(BaseTTS):
    """Sarvam AI Text-to-Speech adapter.

    Args:
        api_key: Sarvam AI subscription key
        model: TTS model identifier (default: "bulbul:v2")
        speaker: Voice speaker identifier (default: "meera")
        endpoint: API endpoint URL
        timeout_seconds: Request timeout in seconds
        max_text_length: Maximum allowed text length
        max_audio_size_mb: Maximum allowed audio response size
        http_client: Optional injected httpx.Client for testing
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_SARVAM_TTS_MODEL,
        speaker: str = DEFAULT_SARVAM_SPEAKER,
        endpoint: str = DEFAULT_SARVAM_TTS_ENDPOINT,
        timeout_seconds: float = 30.0,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        max_audio_size_mb: float = 10.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        if not api_key and http_client is None:
            raise ValueError("api_key is required when no custom http_client is injected")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_text_length <= 0:
            raise ValueError("max_text_length must be positive")
        if max_audio_size_mb <= 0:
            raise ValueError("max_audio_size_mb must be positive")

        self._api_key = api_key.strip() if api_key else ""
        self._model = model.strip() or DEFAULT_SARVAM_TTS_MODEL
        self._speaker = speaker.strip() or DEFAULT_SARVAM_SPEAKER
        self._endpoint = endpoint.strip() or DEFAULT_SARVAM_TTS_ENDPOINT
        self._timeout_seconds = timeout_seconds
        self._max_text_length = max_text_length
        self._max_audio_size_mb = max_audio_size_mb
        self._http_client = http_client

    @property
    def model_name(self) -> str:
        """The active Sarvam TTS model identifier."""
        return self._model

    @property
    def provider(self) -> str:
        """Provider name reported on responses."""
        return "sarvam"

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize text into speech audio bytes using Sarvam AI.

        Args:
            request: Validated TTSRequest containing text and voice parameters

        Returns:
            TTSResponse containing raw WAV audio bytes and duration

        Raises:
            TTSError: On network, authentication, or synthesis errors
        """
        # Validate text length and clean content
        validated_text = validate_text(request.text, max_length=self._max_text_length)
        speed = validate_speed(request.speed) if request.speed is not None else 1.0

        target_lang = _map_language_to_sarvam(request.language)
        if request.voice and request.voice.lower() in SARVAM_SPEAKERS:
            speaker = request.voice.lower()
        else:
            speaker = self._speaker

        headers = {
            "api-subscription-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": [validated_text],
            "target_language_code": target_lang,
            "speaker": speaker,
            "pitch": 0.0,
            "pace": speed,
            "loudness": 1.0,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            "model": self._model,
        }

        start_time = time.perf_counter()

        try:
            if self._http_client is not None:
                resp = self._http_client.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    resp = client.post(
                        self._endpoint,
                        headers=headers,
                        json=payload,
                    )
        except httpx.TimeoutException as exc:
            raise TTSError(f"Sarvam TTS request timed out after {self._timeout_seconds}s") from exc
        except Exception as exc:
            safe_msg = _redact_key(str(exc), self._api_key)
            raise TTSError(f"Sarvam TTS network error: {safe_msg}") from exc

        duration_sec = time.perf_counter() - start_time

        if resp.status_code != 200:
            safe_body = _redact_key(resp.text, self._api_key)
            if resp.status_code in (401, 403):
                raise TTSError(f"Sarvam TTS authentication failed (status {resp.status_code}): {safe_body}")
            raise TTSError(f"Sarvam TTS failed with status {resp.status_code}: {safe_body}")

        try:
            res_data = resp.json()
        except Exception as exc:
            raise TTSError(f"Failed to parse Sarvam TTS JSON response: {exc}") from exc

        audios = res_data.get("audios", [])
        if not audios or not isinstance(audios, list):
            raise TTSError("Sarvam TTS returned empty audio payload")

        try:
            audio_bytes = base64.b64decode(audios[0])
        except Exception as exc:
            raise TTSError(f"Failed to decode base64 audio from Sarvam TTS: {exc}") from exc

        # Validate container bytes
        validate_tts_audio(
            audio_bytes,
            "wav",
            max_bytes=int(self._max_audio_size_mb * 1024 * 1024),
        )

        return TTSResponse(
            audio=audio_bytes,
            content_type="audio/wav",
            format="wav",
            model=self._model,
            provider=self.provider,
            latency_ms=round(duration_sec * 1000.0, 2),
            character_count=len(validated_text),
        )
