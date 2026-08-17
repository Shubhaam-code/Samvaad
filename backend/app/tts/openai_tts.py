"""OpenAI-compatible TTS provider adapter.

Implements the ``BaseTTS`` contract on top of the official OpenAI Python
SDK v1.x (``client.audio.speech.create``).

Endpoint behavior:
- ``TTS_BASE_URL`` absent / empty -> official OpenAI API (https://api.openai.com/v1).
  Requires ``TTS_API_KEY`` (or falls back to ``LLM_API_KEY``).
- ``TTS_BASE_URL`` set to any other URL -> OpenAI-compatible TTS endpoint
  (self-hosted Kokoro, Piper, vLLM TTS, etc.). Key is optional when local
  server requires no auth.

Audio handling:
- Audio is kept in memory as transient bytes (never stored on disk).
- Validated strictly (size bounds, container magic-byte verification).

Credential safety:
- API keys come from environment / settings only.
- API keys are never logged, echoed in exception text, or exposed in repr.
- If the SDK echoes the key in an error message, it is redacted to ``[REDACTED]``.

Testability:
- Pass ``client=<stub>`` to bypass SDK client construction entirely.
- Zero network calls during tests.
"""

from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI

from .base import (
    DEFAULT_MAX_TEXT_LENGTH,
    MAX_SPEED,
    MIN_SPEED,
    BaseTTS,
    TTSError,
    validate_language,
    validate_output_format,
    validate_speed,
    validate_text,
    validate_voice,
)
from .models import TTSResponse
from .validation import validate_tts_audio

_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

DEFAULT_BASE_URL: str = _OPENAI_DEFAULT_BASE_URL
DEFAULT_MODEL_NAME: str = "tts-1"
DEFAULT_VOICE: str = "alloy"
DEFAULT_OUTPUT_FORMAT: str = "mp3"
DEFAULT_SPEED: float = 1.0
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_AUDIO_SIZE_MB: float = 10.0

_MS_ROUND = 4


def is_openai_tts_configured(*, api_key: Optional[str], base_url: str) -> bool:
    """Return True when the provider has enough configuration to function.

    Rules (mirrors is_openai_compatible_configured / is_openai_whisper_configured):
    - A non-empty API key always counts as configured.
    - A non-default base URL (any OpenAI-compatible server) counts
      as configured even without a key — many local servers need no auth.
    - Missing key AND the default OpenAI base URL -> NOT configured.

    Args:
        api_key: Resolved API key string, or None / empty string.
        base_url: Resolved base URL string.

    Returns:
        True when a usable configuration exists.
    """
    if api_key and api_key.strip():
        return True
    return base_url.rstrip("/") != _OPENAI_DEFAULT_BASE_URL.rstrip("/")


class OpenAITTS(BaseTTS):
    """OpenAI Text-to-Speech API provider.

    Supports both the official OpenAI TTS API (tts-1, tts-1-hd) and any
    OpenAI-compatible TTS endpoint via ``base_url``.

    Args:
        api_key: Provider API key. Required for default OpenAI base URL;
            optional for compatible local endpoints.
        base_url: Endpoint base URL override. None -> official OpenAI API.
        model: TTS model identifier (default: 'tts-1').
        voice: Default voice identifier (default: 'alloy').
        output_format: Default audio output format ('mp3', 'opus', 'aac', 'flac', 'wav', 'pcm').
        speed: Default playback speed multiplier (default: 1.0).
        timeout_seconds: Hard timeout for API calls in seconds.
        max_text_length: Maximum allowed text character length.
        max_audio_size_mb: Maximum accepted audio size in MB.
        client: Injected SDK-compatible object for offline tests. When provided,
            no real SDK client is created and zero network calls occur.

    Raises:
        ValueError: If arguments are invalid or key is missing for default base URL.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL_NAME,
        voice: str = DEFAULT_VOICE,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        speed: float = DEFAULT_SPEED,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        max_audio_size_mb: float = DEFAULT_MAX_AUDIO_SIZE_MB,
        client: Optional[object] = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"model must be a non-empty string, got {model!r}")
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
            not isinstance(max_text_length, int)
            or max_text_length <= 0
        ):
            raise ValueError(
                f"max_text_length must be a positive integer, got {max_text_length!r}"
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
            raise ValueError(f"api_key must be a string or None, got {type(api_key).__name__}")

        self._api_key: str = (api_key or "").strip()
        self._base_url: str = (
            base_url.strip() if (base_url and base_url.strip()) else _OPENAI_DEFAULT_BASE_URL
        )
        self._model = model.strip()
        self._voice = validate_voice(voice)
        self._output_format = validate_output_format(output_format)
        self._speed = validate_speed(speed)
        self._timeout_seconds = float(timeout_seconds)
        self._max_text_length = max_text_length
        self._max_audio_size_mb = float(max_audio_size_mb)

        if client is not None:
            # Injected client for offline tests — zero network calls possible
            self._client = client
        else:
            # Production path
            if not self._api_key and self._base_url == _OPENAI_DEFAULT_BASE_URL:
                raise ValueError(
                    "api_key is required when using the default OpenAI base URL "
                    "(or set TTS_BASE_URL to a local OpenAI-compatible endpoint)"
                )
            self._client = OpenAI(
                api_key=self._api_key or None,
                base_url=self._base_url,
                timeout=self._timeout_seconds,
            )

    @property
    def provider(self) -> str:
        """Provider name reported on responses."""
        return "openai_tts"

    @property
    def model_name(self) -> str:
        """TTS model identifier reported on responses."""
        return self._model

    def synthesize(self, request: object) -> TTSResponse:
        """Synthesize a text request into audio via the OpenAI TTS API.

        Args:
            request: Object with ``text`` attribute. Optional: ``voice``,
                ``model``, ``output_format``, ``speed``, ``language``.

        Returns:
            TTSResponse with audio bytes, content_type, format, model, provider,
            latency_ms, character_count, and metadata.

        Raises:
            ValueError: If request attributes are missing or invalid.
            TTSError: If the provider API call fails or times out.
        """
        if not hasattr(request, "text"):
            raise ValueError("request must have a 'text' attribute")

        raw_text = getattr(request, "text")
        text = validate_text(raw_text, max_length=self._max_text_length)

        req_voice = getattr(request, "voice", None)
        effective_voice = validate_voice(req_voice) if req_voice is not None else self._voice

        req_model = getattr(request, "model", None)
        effective_model = (
            req_model.strip()
            if (req_model and isinstance(req_model, str) and req_model.strip())
            else self._model
        )

        req_format = getattr(request, "output_format", None)
        effective_format = (
            validate_output_format(req_format) if req_format is not None else self._output_format
        )

        req_speed = getattr(request, "speed", None)
        effective_speed = (
            validate_speed(req_speed) if req_speed is not None else self._speed
        )

        req_language = getattr(request, "language", None)
        if req_language is not None:
            validate_language(req_language)

        # Build kwargs for OpenAI audio.speech.create
        kwargs: dict[str, object] = {
            "model": effective_model,
            "input": text,
            "voice": effective_voice,
            "response_format": effective_format,
            "speed": effective_speed,
            "timeout": self._timeout_seconds,
        }

        # Provider call with latency measurement
        start = time.perf_counter()
        try:
            raw = self._client.audio.speech.create(**kwargs)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — wrap all SDK failures
            raise self._wrap_provider_error(exc) from exc
        latency_ms = round((time.perf_counter() - start) * 1000.0, _MS_ROUND)

        return self._map_response(
            raw,
            text=text,
            model=effective_model,
            voice=effective_voice,
            output_format=effective_format,
            speed=effective_speed,
            latency_ms=latency_ms,
        )

    def synthesize_batch(self, requests: list[object]) -> list[TTSResponse]:
        """Synthesize multiple text requests in batch sequentially preserving order.

        Args:
            requests: List of request objects with ``text``.

        Returns:
            List of TTSResponse objects in matching order.
        """
        if not isinstance(requests, list):
            raise ValueError(f"requests must be a list, got {type(requests).__name__}")
        return [self.synthesize(req) for req in requests]

    def _map_response(
        self,
        raw: object,
        *,
        text: str,
        model: str,
        voice: str,
        output_format: str,
        speed: float,
        latency_ms: float,
    ) -> TTSResponse:
        """Map raw provider response into a canonical TTSResponse."""
        # Extract audio bytes
        audio_bytes: Optional[bytes] = None
        if hasattr(raw, "content") and isinstance(raw.content, bytes):  # type: ignore[union-attr]
            audio_bytes = raw.content  # type: ignore[union-attr]
        elif hasattr(raw, "read") and callable(raw.read):  # type: ignore[union-attr]
            audio_bytes = raw.read()  # type: ignore[union-attr]
        elif isinstance(raw, bytes):
            audio_bytes = raw
        elif isinstance(raw, dict) and "audio" in raw and isinstance(raw["audio"], bytes):
            audio_bytes = raw["audio"]

        if not audio_bytes:
            raise TTSError("OpenAI TTS provider returned empty audio content")

        max_bytes = int(self._max_audio_size_mb * 1024 * 1024)
        try:
            validated = validate_tts_audio(
                audio_bytes,
                expected_format=output_format,
                max_bytes=max_bytes,
            )
        except ValueError as exc:
            raise TTSError(f"TTS audio validation failure: {exc}") from exc

        return TTSResponse(
            audio=audio_bytes,
            content_type=validated.content_type,
            format=validated.format,
            model=model,
            provider=self.provider,
            latency_ms=latency_ms,
            character_count=len(text),
            metadata={"voice": voice, "speed": speed},
        )

    def _wrap_provider_error(self, exc: Exception) -> TTSError:
        """Wrap SDK exception into a sanitized TTSError without leaking secrets."""
        text = str(exc)
        if self._api_key and self._api_key in text:
            text = text.replace(self._api_key, "[REDACTED]")
        return TTSError(f"OpenAI TTS provider error ({type(exc).__name__}): {text}")

    def __repr__(self) -> str:
        return (
            f"OpenAITTS(model={self._model!r}, "
            f"voice={self._voice!r}, "
            f"output_format={self._output_format!r}, "
            f"base_url={self._base_url!r}, "
            f"timeout_seconds={self._timeout_seconds})"
        )


def create_openai_tts(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = DEFAULT_MODEL_NAME,
    voice: str = DEFAULT_VOICE,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    speed: float = DEFAULT_SPEED,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
    max_audio_size_mb: float = DEFAULT_MAX_AUDIO_SIZE_MB,
    client: Optional[object] = None,
) -> OpenAITTS:
    """Factory helper to create an OpenAITTS instance."""
    return OpenAITTS(
        api_key=api_key,
        base_url=base_url,
        model=model,
        voice=voice,
        output_format=output_format,
        speed=speed,
        timeout_seconds=timeout_seconds,
        max_text_length=max_text_length,
        max_audio_size_mb=max_audio_size_mb,
        client=client,
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_VOICE",
    "DEFAULT_OUTPUT_FORMAT",
    "DEFAULT_SPEED",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_TEXT_LENGTH",
    "DEFAULT_MAX_AUDIO_SIZE_MB",
    "OpenAITTS",
    "create_openai_tts",
    "is_openai_tts_configured",
]
