"""Fake TTS provider for tests and offline development.

IMPORTANT: This provider exists exclusively for unit tests and offline development.
It MUST NEVER be returned by get_tts() in app/api/dependencies.py or used in production.

Provides a deterministic, offline implementation of the BaseTTS contract:
- ``synthesize(request)``: strictly validates the request, then returns a
  configurable deterministic TTSResponse without network calls or external SDKs.
- ``synthesize_batch(requests)``: sequential batch synthesis preserving order.
- Supports canned audio responses keyed by input text, custom default audio bytes,
  simulated latency, and error injection.
"""

from __future__ import annotations

import time
from typing import Optional

from .base import (
    DEFAULT_MAX_TEXT_LENGTH,
    BaseTTS,
    TTSError,
    validate_language,
    validate_output_format,
    validate_speed,
    validate_text,
    validate_voice,
)
from .models import TTSResponse
from .validation import (
    FORMAT_TO_MIME,
    validate_tts_audio,
)

# Minimal valid audio byte templates for fake synthesis
_MINIMAL_AUDIO: dict[str, bytes] = {
    "mp3": b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x00fake-mp3-audio-bytes",
    "wav": (
        b"RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        b"\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00"
        b"\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00"
    ),
    "opus": b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00fake-opus-bytes",
    "flac": b"fLaC\x00\x00\x00\x22fake-flac-audio-bytes",
    "aac": b"\xff\xf1\x50\x80\x00\x1f\xfcfake-aac-bytes",
    "pcm": b"\x00\x00\x01\x00\x02\x00fake-raw-pcm-bytes",
}


class FakeTTS(BaseTTS):
    """Deterministic offline TTS provider for unit tests and local development.

    DO NOT use in production. get_tts() will never return this class.

    Args:
        canned_responses: Mapping from input text string to specific audio bytes.
        default_audio: Fallback audio bytes when text is not in canned_responses.
        default_format: Default output format when not specified in request ('mp3').
        model_name: Model identifier reported on responses ('fake-tts-1').
        simulate_latency_ms: Additional artificial delay in milliseconds (>= 0).
        should_raise: When set, synthesize() raises this exception instead.
        max_text_length: Maximum accepted text length (default: 4096).
    """

    def __init__(
        self,
        *,
        canned_responses: Optional[dict[str, bytes]] = None,
        default_audio: Optional[bytes] = None,
        default_format: str = "mp3",
        model_name: str = "fake-tts-1",
        simulate_latency_ms: float = 0.0,
        should_raise: Optional[Exception] = None,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
    ) -> None:
        if (
            not isinstance(simulate_latency_ms, (int, float))
            or isinstance(simulate_latency_ms, bool)
            or simulate_latency_ms < 0
        ):
            raise ValueError("simulate_latency_ms must be a non-negative number")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(max_text_length, int) or max_text_length <= 0:
            raise ValueError("max_text_length must be a positive integer")

        self._canned_responses: dict[str, bytes] = canned_responses or {}
        self._default_audio = default_audio
        self._default_format = validate_output_format(default_format)
        self._model_name = model_name.strip()
        self._simulate_latency_ms = float(simulate_latency_ms)
        self._should_raise = should_raise
        self._max_text_length = max_text_length

    @property
    def provider(self) -> str:
        """Provider name reported on responses."""
        return "fake"

    @property
    def model_name(self) -> str:
        """Model identifier reported on responses."""
        return self._model_name

    def synthesize(self, request: object) -> TTSResponse:
        """Synthesize a text request deterministically.

        Enforces strict input validation rules so tests exercise the real validation path.

        Args:
            request: Object with at least a ``text`` attribute. May also have
                ``voice``, ``model``, ``output_format``, ``speed``, ``language``.

        Returns:
            TTSResponse containing audio bytes and metadata.

        Raises:
            ValueError: If request attributes are missing or invalid.
            TTSError: If error injection is configured via should_raise.
        """
        if not hasattr(request, "text"):
            raise ValueError("request must have a 'text' attribute")

        raw_text = getattr(request, "text")
        text = validate_text(raw_text, max_length=self._max_text_length)

        voice = getattr(request, "voice", "alloy")
        if voice is not None:
            validate_voice(voice)

        speed = getattr(request, "speed", 1.0)
        if speed is not None:
            validate_speed(speed)

        output_format = getattr(request, "output_format", self._default_format)
        canonical_fmt = validate_output_format(output_format)

        language = getattr(request, "language", None)
        if language is not None:
            validate_language(language)

        model_override = getattr(request, "model", None)
        effective_model = (
            model_override.strip()
            if (model_override and isinstance(model_override, str) and model_override.strip())
            else self._model_name
        )

        # Inject error before response generation
        if self._should_raise is not None:
            if isinstance(self._should_raise, TTSError):
                raise self._should_raise
            raise TTSError(
                f"FakeTTS injected error: {self._should_raise}"
            ) from self._should_raise

        # Measure elapsed time including simulated delay
        start = time.perf_counter()
        if self._simulate_latency_ms > 0:
            time.sleep(self._simulate_latency_ms / 1000.0)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 4)

        # Resolve audio bytes: canned response -> custom default -> format template
        if text in self._canned_responses:
            audio_bytes = self._canned_responses[text]
        elif self._default_audio is not None:
            audio_bytes = self._default_audio
        else:
            audio_bytes = _MINIMAL_AUDIO.get(
                canonical_fmt,
                _MINIMAL_AUDIO["mp3"],
            )

        validated_audio = validate_tts_audio(audio_bytes, expected_format=canonical_fmt)

        return TTSResponse(
            audio=audio_bytes,
            content_type=validated_audio.content_type,
            format=canonical_fmt,
            model=effective_model,
            provider=self.provider,
            latency_ms=latency_ms,
            character_count=len(text),
            metadata={"voice": voice or "alloy", "speed": speed or 1.0},
        )

    def synthesize_batch(self, requests: list[object]) -> list[TTSResponse]:
        """Synthesize multiple text requests sequentially in batch.

        Args:
            requests: List of request objects with ``text``.

        Returns:
            List of TTSResponse objects in matching order.
        """
        if not isinstance(requests, list):
            raise ValueError(f"requests must be a list, got {type(requests).__name__}")
        return [self.synthesize(req) for req in requests]

    def __repr__(self) -> str:
        return (
            f"FakeTTS(model_name={self._model_name!r}, "
            f"default_format={self._default_format!r})"
        )


def create_fake_tts(
    *,
    canned_responses: Optional[dict[str, bytes]] = None,
    default_audio: Optional[bytes] = None,
    default_format: str = "mp3",
    model_name: str = "fake-tts-1",
    simulate_latency_ms: float = 0.0,
    should_raise: Optional[Exception] = None,
    max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
) -> FakeTTS:
    """Factory helper to create a FakeTTS provider instance for tests.

    DO NOT wire the return value into production code paths.

    Args:
        canned_responses: Mapping from text to audio bytes.
        default_audio: Fallback audio bytes.
        default_format: Default output format ('mp3').
        model_name: Model identifier.
        simulate_latency_ms: Artificial latency in milliseconds (>= 0).
        should_raise: Exception to raise on synthesize().
        max_text_length: Maximum allowed text character length.

    Returns:
        Configured FakeTTS instance.
    """
    return FakeTTS(
        canned_responses=canned_responses,
        default_audio=default_audio,
        default_format=default_format,
        model_name=model_name,
        simulate_latency_ms=simulate_latency_ms,
        should_raise=should_raise,
        max_text_length=max_text_length,
    )


__all__ = [
    "FakeTTS",
    "create_fake_tts",
]
