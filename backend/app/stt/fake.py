"""Fake STT provider for tests and offline development.

IMPORTANT: This provider exists exclusively for tests and offline development.
It MUST NEVER be wired into production through get_stt() or any real
dependency path. get_stt() in app/api/dependencies.py only ever returns
None or OpenAIWhisperSTT.

Provides a deterministic, offline implementation of the BaseSTT contract:

- ``transcribe(request)``: validates audio strictly (same rules as real
  providers), then returns a configurable deterministic STTResponse without
  network calls or external SDKs.
- Supports canned responses keyed by audio bytes, custom default text,
  simulated latency, and error injection.
- Preserves the BaseSTT / STTProtocol contract so callers can test full
  pipelines using dependency_overrides.
"""

from __future__ import annotations

import time
from typing import Optional

from .base import BaseSTT, STTError, validate_language
from .models import STTResponse
from .validation import validate_audio


class FakeSTT(BaseSTT):
    """Deterministic offline STT provider for unit tests and local development.

    DO NOT use in production. get_stt() will never return this class.

    Args:
        canned_responses: Mapping from audio bytes to specific transcription
            text. When audio matches a key exactly, that text is returned.
        default_text: Fallback transcription text used when the audio bytes
            are not found in canned_responses.
        language: Default language code (ISO 639-1) reported when no language
            hint is supplied by the request. Defaults to "en".
        simulate_latency_ms: Additional artificial delay in milliseconds
            added on top of real execution time, to simulate provider
            round-trips in latency tests. Must be >= 0.
        should_raise: When set, transcribe() raises this exception instead
            of returning a response. Used for error-path testing.
    """

    def __init__(
        self,
        *,
        canned_responses: Optional[dict[bytes, str]] = None,
        default_text: str = "This is a fake transcription of the provided audio.",
        language: Optional[str] = None,
        simulate_latency_ms: float = 0.0,
        should_raise: Optional[Exception] = None,
    ) -> None:
        if not isinstance(default_text, str) or not default_text.strip():
            raise ValueError("default_text must be a non-empty string")
        if (
            not isinstance(simulate_latency_ms, (int, float))
            or isinstance(simulate_latency_ms, bool)
            or simulate_latency_ms < 0
        ):
            raise ValueError("simulate_latency_ms must be a non-negative number")

        self._canned_responses: dict[bytes, str] = canned_responses or {}
        self._default_text = default_text
        self._language = validate_language(language)
        self._simulate_latency_ms = float(simulate_latency_ms)
        self._should_raise = should_raise

    @property
    def provider(self) -> str:
        """Provider name reported on responses."""
        return "fake"

    @property
    def model_name(self) -> str:
        """Model identifier reported on responses."""
        return "fake-whisper"

    def transcribe(self, request: object) -> STTResponse:
        """Transcribe an audio request deterministically.

        Enforces the same strict audio validation rules used by real providers
        (size, extension, MIME, magic bytes) so that test code exercises the
        same validation path.

        Args:
            request: Object with at least ``audio`` (bytes) and ``filename``
                (str) attributes. content_type, language, and prompt are
                read when present.

        Returns:
            STTResponse with deterministic transcription and metadata.

        Raises:
            ValueError: If request attributes are missing or audio is invalid.
            STTError: If error injection is configured via should_raise.
        """
        if not hasattr(request, "audio") or not hasattr(request, "filename"):
            raise ValueError("request must have 'audio' and 'filename' attributes")

        audio_bytes: bytes = getattr(request, "audio")
        filename: str = getattr(request, "filename")
        content_type: Optional[str] = getattr(request, "content_type", None)
        req_language: Optional[str] = getattr(request, "language", None)

        # Enforce strict audio validation (same rules as real providers)
        validate_audio(audio_bytes, filename=filename, content_type=content_type)

        # Inject error before any response is produced
        if self._should_raise is not None:
            if isinstance(self._should_raise, STTError):
                raise self._should_raise
            raise STTError(
                f"FakeSTT injected error: {self._should_raise}"
            ) from self._should_raise

        # Measure total elapsed time, including any artificial delay
        start = time.perf_counter()
        if self._simulate_latency_ms > 0:
            time.sleep(self._simulate_latency_ms / 1000.0)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 4)

        text = self._canned_responses.get(audio_bytes, self._default_text)
        resolved_language = validate_language(req_language) or self._language or "en"

        return STTResponse(
            text=text,
            language=resolved_language,
            provider=self.provider,
            model=self.model_name,
            latency_ms=latency_ms,
            # Intentional simulation values - real providers may not return these
            duration_seconds=1.5,
            confidence=0.95,
        )

    def __repr__(self) -> str:
        return (
            f"FakeSTT(model_name={self.model_name!r}, "
            f"default_text={self._default_text!r}, "
            f"language={self._language!r})"
        )


def create_fake_stt(
    *,
    canned_responses: Optional[dict[bytes, str]] = None,
    default_text: str = "This is a fake transcription of the provided audio.",
    language: Optional[str] = None,
    simulate_latency_ms: float = 0.0,
    should_raise: Optional[Exception] = None,
) -> FakeSTT:
    """Create a FakeSTT provider instance for tests and offline development.

    DO NOT wire the return value into production code paths.

    Args:
        canned_responses: Mapping from audio bytes to specific text responses.
        default_text: Fallback text when audio bytes are not matched.
        language: Default language code reported on responses.
        simulate_latency_ms: Artificial delay in milliseconds (>= 0).
        should_raise: Exception to raise on every transcribe() call.

    Returns:
        Configured FakeSTT instance.
    """
    return FakeSTT(
        canned_responses=canned_responses,
        default_text=default_text,
        language=language,
        simulate_latency_ms=simulate_latency_ms,
        should_raise=should_raise,
    )


__all__ = [
    "FakeSTT",
    "create_fake_stt",
]
