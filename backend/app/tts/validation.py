"""Strict audio output validation for the TTS layer.

Validates generated audio from TTS providers:
1. Type & size bounds (bounded audio - non-empty, within max size limit).
2. Supported format check ('mp3', 'opus', 'aac', 'flac', 'wav', 'pcm').
3. Format-to-MIME mapping consistency.
4. Container magic-byte sniffing (catches empty/corrupt/mismatched audio).

All audio is kept transient in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import SUPPORTED_FORMATS
from .types import TTSAudio

# Canonical MIME mappings for TTS formats
FORMAT_TO_MIME: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# Supported MIME types set
SUPPORTED_TTS_MIMES: frozenset[str] = frozenset(FORMAT_TO_MIME.values())

# Magic-byte container signatures
_WAV_RIFF = b"RIFF"
_WAV_FMT = b"WAVE"
_OGG_MAGIC = b"OggS"
_FLAC_MAGIC = b"fLaC"
_MP3_ID3 = b"ID3"

DEFAULT_MAX_TTS_AUDIO_BYTES: int = 10 * 1024 * 1024  # 10 MB


@dataclass(frozen=True)
class ValidatedTTSAudio:
    """Result of successful TTS audio output validation.

    Attributes:
        content_type: Canonical MIME type of the audio
        format: Canonical format identifier (e.g. 'mp3', 'wav')
        size_bytes: Size of the audio in bytes
    """

    content_type: str
    format: str
    size_bytes: int


def sniff_tts_audio_format(audio: TTSAudio) -> Optional[str]:
    """Detect audio container format from magic bytes.

    Returns the canonical format identifier ('mp3', 'opus', 'aac', 'flac', 'wav')
    when a recognized container signature is found, or None if unrecognized/raw.

    Args:
        audio: Raw audio bytes

    Returns:
        Canonical format identifier or None
    """
    if len(audio) >= 12 and audio[:4] == _WAV_RIFF and audio[8:12] == _WAV_FMT:
        return "wav"
    if len(audio) >= 4 and audio[:4] == _OGG_MAGIC:
        return "opus"
    if len(audio) >= 4 and audio[:4] == _FLAC_MAGIC:
        return "flac"
    if len(audio) >= 3 and audio[:3] == _MP3_ID3:
        return "mp3"
    # MP3 frame sync: 11 bits set (0xFF followed by high 3 bits set, 0xE0 mask)
    if len(audio) >= 2 and audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0:
        # Check ADTS (AAC) sync word: 12 bits 0xFFF (0xFF followed by 0xF0..0xF9)
        if (audio[1] & 0xF6) == 0xF0:
            return "aac"
        return "mp3"
    return None


def validate_tts_audio(
    audio: object,
    expected_format: str,
    max_bytes: int = DEFAULT_MAX_TTS_AUDIO_BYTES,
) -> ValidatedTTSAudio:
    """Validate synthesized TTS audio bytes.

    Enforces:
    - Type check: must be bytes
    - Non-empty: length > 0
    - Size check: length <= max_bytes
    - Format support check: expected_format in SUPPORTED_FORMATS
    - Container magic-byte validation (where container headers apply)

    Args:
        audio: Synthesized audio bytes
        expected_format: Requested audio format identifier
        max_bytes: Maximum allowed audio byte size

    Returns:
        ValidatedTTSAudio with canonical content_type, format, and size_bytes.

    Raises:
        ValueError: If audio is invalid, empty, oversized, or format mismatches.
    """
    if not isinstance(audio, bytes):
        raise ValueError(f"TTS audio must be bytes, got {type(audio).__name__}")
    if not audio:
        raise ValueError("TTS audio cannot be empty")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError(f"max_bytes must be a positive integer, got {max_bytes!r}")
    if len(audio) > max_bytes:
        mb_size = max_bytes / (1024 * 1024)
        raise ValueError(
            f"TTS audio exceeds maximum allowed size of {mb_size:g} MB ({len(audio)} bytes)"
        )

    if not isinstance(expected_format, str):
        raise ValueError(
            f"expected_format must be a string, got {type(expected_format).__name__}"
        )
    canonical_fmt = expected_format.strip().lower()
    if canonical_fmt not in SUPPORTED_FORMATS:
        supported_str = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"unsupported TTS output format: {expected_format!r} (supported: {supported_str})"
        )

    # Magic-byte container sniffing for formats with standard headers
    # PCM has no container header (raw L16/S16 samples), so container sniffing is skipped.
    if canonical_fmt != "pcm":
        sniffed = sniff_tts_audio_format(audio)
        if sniffed is not None and sniffed != canonical_fmt:
            raise ValueError(
                f"TTS audio content does not match expected format: "
                f"sniffed {sniffed!r} but expected {canonical_fmt!r}"
            )

    content_type = FORMAT_TO_MIME[canonical_fmt]
    return ValidatedTTSAudio(
        content_type=content_type,
        format=canonical_fmt,
        size_bytes=len(audio),
    )


__all__ = [
    "FORMAT_TO_MIME",
    "SUPPORTED_TTS_MIMES",
    "DEFAULT_MAX_TTS_AUDIO_BYTES",
    "ValidatedTTSAudio",
    "sniff_tts_audio_format",
    "validate_tts_audio",
]
