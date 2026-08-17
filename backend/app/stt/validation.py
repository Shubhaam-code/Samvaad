"""Strict audio validation for the STT layer.

Treats uploaded audio as untrusted input. Validation is layered:

1. Type/size checks (bounded uploads - no arbitrarily large audio)
2. Extension checks against the supported format set
3. MIME type checks (aliases canonicalized; parameters stripped)
4. Extension/MIME cross-check (declared type must be self-consistent)
5. Magic-byte sniffing (catches truncated/mismatched/corrupt audio)

Sniffing validates container signatures (RIFF/WAVE, OggS, EBML, ftyp,
ID3/frame-sync, ADTS) without decoding the audio - it is a cheap
corruption/mismatch guard, not a full decoder. Deep decode validation
would require ffmpeg (documented as a limitation; the selected hosted
provider performs its own decoding).

No permanent storage: audio stays in memory for the duration of a
request and is never written to disk by this package.

Phase 7.2: Audio validation for the production STT provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .types import STTAudio

# Canonical MIME types actually accepted by the pipeline.
# Browser MediaRecorder output (webm/ogg/wav) is supported directly.
SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "audio/wav",
        "audio/mpeg",
        "audio/mp4",
        "audio/aac",
        "audio/webm",
        "audio/ogg",
    }
)

# Supported file extensions (canonical, lowercase, with leading dot).
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".aac",
        ".webm",
        ".ogg",
    }
)

# MIME aliases mapped onto their canonical form (parameters stripped).
_MIME_ALIASES: dict[str, str] = {
    "audio/wav": "audio/wav",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/mpeg": "audio/mpeg",
    "audio/mp3": "audio/mpeg",
    "audio/x-mpeg": "audio/mpeg",
    "audio/mp4": "audio/mp4",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
    "audio/aac": "audio/aac",
    "audio/webm": "audio/webm",
    "audio/ogg": "audio/ogg",
    "audio/opus": "audio/ogg",
}

_EXTENSION_TO_MIME: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
}

# Magic-byte signatures used for corruption/mismatch sniffing.
_WAV_MAGIC = b"RIFF"
_WAV_FMT = b"WAVE"
_OGG_MAGIC = b"OggS"
_WEBM_MAGIC = b"\x1a\x45\xdf\xa3"
_MP4_FTYP = b"ftyp"
_MP3_ID3 = b"ID3"

# Default upload bound: 10 MB (well under the hosted provider's 25 MB
# limit; configurable via STT_MAX_AUDIO_SIZE_MB).
DEFAULT_MAX_AUDIO_BYTES: int = 10 * 1024 * 1024


@dataclass(frozen=True)
class ValidatedAudio:
    """Result of a successful audio validation.

    Attributes:
        content_type: Canonical MIME type of the audio
        extension: Canonical extension (with leading dot)
    """

    content_type: str
    extension: str


def canonicalize_mime(content_type: str) -> Optional[str]:
    """Map a declared MIME type onto its canonical form.

    Parameters (e.g. ``;codecs=opus``) are stripped and well-known
    aliases are normalized.

    Args:
        content_type: Declared MIME type

    Returns:
        The canonical MIME type, or None when unsupported
    """
    if not isinstance(content_type, str) or not content_type.strip():
        return None
    base = content_type.strip().split(";", 1)[0].strip().lower()
    return _MIME_ALIASES.get(base)


def sniff_audio_format(audio: STTAudio) -> Optional[str]:
    """Detect the audio container from magic bytes.

    Returns the canonical MIME type when a known signature is found,
    or None when the blob is unrecognized/corrupt.

    Args:
        audio: Raw audio bytes

    Returns:
        Canonical MIME type, or None when unrecognized
    """
    if len(audio) >= 12 and audio[:4] == _WAV_MAGIC and audio[8:12] == _WAV_FMT:
        return "audio/wav"
    if len(audio) >= 4 and audio[:4] == _OGG_MAGIC:
        return "audio/ogg"
    if len(audio) >= 4 and audio[:4] == _WEBM_MAGIC:
        return "audio/webm"
    if len(audio) >= 12 and audio[4:8] == _MP4_FTYP:
        return "audio/mp4"
    if len(audio) >= 3 and audio[:3] == _MP3_ID3:
        return "audio/mpeg"
    if len(audio) >= 2 and audio[0] == 0xFF and (audio[1] & 0xF6) == 0xF0:
        return "audio/aac"
    if len(audio) >= 2 and audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    return None


def validate_audio(
    audio: object,
    filename: object,
    content_type: Optional[object] = None,
    max_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
) -> ValidatedAudio:
    """Validate an untrusted audio upload.

    Enforces, in order: type/size bounds, supported extension,
    supported MIME type (inferred from the extension when absent),
    extension/MIME consistency, and container signature sniffing.

    Args:
        audio: Raw audio bytes
        filename: Original filename (extension checked)
        content_type: Optional declared MIME type (aliases accepted)
        max_bytes: Maximum accepted audio size in bytes

    Returns:
        A ValidatedAudio with the canonical content type and extension

    Raises:
        ValueError: If any validation layer rejects the input
    """
    if not isinstance(audio, bytes):
        raise ValueError(f"STT audio must be bytes, got {type(audio).__name__}")
    if not audio:
        raise ValueError("STT audio cannot be empty")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError(f"max_bytes must be a positive integer, got {max_bytes!r}")
    if len(audio) > max_bytes:
        raise ValueError(
            f"audio exceeds the maximum allowed size of "
            f"{_format_mb(max_bytes)} MB ({len(audio)} bytes)"
        )
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")

    extension = os.path.splitext(filename)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported audio format: extension {extension!r} "
            f"(supported: {_format_extensions()})"
        )

    expected_mime = _EXTENSION_TO_MIME[extension]

    if content_type is not None:
        if not isinstance(content_type, str):
            raise ValueError(
                f"content_type must be a string or None, got {type(content_type).__name__}"
            )
        canonical = canonicalize_mime(content_type)
        if canonical is None:
            raise ValueError(
                f"unsupported audio MIME type: {content_type!r} "
                f"(supported: {_format_mimes()})"
            )
        if canonical != expected_mime:
            raise ValueError(
                f"audio MIME type {canonical!r} does not match the file "
                f"extension {extension!r}"
            )
    else:
        canonical = expected_mime

    sniffed = sniff_audio_format(audio)
    if sniffed is None:
        raise ValueError(
            "audio content is unrecognized or corrupt (no known container signature)"
        )
    if sniffed != canonical:
        raise ValueError(
            f"audio content does not match the declared format: "
            f"sniffed {sniffed!r} but declared {canonical!r}"
        )

    return ValidatedAudio(content_type=canonical, extension=extension)


def _format_mb(max_bytes: int) -> str:
    """Format a byte count as MB for error messages."""
    return f"{max_bytes / (1024 * 1024):g}"


def _format_extensions() -> str:
    """Human-readable supported extension list."""
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


def _format_mimes() -> str:
    """Human-readable supported MIME list."""
    return ", ".join(sorted(SUPPORTED_MIME_TYPES))


__all__ = [
    "SUPPORTED_MIME_TYPES",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_MAX_AUDIO_BYTES",
    "ValidatedAudio",
    "canonicalize_mime",
    "sniff_audio_format",
    "validate_audio",
]