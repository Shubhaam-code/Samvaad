"""Predictable type aliases for text-to-speech.

Defines the canonical types used across the TTS layer:

- Input text is a plain string: ``TTSText``
- Output audio is a plain ``bytes`` blob: ``TTSAudio``
- Voice name is a string identifier: ``TTSVoice``
- Model name is a string identifier: ``TTSModel``
- Output format is a string identifier (e.g. ``mp3``, ``wav``): ``TTSFormat``
- Language code is a string (ISO 639-1/2, e.g. ``en``, ``hi``): ``TTSLanguage``

These aliases keep the interface provider-agnostic: any future provider
(hosted API, local model) can accept/return the same shapes without
changing callers.
"""

from __future__ import annotations

from typing import TypeAlias

TTSText: TypeAlias = str
"""A single text input for synthesis."""

TTSAudio: TypeAlias = bytes
"""A single synthesized audio output as a ``bytes`` blob (transient, in-memory)."""

TTSVoice: TypeAlias = str
"""A voice identifier (e.g. ``alloy``, ``echo``, ``fable``, ``onyx``, ``nova``, ``shimmer``)."""

TTSModel: TypeAlias = str
"""A TTS model identifier (e.g. ``tts-1``, ``tts-1-hd``)."""

TTSFormat: TypeAlias = str
"""An audio output format (e.g. ``mp3``, ``opus``, ``aac``, ``flac``, ``wav``, ``pcm``)."""

TTSLanguage: TypeAlias = str
"""An optional ISO 639-1/2 language code (e.g. ``en``, ``hi``)."""

__all__ = [
    "TTSText",
    "TTSAudio",
    "TTSVoice",
    "TTSModel",
    "TTSFormat",
    "TTSLanguage",
]
