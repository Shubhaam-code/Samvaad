"""Predictable type aliases for speech-to-text.

Defines the canonical types used across the STT layer:

- A single audio input is a plain ``bytes`` blob: ``STTAudio``
- A transcription output is a plain string: ``STTText``
- A language code is a plain string (ISO 639-1/2, e.g. ``en`` / ``hi``)

These aliases keep the interface provider-agnostic: any future provider
(hosted API, local model) can accept/return the same shapes without
changing callers.

Phase 7.1: STT interface/types only.
"""

from __future__ import annotations

from typing import TypeAlias

STTAudio: TypeAlias = bytes
"""A single raw audio input as a ``bytes`` blob (transient, never stored)."""

STTText: TypeAlias = str
"""A single transcription output as a plain string."""

STTLanguage: TypeAlias = str
"""An ISO 639-1/2 language code (e.g. ``en``, ``hi``)."""

__all__ = [
    "STTAudio",
    "STTText",
    "STTLanguage",
]