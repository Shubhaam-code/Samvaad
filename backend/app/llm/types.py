"""Predictable type aliases for LLM text generation.

Defines the canonical string representations used across the LLM layer:

- A single prompt is a plain string: ``str``
- A generated text response is a plain string: ``str``

These aliases keep the interface provider-agnostic: any future provider
(OpenAI-compatible API, Gemini, local model) can accept/return plain
strings without changing callers.

Phase 6.1: LLM interface/types only (no real provider).
"""

from __future__ import annotations

from typing import TypeAlias

LLMPrompt: TypeAlias = str
"""A single user prompt / generation input as a plain string."""

LLMText: TypeAlias = str
"""A single generated text output as a plain string."""

__all__ = [
    "LLMPrompt",
    "LLMText",
]
