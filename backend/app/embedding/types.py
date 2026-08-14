"""Predictable type aliases for embedding vectors.

Defines the canonical representation used across the embedding layer:

- A single embedding is a flat list of floats: ``list[float]``
- A batch of embeddings is a list of those: ``list[list[float]]``

These aliases keep the interface provider-agnostic: any future provider
(HuggingFace / Sentence Transformers, local ONNX, API-based) can convert
its native output into ``EmbeddingVector`` without changing callers.

Phase 4.1: Embedding interface/types only (no real model).
"""

from __future__ import annotations

from typing import TypeAlias

EmbeddingVector: TypeAlias = list[float]
"""A single dense embedding vector represented as a flat list of floats."""

EmbeddingBatch: TypeAlias = list[EmbeddingVector]
"""A batch of embedding vectors, preserving input order."""

__all__ = [
    "EmbeddingVector",
    "EmbeddingBatch",
]