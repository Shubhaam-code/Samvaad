"""Deterministic fake embedder for offline testing.

This embedder NEVER contacts the network and NEVER loads a model.
Vectors are derived purely from the SHA-256 hash of the input text,
so they are:

- Deterministic: same text always produces the same vector
  (across calls, instances, and process runs)
- Stable: vectors are reproducible on any machine
- Offline: no model download, no API calls, no external dependencies

The fake vectors are L2-normalized unit vectors in a fixed dimension,
which makes them realistic stand-ins for real embeddings in tests of
downstream logic (ordering, dimensions, indexing) without needing a
production model.

The production embedding model is deliberately NOT chosen here;
selection happens in Phase 4.2.

Phase 4.1: Fake embedder only.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Optional

from .base import (
    BaseEmbedder,
    validate_batch,
    validate_batch_size,
    validate_text,
)
from .types import EmbeddingBatch, EmbeddingVector

_DEFAULT_DIMENSION = 16
_DEFAULT_BATCH_SIZE = 32


class FakeEmbedder(BaseEmbedder):
    """Deterministic, offline, hash-based embedder for testing.

    Args:
        dimension: Fixed vector dimension (>= 1)
        batch_size: Maximum texts allowed per encode_batch() call (>= 1)

    Raises:
        ValueError: If dimension or batch_size are invalid
    """

    def __init__(self, dimension: int = _DEFAULT_DIMENSION, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError(f"dimension must be an integer >= 1, got {dimension!r}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be an integer >= 1, got {batch_size!r}")
        self._dimension = dimension
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        """Fixed vector dimension of this embedder."""
        return self._dimension

    @property
    def batch_size(self) -> int:
        """Maximum texts allowed per encode_batch() call."""
        return self._batch_size

    def encode(self, text: str) -> EmbeddingVector:
        """Embed a single text into a deterministic unit vector.

        Args:
            text: Non-empty text to embed

        Returns:
            L2-normalized vector of floats with fixed dimension

        Raises:
            ValueError: If text is empty or whitespace-only
        """
        validate_text(text)
        return self._deterministic_vector(text)

    def encode_batch(self, texts: list[str]) -> EmbeddingBatch:
        """Embed a batch of texts, preserving input order.

        ``["A", "B", "C"]`` always produces ``[vector(A), vector(B), vector(C)]``.

        Args:
            texts: Non-empty list of texts to embed

        Returns:
            List of vectors in exactly the same order as the input

        Raises:
            ValueError: If the batch is empty, contains empty/whitespace text,
                        or exceeds the configured batch_size
        """
        validate_batch(texts)
        validate_batch_size(texts, self._batch_size)
        return [self._deterministic_vector(text) for text in texts]

    def _deterministic_vector(self, text: str) -> EmbeddingVector:
        """Generate a deterministic unit vector from text.

        Uses a SHA-256 digest of the UTF-8 encoded text as the seed for
        Python's Mersenne Twister PRNG, which is specified to be
        reproducible across runs and platforms.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:16], byteorder="big")
        rng = random.Random(seed)

        raw = [rng.uniform(-1.0, 1.0) for _ in range(self._dimension)]
        norm = math.sqrt(sum(value * value for value in raw))
        if norm == 0.0:
            raw[0] = 1.0
            norm = 1.0
        return [value / norm for value in raw]

    def __repr__(self) -> str:
        return (
            f"FakeEmbedder(dimension={self._dimension}, "
            f"batch_size={self._batch_size})"
        )


def create_fake_embedder(
    dimension: int = _DEFAULT_DIMENSION,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> FakeEmbedder:
    """Create a FakeEmbedder for testing and offline development."""
    return FakeEmbedder(dimension=dimension, batch_size=batch_size)


__all__ = [
    "FakeEmbedder",
    "create_fake_embedder",
]