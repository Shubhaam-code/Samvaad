"""Base embedding interface and shared validation rules.

This module defines the provider-agnostic embedder contract that all
concrete implementations (fake, HuggingFace, local, API-based) must follow.

The interface is intentionally small:

- ``encode(text)``: one text -> one vector
- ``encode_batch(texts)``: many texts -> many vectors (order preserved)
- ``dimension``: optional vector size (None until the model is known)

Shared validation helpers are provided as module-level functions so that
future providers and callers can reuse the exact same rules.

Phase 4.1: Interface definition + validation only (no production model).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Protocol

from .types import EmbeddingBatch, EmbeddingVector


def validate_text(text: str) -> str:
    """Validate a single input text for embedding.

    Rules:
    - Must be a string
    - Must not be empty
    - Must not be whitespace-only

    Args:
        text: Text to embed

    Returns:
        The validated text (unchanged)

    Raises:
        ValueError: If text is not a string, empty, or whitespace-only
    """
    if not isinstance(text, str):
        raise ValueError(f"Embedding input must be a string, got {type(text).__name__}")
    if not text:
        raise ValueError("Embedding input text cannot be empty")
    if not text.strip():
        raise ValueError("Embedding input text cannot be whitespace-only")
    return text


def validate_batch(texts: list[str]) -> list[str]:
    """Validate a batch of input texts for embedding.

    Rules:
    - Must be a list
    - Must not be empty (at least one text required)
    - Every item is validated via validate_text()

    Args:
        texts: List of texts to embed

    Returns:
        The validated batch (unchanged)

    Raises:
        ValueError: If batch is not a list, empty, or contains invalid text
    """
    if not isinstance(texts, list):
        raise ValueError(f"Embedding batch must be a list, got {type(texts).__name__}")
    if not texts:
        raise ValueError("Embedding batch cannot be empty")
    for text in texts:
        validate_text(text)
    return texts


def validate_batch_size(texts: list[str], batch_size: int) -> int:
    """Validate a batch does not exceed the configured batch size.

    Rules:
    - batch_size must be a positive integer
    - len(texts) must not exceed batch_size

    Args:
        texts: List of texts to embed
        batch_size: Maximum number of texts per encode_batch() call

    Returns:
        The validated batch size

    Raises:
        ValueError: If batch_size is invalid or the batch is too large
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError(f"batch_size must be an integer, got {type(batch_size).__name__}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if len(texts) > batch_size:
        raise ValueError(
            f"Batch of {len(texts)} texts exceeds configured batch_size of {batch_size}"
        )
    return batch_size


def validate_embeddings(
    vectors: EmbeddingBatch,
    expected_dimension: Optional[int] = None,
) -> EmbeddingBatch:
    """Validate a produced batch of embedding vectors.

    Rules:
    - Must be a list of vectors
    - Each vector must be a list of floats
    - All vectors must share the same dimension
    - If expected_dimension is given, every vector must match it
    - All values must be finite (no NaN, no infinity)

    Args:
        vectors: Embedding vectors to validate
        expected_dimension: Optional expected vector size; when provided,
                            every vector must have exactly this length

    Returns:
        The validated batch (unchanged)

    Raises:
        ValueError: If vectors violate any rule above
    """
    if not isinstance(vectors, list):
        raise ValueError(f"Embedding output must be a list, got {type(vectors).__name__}")
    if not vectors:
        raise ValueError("Embedding output batch cannot be empty")

    first_dim = len(vectors[0])
    if expected_dimension is not None and first_dim != expected_dimension:
        raise ValueError(
            f"Vector dimension {first_dim} does not match expected dimension "
            f"{expected_dimension}"
        )

    for i, vector in enumerate(vectors):
        if not isinstance(vector, list):
            raise ValueError(
                f"Vector at index {i} must be a list of floats, got {type(vector).__name__}"
            )
        if len(vector) != first_dim:
            raise ValueError(
                f"Inconsistent vector dimensions: vector 0 has {first_dim} "
                f"dimensions but vector {i} has {len(vector)}"
            )
        for j, value in enumerate(vector):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"Vector {i} value at index {j} must be a number, "
                    f"got {type(value).__name__}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Vector {i} value at index {j} is not finite: {value!r}"
                )
    return vectors


class BaseEmbedder(ABC):
    """Abstract base class for all embedding providers.

    Concrete implementations will include:
    - FakeEmbedder: deterministic hash-based vectors for tests (Phase 4.1)
    - HuggingFace/Sentence Transformers provider (Phase 4.2)
    - API-based provider (Phase 4.2)

    All embedders must implement encode(), encode_batch() and the
    dimension property, and must preserve input ordering in encode_batch().

    Phase 4.1: Base interface only (no production model).
    """

    @abstractmethod
    def encode(self, text: str) -> EmbeddingVector:
        """Embed a single text into a vector.

        Args:
            text: Non-empty text to embed

        Returns:
            A list of floats representing the text

        Raises:
            ValueError: If text is empty or whitespace-only
        """
        pass

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> EmbeddingBatch:
        """Embed a batch of texts into vectors, preserving input order.

        ``["A", "B", "C"]`` must produce ``[vector(A), vector(B), vector(C)]``.

        Args:
            texts: Non-empty list of texts to embed

        Returns:
            List of vectors in exactly the same order as the input

        Raises:
            ValueError: If batch is empty, contains empty/whitespace text,
                        or exceeds the configured batch size
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> Optional[int]:
        """Vector dimension produced by this embedder.

        May be None until the production model is selected (Phase 4.2).
        """
        pass


class EmbedderProtocol(Protocol):
    """Protocol defining the embedder interface for type checking.

    Allows duck-typed embedder implementations that don't explicitly
    inherit from BaseEmbedder but still follow the contract.
    """

    def encode(self, text: str) -> EmbeddingVector:
        """Embed a single text into a vector."""
        ...

    def encode_batch(self, texts: list[str]) -> EmbeddingBatch:
        """Embed a batch of texts into vectors (order preserved)."""
        ...

    @property
    def dimension(self) -> Optional[int]:
        """Vector dimension produced by this embedder."""
        ...


__all__ = [
    "BaseEmbedder",
    "EmbedderProtocol",
    "validate_text",
    "validate_batch",
    "validate_batch_size",
    "validate_embeddings",
]