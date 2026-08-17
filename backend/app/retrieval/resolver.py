"""Chunk resolver: chunk_id -> actual Chunk evidence.

Provides the repository abstraction that maps vector store search hit
ids back to the actual ``Chunk`` objects carrying ``chunk_text``. The
vector store metadata only carries ids (``VectorRecord.chunk_id``);
chunk text lives on ``Chunk``, which is the canonical evidence unit
consumed by GroundingVerifier.

The interface is intentionally small:

- ``resolve(chunk_ids)``: many ids -> many Chunks (order preserved,
  unresolved ids silently absent)

Concrete implementations include:

- ``DictChunkResolver``: in-memory dict-backed repository, suitable for
  tests, offline development, and small in-process corpora.

Phase 5.2: Resolver interface + in-memory implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Protocol

from app.chunking.models import Chunk


def validate_chunk_ids(chunk_ids: list[str]) -> list[str]:
    """Validate a list of chunk ids for resolution.

    Rules:
    - Must be a list
    - Must not be empty
    - Every item must be a non-empty string

    Args:
        chunk_ids: Chunk ids to resolve

    Returns:
        The validated list (unchanged)

    Raises:
        ValueError: If chunk_ids is not a list, is empty, or contains
                    an empty/whitespace id
    """
    if not isinstance(chunk_ids, list):
        raise ValueError(f"chunk_ids must be a list, got {type(chunk_ids).__name__}")
    if not chunk_ids:
        raise ValueError("chunk_ids cannot be empty")
    for index, chunk_id in enumerate(chunk_ids):
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"chunk_id at index {index} must be a non-empty string")
    return chunk_ids


class ChunkResolver(ABC):
    """Abstract base class for chunk id -> Chunk resolution.

    All resolvers must implement resolve() and preserve input ordering:
    ids that resolve are returned in the same relative order as the
    input list. Unresolved ids are simply absent from the result.

    Phase 5.2: Interface only.
    """

    @abstractmethod
    def resolve(self, chunk_ids: list[str]) -> list[Chunk]:
        """Resolve chunk ids to actual Chunk evidence objects.

        Args:
            chunk_ids: Non-empty list of chunk ids to resolve

        Returns:
            List of Chunk objects in input order; ids that could not be
            resolved are simply absent (never a partial placeholder)

        Raises:
            ValueError: If chunk_ids is empty or contains invalid ids
        """
        pass


class ChunkResolverProtocol(Protocol):
    """Protocol defining the resolver interface for type checking.

    Allows duck-typed resolver implementations that don't explicitly
    inherit from ChunkResolver but still follow the contract.
    """

    def resolve(self, chunk_ids: list[str]) -> list[Chunk]:
        """Resolve chunk ids to actual Chunk evidence objects."""
        ...


class DictChunkResolver(ChunkResolver):
    """In-memory dict-backed chunk repository.

    Maps chunk_id to Chunk objects. Deterministic, offline, and
    dependency-free; suitable for tests, offline development, and
    small in-process corpora.

    Args:
        chunks: Optional initial mapping of chunk_id -> Chunk

    Raises:
        ValueError: If initial chunks contain empty ids or non-Chunk values
    """

    def __init__(self, chunks: Optional[Dict[str, Chunk]] = None) -> None:
        self._chunks: Dict[str, Chunk] = {}
        if chunks:
            self.add_many(list(chunks.values()))

    @property
    def count(self) -> int:
        """Number of Chunk objects currently registered."""
        return len(self._chunks)

    @property
    def chunk_ids(self) -> list[str]:
        """Chunk ids currently registered, in insertion order."""
        return list(self._chunks.keys())

    def add(self, chunk: Chunk) -> None:
        """Register a single Chunk under its chunk_id.

        Args:
            chunk: Chunk to register

        Raises:
            ValueError: If chunk is not a Chunk or has an empty chunk_id
        """
        if not isinstance(chunk, Chunk):
            raise ValueError(f"chunk must be a Chunk, got {type(chunk).__name__}")
        if not chunk.chunk_id or not chunk.chunk_id.strip():
            raise ValueError("chunk must have a non-empty chunk_id")
        self._chunks[chunk.chunk_id] = chunk

    def add_many(self, chunks: list[Chunk]) -> None:
        """Register multiple Chunks under their chunk_ids.

        Args:
            chunks: Chunks to register

        Raises:
            ValueError: If chunks is empty or contains an invalid Chunk
        """
        if not isinstance(chunks, list):
            raise ValueError(f"chunks must be a list, got {type(chunks).__name__}")
        if not chunks:
            raise ValueError("chunks cannot be empty")
        for chunk in chunks:
            self.add(chunk)

    def resolve(self, chunk_ids: list[str]) -> list[Chunk]:
        """Resolve chunk ids to registered Chunk objects.

        Ordering: results follow the relative order of resolvable ids
        in the input list. Unresolvable ids are silently absent.

        Args:
            chunk_ids: Non-empty list of chunk ids to resolve

        Returns:
            List of Chunk objects in input order (resolvable ids only)

        Raises:
            ValueError: If chunk_ids is empty or contains invalid ids
        """
        validate_chunk_ids(chunk_ids)
        resolved: list[Chunk] = []
        for chunk_id in chunk_ids:
            chunk = self._chunks.get(chunk_id)
            if chunk is not None:
                resolved.append(chunk)
        return resolved

    def __repr__(self) -> str:
        return f"DictChunkResolver(count={self.count})"


__all__ = [
    "ChunkResolver",
    "ChunkResolverProtocol",
    "DictChunkResolver",
    "validate_chunk_ids",
]
