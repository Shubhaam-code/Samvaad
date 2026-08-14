"""Base interface for chunking strategies.

This module defines the generic chunker contract that all concrete
chunking implementations must follow.

Phase 3.1: Interface definition (no concrete implementations yet).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from ..dataset.models import CanonicalPassage
from .models import Chunk


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies.
    
    Concrete implementations will include:
    - PassageChunker: Keep entire passage as single chunk
    - SentenceChunker: Split by sentence boundaries
    - TokenChunker: Split by token count with overlap
    - AdaptiveChunker: Context-aware splitting
    
    All chunkers must implement the chunk() method that transforms
    a CanonicalPassage into a list of retrieval-ready Chunks.
    
    Phase 3.1: Base interface only (no implementations).
    """
    
    @abstractmethod
    def chunk(self, passage: CanonicalPassage) -> list[Chunk]:
        """Transform a CanonicalPassage into retrieval-ready Chunks.
        
        This method must:
        - Preserve all source metadata from the passage
        - Generate deterministic chunk IDs
        - Return at least one chunk (even if unchanged passage)
        - Never return an empty list for valid passages
        
        Args:
            passage: Source CanonicalPassage to chunk
        
        Returns:
            List of Chunk objects derived from the passage
        
        Raises:
            ValueError: If passage is invalid or cannot be chunked
        """
        pass
    
    @abstractmethod
    def chunk_batch(self, passages: list[CanonicalPassage]) -> list[Chunk]:
        """Transform a batch of CanonicalPassages into Chunks.
        
        Default implementation simply calls chunk() for each passage
        and concatenates results, but implementations may override
        for batch optimization.
        
        Args:
            passages: List of CanonicalPassages to chunk
        
        Returns:
            List of all Chunks from all passages
        """
        pass


class ChunkerProtocol(Protocol):
    """Protocol defining the chunker interface for type checking.
    
    This allows duck-typed chunker implementations that don't
    explicitly inherit from BaseChunker but still follow the contract.
    """
    
    def chunk(self, passage: CanonicalPassage) -> list[Chunk]:
        """Transform a CanonicalPassage into Chunks."""
        ...
    
    def chunk_batch(self, passages: list[CanonicalPassage]) -> list[Chunk]:
        """Transform a batch of CanonicalPassages into Chunks."""
        ...
