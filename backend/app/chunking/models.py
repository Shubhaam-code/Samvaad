"""Canonical Chunk data model for retrieval-ready text segments.

This module defines the internal representation of a text chunk derived from
CanonicalPassage records. Chunks are the atomic units for embedding and retrieval.

Phase 3.1: Chunk schema and architecture (no chunking algorithms yet).
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChunkingStrategy(str, Enum):
    """Enumeration of supported chunking strategies.
    
    Future implementations will provide these strategies:
    - PASSAGE: Keep entire passage as single chunk (no splitting)
    - SENTENCE: Split by sentence boundaries
    - TOKEN: Split by token count with overlap
    - ADAPTIVE: Context-aware splitting based on content structure
    """
    PASSAGE = "passage"
    SENTENCE = "sentence"
    TOKEN = "token"
    ADAPTIVE = "adaptive"


class Chunk(BaseModel):
    """A single retrieval-ready text chunk derived from a CanonicalPassage.
    
    Each Chunk represents a segment of text that will be:
    - Embedded into a vector
    - Indexed in a vector database
    - Retrieved in response to queries
    - Cited in generated answers
    
    The chunk preserves complete traceability to its source passage and query.
    
    Phase 3.1: Data model only (no chunking algorithms implemented).
    """
    
    # Chunk identification
    chunk_id: str = Field(
        ...,
        description="Deterministic unique identifier: hash of document_id + strategy + chunk_index",
    )
    document_id: str = Field(
        ...,
        description="Source document ID from CanonicalPassage",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based index of this chunk within the source passage",
    )
    
    # Chunking metadata
    strategy: ChunkingStrategy = Field(
        ...,
        description="Chunking strategy used to create this chunk",
    )
    
    # Chunk content
    chunk_text: str = Field(
        ...,
        min_length=1,
        description="The actual text content of this chunk",
    )
    
    # Content metrics (optional, populated when available)
    character_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of characters in chunk_text",
    )
    token_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of tokens in chunk_text (if tokenized)",
    )
    
    # Position within source passage (optional)
    start_offset: Optional[int] = Field(
        None,
        ge=0,
        description="Character offset where chunk starts in source passage",
    )
    end_offset: Optional[int] = Field(
        None,
        ge=0,
        description="Character offset where chunk ends in source passage",
    )
    
    # Overlap information (optional, for strategies that use overlap)
    overlap_before: Optional[int] = Field(
        None,
        ge=0,
        description="Number of overlapping characters with previous chunk",
    )
    overlap_after: Optional[int] = Field(
        None,
        ge=0,
        description="Number of overlapping characters with next chunk",
    )
    
    # Source passage metadata (preserved for traceability)
    query_id: int = Field(
        ...,
        description="Query ID from source CanonicalPassage",
    )
    passage_index: int = Field(
        ...,
        ge=0,
        description="Passage index from source CanonicalPassage",
    )
    
    # Language metadata
    target_lang: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Target language code",
    )
    source_lang: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Source language code (typically 'en')",
    )
    
    # Query context (preserved for relevance)
    query: str = Field(
        ...,
        min_length=1,
        description="User query from source passage",
    )
    eng_query: str = Field(
        ...,
        min_length=1,
        description="English version of query",
    )
    query_type: Optional[str] = Field(
        None,
        description="Query category/type if available",
    )
    
    # Answer context (optional, preserved for grounding)
    answer: Optional[str] = Field(
        None,
        description="Answer in target language",
    )
    eng_answer: Optional[str] = Field(
        None,
        description="Answer in English",
    )
    
    # Relevance label
    is_selected: bool = Field(
        ...,
        description="Whether source passage was selected as relevant",
    )
    
    @field_validator("chunk_text", "query", "eng_query")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        """Ensure required text fields are not empty after stripping."""
        if not v or not v.strip():
            raise ValueError("Required text field cannot be empty")
        return v
    
    @field_validator("end_offset")
    @classmethod
    def validate_end_offset_after_start(cls, v: Optional[int], info) -> Optional[int]:
        """Ensure end_offset is after start_offset if both are present."""
        if v is not None and "start_offset" in info.data:
            start = info.data.get("start_offset")
            if start is not None and v < start:
                raise ValueError(f"end_offset ({v}) must be >= start_offset ({start})")
        return v
    
    @staticmethod
    def generate_chunk_id(
        document_id: str,
        strategy: ChunkingStrategy,
        chunk_index: int,
    ) -> str:
        """Generate deterministic chunk ID.
        
        Uses SHA-256 hash of the canonical string representation to ensure:
        - Deterministic: same inputs always produce same ID
        - Unique: different inputs produce different IDs
        - Stable: IDs remain consistent across chunking runs
        
        Args:
            document_id: Source document ID
            strategy: Chunking strategy used
            chunk_index: Zero-based chunk index
        
        Returns:
            Hex string chunk ID (64 characters)
        
        Example:
            >>> Chunk.generate_chunk_id("doc123", ChunkingStrategy.SENTENCE, 0)
            'a1b2c3d4...'
        """
        # Create canonical string from key identifiers
        canonical_string = (
            f"document_id={document_id}:"
            f"strategy={strategy.value}:"
            f"chunk_index={chunk_index}"
        )
        
        # Generate SHA-256 hash
        hash_bytes = hashlib.sha256(canonical_string.encode("utf-8")).digest()
        return hash_bytes.hex()
    
    @classmethod
    def from_passage_segment(
        cls,
        document_id: str,
        chunk_index: int,
        strategy: ChunkingStrategy,
        chunk_text: str,
        query_id: int,
        passage_index: int,
        target_lang: str,
        source_lang: str,
        query: str,
        eng_query: str,
        query_type: Optional[str],
        answer: Optional[str],
        eng_answer: Optional[str],
        is_selected: bool,
        character_count: Optional[int] = None,
        token_count: Optional[int] = None,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
        overlap_before: Optional[int] = None,
        overlap_after: Optional[int] = None,
    ) -> Chunk:
        """Factory method to create a Chunk from passage segment data.
        
        This is the primary interface for constructing chunks during chunking.
        It automatically generates the deterministic chunk_id and ensures
        metadata is preserved correctly.
        
        Args:
            document_id: Source document ID
            chunk_index: Zero-based chunk index
            strategy: Chunking strategy
            chunk_text: Actual chunk text content
            query_id: Query ID from source passage
            passage_index: Passage index from source passage
            target_lang: Target language code
            source_lang: Source language code
            query: Query text in target language
            eng_query: Query text in English
            query_type: Query category/type
            answer: Answer in target language
            eng_answer: Answer in English
            is_selected: Relevance label
            character_count: Character count (optional)
            token_count: Token count (optional)
            start_offset: Start offset (optional)
            end_offset: End offset (optional)
            overlap_before: Overlap before (optional)
            overlap_after: Overlap after (optional)
        
        Returns:
            Chunk instance with generated chunk_id
        """
        chunk_id = cls.generate_chunk_id(document_id, strategy, chunk_index)
        
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=chunk_index,
            strategy=strategy,
            chunk_text=chunk_text,
            character_count=character_count,
            token_count=token_count,
            start_offset=start_offset,
            end_offset=end_offset,
            overlap_before=overlap_before,
            overlap_after=overlap_after,
            query_id=query_id,
            passage_index=passage_index,
            target_lang=target_lang,
            source_lang=source_lang,
            query=query,
            eng_query=eng_query,
            query_type=query_type,
            answer=answer,
            eng_answer=eng_answer,
            is_selected=is_selected,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation.
        
        Useful for serialization to JSON or other formats.
        """
        return self.model_dump()
    
    def __repr__(self) -> str:
        """String representation showing key identifiers."""
        return (
            f"Chunk(chunk_id={self.chunk_id[:16]}..., "
            f"document_id={self.document_id[:16]}..., "
            f"chunk_index={self.chunk_index}, "
            f"strategy={self.strategy.value}, "
            f"chars={len(self.chunk_text)}, "
            f"lang={self.target_lang})"
        )
