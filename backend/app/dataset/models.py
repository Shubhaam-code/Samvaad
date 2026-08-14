"""Canonical data model for MSMARCO-XI passage records.

This module defines the internal representation of a single passage/document
extracted from the MSMARCO-XI dataset. Each canonical record represents one
passage that can be:
  * chunked for embedding
  * retrieved in response to queries
  * evaluated for relevance
  * cited in generated answers
  * processed in multilingual contexts

Phase 2.2: Dataset preprocessing model (no processing implementation yet).
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CanonicalPassage(BaseModel):
    """A single passage record normalized from MSMARCO-XI.
    
    Each CanonicalPassage represents one passage from the nested passages list
    in the original MSMARCO-XI format. Multiple passages may share the same
    query_id but have different passage_index values.
    
    The document_id is deterministically derived from query_id and passage_index
    to ensure stable identification across processing runs.
    """
    
    # Stable identifiers
    document_id: str = Field(
        ...,
        description="Deterministic unique identifier: hash of dataset + target_lang + query_id + passage_index",
    )
    query_id: int = Field(
        ...,
        description="Original query ID from MSMARCO-XI dataset",
    )
    passage_index: int = Field(
        ...,
        ge=0,
        description="Zero-based index of this passage within the query's passage list",
    )
    
    # Query information
    query: str = Field(
        ...,
        min_length=1,
        description="User query in target language",
    )
    query_type: str | None = Field(
        None,
        description="Query type/category if available (e.g., DESCRIPTION, ENTITY, NUMERIC)",
    )
    
    # Answer information
    answer: str | None = Field(
        None,
        description="Answer in target language (may be None for some queries)",
    )
    
    # Language metadata
    source_lang: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Source language code (typically 'en' for English)",
    )
    target_lang: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Target language code (e.g., 'hi' for Hindi)",
    )
    
    # English versions
    eng_query: str = Field(
        ...,
        min_length=1,
        description="Query in English (original or back-translated)",
    )
    eng_answer: str | None = Field(
        None,
        description="Answer in English (may be None for some queries)",
    )
    
    # Passage content
    translated_passage: str = Field(
        ...,
        min_length=1,
        description="Passage text in target language",
    )
    english_passage: str = Field(
        ...,
        min_length=1,
        description="Passage text in English",
    )
    
    # Relevance information
    is_selected: bool = Field(
        ...,
        description="Whether this passage was selected as relevant for the query",
    )
    
    @field_validator("query", "eng_query", "translated_passage", "english_passage")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        """Ensure required text fields are not empty after stripping."""
        if not v or not v.strip():
            raise ValueError("Required text field cannot be empty")
        return v
    
    @staticmethod
    def generate_document_id(target_lang: str, query_id: int, passage_index: int) -> str:
        """Generate a deterministic document ID including language identity.
        
        Uses SHA-256 hash of the canonical string representation to ensure:
        * Deterministic: same inputs always produce same ID
        * Unique: different inputs produce different IDs
        * Language-aware: same query_id + passage_index across different languages produce different IDs
        * Stable: IDs remain consistent across processing runs
        
        Args:
            target_lang: Target language code (e.g., 'hi', 'ta', 'bn')
            query_id: MSMARCO-XI query ID
            passage_index: Zero-based passage index within query
        
        Returns:
            Hex string document ID (64 characters)
        
        Example:
            >>> CanonicalPassage.generate_document_id('hi', 123, 0)
            'f8e7c9d1a2b3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9'
        """
        # Use a canonical string format: dataset:lang:query_id:passage_index
        canonical_string = f"MSMARCO-XI:{target_lang}:{query_id}:{passage_index}"
        # Generate SHA-256 hash
        hash_bytes = hashlib.sha256(canonical_string.encode("utf-8")).digest()
        # Return as hex string
        return hash_bytes.hex()
    
    @classmethod
    def from_msmarco_record(
        cls,
        query_id: int,
        query: str,
        query_type: str | None,
        answer: str | None,
        source_lang: str,
        target_lang: str,
        eng_query: str,
        eng_answer: str | None,
        passage_index: int,
        translated_passage: str,
        english_passage: str,
        is_selected: bool,
    ) -> CanonicalPassage:
        """Factory method to create a CanonicalPassage from MSMARCO-XI source data.
        
        This is the primary interface for constructing records during dataset
        preprocessing. It automatically generates the deterministic document_id
        that includes the target language for cross-language uniqueness.
        
        Args:
            query_id: Original MSMARCO-XI query ID
            query: Query text in target language
            query_type: Query category/type
            answer: Answer in target language
            source_lang: Source language code
            target_lang: Target language code
            eng_query: Query in English
            eng_answer: Answer in English
            passage_index: Zero-based passage index
            translated_passage: Passage in target language
            english_passage: Passage in English
            is_selected: Relevance label
        
        Returns:
            CanonicalPassage instance with generated document_id
        """
        document_id = cls.generate_document_id(target_lang, query_id, passage_index)
        
        return cls(
            document_id=document_id,
            query_id=query_id,
            passage_index=passage_index,
            query=query,
            query_type=query_type,
            answer=answer,
            source_lang=source_lang,
            target_lang=target_lang,
            eng_query=eng_query,
            eng_answer=eng_answer,
            translated_passage=translated_passage,
            english_passage=english_passage,
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
            f"CanonicalPassage(document_id={self.document_id[:16]}..., "
            f"query_id={self.query_id}, passage_index={self.passage_index}, "
            f"is_selected={self.is_selected}, lang={self.target_lang})"
        )
