"""Base vector store interface, data models, and shared validation rules.

Phase 4.5: Defines the provider-agnostic vector store contract, records,
and index manifest for vector index persistence and nearest-neighbor search.

Key components:
- ``VectorRecord``: Metadata mapping for a single embedded chunk.
- ``VectorSearchResult``: Search result item containing record, similarity score, and position.
- ``IndexManifest``: Schema manifest describing index metadata.
- ``VectorStoreError``: Custom exception for vector store operations.
- ``BaseVectorStore``: Abstract base class for local vector index implementations.
- Validation functions for vectors, dimensions, metadata, top_k, and search parameters.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class VectorStoreError(Exception):
    """Custom exception raised for vector store operational failures."""
    pass


class IndexManifest(BaseModel):
    """Manifest describing a persisted vector store's architecture and schema."""

    schema_version: str = Field("1.0", description="Schema version identifier")
    vector_store_type: str = Field(..., description="Store backend type: 'faiss' or 'numpy'")
    index_type: str = Field(..., description="Index implementation type (e.g. IndexFlatIP, NumpyDotProduct)")
    dimension: int = Field(..., ge=1, description="Dense vector dimension")
    metric: str = Field(..., description="Distance/similarity metric (e.g. inner_product, dot_product)")
    count: int = Field(..., ge=0, description="Total vectors indexed")
    embedding_model_name: Optional[str] = Field(None, description="Embedding model identifier used")
    normalization_expectation: str = Field("l2_normalized", description="Expected normalization: 'l2_normalized' or 'raw'")
    metadata_schema_version: str = Field("1.0", description="Metadata schema version")


class VectorRecord(BaseModel):
    """Metadata record associated 1:1 with an indexed embedding vector.

    Preserves source metadata to ensure complete traceability from vector index
    back to source document/chunk.
    """

    chunk_id: str = Field(..., min_length=1, description="Source chunk unique ID")
    document_id: str = Field(..., min_length=1, description="Source document ID")
    chunk_index: int = Field(..., ge=0, description="Zero-based chunk index within document")
    query_id: Optional[int] = Field(None, description="Original query ID if available")
    passage_index: Optional[int] = Field(None, description="Original passage index if available")
    target_lang: Optional[str] = Field(None, description="Target language code")
    source_lang: Optional[str] = Field(None, description="Source language code")
    is_selected: Optional[bool] = Field(None, description="Relevance ground-truth flag")
    extra_metadata: Optional[dict[str, Any]] = Field(None, description="Additional custom key-value pairs")


class VectorSearchResult(BaseModel):
    """Result item returned from a nearest-neighbor vector search."""

    chunk_id: str = Field(..., description="Chunk ID of the matched vector")
    score: float = Field(..., description="Similarity score (higher = more similar for Inner Product)")
    position: int = Field(..., ge=0, description="0-based index position of vector in store")
    record: VectorRecord = Field(..., description="Complete metadata record")


def validate_vector(vector: List[float], expected_dim: Optional[int] = None) -> List[float]:
    """Validate a single vector representation.

    Rules:
    - Must be a list of floats/numbers
    - Cannot be empty
    - Must match expected_dim if provided
    - All elements must be finite numbers (no NaN, no Inf)

    Args:
        vector: List of floats
        expected_dim: Optional expected vector dimension

    Returns:
        Validated vector

    Raises:
        ValueError: If vector violates validation rules
    """
    if not isinstance(vector, list):
        raise ValueError(f"Vector must be a list of floats, got {type(vector).__name__}")
    if not vector:
        raise ValueError("Vector cannot be empty")
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(
            f"Vector dimension {len(vector)} does not match expected dimension {expected_dim}"
        )
    for j, val in enumerate(vector):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Vector element at index {j} must be numeric, got {type(val).__name__}")
        if not math.isfinite(float(val)):
            raise ValueError(f"Vector element at index {j} is non-finite: {val!r}")
    return vector


def validate_vectors(
    vectors: List[List[float]], expected_dim: Optional[int] = None
) -> List[List[float]]:
    """Validate a list of embedding vectors for insertion.

    Args:
        vectors: List of float lists
        expected_dim: Optional expected dimension

    Returns:
        Validated vectors list

    Raises:
        ValueError: If vectors list is empty, dimension mismatched, or non-finite
    """
    if not isinstance(vectors, list):
        raise ValueError(f"Vectors batch must be a list, got {type(vectors).__name__}")
    if not vectors:
        raise ValueError("Vectors list for insertion cannot be empty")

    dim = len(vectors[0])
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"Vector dimension {dim} does not match index dimension {expected_dim}")

    for i, vec in enumerate(vectors):
        if not isinstance(vec, list):
            raise ValueError(f"Vector at index {i} must be a list, got {type(vec).__name__}")
        if len(vec) != dim:
            raise ValueError(
                f"Inconsistent vector dimensions: vector 0 has {dim} but vector {i} has {len(vec)}"
            )
        validate_vector(vec, expected_dim=dim)
    return vectors


def validate_records(records: List[VectorRecord], expected_count: int) -> List[VectorRecord]:
    """Validate metadata records for insertion.

    Args:
        records: List of VectorRecord objects
        expected_count: Expected number of records (must match vectors count)

    Returns:
        Validated records list

    Raises:
        ValueError: If records list is invalid or count does not match expected_count
    """
    if not isinstance(records, list):
        raise ValueError(f"Records must be a list, got {type(records).__name__}")
    if len(records) != expected_count:
        raise ValueError(
            f"Record count ({len(records)}) does not match vector count ({expected_count})"
        )
    for i, rec in enumerate(records):
        if not isinstance(rec, VectorRecord):
            raise ValueError(f"Record at index {i} must be a VectorRecord, got {type(rec).__name__}")
        if not rec.chunk_id or not rec.chunk_id.strip():
            raise ValueError(f"Record at index {i} has empty chunk_id")
        if not rec.document_id or not rec.document_id.strip():
            raise ValueError(f"Record at index {i} has empty document_id")
    return records


def validate_top_k(top_k: int) -> int:
    """Validate search top_k parameter.

    Args:
        top_k: Number of nearest neighbors to retrieve

    Returns:
        Validated top_k

    Raises:
        ValueError: If top_k <= 0 or not an integer
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError(f"top_k must be a positive integer, got {type(top_k).__name__}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    return top_k


class BaseVectorStore(ABC):
    """Abstract base class for persistent local vector store implementations.

    Implementations manage nearest-neighbor search, vector storage, metadata mapping,
    and index serialization/deserialization.
    """

    @abstractmethod
    def add(self, vectors: List[List[float]], records: List[VectorRecord]) -> List[int]:
        """Add embedding vectors and associated metadata records to the store."""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[VectorSearchResult]:
        """Perform low-level k-nearest-neighbor search."""
        pass

    @abstractmethod
    def save(self, path: str | os.PathLike, overwrite: bool = False) -> str:
        """Persist vector index and metadata sidecar to disk safely."""
        pass

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str | os.PathLike,
        expected_dimension: Optional[int] = None,
        expected_model_name: Optional[str] = None,
    ) -> BaseVectorStore:
        """Load a persisted vector store from disk with optional validation."""
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Total number of vectors currently indexed in the store."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dense vector dimension of the store."""
        pass

    @property
    @abstractmethod
    def embedding_model_name(self) -> Optional[str]:
        """Optional name/identifier of embedding model used."""
        pass

    @property
    def normalization_expectation(self) -> str:
        """Normalization requirement: 'l2_normalized' or 'raw'."""
        return "l2_normalized"


__all__ = [
    "VectorStoreError",
    "IndexManifest",
    "VectorRecord",
    "VectorSearchResult",
    "BaseVectorStore",
    "validate_vector",
    "validate_vectors",
    "validate_records",
    "validate_top_k",
]
