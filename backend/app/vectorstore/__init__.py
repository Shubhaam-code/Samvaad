"""Vector store and index architecture package.

Phase 4.4: Provides a persistent, reusable local vector index architecture
supporting FAISS and fallback implementations with metadata mapping, ordering stability,
and low-level k-nearest-neighbor search.

Exports:
- ``BaseVectorStore``: Abstract base class interface
- ``VectorRecord``: Metadata mapping for 1:1 vector traceability
- ``VectorSearchResult``: Search result item containing record, score, position
- ``VectorStoreError``: Operational exception
- ``FaissVectorStore``: FAISS IndexFlatIP persistent vector store
- ``NumpyVectorStore``: Pure NumPy persistent vector store fallback
- ``create_vector_store``: Factory function
"""

from typing import Optional

from .base import (
    BaseVectorStore,
    IndexManifest,
    VectorRecord,
    VectorSearchResult,
    VectorStoreError,
    validate_records,
    validate_top_k,
    validate_vector,
    validate_vectors,
)
from .faiss_store import HAS_FAISS, FaissVectorStore
from .lifecycle import delete_index, exists, inspect_manifest, validate_index
from .numpy_store import NumpyVectorStore


def create_vector_store(
    dimension: int,
    store_type: str = "faiss",
    embedding_model_name: Optional[str] = None,
) -> BaseVectorStore:
    """Factory function to create a vector store instance.

    Args:
        dimension: Dense vector dimension (must be positive)
        store_type: Vector store backend ("faiss" or "numpy", default "faiss")
        embedding_model_name: Optional name of embedding model used

    Returns:
        BaseVectorStore implementation

    Raises:
        ValueError: If store_type is unsupported or dimension is invalid
        VectorStoreError: If "faiss" requested but faiss is not installed
    """
    stype = store_type.strip().lower()
    if stype == "faiss":
        if not HAS_FAISS:
            raise VectorStoreError(
                "faiss package is not installed in the environment. "
                "Use store_type='numpy' or install faiss-cpu."
            )
        return FaissVectorStore(dimension=dimension, embedding_model_name=embedding_model_name)
    elif stype == "numpy":
        return NumpyVectorStore(dimension=dimension, embedding_model_name=embedding_model_name)
    else:
        raise ValueError(
            f"Unsupported vector store type: '{store_type}'. Supported types: 'faiss', 'numpy'"
        )


__all__ = [
    "BaseVectorStore",
    "IndexManifest",
    "VectorRecord",
    "VectorSearchResult",
    "VectorStoreError",
    "FaissVectorStore",
    "NumpyVectorStore",
    "HAS_FAISS",
    "create_vector_store",
    "inspect_manifest",
    "exists",
    "validate_index",
    "delete_index",
    "validate_vector",
    "validate_vectors",
    "validate_records",
    "validate_top_k",
]
