"""FAISS-backed local vector store implementation.

Phase 4.4: Provides a persistent local vector index backed by FAISS IndexFlatIP.

Key Features:
- Metric: Inner Product (faiss.IndexFlatIP). When vectors are L2-normalized,
  Inner Product corresponds to Cosine Similarity.
- Explicit Normalization Expectation: Expects caller to provide normalized vectors if
  cosine similarity is desired.
- Safe Serialization: Persists FAISS binary index (index.faiss), metadata sidecar
  (metadata.json), and schema manifest (schema.json) without arbitrary pickle.
- Atomic Persistence: Uses temporary destination directory during save to prevent
  corrupted partial writes.
- Ordering Stability: Vector index position i maps 1:1 to metadata record i.
- Index-level Logging: Progress logged at batch/operation level only (no vector values).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import List, Optional
import numpy as np

from .base import (
    BaseVectorStore,
    VectorRecord,
    VectorSearchResult,
    VectorStoreError,
    validate_records,
    validate_top_k,
    validate_vector,
    validate_vectors,
)

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

SCHEMA_VERSION = "1.0"
DEFAULT_INDEX_FILENAME = "index.faiss"
DEFAULT_METADATA_FILENAME = "metadata.json"
DEFAULT_SCHEMA_FILENAME = "schema.json"


class FaissVectorStore(BaseVectorStore):
    """FAISS-backed persistent vector index using Inner Product metric.

    Expects vectors of fixed dimension. If vectors are L2-normalized,
    inner product corresponds to cosine similarity.

    Args:
        dimension: Dense vector dimension (must be positive)
        embedding_model_name: Optional name/identifier of embedding model used

    Raises:
        VectorStoreError: If faiss package is not installed
        ValueError: If dimension <= 0
    """

    def __init__(self, dimension: int, embedding_model_name: Optional[str] = None):
        if not HAS_FAISS:
            raise VectorStoreError(
                "faiss package is not installed. Install faiss-cpu to use FaissVectorStore."
            )
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError(f"Vector dimension must be a positive integer, got {dimension!r}")

        self._dimension = dimension
        self._embedding_model_name = embedding_model_name
        self._index = faiss.IndexFlatIP(dimension)
        self._records: List[VectorRecord] = []

    @property
    def count(self) -> int:
        """Total number of vectors currently indexed in the store."""
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        """Dense vector dimension of the store."""
        return self._dimension

    @property
    def embedding_model_name(self) -> Optional[str]:
        """Optional embedding model identifier."""
        return self._embedding_model_name

    def add(self, vectors: List[List[float]], records: List[VectorRecord]) -> List[int]:
        """Add embedding vectors and associated metadata records to the FAISS index.

        Args:
            vectors: List of dense embedding float vectors
            records: List of VectorRecord metadata objects (1:1 with vectors)

        Returns:
            List of 0-based index positions assigned to inserted vectors

        Raises:
            ValueError: If input validation fails (dimension mismatch, non-finite, empty)
        """
        validate_vectors(vectors, expected_dim=self._dimension)
        validate_records(records, expected_count=len(vectors))

        start_pos = self.count
        arr = np.array(vectors, dtype=np.float32)

        # FAISS add
        self._index.add(arr)
        self._records.extend(records)

        assigned_positions = list(range(start_pos, start_pos + len(vectors)))
        logger.info(
            f"Added {len(vectors)} vectors to FaissVectorStore "
            f"(new count: {self.count}, dimension: {self._dimension})"
        )
        return assigned_positions

    def search(self, query_vector: List[float], top_k: int) -> List[VectorSearchResult]:
        """Perform k-nearest-neighbor search using Inner Product similarity.

        Args:
            query_vector: Dense query vector
            top_k: Number of nearest neighbors to retrieve

        Returns:
            List of VectorSearchResult objects sorted by similarity score descending

        Raises:
            VectorStoreError: If vector store is empty
            ValueError: If query_vector or top_k validation fails
        """
        if self.count == 0:
            raise VectorStoreError("Cannot search an empty vector store")

        validate_vector(query_vector, expected_dim=self._dimension)
        validate_top_k(top_k)

        # Clamp top_k to available vectors if top_k > count
        k_search = min(top_k, self.count)

        query_arr = np.array([query_vector], dtype=np.float32)
        scores_arr, ids_arr = self._index.search(query_arr, k_search)

        results: List[VectorSearchResult] = []
        raw_scores = scores_arr[0]
        raw_ids = ids_arr[0]

        for score, idx in zip(raw_scores, raw_ids):
            if idx < 0 or idx >= len(self._records):
                continue
            rec = self._records[idx]
            results.append(
                VectorSearchResult(
                    chunk_id=rec.chunk_id,
                    score=float(score),
                    position=int(idx),
                    record=rec,
                )
            )

        logger.info(
            f"Search completed on FaissVectorStore (requested top_k: {top_k}, "
            f"returned: {len(results)})"
        )
        return results

    def save(self, path: str | os.PathLike, overwrite: bool = False) -> str:
        """Persist FAISS index, metadata records, and schema to disk atomically.

        Args:
            path: Target directory path (str or PathLike)
            overwrite: If True, overwrite existing directory; else raise FileExistsError

        Returns:
            Absolute normalized path string where store was saved

        Raises:
            FileExistsError: If path exists and overwrite is False
            VectorStoreError: If saving fails
        """
        abs_path = os.path.abspath(os.fspath(path))

        if os.path.exists(abs_path):
            if not overwrite:
                raise FileExistsError(
                    f"Vector store target path already exists: '{abs_path}'. "
                    f"Set overwrite=True to overwrite."
                )

        # Create temporary working directory for atomic save
        parent_dir = os.path.dirname(abs_path) or "."
        os.makedirs(parent_dir, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix=".tmp_faiss_", dir=parent_dir)

        try:
            # 1. Save FAISS index binary
            faiss_file = os.path.join(tmp_dir, DEFAULT_INDEX_FILENAME)
            faiss.write_index(self._index, faiss_file)

            # 2. Save metadata sidecar
            meta_file = os.path.join(tmp_dir, DEFAULT_METADATA_FILENAME)
            records_data = [rec.model_dump() for rec in self._records]
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(records_data, f, ensure_ascii=False, indent=2)

            # 3. Save schema manifest
            schema_file = os.path.join(tmp_dir, DEFAULT_SCHEMA_FILENAME)
            schema_data = {
                "schema_version": SCHEMA_VERSION,
                "vector_store_type": "faiss",
                "index_type": "IndexFlatIP",
                "dimension": self._dimension,
                "metric": "inner_product",
                "count": self.count,
                "embedding_model_name": self._embedding_model_name,
                "normalization_expectation": self.normalization_expectation,
                "metadata_schema_version": "1.0",
            }
            with open(schema_file, "w", encoding="utf-8") as f:
                json.dump(schema_data, f, ensure_ascii=False, indent=2)

            # Atomic move / replace
            if os.path.exists(abs_path):
                if os.path.isdir(abs_path):
                    shutil.rmtree(abs_path)
                else:
                    os.remove(abs_path)

            shutil.move(tmp_dir, abs_path)
            logger.info(
                f"Saved FaissVectorStore to '{abs_path}' "
                f"(count: {self.count}, dimension: {self._dimension})"
            )
            return abs_path

        except Exception as e:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if isinstance(e, FileExistsError):
                raise
            raise VectorStoreError(f"Failed to save FaissVectorStore to '{abs_path}': {e}") from e

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        expected_dimension: Optional[int] = None,
        expected_model_name: Optional[str] = None,
    ) -> FaissVectorStore:
        """Load a persisted FaissVectorStore from disk.

        Args:
            path: Directory containing persisted index and metadata (str or PathLike)
            expected_dimension: Optional expected dimension to validate against manifest
            expected_model_name: Optional expected model name to validate against manifest

        Returns:
            Restored FaissVectorStore instance

        Raises:
            FileNotFoundError: If path or required files do not exist
            VectorStoreError: If schema/version/dimension/model/count validation fails
        """
        if not HAS_FAISS:
            raise VectorStoreError(
                "faiss package is not installed. Install faiss-cpu to use FaissVectorStore."
            )

        abs_path = os.path.abspath(os.fspath(path))
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Vector store path not found: '{abs_path}'")

        schema_file = os.path.join(abs_path, DEFAULT_SCHEMA_FILENAME)
        faiss_file = os.path.join(abs_path, DEFAULT_INDEX_FILENAME)
        meta_file = os.path.join(abs_path, DEFAULT_METADATA_FILENAME)

        for req_file in [schema_file, faiss_file, meta_file]:
            if not os.path.isfile(req_file):
                raise FileNotFoundError(
                    f"Required vector store file missing: '{req_file}' in '{abs_path}'"
                )

        try:
            # 1. Load schema manifest
            with open(schema_file, "r", encoding="utf-8") as f:
                try:
                    schema_data = json.load(f)
                except json.JSONDecodeError as exc:
                    raise VectorStoreError(f"Malformed JSON in schema file '{schema_file}': {exc}") from exc

            if not isinstance(schema_data, dict):
                raise VectorStoreError(f"Schema manifest in '{schema_file}' must be a JSON object")

            if schema_data.get("schema_version") != SCHEMA_VERSION:
                raise VectorStoreError(
                    f"Unsupported vector store schema version: {schema_data.get('schema_version')!r} "
                    f"(expected '{SCHEMA_VERSION}')"
                )
            if schema_data.get("vector_store_type") != "faiss":
                raise VectorStoreError(
                    f"Incompatible vector_store_type: {schema_data.get('vector_store_type')!r} "
                    f"(expected 'faiss')"
                )

            dimension = schema_data.get("dimension")
            if not isinstance(dimension, int) or dimension <= 0:
                raise VectorStoreError(f"Invalid dimension in schema manifest: {dimension!r}")

            manifest_count = schema_data.get("count")
            if not isinstance(manifest_count, int) or manifest_count < 0:
                raise VectorStoreError(f"Invalid count in schema manifest: {manifest_count!r}")

            model_name = schema_data.get("embedding_model_name")

            # Validate expected dimension if specified
            if expected_dimension is not None and dimension != expected_dimension:
                raise VectorStoreError(
                    f"Dimension mismatch in vector store '{abs_path}': index dimension ({dimension}) "
                    f"does not match expected dimension ({expected_dimension})"
                )

            # Validate expected model name if specified
            if expected_model_name is not None and model_name != expected_model_name:
                raise VectorStoreError(
                    f"Embedding model mismatch in vector store '{abs_path}': index was built with "
                    f"'{model_name}', but expected '{expected_model_name}'"
                )

            # Create store instance
            store = cls(dimension=dimension, embedding_model_name=model_name)

            # 2. Load FAISS index
            loaded_index = faiss.read_index(faiss_file)
            if loaded_index.d != dimension:
                raise VectorStoreError(
                    f"Loaded FAISS index dimension ({loaded_index.d}) "
                    f"does not match schema dimension ({dimension})"
                )
            if loaded_index.ntotal != manifest_count:
                raise VectorStoreError(
                    f"FAISS index count ({loaded_index.ntotal}) "
                    f"does not match manifest count ({manifest_count})"
                )

            store._index = loaded_index

            # 3. Load metadata sidecar
            with open(meta_file, "r", encoding="utf-8") as f:
                try:
                    records_raw = json.load(f)
                except json.JSONDecodeError as exc:
                    raise VectorStoreError(f"Malformed JSON in metadata file '{meta_file}': {exc}") from exc

            if not isinstance(records_raw, list):
                raise VectorStoreError(f"Metadata sidecar in '{meta_file}' must be a JSON list")

            if len(records_raw) != loaded_index.ntotal:
                raise VectorStoreError(
                    f"Metadata record count ({len(records_raw)}) does not match "
                    f"FAISS index total ({loaded_index.ntotal})"
                )

            records: List[VectorRecord] = []
            for idx, item in enumerate(records_raw):
                if not isinstance(item, dict):
                    raise VectorStoreError(f"Metadata record at index {idx} is not a dictionary")
                try:
                    records.append(VectorRecord(**item))
                except Exception as exc:
                    raise VectorStoreError(f"Invalid VectorRecord at index {idx}: {exc}") from exc

            store._records = records

            logger.info(
                f"Loaded FaissVectorStore from '{abs_path}' "
                f"(count: {store.count}, dimension: {store.dimension})"
            )
            return store

        except Exception as e:
            if isinstance(e, (VectorStoreError, FileNotFoundError)):
                raise
            raise VectorStoreError(f"Failed to load FaissVectorStore from '{abs_path}': {e}") from e


__all__ = ["FaissVectorStore", "HAS_FAISS"]
