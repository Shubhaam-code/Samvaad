"""Persisted RAG index manifest: format, contents, and compatibility checks.

The index manifest describes everything the runtime needs to decide
whether a persisted index can be loaded safely:

- index format / schema versions (reject indexes built by incompatible
  tooling)
- dataset / source identifier (where the corpus came from)
- embedding provider + model + dimension (reject indexes built with a
  different model or dimension instead of silently returning wrong
  retrieval results)
- vector store backend
- chunk / document / vector counts
- creation timestamp

The manifest is a single ``manifest.json`` file at the index directory
root. The vector store directory inside the index carries its own
``schema.json`` (vector store schema manifest) - this file is the
index-level manifest on top of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

INDEX_FORMAT = "hhgoa-rag-index"
INDEX_FORMAT_VERSION = 1
INDEX_SCHEMA_VERSION = "1.0"

MANIFEST_FILENAME = "manifest.json"


class IndexCompatibilityError(Exception):
    """Raised when a persisted index is missing, corrupt, or incompatible."""


class DatasetInfo(BaseModel):
    """Dataset/source identifier for the indexed corpus."""

    model_config = ConfigDict(protected_namespaces=())

    source: str = Field(
        ...,
        description="Dataset identifier (e.g. 'msmarco-xi' or 'local-parquet')",
    )
    config: Optional[str] = Field(None, description="Dataset config name")
    split: Optional[str] = Field(None, description="Dataset split name")
    input_path: Optional[str] = Field(None, description="Source path used for the build")
    processed_path: Optional[str] = Field(
        None,
        description="Processed CanonicalPassage parquet used for the build",
    )
    target_lang: Optional[str] = Field(None, description="Target language code")


class EmbeddingInfo(BaseModel):
    """Embedding provider/model/dimension used to build the index."""

    model_config = ConfigDict(protected_namespaces=())

    provider: str = Field(..., description="Embedding provider (e.g. 'huggingface')")
    model: str = Field(..., description="Embedding model identifier")
    dimension: int = Field(..., ge=1, description="Embedding vector dimension")
    normalize: bool = Field(True, description="Whether embeddings are L2-normalized")
    device: Optional[str] = Field(None, description="Inference device used at build time")


class VectorStoreInfo(BaseModel):
    """Vector store backend details for the persisted index."""

    model_config = ConfigDict(protected_namespaces=())

    backend: str = Field(..., description="Vector store backend ('faiss' or 'numpy')")
    index_type: Optional[str] = Field(None, description="Index implementation type")
    metric: Optional[str] = Field(None, description="Similarity metric")
    count: int = Field(0, ge=0, description="Total vectors indexed")


class ChunkingInfo(BaseModel):
    """Chunking configuration used to build the index."""

    model_config = ConfigDict(protected_namespaces=())

    strategy: str = Field("passage", description="Chunking strategy used")
    params: dict[str, Any] = Field(default_factory=dict)


class CountsInfo(BaseModel):
    """Corpus counts recorded in the manifest."""

    model_config = ConfigDict(protected_namespaces=())

    chunks: int = Field(0, ge=0, description="Total chunks in the persisted chunk corpus")
    documents: int = Field(0, ge=0, description="Total distinct source documents")
    vectors: int = Field(0, ge=0, description="Total vectors in the vector store")


class RagIndexManifest(BaseModel):
    """Index-level manifest describing a persisted RAG index."""

    model_config = ConfigDict(protected_namespaces=())

    format: str = Field(INDEX_FORMAT, description="Index format identifier")
    format_version: int = Field(
        INDEX_FORMAT_VERSION,
        description="Index format version (incompatible formats rejected)",
    )
    schema_version: str = Field(
        INDEX_SCHEMA_VERSION,
        description="Manifest schema version",
    )
    created_at: str = Field(..., description="ISO-8601 UTC creation timestamp")
    dataset: DatasetInfo = Field(..., description="Dataset/source identifier")
    embedding: EmbeddingInfo = Field(..., description="Embedding configuration")
    vector_store: VectorStoreInfo = Field(..., description="Vector store details")
    chunking: ChunkingInfo = Field(default_factory=ChunkingInfo)
    counts: CountsInfo = Field(..., description="Corpus counts")
    build: dict[str, Any] = Field(default_factory=dict, description="Build metadata")


def write_manifest(index_dir: str | Path, manifest: RagIndexManifest) -> Path:
    """Write the index manifest to ``manifest.json`` inside ``index_dir``.

    Args:
        index_dir: Index directory (must exist)
        manifest: Manifest to persist

    Returns:
        Path to the written manifest file

    Raises:
        FileNotFoundError: If the index directory does not exist
    """
    index_path = Path(index_dir)
    if not index_path.is_dir():
        raise FileNotFoundError(f"Index directory not found: '{index_path}'")
    manifest_file = index_path / MANIFEST_FILENAME
    manifest_file.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_file


def read_manifest(index_dir: str | Path) -> RagIndexManifest:
    """Read and parse the index manifest from an index directory.

    Args:
        index_dir: Index directory containing ``manifest.json``

    Returns:
        Parsed RagIndexManifest

    Raises:
        FileNotFoundError: If the index directory or manifest file is missing
        IndexCompatibilityError: If the manifest is malformed or invalid
    """
    index_path = Path(index_dir)
    if not index_path.is_dir():
        raise FileNotFoundError(f"Index directory not found: '{index_path}'")
    manifest_file = index_path / MANIFEST_FILENAME
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Index manifest missing: '{manifest_file}'")

    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IndexCompatibilityError(
            f"Malformed JSON in index manifest '{manifest_file}': {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise IndexCompatibilityError(
            f"Index manifest '{manifest_file}' must be a JSON object"
        )

    if data.get("format") != INDEX_FORMAT:
        raise IndexCompatibilityError(
            f"Unsupported index format {data.get('format')!r} in '{manifest_file}' "
            f"(expected '{INDEX_FORMAT}')"
        )
    if data.get("format_version") != INDEX_FORMAT_VERSION:
        raise IndexCompatibilityError(
            f"Unsupported index format version {data.get('format_version')!r} "
            f"(expected {INDEX_FORMAT_VERSION})"
        )

    try:
        return RagIndexManifest(**data)
    except Exception as exc:
        raise IndexCompatibilityError(
            f"Invalid index manifest content in '{manifest_file}': {exc}"
        ) from exc


def validate_manifest_compat(
    manifest: RagIndexManifest,
    *,
    expected_model_name: Optional[str] = None,
    expected_dimension: Optional[int] = None,
    expected_backend: Optional[str] = None,
) -> None:
    """Validate a manifest against expected runtime configuration.

    An incompatible index (different embedding model, dimension, or
    vector store backend) must fail loudly instead of silently returning
    bad retrieval results.

    Args:
        manifest: Manifest to validate
        expected_model_name: Embedding model the runtime expects
        expected_dimension: Embedding dimension the runtime expects
        expected_backend: Vector store backend the runtime expects

    Raises:
        IndexCompatibilityError: If any expectation mismatches
    """
    if expected_model_name is not None:
        if manifest.embedding.model != expected_model_name:
            raise IndexCompatibilityError(
                f"Embedding model mismatch: index was built with "
                f"'{manifest.embedding.model}', but runtime expects "
                f"'{expected_model_name}'"
            )

    if expected_dimension is not None:
        if manifest.embedding.dimension != expected_dimension:
            raise IndexCompatibilityError(
                f"Embedding dimension mismatch: index was built with dimension "
                f"{manifest.embedding.dimension}, but runtime expects "
                f"{expected_dimension}"
            )

    if expected_backend is not None:
        if manifest.vector_store.backend != expected_backend:
            raise IndexCompatibilityError(
                f"Vector store backend mismatch: index was built with "
                f"'{manifest.vector_store.backend}', but runtime expects "
                f"'{expected_backend}'"
            )


__all__ = [
    "INDEX_FORMAT",
    "INDEX_FORMAT_VERSION",
    "INDEX_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "CountsInfo",
    "ChunkingInfo",
    "DatasetInfo",
    "EmbeddingInfo",
    "IndexCompatibilityError",
    "RagIndexManifest",
    "VectorStoreInfo",
    "read_manifest",
    "validate_manifest_compat",
    "write_manifest",
]
