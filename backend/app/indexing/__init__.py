"""Persisted RAG index package: manifest, chunk corpus, and runtime loading.

Phase 5.3: Makes a built index loadable by the retrieval layer.

- manifest: index-level manifest (format, dataset, embedding, vector
  store, counts) + compatibility validation
- chunk_store: persisted chunk corpus (chunks.jsonl + offset index) and
  a lazy chunk_id -> Chunk resolver
- loader: load_index() -> (vector store, chunk resolver, manifest) with
  full compatibility/integrity checks, used by get_orchestrator()
"""

from .chunk_store import (
    CHUNKS_FILENAME,
    CHUNKS_INDEX_FILENAME,
    JsonlChunkResolver,
    JsonlChunkStore,
)
from .loader import (
    VECTORSTORE_DIRNAME,
    index_exists,
    load_index,
    resolve_index_dir,
)
from .manifest import (
    INDEX_FORMAT,
    INDEX_FORMAT_VERSION,
    INDEX_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    ChunkingInfo,
    CountsInfo,
    DatasetInfo,
    EmbeddingInfo,
    IndexCompatibilityError,
    RagIndexManifest,
    VectorStoreInfo,
    read_manifest,
    validate_manifest_compat,
    write_manifest,
)

__all__ = [
    # Manifest
    "INDEX_FORMAT",
    "INDEX_FORMAT_VERSION",
    "INDEX_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "RagIndexManifest",
    "DatasetInfo",
    "EmbeddingInfo",
    "VectorStoreInfo",
    "ChunkingInfo",
    "CountsInfo",
    "IndexCompatibilityError",
    "read_manifest",
    "write_manifest",
    "validate_manifest_compat",
    # Chunk corpus
    "CHUNKS_FILENAME",
    "CHUNKS_INDEX_FILENAME",
    "JsonlChunkStore",
    "JsonlChunkResolver",
    # Loader
    "VECTORSTORE_DIRNAME",
    "index_exists",
    "load_index",
    "resolve_index_dir",
]