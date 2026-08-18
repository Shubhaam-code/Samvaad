"""Runtime loading of persisted RAG indexes.

Wires the persisted pieces back together:

    index directory
        -> read + validate index manifest (format, model, dimension, backend)
        -> load the persisted vector store (FaissVectorStore / NumpyVectorStore)
        -> load the persisted chunk resolver (chunk_id -> Chunk with chunk_text)
        -> cross-check counts (store == manifest == chunk corpus)

This is the loader used by ``get_orchestrator()`` in ``app.api.dependencies``.
Incompatible or corrupt indexes raise instead of silently returning bad
retrieval results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.indexing.chunk_store import (
    CHUNKS_FILENAME,
    CHUNKS_INDEX_FILENAME,
    JsonlChunkResolver,
)
from app.indexing.manifest import (
    MANIFEST_FILENAME,
    IndexCompatibilityError,
    RagIndexManifest,
    read_manifest,
    validate_manifest_compat,
)
from app.vectorstore import (
    BaseVectorStore,
    FaissVectorStore,
    NumpyVectorStore,
)

logger = logging.getLogger(__name__)

VECTORSTORE_DIRNAME = "vectorstore"


def resolve_index_dir(path: Optional[str | Path]) -> Optional[Path]:
    """Resolve a configured index directory path.

    Empty / None values resolve to None (index not configured). Relative
    paths are resolved against the current working directory.

    Args:
        path: Configured index directory

    Returns:
        Absolute Path, or None if not configured
    """
    if path is None:
        return None
    value = str(path).strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def index_exists(index_dir: str | Path) -> bool:
    """Check whether a complete, loadable index directory exists.

    A directory only counts as an index when it contains the index
    manifest, the chunk corpus, and a vector store directory.

    Args:
        index_dir: Candidate index directory

    Returns:
        True if the directory looks like a complete persisted index
    """
    path = Path(index_dir)
    if not path.is_dir():
        return False
    if not (path / MANIFEST_FILENAME).is_file():
        return False
    if not (path / CHUNKS_FILENAME).is_file():
        return False
    store_dir = path / VECTORSTORE_DIRNAME
    if not store_dir.is_dir():
        return False
    if not (store_dir / "schema.json").is_file():
        return False
    if not (store_dir / "metadata.json").is_file():
        return False
    has_faiss = (store_dir / "index.faiss").is_file()
    has_numpy = (store_dir / "vectors.npy").is_file()
    return has_faiss or has_numpy


def load_index(
    index_dir: str | Path,
    *,
    expected_model_name: Optional[str] = None,
    expected_dimension: Optional[int] = None,
    expected_backend: Optional[str] = None,
    lazy_chunks: bool = True,
) -> tuple[BaseVectorStore, JsonlChunkResolver, RagIndexManifest]:
    """Load a persisted RAG index: vector store + chunk resolver + manifest.

    Performs full compatibility validation before returning:

    1. Manifest format/version/schema validation.
    2. Embedding model / dimension / backend compatibility checks.
    3. Vector store load (its own schema.json is validated inside).
    4. Chunk resolver load.
    5. Cross-checks: store.count == manifest.counts.vectors,
       resolver.count == manifest.counts.chunks, and store count ==
       chunk count.

    Args:
        index_dir: Index directory to load
        expected_model_name: Expected embedding model (runtime config)
        expected_dimension: Expected embedding dimension (runtime config)
        expected_backend: Expected vector store backend (runtime config)
        lazy_chunks: If True (default), the chunk resolver reads chunk
            lines on demand instead of materializing the full corpus

    Returns:
        Tuple of (vector_store, chunk_resolver, manifest)

    Raises:
        FileNotFoundError: If the index directory or required files are missing
        IndexCompatibilityError: If the index is incompatible or corrupt
        VectorStoreError: If the persisted vector store fails validation
    """
    index_path = Path(index_dir)
    if not index_path.is_dir():
        raise FileNotFoundError(f"Index directory not found: '{index_path}'")

    manifest = read_manifest(index_path)
    validate_manifest_compat(
        manifest,
        expected_model_name=expected_model_name,
        expected_dimension=expected_dimension,
        expected_backend=expected_backend,
    )

    store_dir = index_path / VECTORSTORE_DIRNAME
    if not store_dir.is_dir():
        raise IndexCompatibilityError(
            f"Vector store directory missing in index '{index_path}': '{store_dir}'"
        )

    store = _load_vector_store(
        store_dir,
        backend=manifest.vector_store.backend,
        expected_dimension=manifest.embedding.dimension,
        expected_model_name=manifest.embedding.model,
    )

    resolver = JsonlChunkResolver(
        index_path / CHUNKS_FILENAME,
        index_path / CHUNKS_INDEX_FILENAME,
        lazy=lazy_chunks,
    )

    if store.count != manifest.counts.vectors:
        raise IndexCompatibilityError(
            f"Vector count mismatch in index '{index_path}': store has "
            f"{store.count} vectors but the manifest records "
            f"{manifest.counts.vectors}"
        )
    if resolver.count != manifest.counts.chunks:
        raise IndexCompatibilityError(
            f"Chunk count mismatch in index '{index_path}': chunk corpus has "
            f"{resolver.count} chunks but the manifest records "
            f"{manifest.counts.chunks}"
        )
    if store.count != resolver.count:
        raise IndexCompatibilityError(
            f"Index integrity mismatch in '{index_path}': vector store has "
            f"{store.count} vectors but the chunk corpus has {resolver.count} chunks"
        )

    logger.info(
        "Loaded index '%s' (backend=%s, dimension=%d, vectors=%d, chunks=%d)",
        index_path,
        manifest.vector_store.backend,
        manifest.embedding.dimension,
        store.count,
        resolver.count,
    )
    return store, resolver, manifest


def _load_vector_store(
    store_dir: Path,
    *,
    backend: str,
    expected_dimension: int,
    expected_model_name: str,
) -> BaseVectorStore:
    """Load the persisted vector store for the manifest backend."""
    if backend == "faiss":
        return FaissVectorStore.load(
            store_dir,
            expected_dimension=expected_dimension,
            expected_model_name=expected_model_name,
        )
    if backend == "numpy":
        return NumpyVectorStore.load(
            store_dir,
            expected_dimension=expected_dimension,
            expected_model_name=expected_model_name,
        )
    raise IndexCompatibilityError(
        f"Unknown vector store backend in manifest: '{backend}'"
    )


__all__ = [
    "VECTORSTORE_DIRNAME",
    "index_exists",
    "load_index",
    "resolve_index_dir",
]