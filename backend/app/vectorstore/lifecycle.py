"""Vector store index lifecycle, inspection, and safety helpers.

Phase 4.5: Provides lightweight index lifecycle management:
- ``inspect_manifest(path)``: Read index metadata manifest without loading vectors.
- ``exists(path)``: Check if a valid vector index directory exists at path.
- ``validate_index(path, ...)``: Validate index manifest & files without loading full vectors.
- ``delete_index(path, confirm=True)``: Safe deletion helper requiring explicit confirmation flag.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, Optional

from .base import IndexManifest, VectorStoreError

DEFAULT_SCHEMA_FILENAME = "schema.json"
DEFAULT_METADATA_FILENAME = "metadata.json"
DEFAULT_FAISS_FILENAME = "index.faiss"
DEFAULT_NUMPY_FILENAME = "vectors.npy"


def inspect_manifest(path: str | os.PathLike) -> IndexManifest:
    """Lightweight manifest inspection without loading vector data into Python memory.

    Args:
        path: Path to vector store directory

    Returns:
        IndexManifest dataclass containing index metadata and architecture properties

    Raises:
        FileNotFoundError: If path or schema.json does not exist
        VectorStoreError: If schema JSON is malformed or invalid
    """
    abs_path = os.path.abspath(os.fspath(path))
    schema_file = os.path.join(abs_path, DEFAULT_SCHEMA_FILENAME)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Vector store path not found: '{abs_path}'")
    if not os.path.isfile(schema_file):
        raise FileNotFoundError(f"Schema manifest missing: '{schema_file}'")

    try:
        with open(schema_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise VectorStoreError(f"Malformed JSON in schema manifest '{schema_file}': {exc}") from exc
    except Exception as exc:
        raise VectorStoreError(f"Error reading schema manifest '{schema_file}': {exc}") from exc

    if not isinstance(data, dict):
        raise VectorStoreError(f"Schema manifest in '{schema_file}' must be a JSON object")

    try:
        manifest = IndexManifest(**data)
    except Exception as exc:
        raise VectorStoreError(f"Invalid schema manifest content in '{schema_file}': {exc}") from exc

    return manifest


def exists(path: str | os.PathLike) -> bool:
    """Check if a valid vector store directory exists at the given path.

    Args:
        path: Directory path to check

    Returns:
        True if path exists as a directory containing schema.json, metadata.json,
        and an index file (index.faiss or vectors.npy); False otherwise.
    """
    abs_path = os.path.abspath(os.fspath(path))
    if not os.path.isdir(abs_path):
        return False

    schema_file = os.path.join(abs_path, DEFAULT_SCHEMA_FILENAME)
    metadata_file = os.path.join(abs_path, DEFAULT_METADATA_FILENAME)
    faiss_file = os.path.join(abs_path, DEFAULT_FAISS_FILENAME)
    numpy_file = os.path.join(abs_path, DEFAULT_NUMPY_FILENAME)

    has_schema = os.path.isfile(schema_file)
    has_metadata = os.path.isfile(metadata_file)
    has_vector_file = os.path.isfile(faiss_file) or os.path.isfile(numpy_file)

    return has_schema and has_metadata and has_vector_file


def validate_index(
    path: str | os.PathLike,
    expected_dimension: Optional[int] = None,
    expected_model_name: Optional[str] = None,
) -> bool:
    """Validate index directory structure, schema manifest, and metadata without loading vectors.

    Args:
        path: Vector store directory path
        expected_dimension: Optional expected dimension to verify
        expected_model_name: Optional expected embedding model name to verify

    Returns:
        True if index is fully valid

    Raises:
        FileNotFoundError: If path or required files are missing
        VectorStoreError: If manifest, dimension, model name, or metadata validation fails
    """
    abs_path = os.path.abspath(os.fspath(path))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Vector store directory not found: '{abs_path}'")

    # 1. Inspect manifest
    manifest = inspect_manifest(abs_path)

    # 2. Check dimension compatibility if expected
    if expected_dimension is not None:
        if manifest.dimension != expected_dimension:
            raise VectorStoreError(
                f"Dimension mismatch in vector store '{abs_path}': index has dimension "
                f"{manifest.dimension}, but expected {expected_dimension}"
            )

    # 3. Check model compatibility if expected
    if expected_model_name is not None:
        if manifest.embedding_model_name != expected_model_name:
            raise VectorStoreError(
                f"Embedding model mismatch in vector store '{abs_path}': index was built with "
                f"'{manifest.embedding_model_name}', but expected '{expected_model_name}'"
            )

    # 4. Check metadata file exists and count matches manifest
    metadata_file = os.path.join(abs_path, DEFAULT_METADATA_FILENAME)
    if not os.path.isfile(metadata_file):
        raise FileNotFoundError(f"Metadata file missing: '{metadata_file}'")

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            records_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise VectorStoreError(f"Malformed JSON in metadata file '{metadata_file}': {exc}") from exc
    except Exception as exc:
        raise VectorStoreError(f"Error reading metadata file '{metadata_file}': {exc}") from exc

    if not isinstance(records_data, list):
        raise VectorStoreError(f"Metadata in '{metadata_file}' must be a JSON list")

    if len(records_data) != manifest.count:
        raise VectorStoreError(
            f"Metadata count mismatch in '{abs_path}': metadata records count ({len(records_data)}) "
            f"does not match manifest count ({manifest.count})"
        )

    # 5. Check index file exists
    faiss_file = os.path.join(abs_path, DEFAULT_FAISS_FILENAME)
    numpy_file = os.path.join(abs_path, DEFAULT_NUMPY_FILENAME)
    if manifest.vector_store_type == "faiss":
        if not os.path.isfile(faiss_file):
            raise FileNotFoundError(f"FAISS index binary missing: '{faiss_file}'")
    elif manifest.vector_store_type == "numpy":
        if not os.path.isfile(numpy_file):
            raise FileNotFoundError(f"NumPy vector matrix missing: '{numpy_file}'")
    else:
        raise VectorStoreError(f"Unknown vector store type in manifest: '{manifest.vector_store_type}'")

    return True


def delete_index(path: str | os.PathLike, confirm: bool = False) -> bool:
    """Safe helper to delete a local vector store directory.

    Requires explicit confirm=True flag to prevent accidental deletion.

    Args:
        path: Path to vector store directory
        confirm: Must be explicitly set to True to execute deletion

    Returns:
        True if index directory was deleted; False if path did not exist

    Raises:
        ValueError: If confirm is False
    """
    if not confirm:
        raise ValueError(
            f"delete_index() requires explicit confirm=True flag to delete index at '{path}'"
        )

    abs_path = os.path.abspath(os.fspath(path))
    if not os.path.exists(abs_path):
        return False

    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path)
    else:
        os.remove(abs_path)
    return True


__all__ = [
    "inspect_manifest",
    "exists",
    "validate_index",
    "delete_index",
]
