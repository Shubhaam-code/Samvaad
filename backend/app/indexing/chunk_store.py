"""Persisted chunk corpus store and resolver (chunk_id -> Chunk).

The vector store metadata sidecar only carries ids
(``VectorRecord.chunk_id``); the actual ``Chunk`` objects with
``chunk_text`` live here so the runtime can reconstruct full Chunk
evidence for GroundingVerifier.

Format
------
- ``chunks.jsonl``: one JSON-encoded ``Chunk`` per line (newlines inside
  text are JSON-escaped, so every chunk is exactly one physical line).
- ``chunks_index.json``: ``{chunk_id: byte_offset}`` map written by the
  builder for O(1) random access at runtime.

Builder side: ``JsonlChunkStore`` appends chunks while writing the
corpus; ``finalize()`` persists the offset index.

Runtime side: ``JsonlChunkResolver`` resolves chunk ids to ``Chunk``
objects. It is lazy by default (only the offset map is kept in memory,
chunk lines are read on demand), which keeps startup memory bounded for
large corpora. With ``lazy=False`` the entire corpus is loaded into
memory as a plain dict, matching the in-memory ``DictChunkResolver``
behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.chunking.models import Chunk
from app.retrieval.resolver import ChunkResolver, validate_chunk_ids

CHUNKS_FILENAME = "chunks.jsonl"
CHUNKS_INDEX_FILENAME = "chunks_index.json"


class JsonlChunkStore:
    """Incremental writer for the persisted chunk corpus.

    Args:
        path: Path to the ``chunks.jsonl`` file to write

    Raises:
        ValueError: If the parent directory does not exist
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.parent.is_dir():
            raise ValueError(
                f"Chunk store parent directory does not exist: '{self._path.parent}'"
            )
        # Binary mode so tell() yields true byte offsets that match the
        # resolver's random-access binary reads.
        self._file = self._path.open("wb")
        self._offsets: dict[str, int] = {}
        self._count = 0

    @property
    def path(self) -> Path:
        """Path to the chunk corpus file."""
        return self._path

    @property
    def count(self) -> int:
        """Number of chunks written so far."""
        return self._count

    def __enter__(self) -> JsonlChunkStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying file handle (safe to call repeatedly)."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def append(self, chunks: list[Chunk]) -> int:
        """Append a batch of chunks to the corpus.

        Args:
            chunks: Chunks to append

        Returns:
            Number of chunks appended

        Raises:
            ValueError: If chunks is not a list or contains invalid Chunks
            RuntimeError: If the store is closed
        """
        if not isinstance(chunks, list):
            raise ValueError(f"chunks must be a list, got {type(chunks).__name__}")
        if self._file is None:
            raise RuntimeError(f"Chunk store is closed: '{self._path}'")
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise ValueError(
                    f"Chunk store requires Chunk instances, got {type(chunk).__name__}"
                )
            offset = self._file.tell()
            self._file.write(chunk.model_dump_json().encode("utf-8") + b"\n")
            self._offsets[chunk.chunk_id] = offset
            self._count += 1
        return len(chunks)

    def finalize(self, index_path: str | Path | None = None) -> Path:
        """Write the offset index file and close the store.

        Args:
            index_path: Optional path for the offset index
                (defaults to ``chunks_index.json`` next to the corpus)

        Returns:
            Path to the written offset index file
        """
        self.close()
        index_file = Path(index_path) if index_path is not None else (
            self._path.with_name(CHUNKS_INDEX_FILENAME)
        )
        index_file.write_text(
            json.dumps(self._offsets, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return index_file


class JsonlChunkResolver(ChunkResolver):
    """chunk_id -> Chunk resolver backed by the persisted JSONL corpus.

    Args:
        jsonl_path: Path to ``chunks.jsonl``
        index_path: Optional path to the offset index
            (defaults to ``chunks_index.json`` next to the corpus)
        lazy: If True (default), only the offset map is kept in memory
            and chunk lines are read on demand; if False, the whole
            corpus is loaded into memory

    Raises:
        FileNotFoundError: If the corpus or index file is missing
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        index_path: str | Path | None = None,
        lazy: bool = True,
    ) -> None:
        self._jsonl_path = Path(jsonl_path)
        if not self._jsonl_path.is_file():
            raise FileNotFoundError(
                f"Chunk corpus file not found: '{self._jsonl_path}'"
            )

        if index_path is not None:
            self._index_path = Path(index_path)
        else:
            self._index_path = self._jsonl_path.with_name(CHUNKS_INDEX_FILENAME)

        if self._index_path.is_file():
            self._offsets = json.loads(self._index_path.read_text(encoding="utf-8"))
            if not isinstance(self._offsets, dict):
                raise ValueError(
                    f"Chunk offset index '{self._index_path}' must be a JSON object"
                )
        else:
            self._offsets = self._scan_offsets()

        self._lazy = lazy
        self._chunks: Optional[dict[str, Chunk]] = None
        if not lazy:
            self._load_all()

    def _scan_offsets(self) -> dict[str, int]:
        """Build the offset map by scanning the corpus once."""
        offsets: dict[str, int] = {}
        with self._jsonl_path.open("rb") as f:
            offset = 0
            for raw_line in f:
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid chunk line at byte offset {offset} in "
                        f"'{self._jsonl_path}': {exc}"
                    ) from exc
                if not isinstance(record, dict) or not record.get("chunk_id"):
                    raise ValueError(
                        f"Invalid chunk line at byte offset {offset} in "
                        f"'{self._jsonl_path}'"
                    )
                offsets[record["chunk_id"]] = offset
                offset = f.tell()
        return offsets

    def _load_all(self) -> None:
        """Load the entire corpus into an in-memory dict."""
        chunks: dict[str, Chunk] = {}
        with self._jsonl_path.open("rb") as f:
            for raw_line in f:
                chunk = Chunk.model_validate_json(raw_line.decode("utf-8"))
                chunks[chunk.chunk_id] = chunk
        self._chunks = chunks

    @property
    def count(self) -> int:
        """Number of chunks in the corpus."""
        return len(self._offsets)

    @property
    def chunk_ids(self) -> list[str]:
        """Chunk ids in the corpus, in insertion order."""
        return list(self._offsets.keys())

    def resolve(self, chunk_ids: list[str]) -> list[Chunk]:
        """Resolve chunk ids to actual Chunk evidence objects.

        Ordering: results follow the relative order of resolvable ids in
        the input list. Unresolvable ids are silently absent.

        Args:
            chunk_ids: Non-empty list of chunk ids to resolve

        Returns:
            List of Chunk objects in input order (resolvable ids only)

        Raises:
            ValueError: If chunk_ids is empty or contains invalid ids
        """
        validate_chunk_ids(chunk_ids)
        if not self._lazy:
            resolved: list[Chunk] = []
            for chunk_id in chunk_ids:
                chunk = self._chunks.get(chunk_id)
                if chunk is not None:
                    resolved.append(chunk)
            return resolved

        resolved = []
        with self._jsonl_path.open("rb") as f:
            for chunk_id in chunk_ids:
                offset = self._offsets.get(chunk_id)
                if offset is None:
                    continue
                f.seek(offset)
                raw = f.readline()
                if not raw:
                    continue
                chunk = Chunk.model_validate_json(raw.decode("utf-8"))
                resolved.append(chunk)
        return resolved

    def to_dict(self) -> dict[str, Chunk]:
        """Return the full in-memory chunk mapping (materializes the corpus)."""
        if self._chunks is None:
            self._load_all()
        return self._chunks

    def __repr__(self) -> str:
        return (
            f"JsonlChunkResolver(path={self._jsonl_path.name!r}, "
            f"count={self.count}, lazy={self._lazy})"
        )


__all__ = [
    "CHUNKS_FILENAME",
    "CHUNKS_INDEX_FILENAME",
    "JsonlChunkResolver",
    "JsonlChunkStore",
]
