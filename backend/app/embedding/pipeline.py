"""Memory-safe batch embedding pipeline.

Phase 4.3: Orchestrates embedding of already-created Chunk objects in
configurable batches, without ever materializing the whole dataset in
memory at once.

Design goals:
- Provider-agnostic: works with any embedder implementing the Phase 4.1
  interface (encode / encode_batch / dimension), e.g. FakeEmbedder or
  HuggingFaceEmbedder, with no pipeline changes.
- Memory-safe: chunks are embedded batch-by-batch via the embedder's
  encode_batch(); only one batch of vectors exists at a time when using
  the streaming embed_batches() iterator. No per-chunk encode() calls.
- Ordering: results map 1:1 to input chunks in exact input order.
- Errors: fail-fast by default; optionally report via an on_error
  callback while recording failures. Failed batches never silently
  produce corrupted output, and a vector-count mismatch raises.
- Logging: batch-level progress only (never per-chunk, never vectors).

No vector database, index, persistence or retrieval is created here
(Phase 4.4).
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Iterator, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .types import EmbeddingVector

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


class EmbeddingResult(BaseModel):
    """Embedding record for a single chunk.

    Attributes:
        chunk_id: Chunk identifier the embedding was produced for
        embedding: Dense vector as plain Python floats
        dimension: Vector length (== len(embedding))
        model_name: Embedding model identifier, if the embedder exposes one
        provider: Embedder provider name (class name unless overridden)
    """

    model_config = ConfigDict(protected_namespaces=())

    chunk_id: str = Field(..., description="Chunk identifier")
    embedding: EmbeddingVector = Field(..., description="Dense vector of floats")
    dimension: int = Field(..., ge=1, description="Vector length")
    model_name: Optional[str] = Field(None, description="Embedding model identifier")
    provider: Optional[str] = Field(None, description="Embedder provider name")


class EmbeddingFailure(BaseModel):
    """Report of a failed embedding batch.

    A batch fails as a whole (provider error, dimension mismatch,
    non-finite values, wrong vector count). Failed chunks are never
    silently skipped: the failure is recorded here and, optionally,
    forwarded to the configured on_error callback.

    Attributes:
        batch_index: 1-based index of the failed batch
        chunk_ids: Chunk identifiers belonging to the failed batch
        error: Error message describing the failure
    """

    batch_index: int = Field(..., ge=1, description="1-based batch index")
    chunk_ids: List[str] = Field(..., description="Chunk ids in the failed batch")
    error: str = Field(..., description="Failure description")


class EmbeddingPipelineError(Exception):
    """Raised when a batch fails and fail_fast=True."""


class EmbeddingPipeline:
    """Memory-safe batch embedding pipeline.

    Args:
        embedder: Any embedder implementing encode(text) -> list[float]
                  and encode_batch(texts) -> list[list[float]], with an
                  optional ``dimension`` and ``batch_size`` attribute
        batch_size: Maximum chunks embedded per encode_batch() call
                    (must not exceed the embedder's own batch size)
        fail_fast: If True (default), raise EmbeddingPipelineError on the
                   first failed batch. If False, report failures via
                   on_error/errors and continue with remaining batches.
        on_error: Optional callback invoked with an EmbeddingFailure for
                  every failed batch (before raising when fail_fast=True).

    Raises:
        ValueError: If embedder is missing, batch_size is invalid, or
                    batch_size exceeds the embedder's batch size
    """

    def __init__(
        self,
        embedder: object,
        batch_size: int = DEFAULT_BATCH_SIZE,
        fail_fast: bool = True,
        on_error: Optional[Callable[[EmbeddingFailure], None]] = None,
    ) -> None:
        if embedder is None:
            raise ValueError("EmbeddingPipeline requires an embedder")
        if not callable(getattr(embedder, "encode_batch", None)):
            raise ValueError(
                "Embedder must implement encode_batch(texts) -> list[list[float]]"
            )
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be an integer >= 1, got {batch_size!r}")

        embedder_batch_size = getattr(embedder, "batch_size", None)
        if embedder_batch_size is not None and batch_size > embedder_batch_size:
            raise ValueError(
                f"pipeline batch_size ({batch_size}) exceeds the embedder's "
                f"batch_size ({embedder_batch_size}); configure a smaller batch"
            )

        self._embedder = embedder
        self._batch_size = batch_size
        self._fail_fast = fail_fast
        self._on_error = on_error
        self._errors: List[EmbeddingFailure] = []
        self._expected_dimension = getattr(embedder, "dimension", None)
        self._model_name = getattr(embedder, "model_name", None)
        self._provider = getattr(embedder, "provider", None) or type(embedder).__name__

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_batches(self, chunks: list[object]) -> Iterator[List[EmbeddingResult]]:
        """Embed chunks batch-by-batch, yielding one result list per batch.

        Streaming interface: only one batch of results exists in memory
        at a time. Yields lists of EmbeddingResult in exact input order.

        Args:
            chunks: Chunk objects (or duck-typed chunk-like objects with
                    ``chunk_id`` and ``chunk_text`` attributes)

        Yields:
            One list of EmbeddingResult per batch, in input order

        Raises:
            ValueError: If input is empty or contains malformed chunks
            EmbeddingPipelineError: If a batch fails and fail_fast=True
        """
        chunks = self._validate_chunks(chunks)
        self._errors = []

        total = len(chunks)
        num_batches = math.ceil(total / self._batch_size)
        logger.info(
            "Embedding %d chunks in batches of %d (%d batch(es)) with provider '%s'",
            total, self._batch_size, num_batches, self._provider,
        )

        embedded_so_far = 0
        for i in range(0, total, self._batch_size):
            batch = chunks[i:i + self._batch_size]
            batch_num = i // self._batch_size + 1
            results = self._process_batch(batch, batch_num, num_batches)
            if results is None:
                continue  # failed batch; already reported (fail_fast=False)
            embedded_so_far += len(results)
            logger.info(
                "Embedded batch %d/%d (%d/%d chunks)",
                batch_num, num_batches, embedded_so_far, total,
            )
            yield results

    def embed_batch(self, chunks: list[object]) -> List[EmbeddingResult]:
        """Embed a single batch (convenience API).

        Args:
            chunks: Chunk objects for exactly one batch
                    (must not exceed the configured batch_size)

        Returns:
            List of EmbeddingResult in input order

        Raises:
            ValueError: If input is empty, malformed, or exceeds batch_size
            EmbeddingPipelineError: If the batch fails and fail_fast=True
        """
        chunks = self._validate_chunks(chunks)
        if len(chunks) > self._batch_size:
            raise ValueError(
                f"embed_batch() accepts at most batch_size ({self._batch_size}) "
                f"chunks, got {len(chunks)}"
            )
        self._errors = []
        results = self._process_batch(chunks, 1, 1)
        return results or []

    def embed_all(self, chunks: list[object]) -> List[EmbeddingResult]:
        """Embed all chunks and return every result (convenience API).

        Note: accumulates all results in memory; prefer embed_batches()
        for large inputs.

        Args:
            chunks: Chunk objects to embed

        Returns:
            All EmbeddingResult objects in input order
        """
        results: List[EmbeddingResult] = []
        for batch in self.embed_batches(chunks):
            results.extend(batch)
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedder(self) -> object:
        """The configured embedder instance."""
        return self._embedder

    @property
    def batch_size(self) -> int:
        """Maximum chunks embedded per encode_batch() call."""
        return self._batch_size

    @property
    def fail_fast(self) -> bool:
        """Whether a failed batch raises immediately."""
        return self._fail_fast

    @property
    def errors(self) -> List[EmbeddingFailure]:
        """Failures recorded during the most recent run (empty if none)."""
        return list(self._errors)

    @property
    def model_name(self) -> Optional[str]:
        """Model identifier reported on results (embedder-provided)."""
        return self._model_name

    @property
    def provider(self) -> str:
        """Provider name reported on results."""
        return self._provider

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_chunks(self, chunks: list[object]) -> list[object]:
        """Validate the input list and every chunk (structural rules).

        These rules always raise (regardless of fail_fast): a malformed
        input can never produce meaningful output.

        Args:
            chunks: Input list of chunk-like objects

        Returns:
            The validated list (unchanged, no copies)

        Raises:
            ValueError: If input is not a list, is empty, or contains a
                        chunk missing/empty chunk_id or chunk_text
        """
        if not isinstance(chunks, list):
            raise ValueError(
                f"chunks must be a list, got {type(chunks).__name__}"
            )
        if not chunks:
            raise ValueError("chunks cannot be empty")

        for index, chunk in enumerate(chunks):
            if not hasattr(chunk, "chunk_id") or not hasattr(chunk, "chunk_text"):
                raise ValueError(
                    f"Item at index {index} is not a chunk-like object: "
                    f"expected chunk_id and chunk_text attributes, got "
                    f"{type(chunk).__name__}"
                )
            chunk_id = chunk.chunk_id
            chunk_text = chunk.chunk_text
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"Chunk at index {index} has a missing or empty chunk_id"
                )
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                raise ValueError(
                    f"Chunk '{chunk_id}' has a missing or whitespace-only chunk_text"
                )
        return chunks

    def _process_batch(
        self,
        batch: list[object],
        batch_num: int,
        num_batches: int,
    ) -> Optional[List[EmbeddingResult]]:
        """Embed one batch, applying the error-handling strategy.

        Args:
            batch: Chunks of this batch (already validated)
            batch_num: 1-based batch index (for logging/reporting)
            num_batches: Total batch count (for logging/reporting)

        Returns:
            List of EmbeddingResult, or None if the batch failed and
            fail_fast=False (failure already reported)

        Raises:
            EmbeddingPipelineError: If the batch failed and fail_fast=True
        """
        try:
            vectors = self._embedder.encode_batch(
                [chunk.chunk_text for chunk in batch]
            )
            if len(vectors) != len(batch):
                raise ValueError(
                    f"Embedder returned {len(vectors)} vectors for "
                    f"{len(batch)} inputs; refusing to produce misaligned output"
                )
            results = [
                self._build_result(chunk, vector)
                for chunk, vector in zip(batch, vectors)
            ]
        except Exception as e:
            failure = EmbeddingFailure(
                batch_index=batch_num,
                chunk_ids=[chunk.chunk_id for chunk in batch],
                error=str(e),
            )
            self._errors.append(failure)
            if self._on_error is not None:
                try:
                    self._on_error(failure)
                except Exception as callback_error:
                    logger.error(
                        "on_error callback failed for batch %d/%d: %s",
                        batch_num, num_batches, callback_error,
                    )
            if self._fail_fast:
                logger.error(
                    "Embedding failed for batch %d/%d (%d chunks): %s",
                    batch_num, num_batches, len(batch), e,
                )
                raise EmbeddingPipelineError(
                    f"Embedding batch {batch_num}/{num_batches} failed "
                    f"(chunks: {', '.join(failure.chunk_ids)}): {e}"
                ) from e
            logger.warning(
                "Embedding failed for batch %d/%d (%d chunks); batch skipped "
                "and reported via errors/on_error: %s",
                batch_num, num_batches, len(batch), e,
            )
            return None
        return results

    def _build_result(self, chunk: object, vector: EmbeddingVector) -> EmbeddingResult:
        """Validate one vector and build its EmbeddingResult.

        Args:
            chunk: Source chunk (validated earlier)
            vector: Vector produced by the embedder for this chunk

        Returns:
            EmbeddingResult with metadata attached

        Raises:
            ValueError: If the vector has a mismatched dimension, contains
                        non-finite values, or is not a flat list of numbers
        """
        if not isinstance(vector, list):
            raise ValueError(
                f"Embedder returned {type(vector).__name__} for chunk "
                f"'{chunk.chunk_id}'; expected list[float]"
            )
        dimension = len(vector)
        if self._expected_dimension is not None and dimension != self._expected_dimension:
            raise ValueError(
                f"Chunk '{chunk.chunk_id}' embedding dimension {dimension} "
                f"does not match embedder dimension {self._expected_dimension}"
            )
        for index, value in enumerate(vector):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"Chunk '{chunk.chunk_id}' embedding value at index {index} "
                    f"is not a number: {type(value).__name__}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Chunk '{chunk.chunk_id}' embedding value at index {index} "
                    f"is not finite: {value!r}"
                )
        return EmbeddingResult(
            chunk_id=chunk.chunk_id,
            embedding=vector,
            dimension=dimension,
            model_name=self._model_name,
            provider=self._provider,
        )

    def __repr__(self) -> str:
        return (
            f"EmbeddingPipeline(provider={self._provider!r}, "
            f"batch_size={self._batch_size}, fail_fast={self._fail_fast})"
        )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EmbeddingFailure",
    "EmbeddingPipeline",
    "EmbeddingPipelineError",
    "EmbeddingResult",
]