"""Retrieval orchestration layer.

Connects the existing real components into a single retrieval pipeline:

    query
      -> InputGuardrail (pre-retrieval safety check)
      -> embed query (real embedder)
      -> vector store search (real vector store)
      -> resolve chunk ids to Chunk evidence (resolver)
      -> structured RetrievalResult

Guarantees:

- Rejected input short-circuits BEFORE embedder.encode(), vector
  store search, and resolver resolution: a query rejected by the input
  guardrail never touches the embedding model or the vector store.
- Search-result ordering and scores are preserved end-to-end.
- Every returned hit carries its actual Chunk (with chunk_text) so the
  result can be passed directly to GroundingVerifier post-generation.
- Unresolved hit ids are preserved in missing_chunk_ids (never silently
  dropped or replaced with placeholders).
- Real per-stage latencies are recorded (guardrail, embedding, search,
  resolution) for downstream analytics.

All dependencies are injected: embedder, vector store, resolver, and
guardrail pipeline. Retrieval itself is real - no fakes, no stubs.

Phase 5.2: Retrieval orchestration only (no endpoints, no generation).
"""

from __future__ import annotations

import time
from typing import Optional

from app.guardrails.models import GuardrailVerdict
from app.guardrails.pipeline import GuardrailPipeline
from app.retrieval.models import RetrievalResult, RetrievedChunk
from app.retrieval.resolver import ChunkResolverProtocol
from app.vectorstore.base import validate_top_k


class RetrievalError(Exception):
    """Custom exception raised for retrieval orchestration failures.

    Wraps underlying provider failures (embedder, vector store, resolver)
    so callers never depend on provider-specific exception types.
    """
    pass


def validate_query(query: str) -> str:
    """Validate a single retrieval query.

    Rules:
    - Must be a string
    - Must not be empty
    - Must not be whitespace-only

    Args:
        query: Query text to validate

    Returns:
        The validated query (unchanged)

    Raises:
        ValueError: If query is not a string, empty, or whitespace-only
    """
    if not isinstance(query, str):
        raise ValueError(f"Query must be a string, got {type(query).__name__}")
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace-only")
    return query


class RetrievalOrchestrator:
    """Orchestrates guardrail -> embedding -> search -> resolution.

    Args:
        embedder: Any embedder implementing encode(text) -> list[float]
        vector_store: Any vector store implementing
            search(query_vector, top_k) -> list[VectorSearchResult]
        resolver: Any resolver implementing resolve(chunk_ids) -> list[Chunk]
        guardrail_pipeline: Optional GuardrailPipeline (creates a real
            default GuardrailPipeline if None)
        top_k: Default number of nearest neighbors to retrieve (>= 1)

    Raises:
        ValueError: If embedder, vector_store, or resolver is missing or
                    lacks the required interface, or top_k is invalid
    """

    def __init__(
        self,
        embedder: object,
        vector_store: object,
        resolver: ChunkResolverProtocol,
        guardrail_pipeline: Optional[GuardrailPipeline] = None,
        top_k: int = 5,
    ) -> None:
        if embedder is None or not callable(getattr(embedder, "encode", None)):
            raise ValueError(
                "embedder must implement encode(text) -> list[float]"
            )
        if vector_store is None or not callable(getattr(vector_store, "search", None)):
            raise ValueError(
                "vector_store must implement search(query_vector, top_k) -> list[VectorSearchResult]"
            )
        if resolver is None or not callable(getattr(resolver, "resolve", None)):
            raise ValueError(
                "resolver must implement resolve(chunk_ids) -> list[Chunk]"
            )
        validate_top_k(top_k)

        self._embedder = embedder
        self._vector_store = vector_store
        self._resolver = resolver
        self._guardrail_pipeline = guardrail_pipeline or GuardrailPipeline()
        self._top_k = top_k

    @property
    def embedder(self) -> object:
        """The configured embedder instance."""
        return self._embedder

    @property
    def vector_store(self) -> object:
        """The configured vector store instance."""
        return self._vector_store

    @property
    def resolver(self) -> ChunkResolverProtocol:
        """The configured chunk resolver instance."""
        return self._resolver

    @property
    def guardrail_pipeline(self) -> GuardrailPipeline:
        """The configured guardrail pipeline instance."""
        return self._guardrail_pipeline

    @property
    def top_k(self) -> int:
        """Default number of nearest neighbors to retrieve."""
        return self._top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Run the full retrieval pipeline for a single query.

        Stages (all real):
        1. Validate the query (ValueError on invalid input).
        2. Run the input guardrail. If the query is rejected
           (OFF_TOPIC_REJECTED), return immediately: embedding, vector
           search, and resolution are NEVER called.
        3. Embed the query with the configured embedder.
        4. Search the vector store for top_k nearest neighbors.
        5. Resolve hit chunk ids to actual Chunk evidence.
        6. Return a RetrievalResult preserving search order/scores,
           carrying missing_chunk_ids, and recording per-stage latencies.

        Args:
            query: User query to retrieve evidence for
            top_k: Optional number of neighbors to retrieve
                (defaults to the orchestrator's configured top_k)

        Returns:
            RetrievalResult with guardrail verdict, resolved evidence,
            unresolved ids, and real per-stage latencies

        Raises:
            ValueError: If query is invalid or top_k is invalid
            RetrievalError: If embedding, search, or resolution fails
        """
        validate_query(query)
        k = self._top_k if top_k is None else top_k
        validate_top_k(k)

        latencies: dict[str, float] = {}

        # Stage 1: pre-retrieval input guardrail
        start = time.perf_counter()
        try:
            guardrail_result = self._guardrail_pipeline.check_input(query)
        except Exception as exc:
            raise RetrievalError(f"Input guardrail failed for query: {exc}") from exc
        latencies["guardrail_ms"] = _elapsed_ms(start)

        if guardrail_result.verdict == GuardrailVerdict.OFF_TOPIC_REJECTED:
            # Short-circuit: embedding, search, and resolution are NEVER called.
            return RetrievalResult(
                query=query,
                allowed=False,
                guardrail=guardrail_result,
                latencies_ms=latencies,
            )

        # Stage 2: embed the query
        start = time.perf_counter()
        try:
            query_vector = self._embedder.encode(query)
        except Exception as exc:
            raise RetrievalError(f"Query embedding failed: {exc}") from exc
        latencies["embedding_ms"] = _elapsed_ms(start)

        # Stage 3: vector store search
        start = time.perf_counter()
        try:
            search_results = self._vector_store.search(query_vector, k)
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc
        latencies["search_ms"] = _elapsed_ms(start)

        hit_ids = [result.chunk_id for result in search_results]

        # Stage 4: resolve chunk ids to actual Chunk evidence
        start = time.perf_counter()
        try:
            resolved_chunks = self._resolver.resolve(hit_ids)
        except Exception as exc:
            raise RetrievalError(f"Chunk resolution failed: {exc}") from exc
        latencies["resolution_ms"] = _elapsed_ms(start)

        chunk_by_id = {chunk.chunk_id: chunk for chunk in resolved_chunks}

        retrieved: list[RetrievedChunk] = []
        missing: list[str] = []
        for result in search_results:
            chunk = chunk_by_id.get(result.chunk_id)
            if chunk is None:
                missing.append(result.chunk_id)
            else:
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=result.chunk_id,
                        score=result.score,
                        position=result.position,
                        chunk=chunk,
                    )
                )

        return RetrievalResult(
            query=query,
            allowed=True,
            guardrail=guardrail_result,
            retrieved_chunks=retrieved,
            missing_chunk_ids=missing,
            latencies_ms=latencies,
        )

    def __repr__(self) -> str:
        return (
            f"RetrievalOrchestrator(embedder={type(self._embedder).__name__}, "
            f"vector_store={type(self._vector_store).__name__}, "
            f"resolver={type(self._resolver).__name__}, "
            f"top_k={self._top_k})"
        )


def _elapsed_ms(start: float) -> float:
    """Seconds elapsed since start, converted to milliseconds."""
    return (time.perf_counter() - start) * 1000.0


__all__ = [
    "RetrievalError",
    "RetrievalOrchestrator",
    "validate_query",
]
