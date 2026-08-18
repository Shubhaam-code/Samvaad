"""Data models for the retrieval orchestration layer.

Defines the canonical result structures produced by the retrieval
orchestrator:

- ``RetrievedChunk``: a single search hit resolved back to its actual
  ``Chunk`` evidence object (the canonical evidence unit consumed by
  GroundingVerifier).
- ``RetrievalResult``: the full outcome of one query, including the
  guardrail decision, retrieved evidence, unresolved chunk ids, and
  real per-stage latencies.

Phase 5.2: Retrieval orchestration data models only.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.chunking.models import Chunk
from app.guardrails.models import GuardrailResult


class RetrievedChunk(BaseModel):
    """A single search hit resolved back to its actual Chunk evidence.

    Attributes:
        chunk_id: Chunk identifier of the matched vector
        score: Similarity score from the vector search (higher = more similar)
        position: 0-based index position of the vector in the store
        chunk: The actual Chunk object carrying chunk_text evidence
    """

    chunk_id: str = Field(..., min_length=1, description="Chunk identifier of the matched vector")
    score: float = Field(..., description="Similarity score from the vector search")
    position: int = Field(..., ge=0, description="0-based index position of the vector in the store")
    chunk: Chunk = Field(..., description="Actual Chunk object carrying chunk_text evidence")


class RetrievalResult(BaseModel):
    """Outcome of a single retrieval query.

    Attributes:
        query: The normalized user query that was processed
        allowed: Whether the query passed the input guardrail and proceeded
            to retrieval (False for rejected queries)
        guardrail: Full GuardrailResult from the pre-retrieval input check
        retrieved_chunks: Resolved evidence chunks in search-result order
        missing_chunk_ids: Search hit chunk ids that could not be resolved
            to Chunk evidence (unresolved by the resolver)
        latencies_ms: Real per-stage latencies in milliseconds
            (guardrail_ms, embedding_ms, search_ms, resolution_ms)
    """

    query: str = Field(..., min_length=1, description="The user query that was processed")
    allowed: bool = Field(..., description="Whether the query passed the input guardrail")
    guardrail: GuardrailResult = Field(..., description="Full pre-retrieval guardrail result")
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Resolved evidence chunks in search-result order",
    )
    missing_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Search hit chunk ids that could not be resolved to Chunk evidence",
    )
    latencies_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Real per-stage latencies in milliseconds",
    )


__all__ = [
    "RetrievedChunk",
    "RetrievalResult",
]
