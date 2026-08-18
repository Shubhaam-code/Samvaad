"""Retrieval orchestration package.

Phase 5.2: Provider-agnostic retrieval orchestration layer that connects
the existing real components (guardrails, embedding, vector store) with
a chunk resolver into a single structured retrieval pipeline.

- models: RetrievedChunk / RetrievalResult (structured outcomes)
- resolver: ChunkResolver ABC, ChunkResolverProtocol, DictChunkResolver
- orchestrator: RetrievalOrchestrator - guardrail -> embed -> search ->
  resolve, with rejection short-circuiting, real stage latencies, and
  missing-id preservation

No FastAPI endpoints, no answer generation, no STT/TTS. Retrieval is
real: only the existing real embedder and vector store are used.
"""

from .models import RetrievalResult, RetrievedChunk
from .orchestrator import RetrievalError, RetrievalOrchestrator, validate_query
from .reranker import (
    BaseReranker,
    FastReranker,
    PassThroughReranker,
    RerankerProtocol,
)
from .resolver import (
    ChunkResolver,
    ChunkResolverProtocol,
    DictChunkResolver,
    validate_chunk_ids,
)

__all__ = [
    # Data models
    "RetrievedChunk",
    "RetrievalResult",
    # Reranker
    "BaseReranker",
    "FastReranker",
    "PassThroughReranker",
    "RerankerProtocol",
    # Resolver
    "ChunkResolver",
    "ChunkResolverProtocol",
    "DictChunkResolver",
    "validate_chunk_ids",
    # Orchestrator
    "RetrievalError",
    "RetrievalOrchestrator",
    "validate_query",
]
