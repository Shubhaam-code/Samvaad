"""Embedding package for text-to-vector conversion.

Phase 4.1: Provider-agnostic embedding interface and configuration only.

- types: predictable vector type aliases (list[float], list[list[float]])
- base:  BaseEmbedder ABC, EmbedderProtocol, shared validation rules
- config: EmbeddingConfig / EmbeddingProvider
- fake:  FakeEmbedder - deterministic, offline, hash-based (tests only)

Phase 4.2: Production multilingual embedding integration.

- huggingface: HuggingFaceEmbedder adapter for the selected production
  model (intfloat/multilingual-e5-small) using transformers + torch.

Phase 4.3: Memory-safe batch embedding pipeline.

- pipeline: EmbeddingPipeline - batches Chunk objects through any
  embedder (FakeEmbedder or HuggingFaceEmbedder) with ordering,
  validation, error reporting, and batch-level progress logging.
  No vector database / index / persistence is created here (Phase 4.4).

The production model is never downloaded automatically; loading is
restricted to the local cache unless explicitly allowed (see
scripts/test_production_embedding.py --allow-download).
"""

from .base import (
    BaseEmbedder,
    EmbedderProtocol,
    validate_batch,
    validate_batch_size,
    validate_embeddings,
    validate_text,
)
from .config import EmbeddingConfig, EmbeddingProvider
from .fake import FakeEmbedder, create_fake_embedder
from .huggingface import (
    DEFAULT_MODEL_NAME,
    HuggingFaceEmbedder,
    create_huggingface_embedder,
    is_model_cached,
)
from .pipeline import (
    DEFAULT_BATCH_SIZE,
    EmbeddingFailure,
    EmbeddingPipeline,
    EmbeddingPipelineError,
    EmbeddingResult,
)
from .types import EmbeddingBatch, EmbeddingVector

__all__ = [
    # Type aliases
    "EmbeddingVector",
    "EmbeddingBatch",
    # Base interface
    "BaseEmbedder",
    "EmbedderProtocol",
    # Validation rules
    "validate_text",
    "validate_batch",
    "validate_batch_size",
    "validate_embeddings",
    # Configuration
    "EmbeddingConfig",
    "EmbeddingProvider",
    # Fake embedder (tests/offline dev)
    "FakeEmbedder",
    "create_fake_embedder",
    # Production embedder (Phase 4.2)
    "DEFAULT_MODEL_NAME",
    "HuggingFaceEmbedder",
    "create_huggingface_embedder",
    "is_model_cached",
    # Batch pipeline (Phase 4.3)
    "EmbeddingPipeline",
    "EmbeddingPipelineError",
    "EmbeddingResult",
    "EmbeddingFailure",
    "DEFAULT_BATCH_SIZE",
]