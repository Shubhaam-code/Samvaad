"""Chunking package for text segmentation and retrieval preparation.

Phase 3.1: Chunk schema and base architecture.
Phase 3.2: PassageChunker (passage-preserving).
Phase 3.3: SentenceChunker (sentence-aware with overlap).
Phase 3.4: TokenChunker + tokenizer abstraction.
Phase 3.5: AdaptiveChunker (rule-based strategy selection).
Phase 3.6: ChunkingEngine (multi-strategy registry/factory).
Phase 3.7: Benchmark (offline synthetic chunk quality metrics).
"""

from .adaptive_chunker import AdaptiveChunker
from .base import BaseChunker, ChunkerProtocol
from .benchmark import BenchmarkReport, StrategyBenchmarkResult, run_benchmark
from .engine import ChunkingEngine
from .models import Chunk, ChunkingStrategy
from .passage_chunker import PassageChunker
from .sentence_chunker import SentenceChunker
from .token_chunker import TokenChunker
from .tokenizer import (
    HuggingFaceTokenizerAdapter,
    SimpleWhitespaceTokenizer,
    TokenizerProtocol,
    create_default_tokenizer,
    create_fallback_tokenizer,
    create_huggingface_tokenizer,
    create_test_tokenizer,
)

__all__ = [
    # Models
    "Chunk",
    "ChunkingStrategy",
    # Base interface
    "BaseChunker",
    "ChunkerProtocol",
    # Concrete implementations (Phase 3.2–3.5)
    "PassageChunker",
    "SentenceChunker",
    "TokenChunker",
    "AdaptiveChunker",
    # Engine (Phase 3.6)
    "ChunkingEngine",
    # Tokenizer abstraction (Phase 3.4)
    "TokenizerProtocol",
    "HuggingFaceTokenizerAdapter",
    "SimpleWhitespaceTokenizer",
    "create_default_tokenizer",
    "create_huggingface_tokenizer",
    "create_test_tokenizer",
    "create_fallback_tokenizer",
    # Benchmark (Phase 3.7)
    "run_benchmark",
    "BenchmarkReport",
    "StrategyBenchmarkResult",
]
