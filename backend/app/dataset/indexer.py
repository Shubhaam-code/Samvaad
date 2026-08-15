"""Full dataset indexing pipeline engine for MSMARCO-XI.

Phase 5.1 (Issue #1): Connects dataset ingestion, passage flattening,
multi-strategy chunking, batch embedding, and FAISS vector store persistence.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.chunking.engine import ChunkingEngine
from app.chunking.models import Chunk, ChunkingStrategy
from app.chunking.tokenizer import create_default_tokenizer
from app.dataset.deduplicator import IncrementalDeduplicator
from app.dataset.models import CanonicalPassage
from app.dataset.passage_flattener import flatten_msmarco_record
from app.embedding.base import BaseEmbedder
from app.embedding.config import EmbeddingConfig, EmbeddingProvider
from app.embedding.fake import FakeEmbedder
from app.embedding.huggingface import (
    DEFAULT_MODEL_NAME,
    HuggingFaceEmbedder,
    create_huggingface_embedder,
    is_model_cached,
)
from app.embedding.pipeline import EmbeddingPipeline, EmbeddingResult
from app.vectorstore.base import BaseVectorStore, VectorRecord, VectorStoreError
from app.vectorstore.faiss_store import HAS_FAISS, FaissVectorStore
from app.vectorstore.lifecycle import validate_index
from app.vectorstore.numpy_store import NumpyVectorStore

logger = logging.getLogger(__name__)

# Map common language codes to MSMARCO-XI parquet filenames
LANGUAGE_TO_FILENAME = {
    "as": ("asmtrain.parquet", "asmval.parquet"),
    "bn": ("bentrain.parquet", "benval.parquet"),
    "gu": ("gujtrain.parquet", "gujval.parquet"),
    "hi": ("hintrain.parquet", "hinval.parquet"),
    "kn": ("kantrain.parquet", "kanval.parquet"),
    "ml": ("maltrain.parquet", "malval.parquet"),
    "mr": ("martrain.parquet", "marval.parquet"),
    "ne": ("neptrain.parquet", "nepval.parquet"),
    "or": ("oritrain.parquet", "orval.parquet"),
    "pa": ("pantrain.parquet", "panval.parquet"),
    "sa": ("santrain.parquet", "sanval.parquet"),
    "ta": ("tamtrain.parquet", "tamval.parquet"),
    "te": ("teltrain.parquet", "telval.parquet"),
    "ur": ("urdtrain.parquet", "urdval.parquet"),
}


class DatasetIndexerConfig(BaseModel):
    """Configuration options for the full dataset indexing pipeline."""

    lang: str = Field("hi", description="Target Indic language code (e.g. 'hi', 'bn', 'ta', 'mr')")
    split: str = Field("validation", description="Dataset split ('validation' or 'train')")
    max_samples: Optional[int] = Field(None, ge=1, description="Maximum raw query records to index (None = all)")
    batch_size: int = Field(64, ge=1, le=512, description="Batch size for embedding and vector store writes")
    chunk_strategy: ChunkingStrategy = Field(
        ChunkingStrategy.ADAPTIVE,
        description="Chunking strategy to apply (ADAPTIVE, SENTENCE, PASSAGE, TOKEN)",
    )
    store_type: str = Field("faiss", description="Vector store backend ('faiss' or 'numpy')")
    output_dir: Path = Field(Path("data/processed"), description="Directory to persist vector index and metadata")
    device: str = Field("auto", description="Compute device for embedding model ('auto', 'cpu', 'cuda')")
    dry_run: bool = Field(False, description="If True, uses deterministic FakeEmbedder for fast offline testing")
    overwrite: bool = Field(True, description="Overwrite existing index in output_dir")
    local_parquet_path: Optional[Path] = Field(
        None, description="Optional path to local Parquet file instead of remote Hugging Face streaming"
    )


@dataclass
class IndexingStatistics:
    """Execution telemetry and metrics produced during an indexing run."""

    raw_records_read: int = 0
    canonical_passages_created: int = 0
    duplicates_skipped: int = 0
    chunks_created: int = 0
    vectors_indexed: int = 0
    duration_seconds: float = 0.0
    indexing_rate_chunks_per_sec: float = 0.0
    output_directory: str = ""
    store_type: str = ""
    embedding_model: str = ""


class DatasetIndexer:
    """End-to-end dataset ingestion and vector store indexing orchestrator.

    Integrates:
    1. Streaming ingestion from MSMARCO-XI Parquet or Hugging Face dataset
    2. Record flattening and normalization into CanonicalPassage
    3. Incremental deduplication
    4. Multi-strategy chunking (Passage, Sentence, Token, Adaptive)
    5. Batch embedding with HuggingFaceEmbedder (intfloat/multilingual-e5-small)
    6. VectorStore population (FaissVectorStore / NumpyVectorStore)
    7. Atomic persistence and index verification
    """

    def __init__(self, config: DatasetIndexerConfig):
        self.config = config
        self.stats = IndexingStatistics(
            output_directory=str(config.output_dir),
            store_type=config.store_type,
        )

        # Initialize Tokenizer and Chunking Engine
        self.tokenizer = create_default_tokenizer()
        self.chunking_engine = ChunkingEngine(tokenizer=self.tokenizer)

        # Initialize Deduplicator
        self.deduplicator = IncrementalDeduplicator(keep_relevance_priority=True)

        # Initialize Embedder
        self.embedder, self.dimension = self._initialize_embedder()
        self.stats.embedding_model = getattr(self.embedder, "model_name", "fake-embedder")

        # Initialize Vector Store
        self.vector_store = self._initialize_vector_store()

    def _initialize_embedder(self) -> Tuple[BaseEmbedder, int]:
        """Initialize the configured embedding model."""
        if self.config.dry_run:
            logger.info("Dry-run mode enabled: Initializing FakeEmbedder (dim=384)")
            dim = 384
            embedder = FakeEmbedder(dimension=dim, batch_size=self.config.batch_size)
            return embedder, dim

        logger.info(
            f"Initializing production HuggingFaceEmbedder ({DEFAULT_MODEL_NAME}) on device '{self.config.device}'"
        )
        embedder = create_huggingface_embedder(
            model_name=DEFAULT_MODEL_NAME,
            device=self.config.device,
            batch_size=self.config.batch_size,
            local_files_only=False,
        )
        return embedder, embedder.dimension

    def _initialize_vector_store(self) -> BaseVectorStore:
        """Initialize the target vector store backend."""
        stype = self.config.store_type.strip().lower()
        if stype == "faiss":
            if not HAS_FAISS:
                logger.warning("FAISS not installed, falling back to NumpyVectorStore")
                return NumpyVectorStore(
                    dimension=self.dimension,
                    embedding_model_name=self.stats.embedding_model,
                )
            return FaissVectorStore(
                dimension=self.dimension,
                embedding_model_name=self.stats.embedding_model,
            )
        elif stype == "numpy":
            return NumpyVectorStore(
                dimension=self.dimension,
                embedding_model_name=self.stats.embedding_model,
            )
        else:
            raise ValueError(f"Unsupported store type: {self.config.store_type}")

    def _iter_raw_records(self) -> Iterator[dict[str, Any]]:
        """Iterate raw records from local Parquet file or cached Hugging Face dataset."""
        import pyarrow.parquet as pq

        # Path 1: User-specified local Parquet file
        if self.config.local_parquet_path and self.config.local_parquet_path.exists():
            logger.info(f"Reading from local Parquet: {self.config.local_parquet_path}")
            parquet_file = pq.ParquetFile(str(self.config.local_parquet_path))
            for batch in parquet_file.iter_batches(batch_size=self.config.batch_size):
                for record in batch.to_pylist():
                    yield record
            return

        # Path 2: Download or load from HuggingFace cache via hf_hub_download
        lang_files = LANGUAGE_TO_FILENAME.get(self.config.lang.lower())
        if not lang_files:
            raise ValueError(f"Unsupported language code for MSMARCO-XI: {self.config.lang}")

        split_file = lang_files[0] if self.config.split.lower() == "train" else lang_files[1]
        relative_path = f"{self.config.split}/{split_file}"

        logger.info(f"Fetching MSMARCO-XI Parquet ({relative_path}) via HuggingFace Hub...")
        try:
            from huggingface_hub import hf_hub_download

            local_file_path = hf_hub_download(
                repo_id="ai4bharat/MSMARCO-XI",
                filename=relative_path,
                repo_type="dataset",
            )
            logger.info(f"Reading Parquet stream from: {local_file_path}")
            parquet_file = pq.ParquetFile(local_file_path)
            for batch in parquet_file.iter_batches(batch_size=self.config.batch_size):
                for record in batch.to_pylist():
                    yield record
            return
        except Exception as e:
            logger.error(f"Failed to fetch/read Parquet from HuggingFace Hub: {e}")
            raise

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> IndexingStatistics:
        """Execute the full indexing pipeline.

        Args:
            progress_callback: Optional callback(indexed_vectors, total_chunks, elapsed_sec)

        Returns:
            IndexingStatistics detailing performance and record counts.
        """
        start_time = time.perf_counter()
        logger.info(
            f"Starting Dataset Indexer | Lang: {self.config.lang} | Split: {self.config.split} "
            f"| Strategy: {self.config.chunk_strategy.value} | BatchSize: {self.config.batch_size}"
        )

        chunk_buffer: List[Chunk] = []
        passage_buffer: List[CanonicalPassage] = []

        def flush_chunk_batch(chunks_to_embed: List[Chunk]):
            if not chunks_to_embed:
                return

            # Extract texts for embedding
            texts = [c.chunk_text for c in chunks_to_embed]

            # Generate embeddings
            vectors = self.embedder.encode_batch(texts)

            # Build VectorRecords preserving complete metadata traceability
            records: List[VectorRecord] = []
            for chunk in chunks_to_embed:
                extra = {
                    "chunk_text": chunk.chunk_text,
                    "strategy": chunk.strategy.value,
                    "token_count": chunk.token_count,
                    "character_count": chunk.character_count,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "query": chunk.query,
                    "eng_query": chunk.eng_query,
                    "query_type": chunk.query_type,
                    "answer": chunk.answer,
                    "eng_answer": chunk.eng_answer,
                }

                record = VectorRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    query_id=chunk.query_id,
                    passage_index=chunk.passage_index,
                    target_lang=chunk.target_lang,
                    source_lang=chunk.source_lang,
                    is_selected=chunk.is_selected,
                    extra_metadata=extra,
                )
                records.append(record)

            # Insert into Vector Store
            self.vector_store.add(vectors=vectors, records=records)
            self.stats.vectors_indexed += len(vectors)

            if progress_callback:
                elapsed = time.perf_counter() - start_time
                progress_callback(self.stats.vectors_indexed, self.stats.chunks_created, elapsed)

        # Main Streaming Loop
        for raw_record in self._iter_raw_records():
            self.stats.raw_records_read += 1

            # 1. Flatten nested passages
            try:
                passages = flatten_msmarco_record(raw_record, normalize=True)
            except Exception as ex:
                logger.warning(f"Error flattening record #{self.stats.raw_records_read}: {ex}")
                continue

            # 2. Deduplicate passages
            unique_passages = self.deduplicator.process_batch(passages)
            self.stats.duplicates_skipped += len(passages) - len(unique_passages)

            for passage in unique_passages:
                self.stats.canonical_passages_created += 1

                # 3. Multi-strategy chunking
                chunks = self.chunking_engine.chunk(passage, strategy=self.config.chunk_strategy)
                for chunk in chunks:
                    self.stats.chunks_created += 1
                    chunk_buffer.append(chunk)

                    # Flush when batch size is reached
                    if len(chunk_buffer) >= self.config.batch_size:
                        flush_chunk_batch(chunk_buffer)
                        chunk_buffer = []

            # Check max samples limit
            if self.config.max_samples and self.stats.raw_records_read >= self.config.max_samples:
                logger.info(f"Reached max samples limit: {self.config.max_samples}")
                break

        # Flush any remaining chunks
        if chunk_buffer:
            flush_chunk_batch(chunk_buffer)
            chunk_buffer = []

        # 4. Save Vector Store atomically
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(output_dir, overwrite=self.config.overwrite)

        # 5. Validate Index Integrity
        validation_report = validate_index(
            output_dir,
            expected_dimension=self.dimension,
            expected_model_name=self.stats.embedding_model,
        )
        logger.info(f"Vector index saved & validated successfully: {validation_report}")

        # Compute final statistics
        end_time = time.perf_counter()
        self.stats.duration_seconds = round(end_time - start_time, 2)
        if self.stats.duration_seconds > 0:
            self.stats.indexing_rate_chunks_per_sec = round(
                self.stats.chunks_created / self.stats.duration_seconds, 2
            )

        logger.info(
            f"Indexing Complete! Read {self.stats.raw_records_read} queries, "
            f"created {self.stats.chunks_created} chunks, "
            f"indexed {self.stats.vectors_indexed} vectors in {self.stats.duration_seconds}s "
            f"({self.stats.indexing_rate_chunks_per_sec} chunks/sec)"
        )

        return self.stats
