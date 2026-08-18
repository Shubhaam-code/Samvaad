"""Executable production RAG index builder.

Builds the full persisted index pipeline:

    dataset source
      -> preprocessing (flatten, normalize, deduplicate, validate)
      -> canonical passages (parquet)
      -> chunking (ChunkingEngine)
      -> production embeddings (HuggingFaceEmbedder + EmbeddingPipeline)
      -> persistent vector store (FaissVectorStore preferred / NumpyVectorStore)
      -> persisted chunk corpus (chunks.jsonl + offset index)
      -> index manifest (manifest.json)
      -> post-build validation (reload, counts, dimension, retrieval smoke test)
      -> atomic swap into the final index directory

Guarantees:

- The production embedder is ALWAYS the real HuggingFaceEmbedder when
  none is injected; FakeEmbedder is never used by this script.
- The embedding model must be present in the local HuggingFace cache
  (local_files_only=True) unless --allow-download is passed explicitly.
- A failed build never damages an existing valid index: everything is
  written to a temporary directory next to the target and only swapped
  into place after full post-build validation passes.
- Only batch-level progress is printed - never full document contents.

Usage (from the backend/ directory):

    python -m scripts.build_index --help
    python -m scripts.build_index --source-path ../data/raw/hintrain.parquet
    python -m scripts.build_index --hf-split train --limit 100000

Configuration precedence: CLI flags > environment variables
(RAG_INDEX_DIR, RAG_VECTOR_STORE, RAG_EMBEDDING_MODEL, ...) > defaults.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.chunking import (
    ChunkingEngine,
    ChunkingStrategy,
    create_default_tokenizer,
    create_huggingface_tokenizer,
)
from app.dataset.deduplicator import IncrementalDeduplicator
from app.dataset.loader import DATASET_NAME, load_split
from app.dataset.models import CanonicalPassage
from app.dataset.passage_flattener import flatten_msmarco_batch
from app.dataset.parquet_reader import ParquetBatchReader
from app.dataset.preprocessing_pipeline import preprocess_dataset
from app.dataset.processed_writer import ProcessedDatasetWriter
from app.dataset.validator import validate_batch
from app.embedding import (
    DEFAULT_MODEL_NAME,
    EmbeddingPipeline,
    HuggingFaceEmbedder,
    is_model_cached,
)
from app.guardrails.grounding_verifier import GroundingVerifier
from app.indexing.chunk_store import (
    CHUNKS_FILENAME,
    CHUNKS_INDEX_FILENAME,
    JsonlChunkStore,
)
from app.indexing.loader import VECTORSTORE_DIRNAME, load_index, resolve_index_dir
from app.indexing.manifest import (
    ChunkingInfo,
    CountsInfo,
    DatasetInfo,
    EmbeddingInfo,
    RagIndexManifest,
    VectorStoreInfo,
    write_manifest,
)
from app.settings import settings
from app.vectorstore import (
    VectorRecord,
    VectorStoreError,
    create_vector_store,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical default locations (repo-relative, not machine-specific).
DEFAULT_RAW_INPUT = REPO_ROOT / "data" / "raw" / "hintrain.parquet"
DEFAULT_PROCESSED_OUTPUT = REPO_ROOT / "data" / "processed" / "train_processed.parquet"


class IndexBuildError(Exception):
    """Raised when the index build cannot proceed safely."""


class IndexBuildConfig(BaseModel):
    """Configuration for a single index build.

    Exactly one dataset source must be provided: either ``source_path``
    (a local MSMARCO-XI parquet file) or ``hf_split`` (a HuggingFace
    dataset split, e.g. ``"train"`` from ``ai4bharat/MSMARCO-XI``).
    """

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    source_path: Optional[Path] = Field(
        None,
        description="Local MSMARCO-XI parquet file (raw input)",
    )
    processed_output: Optional[Path] = Field(
        None,
        description="Processed CanonicalPassage parquet (writer output)",
    )
    hf_dataset: str = Field(
        DATASET_NAME,
        description="HuggingFace dataset identifier",
    )
    hf_config: str = Field(
        "default",
        description="Dataset config name",
    )
    hf_split: Optional[str] = Field(
        None,
        description="Dataset split to build from (alternative to source_path)",
    )
    index_dir: Path = Field(
        ...,
        description="Final index directory",
    )
    vector_store_backend: str = Field(
        "faiss",
        alias="vector_store",
        description="Vector store backend ('faiss' or 'numpy')",
    )
    top_k: int = Field(
        5,
        ge=1,
        description="Number of neighbors to retrieve during validation smoke test",
    )
    embedding_model: str = Field(
        DEFAULT_MODEL_NAME,
        description="Embedding model identifier",
    )
    embedding_device: str = Field(
        "auto",
        description="Embedding device ('auto', 'cpu', 'cuda')",
    )
    embedding_batch_size: int = Field(
        32,
        ge=1,
        description="Maximum chunks per encode_batch() call",
    )
    chunking_strategy: ChunkingStrategy = Field(
        ChunkingStrategy.PASSAGE,
        description="Chunking strategy",
    )
    tokenizer_model: Optional[str] = Field(
        None,
        description="Tokenizer model for token/adaptive strategies (optional)",
    )
    batch_size: int = Field(
        500,
        ge=1,
        description="Records per preprocessing/indexing batch",
    )
    limit: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum passages to index (bounded sampling)",
    )
    overwrite: bool = Field(
        False,
        description="Allow replacing an existing valid index",
    )
    allow_download: bool = Field(
        False,
        description="Explicitly allow downloading the embedding model",
    )

    @model_validator(mode="after")
    def validate_source(self) -> IndexBuildConfig:
        if (self.source_path is None) == (self.hf_split is None):
            raise ValueError(
                "Exactly one dataset source is required: either "
                "source_path (local parquet) or hf_split (HF dataset)"
            )
        if self.vector_store_backend not in ("faiss", "numpy"):
            raise ValueError(
                f"vector_store_backend must be 'faiss' or 'numpy', "
                f"got {self.vector_store_backend!r}"
            )
        if self.source_path is not None and not self.source_path.is_file():
            raise FileNotFoundError(f"Source parquet file not found: {self.source_path}")
        return self

    @property
    def vector_store(self) -> str:
        """Alias matching the CLI/config term used by the build command."""
        return self.vector_store_backend


@dataclass
class BuildStatistics:
    """Counters collected during a build run."""

    input_records: int = 0
    passages: int = 0
    duplicates_removed: int = 0
    validation_failures: int = 0
    chunks: int = 0
    embedded: int = 0
    vectors_indexed: int = 0
    embedding_failures: int = 0
    documents: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class BuildResult:
    """Outcome of a successful index build."""

    statistics: BuildStatistics
    index_dir: Path
    manifest: RagIndexManifest
    smoke: dict[str, object] = field(default_factory=dict)


def _create_production_embedder(config: IndexBuildConfig) -> HuggingFaceEmbedder:
    """Create the production HuggingFace embedder.

    Never returns a FakeEmbedder. Requires the model to be present in
    the local HuggingFace cache unless allow_download=True.

    Args:
        config: Build configuration

    Returns:
        A HuggingFaceEmbedder instance (model loads lazily on first use)

    Raises:
        IndexBuildError: If the model is not cached and downloads are
            not explicitly allowed
    """
    if not config.allow_download and not is_model_cached(config.embedding_model):
        raise IndexBuildError(
            f"Embedding model '{config.embedding_model}' is not cached locally "
            f"and downloads are disabled. Fetch it once explicitly with "
            f"--allow-download (or the scripts/test_production_embedding.py "
            f"--allow-download helper) before building the index."
        )
    return HuggingFaceEmbedder(
        model_name=config.embedding_model,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
        local_files_only=not config.allow_download,
    )


def _prepare_processed_dataset(
    config: IndexBuildConfig,
    log: Callable[[str], None],
) -> tuple[Path, BuildStatistics]:
    """Produce the processed CanonicalPassage parquet for the build.

    Uses the existing preprocessing pipeline (loader / flattener /
    deduplicator / validator / writer). For local parquet input this is
    ``preprocess_dataset``; for HF splits the same components are
    applied incrementally over the streaming split.

    Args:
        config: Build configuration
        log: Progress printer

    Returns:
        Tuple of (processed parquet path, partial statistics)
    """
    processed_output = config.processed_output
    if processed_output is None:
        processed_output = DEFAULT_PROCESSED_OUTPUT

    stats = BuildStatistics()

    if config.source_path is not None:
        log(f"Preprocessing local parquet: {config.source_path} -> {processed_output}")
        processed_output.parent.mkdir(parents=True, exist_ok=True)
        pre_stats = preprocess_dataset(
            config.source_path,
            processed_output,
            batch_size=config.batch_size,
            overwrite=True,
        )
        stats.input_records = pre_stats.input_records
        stats.passages = pre_stats.records_written
        stats.duplicates_removed = pre_stats.duplicates_removed
        stats.validation_failures = pre_stats.validation_failures
        log(
            f"Preprocessing complete: {stats.input_records:,} records in, "
            f"{stats.passages:,} valid passages out "
            f"(-{stats.duplicates_removed:,} duplicates, "
            f"-{stats.validation_failures:,} invalid)"
        )
        if stats.passages == 0:
            raise IndexBuildError(
                "Build produced zero passages. The dataset source is empty or "
                "produced no valid passages after preprocessing. No index was "
                "created."
            )
        return processed_output, stats

    # HF dataset split path: flatten -> deduplicate -> validate -> write
    # incrementally with the exact same components as preprocess_dataset.
    log(
        f"Loading HF split: {config.hf_dataset} config={config.hf_config} "
        f"split={config.hf_split}"
    )
    processed_output.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_split(
        split=config.hf_split,  # type: ignore[arg-type]
        config_name=config.hf_config,
        streaming=True,
    )
    deduplicator = IncrementalDeduplicator(keep_relevance_priority=True)
    processed = Path(processed_output)
    if processed.exists():
        processed.unlink()
    with ProcessedDatasetWriter(processed, overwrite=True) as writer:
        buffer: list[dict] = []
        seen_records = 0
        for record in dataset:
            buffer.append(record)
            seen_records += 1
            if len(buffer) >= config.batch_size:
                _process_flat_batch(
                    buffer,
                    deduplicator,
                    writer,
                    stats,
                )
                buffer = []
            if config.limit is not None and seen_records >= config.limit:
                break
        if buffer:
            _process_flat_batch(buffer, deduplicator, writer, stats)
    log(
        f"HF split complete: {stats.input_records:,} records in, "
        f"{stats.passages:,} valid passages out "
        f"(-{stats.duplicates_removed:,} duplicates, "
        f"-{stats.validation_failures:,} invalid)"
    )
    return processed, stats


def _process_flat_batch(
    records: list[dict],
    deduplicator: IncrementalDeduplicator,
    writer: ProcessedDatasetWriter,
    stats: BuildStatistics,
) -> None:
    """Run flatten -> deduplicate -> validate -> write for one record batch."""
    stats.input_records += len(records)
    flattened = flatten_msmarco_batch(records, normalize=True)
    unique = deduplicator.process_batch(flattened)
    stats.duplicates_removed += len(flattened) - len(unique)
    if not unique:
        return
    validation = validate_batch(unique)
    stats.validation_failures += validation.invalid_count
    if validation.valid_records:
        writer.write_batch(validation.valid_records)
        stats.passages += len(validation.valid_records)


def _build_chunking_engine(config: IndexBuildConfig) -> ChunkingEngine:
    """Build the chunking engine for the configured strategy.

    Token/adaptive strategies get a tokenizer: the configured model if
    given, otherwise the offline default (local HF tokenizer if cached,
    else the deterministic SimpleWhitespaceTokenizer).
    """
    tokenizer = None
    if config.chunking_strategy in (
        ChunkingStrategy.TOKEN,
        ChunkingStrategy.ADAPTIVE,
    ):
        if config.tokenizer_model:
            tokenizer = create_huggingface_tokenizer(
                config.tokenizer_model,
                local_files_only=not config.allow_download,
            )
        else:
            tokenizer = create_default_tokenizer()
    return ChunkingEngine(tokenizer=tokenizer)


def _run_smoke_test(
    index_dir: Path,
    manifest: RagIndexManifest,
    embedder: object,
    log: Callable[[str], None],
) -> dict[str, object]:
    """Deterministic post-build retrieval smoke test.

    Loads the freshly written index, embeds the first chunk's text as a
    query, searches the persisted vector store, resolves every hit to a
    real Chunk with chunk_text, and runs GroundingVerifier over the
    evidence. All checks must pass or the build is rejected.

    Args:
        index_dir: The (temporary) index directory being validated
        manifest: Manifest written to that directory
        embedder: Embedder used for the build (query embedding)

    Returns:
        Dict describing smoke test results

    Raises:
        IndexBuildError: If any smoke check fails
    """
    store, resolver, loaded_manifest = load_index(
        index_dir,
        expected_model_name=manifest.embedding.model,
        expected_dimension=manifest.embedding.dimension,
        expected_backend=manifest.vector_store.backend,
    )
    if loaded_manifest.counts.chunks == 0:
        raise IndexBuildError("Smoke test failed: index contains no chunks")

    first_id = resolver.chunk_ids[0]
    first_chunk = resolver.resolve([first_id])[0]
    if not first_chunk.chunk_text:
        raise IndexBuildError(
            f"Smoke test failed: chunk '{first_id}' has no chunk_text"
        )

    query_vector = embedder.encode(first_chunk.chunk_text)
    top_k = int(manifest.build.get("top_k", 5))
    hits = store.search(query_vector, top_k=min(top_k, store.count))
    if not hits:
        raise IndexBuildError("Smoke test failed: search returned no results")
    if hits[0].chunk_id != first_id:
        raise IndexBuildError(
            f"Smoke test failed: expected top hit '{first_id}', "
            f"got '{hits[0].chunk_id}'"
        )

    hit_ids = [hit.chunk_id for hit in hits]
    resolved = resolver.resolve(hit_ids)
    if len(resolved) != len(hit_ids):
        missing = set(hit_ids) - {chunk.chunk_id for chunk in resolved}
        raise IndexBuildError(
            f"Smoke test failed: {len(missing)} hit chunk ids did not resolve: "
            f"{sorted(missing)[:5]}"
        )
    for chunk in resolved:
        if not chunk.chunk_text or not chunk.chunk_text.strip():
            raise IndexBuildError(
                f"Smoke test failed: chunk '{chunk.chunk_id}' has empty chunk_text"
            )

    grounding = GroundingVerifier().verify(first_chunk.chunk_text, resolved)
    smoke = {
        "hits": len(hits),
        "resolved": len(resolved),
        "top_hit": hits[0].chunk_id,
        "grounding_verdict": grounding.verdict.value,
        "grounding_score": grounding.score,
    }
    log(f"Smoke test: top_k={smoke['hits']}, resolved={smoke['resolved']}, "
        f"verdict={smoke['grounding_verdict']} (score={smoke['grounding_score']})")
    return smoke


def build_index(
    config: IndexBuildConfig,
    embedder: Optional[object] = None,
    log: Callable[[str], None] = print,
) -> BuildResult:
    """Build a complete persisted RAG index.

    Pipeline: preprocessing -> chunking -> embedding -> vector store ->
    chunk corpus -> manifest -> post-build validation -> atomic swap.

    Args:
        config: Build configuration
        embedder: Embedder to use; when None the production
            HuggingFaceEmbedder is created (never FakeEmbedder)
        log: Progress printer

    Returns:
        BuildResult with statistics, manifest, and smoke test results

    Raises:
        IndexBuildError: If the build cannot proceed safely
        FileNotFoundError / VectorStoreError / RuntimeError: On pipeline failures
    """
    start = time.perf_counter()

    if embedder is None:
        embedder = _create_production_embedder(config)

    if config.overwrite is False:
        final_dir = config.index_dir.resolve()
        if final_dir.exists():
            raise IndexBuildError(
                f"Index already exists at '{final_dir}'. Pass --overwrite to "
                f"rebuild it (the previous index is preserved until the new "
                f"build fully validates)."
            )

    log(
        f"Index build starting: backend={config.vector_store_backend}, "
        f"model={config.embedding_model}, strategy={config.chunking_strategy.value}, "
        f"index={config.index_dir}"
    )

    stats = BuildStatistics()
    processed_path, pre_stats = _prepare_processed_dataset(config, log)
    stats.input_records = pre_stats.input_records
    stats.duplicates_removed = pre_stats.duplicates_removed
    stats.validation_failures = pre_stats.validation_failures

    dimension = int(getattr(embedder, "dimension", None) or 0)
    if dimension <= 0:
        raise IndexBuildError("Embedder did not expose a valid dimension")

    # Create the temporary build directory next to the final index so the
    # final swap stays on the same filesystem (atomic rename).
    final_dir = config.index_dir.resolve()
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".tmp_index_build_{uuid.uuid4().hex[:8]}_",
            dir=str(final_dir.parent),
        )
    )

    try:
        chunk_store: Optional[JsonlChunkStore] = None
        store = create_vector_store(
            dimension=dimension,
            store_type=config.vector_store_backend,
            embedding_model_name=config.embedding_model,
        )
        chunk_store = JsonlChunkStore(tmp_dir / CHUNKS_FILENAME)
        engine = _build_chunking_engine(config)
        pipeline = EmbeddingPipeline(
            embedder=embedder,
            batch_size=config.embedding_batch_size,
            fail_fast=True,
        )

        document_ids: set[str] = set()
        reader = ParquetBatchReader(processed_path, batch_size=config.batch_size)
        for batch_index, batch_table in enumerate(reader, start=1):
            records = batch_table.to_pylist()
            passages = [CanonicalPassage(**record) for record in records]
            validation = validate_batch(passages)
            stats.validation_failures += validation.invalid_count
            passages = validation.valid_records

            if config.limit is not None and stats.passages + len(passages) > config.limit:
                passages = passages[: max(0, config.limit - stats.passages)]
            if not passages:
                continue

            chunks = engine.chunk_batch(passages, config.chunking_strategy)
            if not chunks:
                continue

            batch_vectors: list[list[float]] = []
            for results in pipeline.embed_batches(chunks):
                batch_vectors.extend(result.embedding for result in results)

            records_out = [
                VectorRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    query_id=chunk.query_id,
                    passage_index=chunk.passage_index,
                    target_lang=chunk.target_lang,
                    source_lang=chunk.source_lang,
                    is_selected=chunk.is_selected,
                )
                for chunk in chunks
            ]
            store.add(batch_vectors, records_out)
            chunk_store.append(chunks)
            document_ids.update(chunk.document_id for chunk in chunks)

            stats.passages += len(passages)
            stats.chunks += len(chunks)
            stats.embedded += len(batch_vectors)
            stats.vectors_indexed = store.count
            stats.documents = len(document_ids)
            stats.embedding_failures = len(pipeline.errors)

            log(
                f"Batch {batch_index}: +{len(passages)} passages, "
                f"+{len(chunks)} chunks, +{len(batch_vectors)} embeddings "
                f"(total {store.count:,} vectors, {len(document_ids):,} documents)"
            )

        if stats.chunks == 0:
            raise IndexBuildError(
                "Build produced zero chunks. The dataset source is empty or "
                "produced no valid passages after preprocessing. No index was "
                "created."
            )

        chunk_store.finalize(tmp_dir / CHUNKS_INDEX_FILENAME)

        store_dir = tmp_dir / VECTORSTORE_DIRNAME
        store.save(store_dir, overwrite=True)

        manifest = RagIndexManifest(
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            dataset=DatasetInfo(
                source=config.hf_split or "local-parquet",
                config=config.hf_config if config.hf_split else None,
                split=config.hf_split,
                input_path=str(config.source_path) if config.source_path else None,
                processed_path=str(processed_path),
                target_lang=None,
            ),
            embedding=EmbeddingInfo(
                provider=type(embedder).__name__,
                model=config.embedding_model,
                dimension=dimension,
                normalize=bool(getattr(embedder, "normalize", True)),
                device=getattr(embedder, "device", None),
            ),
            vector_store=VectorStoreInfo(
                backend=config.vector_store_backend,
                count=store.count,
            ),
            chunking=ChunkingInfo(
                strategy=config.chunking_strategy.value,
                params={
                    "batch_size": config.batch_size,
                    "tokenizer_model": config.tokenizer_model,
                },
            ),
            counts=CountsInfo(
                chunks=stats.chunks,
                documents=stats.documents,
                vectors=store.count,
            ),
            build={
                "script": "scripts.build_index",
                "embedding_batch_size": config.embedding_batch_size,
                "limit": config.limit,
                "top_k": config.top_k,
            },
        )
        write_manifest(tmp_dir, manifest)

        smoke = _run_smoke_test(tmp_dir, manifest, embedder, log)
        _finalize_swap(tmp_dir, final_dir)
    except Exception:
        try:
            if "chunk_store" in locals() and chunk_store is not None:
                chunk_store.close()
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    stats.elapsed_seconds = time.perf_counter() - start
    log(
        f"Build complete: {stats.passages:,} passages, {stats.chunks:,} chunks, "
        f"{stats.vectors_indexed:,} vectors, {stats.documents:,} documents, "
        f"{stats.embedding_failures} embedding failures, "
        f"{stats.elapsed_seconds:.1f}s elapsed -> {final_dir}"
    )
    return BuildResult(
        statistics=stats,
        index_dir=final_dir,
        manifest=manifest,
        smoke=smoke,
    )


def _finalize_swap(tmp_dir: Path, final_dir: Path) -> None:
    """Atomically replace the final index directory with the validated build.

    The previous valid index is moved aside first; if the swap fails the
    previous index is restored. Only called after full validation.
    """
    backup_dir: Optional[Path] = None
    if final_dir.exists():
        backup_dir = final_dir.with_name(
            f"{final_dir.name}.old-{uuid.uuid4().hex[:8]}"
        )
        shutil.move(str(final_dir), str(backup_dir))
    try:
        shutil.move(str(tmp_dir), str(final_dir))
    except Exception:
        if backup_dir is not None and backup_dir.exists():
            shutil.move(str(backup_dir), str(final_dir))
        raise
    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="scripts.build_index",
        description=(
            "Build the production RAG index: dataset -> preprocessing -> "
            "chunks -> embeddings -> vector store -> manifest (atomic)."
        ),
    )
    parser.add_argument(
        "--source-path",
        "--input",
        dest="source_path",
        type=Path,
        default=None,
        help="Local MSMARCO-XI parquet file (raw input). Alias: --input. Alternative: --hf-split.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default=None,
        help="HuggingFace dataset split to build from (e.g. 'train', 'validation').",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default="default",
        help="Dataset config name (default 'default').",
    )
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=None,
        help="Processed CanonicalPassage parquet output "
        f"(default {DEFAULT_PROCESSED_OUTPUT}).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help=f"Final index directory (default: RAG_INDEX_DIR or "
        f"{resolve_index_dir(settings.rag_index_dir)}).",
    )
    parser.add_argument(
        "--vector-store",
        dest="vector_store_backend",
        type=str,
        default=None,
        choices=["faiss", "numpy"],
        help=f"Vector store backend (default: RAG_VECTOR_STORE or "
        f"'{settings.rag_vector_store}').",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=f"Embedding model (default: RAG_EMBEDDING_MODEL or "
        f"'{settings.rag_embedding_model}').",
    )
    parser.add_argument(
        "--embedding-device",
        "--device",
        dest="embedding_device",
        type=str,
        default=None,
        help=f"Embedding device auto/cpu/cuda. Alias: --device. Default: RAG_EMBEDDING_DEVICE or "
        f"'{settings.rag_embedding_device}').",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Maximum chunks per encode_batch() call (default 32).",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default="passage",
        choices=["passage", "sentence", "token", "adaptive"],
        help="Chunking strategy (default 'passage').",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Neighbors to retrieve during validation smoke test "
        f"(default: RAG_TOP_K or {settings.rag_top_k}).",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help="Tokenizer model for token/adaptive strategies (optional).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Records per preprocessing/indexing batch (default 500).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum passages to index (bounded sampling for test builds).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing valid index.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow downloading the embedding model (once).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> IndexBuildConfig:
    """Map parsed CLI arguments to an IndexBuildConfig.

    CLI flags take precedence; unset options fall back to the RAG_*
    environment settings, then built-in defaults.
    """
    index_dir = args.index_dir
    if index_dir is None:
        index_dir = resolve_index_dir(settings.rag_index_dir) or (
            REPO_ROOT / "data" / "index"
        )

    source_path = args.source_path
    if source_path is None and args.hf_split is None:
        if DEFAULT_RAW_INPUT.is_file():
            source_path = DEFAULT_RAW_INPUT
            print(
                f"Using default dataset source: {DEFAULT_RAW_INPUT} "
                f"(pass --input or --hf-split to change)"
            )

    try:
        return IndexBuildConfig(
            source_path=source_path,
            processed_output=args.processed_output,
            hf_dataset=DATASET_NAME,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            index_dir=index_dir,
            vector_store_backend=(
                args.vector_store_backend or settings.rag_vector_store
            ),
            top_k=args.top_k or settings.rag_top_k,
            embedding_model=(
                args.embedding_model or settings.rag_embedding_model
            ),
            embedding_device=(
                args.embedding_device or settings.rag_embedding_device
            ),
            embedding_batch_size=args.embedding_batch_size,
            chunking_strategy=ChunkingStrategy(args.chunking_strategy),
            tokenizer_model=args.tokenizer_model,
            batch_size=args.batch_size,
            limit=args.limit,
            overwrite=args.overwrite,
            allow_download=args.allow_download,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise IndexBuildError(f"Invalid build configuration: {exc}") from exc


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:])

    Returns:
        Exit code (0 success, 2 build failure)
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = config_from_args(args)
        result = build_index(config)
    except IndexBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, VectorStoreError, RuntimeError) as exc:
        print(f"ERROR: Index build failed: {exc}", file=sys.stderr)
        return 2

    manifest = result.manifest
    print("=" * 72)
    print("Index build completed")
    print(f"Documents: {result.statistics.documents:,}")
    print(f"Passages: {result.statistics.passages:,}")
    print(f"Chunks: {result.statistics.chunks:,}")
    print(f"Vectors: {result.statistics.vectors_indexed:,}")
    print(f"Embedding model: {manifest.embedding.model}")
    print(f"Dimension: {manifest.embedding.dimension}")
    print(f"Vector store: {manifest.vector_store.backend}")
    print(f"Index directory: {result.index_dir}")
    print(f"Elapsed: {result.statistics.elapsed_seconds:.1f}s")
    print(f"Smoke test: {result.smoke.get('grounding_verdict')} "
          f"(score {result.smoke.get('grounding_score')})")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
