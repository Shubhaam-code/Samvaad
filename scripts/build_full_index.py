"""Build and persist a full MSMARCO-XI FAISS vector index.

Run from repository root:

    # Fast offline dry-run test (0 network, 0 model download):
    python -m scripts.build_full_index --dry-run --max-samples 100

    # Index 1000 Hindi validation queries using production E5 embedder:
    python -m scripts.build_full_index --lang hi --split validation --max-samples 1000

    # Full production indexing:
    python -m scripts.build_full_index --lang hi --split validation

Phase 5.1 (Issue #1): MSMARCO-XI Ingestion & Vector Indexing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure backend package is in python path
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.chunking.models import ChunkingStrategy
from app.dataset.indexer import DatasetIndexer, DatasetIndexerConfig, IndexingStatistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_full_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index the ai4bharat/MSMARCO-XI dataset into a local FAISS vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        help="Target Indic language code (e.g. 'hi', 'bn', 'ta', 'te', 'mr', 'gu', etc.)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "train"],
        help="Dataset split to index ('validation' recommended for dev/testing)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum raw query records to process (default: None, indexes full split)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding and vector store writes",
    )
    parser.add_argument(
        "--chunker",
        type=str,
        default="adaptive",
        choices=["adaptive", "sentence", "passage", "token"],
        help="Chunking strategy to use for segmenting passages",
    )
    parser.add_argument(
        "--store-type",
        type=str,
        default="faiss",
        choices=["faiss", "numpy"],
        help="Vector store backend to build ('faiss' or 'numpy')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory to save index.faiss, metadata.json, and schema.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for embedding model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic FakeEmbedder for rapid offline verification (no model download)",
    )
    parser.add_argument(
        "--local-parquet",
        type=str,
        default=None,
        help="Optional path to a local Parquet file instead of streaming from Hugging Face",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing index if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    strategy_map = {
        "adaptive": ChunkingStrategy.ADAPTIVE,
        "sentence": ChunkingStrategy.SENTENCE,
        "passage": ChunkingStrategy.PASSAGE,
        "token": ChunkingStrategy.TOKEN,
    }
    chunk_strategy = strategy_map[args.chunker.lower()]

    output_dir = REPO_ROOT / args.output_dir
    local_parquet = Path(args.local_parquet) if args.local_parquet else None

    config = DatasetIndexerConfig(
        lang=args.lang,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        chunk_strategy=chunk_strategy,
        store_type=args.store_type,
        output_dir=output_dir,
        device=args.device,
        dry_run=args.dry_run,
        overwrite=not args.no_overwrite,
        local_parquet_path=local_parquet,
    )

    print("=" * 70)
    print(" [*] SAMVAAD - MSMARCO-XI VECTOR INDEXING PIPELINE (Phase 5.1)")
    print("=" * 70)
    print(f" Language Code:     {config.lang}")
    print(f" Dataset Split:     {config.split}")
    print(f" Max Query Records: {config.max_samples or 'ALL (Full Split)'}")
    print(f" Chunking Strategy: {config.chunk_strategy.value}")
    print(f" Vector Store Type: {config.store_type.upper()}")
    print(f" Output Directory:  {config.output_dir}")
    print(f" Dry Run Mode:      {config.dry_run}")
    print("=" * 70)

    try:
        indexer = DatasetIndexer(config)

        def on_progress(indexed: int, chunks: int, elapsed: float):
            rate = indexed / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                f"\r -> Progress: Indexed {indexed} vectors ({chunks} chunks) in {elapsed:.1f}s [{rate:.1f} vec/s]   "
            )
            sys.stdout.flush()

        stats = indexer.run(progress_callback=on_progress)
        print()  # newline after progress bar

        print("-" * 70)
        print(" [+] INDEXING RUN SUMMARY")
        print("-" * 70)
        print(f" Raw Records Processed:   {stats.raw_records_read}")
        print(f" Canonical Passages:      {stats.canonical_passages_created}")
        print(f" Duplicates Removed:      {stats.duplicates_skipped}")
        print(f" Text Chunks Generated:   {stats.chunks_created}")
        print(f" Vectors Indexed:         {stats.vectors_indexed}")
        print(f" Total Elapsed Time:      {stats.duration_seconds}s")
        print(f" Throughput:              {stats.indexing_rate_chunks_per_sec} chunks/sec")
        print(f" Artifacts Persisted In:  {stats.output_directory}")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.exception(f"Indexing pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
