"""Tiny explicit integration smoke test for the embedding pipeline (Phase 4.3).

Uses 5-20 synthetic Chunk objects and FakeEmbedder by default.
It never loads MSMARCO-XI or any real dataset.

Real-model opt-in:
    --real                      use HuggingFaceEmbedder (local cache only;
                                NEVER downloads the model automatically)
    --real --allow-download     explicitly fetch the model once if missing

Usage:
    python scripts/test_embedding_pipeline.py
    python scripts/test_embedding_pipeline.py --real
    python scripts/test_embedding_pipeline.py --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.chunking.models import Chunk, ChunkingStrategy  # noqa: E402
from app.embedding import (  # noqa: E402
    EmbeddingPipeline,
    FakeEmbedder,
    HuggingFaceEmbedder,
    is_model_cached,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

SYNTHETIC_TEXTS = [
    "भारत की राजधानी नई दिल्ली है।",
    "नई दिल्ली भारत की राजधानी है।",
    "India's capital is New Delhi.",
    "The weather is cold today.",
    "गोवा अपने समुद्र तटों के लिए प्रसिद्ध है।",
    "Goa is famous for its beaches.",
    "हिमालय दुनिया की सबसे ऊँची पर्वत श्रृंखला है।",
    "The Himalayas are the highest mountain range.",
    "ताजमहल आगरा में स्थित है।",
    "The Taj Mahal is located in Agra.",
    "वर्षा ऋतु में अच्छी फसल होती है।",
    "Good crops grow during the rainy season.",
]


def make_synthetic_chunks(count: int = 12) -> list[Chunk]:
    """Build tiny synthetic Chunk objects (never real data)."""
    chunks = []
    for i in range(count):
        text = SYNTHETIC_TEXTS[i % len(SYNTHETIC_TEXTS)]
        chunk_id = Chunk.generate_chunk_id(
            document_id=f"smoke-doc-{i}",
            strategy=ChunkingStrategy.PASSAGE,
            chunk_index=0,
        )
        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=f"smoke-doc-{i}",
            chunk_index=0,
            strategy=ChunkingStrategy.PASSAGE,
            chunk_text=text,
            query_id=1,
            passage_index=0,
            target_lang="hi",
            source_lang="en",
            query="स्मोक परीक्षण प्रश्न",
            eng_query="smoke test query",
            is_selected=True,
        ))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tiny explicit embedding pipeline smoke test (synthetic chunks only)."
    )
    parser.add_argument("--count", type=int, default=12,
                        help="Number of synthetic chunks (5-20 recommended)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Pipeline batch size")
    parser.add_argument("--real", action="store_true",
                        help="Use the real HuggingFace model (local cache only)")
    parser.add_argument("--allow-download", action="store_true",
                        help="Explicitly download the real model once if missing")
    args = parser.parse_args()

    if args.count < 1 or args.count > 20:
        print(f"count must be between 1 and 20, got {args.count}")
        return 1

    chunks = make_synthetic_chunks(args.count)
    print(f"Created {len(chunks)} synthetic Chunk objects (no real data).")

    if args.real:
        model = "intfloat/multilingual-e5-small"
        if not is_model_cached(model) and not args.allow_download:
            print("=" * 72)
            print(f"REAL MODEL NOT CACHED: {model}")
            print("=" * 72)
            print("This script will NOT download it automatically.")
            print("Run with --real --allow-download to fetch it once, or")
            print("omit --real to use FakeEmbedder.")
            print("Nothing was downloaded. No real dataset was processed.")
            return 1
        embedder = HuggingFaceEmbedder(
            model_name=model,
            device="auto",
            batch_size=args.batch_size,
            local_files_only=not args.allow_download,
        )
        print(f"Using real model: {model} (device={embedder.device})")
    else:
        embedder = FakeEmbedder(dimension=16, batch_size=args.batch_size)
        print(f"Using FakeEmbedder (dimension={embedder.dimension})")

    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=args.batch_size)

    # ------------------------------------------------------------------
    # Streaming batch pass (the memory-safe API)
    # ------------------------------------------------------------------
    all_results = []
    for batch in pipeline.embed_batches(chunks):
        all_results.extend(batch)

    ok = len(all_results) == len(chunks)
    ordered = [r.chunk_id for r in all_results] == [c.chunk_id for c in chunks]
    dimensions = {r.dimension for r in all_results}
    finite = all(
        math.isfinite(v) for r in all_results for v in r.embedding
    )

    print()
    print(f"Batches processed: {math.ceil(len(chunks) / args.batch_size)}")
    print(f"Total results: {len(all_results)} (expected {len(chunks)}) -> "
          f"{'OK' if ok else 'FAIL'}")
    print(f"Ordering preserved: {'OK' if ordered else 'FAIL'}")
    print(f"Dimensions seen: {sorted(dimensions)} -> "
          f"{'OK' if len(dimensions) == 1 else 'FAIL'}")
    print(f"All values finite: {'OK' if finite else 'FAIL'}")
    print(f"Pipeline provider: {pipeline.provider}")
    print(f"Model metadata on results: {all_results[0].model_name!r}")

    # ------------------------------------------------------------------
    # Determinism check
    # ------------------------------------------------------------------
    again = EmbeddingPipeline(embedder=embedder, batch_size=args.batch_size)
    second_run = again.embed_all(chunks)
    deterministic = second_run == all_results
    print(f"Deterministic across runs: {'OK' if deterministic else 'FAIL'}")

    print()
    print("=" * 72)
    print("SMOKE TEST COMPLETE - only synthetic Chunk objects were processed.")
    print("No MSMARCO-XI data. No vector DB / index / retrieval.")
    print("=" * 72)

    return 0 if (ok and ordered and len(dimensions) == 1 and finite) else 1


if __name__ == "__main__":
    sys.exit(main())