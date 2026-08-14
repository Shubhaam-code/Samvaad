"""Explicit production embedding smoke test (Phase 4.2).

Loads the selected production model (intfloat/multilingual-e5-small),
encodes a handful of tiny Hindi/English sentences, and prints basic
sanity metrics:

- model name, embedding dimension, device, normalization flag
- vector length and finiteness of values
- cosine similarities for obviously related/unrelated examples

This is ONLY a sanity check that the model is functioning. No scientific
benchmark claims are made from these tiny examples.

This script NEVER loads MSMARCO-XI or any real dataset; it processes a
handful of hard-coded synthetic strings only.

Download behavior:
- By default the script only uses the local HuggingFace cache.
- If the model is not cached, it reports exactly what is missing and
  exits WITHOUT downloading.
- Pass --allow-download to explicitly fetch the model once (~0.5 GB).

Usage:
    python scripts/test_production_embedding.py
    python scripts/test_production_embedding.py --allow-download
    python scripts/test_production_embedding.py --device cpu
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

# Allow running from anywhere: add the backend directory to the path
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.embedding import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    HuggingFaceEmbedder,
    is_model_cached,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("embedding_smoke")

# Tiny synthetic examples only (requirement 15)
HINDI_CAPITAL_1 = "भारत की राजधानी नई दिल्ली है।"
HINDI_CAPITAL_2 = "नई दिल्ली भारत की राजधानी है।"
ENGLISH_CAPITAL = "India's capital is New Delhi."
ENGLISH_UNRELATED = "The weather is cold today."
HINDI_UNRELATED = "मौसम आज बहुत ठंडा है।"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (no external deps)."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicit production embedding smoke test (tiny synthetic strings only)."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help="HuggingFace model id or local path")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cpu', or 'cuda'/'cuda:N'")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Maximum texts per encode_batch() call")
    parser.add_argument("--allow-download", action="store_true",
                        help="Explicitly download the model if not cached (once)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: check local cache before doing anything
    # ------------------------------------------------------------------
    cached = is_model_cached(args.model)
    if not cached:
        print("=" * 72)
        print(f"MODEL NOT CACHED LOCALLY: {args.model}")
        print("=" * 72)
        print("The model is not present in the local HuggingFace cache.")
        print("This script will NOT download it automatically.")
        print()
        print("What is missing:")
        print(f"  - HuggingFace snapshot for repo '{args.model}'")
        print(f"    (expected at: ~/.cache/huggingface/hub/models--{args.model.replace('/', '--')})")
        print()
        print("To fetch it once, explicitly run:")
        print(f"  python scripts/test_production_embedding.py --allow-download")
        print()
        print("Nothing was downloaded. No real dataset was processed.")
        return 1

    # ------------------------------------------------------------------
    # Step 2: load the model explicitly
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    print(f"Device: {args.device} | local_files_only={not args.allow_download}")
    embedder = HuggingFaceEmbedder(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=not args.allow_download,
    )

    dimension = embedder.dimension
    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {dimension}")
    print(f"Resolved device: {embedder.device}")
    print(f"Normalization: L2-normalized={embedder.normalize}")
    print()

    # ------------------------------------------------------------------
    # Step 3: encode tiny examples (single + batch)
    # ------------------------------------------------------------------
    samples = [HINDI_CAPITAL_1, HINDI_CAPITAL_2, ENGLISH_CAPITAL,
               ENGLISH_UNRELATED, HINDI_UNRELATED]
    names = ["hi-capital-1", "hi-capital-2", "en-capital",
             "en-unrelated", "hi-unrelated"]

    vectors_batch = embedder.encode_batch(samples)
    vectors_single = [embedder.encode(t) for t in samples]

    print(f"{'sample':<14} {'length':<8} {'finite':<8} match-single")
    print("-" * 48)
    all_finite = True
    order_ok = True
    for name, vector, single in zip(names, vectors_batch, vectors_single):
        finite = all(math.isfinite(v) for v in vector)
        all_finite = all_finite and finite
        same = vector == single
        order_ok = order_ok and same
        print(f"{name:<14} {len(vector):<8} {str(finite):<8} {same}")
    print()
    print(f"All values finite: {all_finite}")
    print(f"Single == batch (identical path): {order_ok}")
    print()

    # ------------------------------------------------------------------
    # Step 4: cosine similarity sanity (related vs unrelated)
    # ------------------------------------------------------------------
    sims = {
        "hi-capital-1 vs hi-capital-2 (related, Hindi)": cosine_similarity(
            vectors_batch[0], vectors_batch[1]),
        "en-capital vs hi-capital-1 (related, cross-lingual)": cosine_similarity(
            vectors_batch[2], vectors_batch[0]),
        "en-unrelated vs hi-capital-1 (unrelated, cross-lingual)": cosine_similarity(
            vectors_batch[3], vectors_batch[0]),
        "en-unrelated vs en-capital (unrelated, English)": cosine_similarity(
            vectors_batch[3], vectors_batch[2]),
        "hi-unrelated vs hi-capital-1 (unrelated, Hindi)": cosine_similarity(
            vectors_batch[4], vectors_batch[0]),
    }

    print("Cosine similarities (tiny sanity check, NOT a benchmark):")
    for label, sim in sims.items():
        print(f"  {label:<52} {sim:.4f}")

    related = [sims["hi-capital-1 vs hi-capital-2 (related, Hindi)"],
               sims["en-capital vs hi-capital-1 (related, cross-lingual)"]]
    unrelated = [sims["en-unrelated vs hi-capital-1 (unrelated, cross-lingual)"],
                 sims["en-unrelated vs en-capital (unrelated, English)"],
                 sims["hi-unrelated vs hi-capital-1 (unrelated, Hindi)"]]
    sanity_ok = min(related) > max(unrelated)
    print()
    print(f"Related pairs (min={min(related):.4f}) vs unrelated pairs (max={max(unrelated):.4f})")
    print(f"Sanity check (related > unrelated): {'PASS' if sanity_ok else 'FAIL'}")
    if not sanity_ok:
        print("NOTE: with tiny random-like examples this may occasionally fail;")
        print("      it is not a benchmark claim, only a functioning check.")
    print()

    print("=" * 72)
    print("SMOKE TEST COMPLETE - only tiny synthetic strings were processed.")
    print("No MSMARCO-XI data was loaded. No vector DB/retrieval was used.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())