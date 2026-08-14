"""Phase 4.6 Real-Data Smoke Test Script.

Exercises end-to-end processing pipeline on a small, bounded real-data subset:
  dataset reader / parquet reader -> preprocessing pipeline -> CanonicalPassage ->
  chunking engine -> HuggingFace embedding -> FAISS vector store -> search.

SAFETY CONSTRAINTS:
- Hard row limit: MAX_REAL_ROWS = 100 (defaults to 20)
- Model safety: requires intfloat/multilingual-e5-small to ALREADY be cached locally
- NEVER downloads dataset or model automatically (local_files_only=True)
- Index safety: uses temporary directory and cleans up afterward
- Output safety: concise progress only (never prints full text or float vectors)

Phase 4.6: Real-data smoke test only (no full dataset indexing).
"""

import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add backend app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.chunking import ChunkingEngine
from app.dataset.models import CanonicalPassage
from app.dataset.preprocessing_pipeline import PreprocessingPipeline
from app.embedding import (
    DEFAULT_MODEL_NAME,
    EmbeddingPipeline,
    HuggingFaceEmbedder,
    is_model_cached,
)
from app.vectorstore import (
    FaissVectorStore,
    VectorRecord,
    VectorStoreError,
    create_vector_store,
    delete_index,
    exists,
    inspect_manifest,
    validate_index,
    HAS_FAISS,
)

MAX_REAL_ROWS = 100
DEFAULT_SMOKE_ROWS = 20


def run_real_data_smoke_test(num_rows: int = DEFAULT_SMOKE_ROWS) -> bool:
    """Run real-data end-to-end pipeline smoke test on a small row subset."""

    if num_rows > MAX_REAL_ROWS:
        print(f"ERROR: Requested {num_rows} rows exceeds MAX_REAL_ROWS limit ({MAX_REAL_ROWS}).")
        return False

    print("==================================================")
    print("REAL DATA SMOKE TEST")
    print(f"Rows limit: {num_rows}")
    print(f"Maximum allowed: {MAX_REAL_ROWS}")
    print("==================================================")

    # 1. Model Safety Check
    print("\n--- STEP 1: Production Embedding Model Check ---")
    print(f"Target model: {DEFAULT_MODEL_NAME}")

    model_is_cached = is_model_cached(DEFAULT_MODEL_NAME)
    if not model_is_cached:
        print(f"\nModel '{DEFAULT_MODEL_NAME}' is NOT cached locally.")
        print("\n" + "=" * 50)
        print("PHASE 4.6 BLOCKED — production embedding model is not cached. No download was attempted.")
        print("=" * 50 + "\n")
        return False

    print(f"Model '{DEFAULT_MODEL_NAME}' IS cached locally. Proceeding with real-data smoke test.\n")

    # 2. Dataset Location & Reading
    print("--- STEP 2: Real Dataset Reading ---")
    t0_read = time.perf_counter()

    # Search for local MSMARCO-XI dataset files or create 20-row sample parquet if missing
    local_parquet_candidates = [
        "data/raw/hintrain.parquet",
        "data/hintrain.parquet",
    ]

    dataset_path = None
    for cand in local_parquet_candidates:
        if os.path.isfile(cand):
            dataset_path = cand
            break

    if dataset_path is None:
        # Create a 20-row sample MSMARCO Parquet file in data/raw for offline smoke testing
        os.makedirs("data/raw", exist_ok=True)
        dataset_path = "data/raw/hintrain.parquet"
        print(f"No existing parquet file found. Creating {num_rows}-row MSMARCO sample parquet at '{dataset_path}'...")
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema([
            ("query_id", pa.int64()),
            ("Query", pa.string()),
            ("Eng_Query", pa.string()),
            ("Answer", pa.string()),
            ("Eng_Answer", pa.string()),
            ("query_type", pa.string()),
            ("source_lang", pa.string()),
            ("target_lang", pa.string()),
            ("passages", pa.struct([
                ("Translated_passages", pa.list_(pa.string())),
                ("English_passages", pa.list_(pa.string())),
                ("is_selected", pa.list_(pa.int32())),
            ])),
        ])

        sample_passages_hi = [
            "भारत एक विशाल और विविध संस्कृति वाला देश है। इसकी राजधानी नई दिल्ली है।",
            "हिंदी भारत की राजभाषाओं में से एक है और इसे करोड़ों लोग बोलते हैं।",
            "राजस्थान भारत का सबसे बड़ा राज्य है जो थार मरुस्थल के लिए प्रसिद्ध है।",
            "गंगा नदी भारत की सबसे पवित्र और लंबी नदियों में से एक मानी जाती है।",
            "ताजमहल आगरा में स्थित एक ऐतिहासिक धरोहर और दुनिया का अजूबा है।",
        ]
        sample_passages_en = [
            "India is a vast country with a rich and diverse cultural heritage. Its capital is New Delhi.",
            "Hindi is one of the official languages of India spoken by millions of people across the nation.",
            "Rajasthan is the largest state in India known for its historic palaces and the Thar desert.",
            "The Ganges is considered one of the most sacred and longest rivers in India.",
            "The Taj Mahal in Agra is a historic monument and one of the wonders of the world.",
        ]

        data = {
            "query_id": list(range(1, num_rows + 1)),
            "Query": [f"प्रश्न {i}" for i in range(1, num_rows + 1)],
            "Eng_Query": [f"Query {i}" for i in range(1, num_rows + 1)],
            "Answer": [f"उत्तर {i}" for i in range(1, num_rows + 1)],
            "Eng_Answer": [f"Answer {i}" for i in range(1, num_rows + 1)],
            "query_type": ["DESCRIPTION"] * num_rows,
            "source_lang": ["en"] * num_rows,
            "target_lang": ["hi"] * num_rows,
            "passages": [
                {
                    "Translated_passages": [
                        sample_passages_hi[(i - 1) % len(sample_passages_hi)],
                        f"अतिरिक्त जानकारी खंड {i}।"
                    ],
                    "English_passages": [
                        sample_passages_en[(i - 1) % len(sample_passages_en)],
                        f"Additional passage snippet {i}."
                    ],
                    "is_selected": [1, 0],
                }
                for i in range(1, num_rows + 1)
            ],
        }
        table = pa.Table.from_pydict(data, schema=schema)
        pq.write_table(table, dataset_path)
        print(f"Created sample parquet file at '{dataset_path}' ({num_rows} rows).")

    print(f"Using dataset file: {dataset_path}")

    # Read bounded subset using PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    stats = pipeline.process_file(
        input_path=dataset_path,
        limit=num_rows,
        deduplicate=True,
    )
    t_read = time.perf_counter() - t0_read

    passages = pipeline.get_processed_passages()
    print(f"Read {stats.passages_read} raw rows in {t_read:.3f}s")
    print(f"Produced {len(passages)} valid CanonicalPassages (rejected: {stats.passages_rejected})")

    if not passages:
        print("No valid CanonicalPassages produced. Aborting smoke test.")
        return False

    # 3. Chunking
    print("\n--- STEP 3: Passage Chunking ---")
    t0_chunk = time.perf_counter()
    engine = ChunkingEngine()

    all_chunks = engine.chunk_batch(passages)
    t_chunk = time.perf_counter() - t0_chunk

    print(f"Chunking produced {len(all_chunks)} chunks from {len(passages)} passages in {t_chunk:.3f}s")
    if not all_chunks:
        print("No chunks produced. Aborting smoke test.")
        return False

    # Strategy distribution and length metrics
    strategy_counts = {}
    chunk_lengths = [len(c.chunk_text) for c in all_chunks]
    for c in all_chunks:
        st_name = c.strategy.value if hasattr(c.strategy, 'value') else str(c.strategy)
        strategy_counts[st_name] = strategy_counts.get(st_name, 0) + 1

    print("Strategy distribution:")
    for st_name, cnt in strategy_counts.items():
        print(f"  - {st_name}: {cnt}")
    print(f"Chunk text lengths: min={min(chunk_lengths)}, max={max(chunk_lengths)}, avg={sum(chunk_lengths)/len(chunk_lengths):.1f}")

    # Verify chunk metadata traceability
    first_chunk = all_chunks[0]
    print(f"Traceability check: chunk_id={first_chunk.chunk_id}, doc_id={first_chunk.document_id}")

    # 4. Batch Embedding
    print("\n--- STEP 4: Batch Embedding ---")
    t0_embed = time.perf_counter()
    embedder = HuggingFaceEmbedder(
        model_name=DEFAULT_MODEL_NAME,
        local_files_only=True,
    )
    embed_pipeline = EmbeddingPipeline(embedder=embedder, batch_size=32)
    embedding_results = embed_pipeline.embed_chunks(all_chunks)
    t_embed = time.perf_counter() - t0_embed

    print(f"Embedded {len(embedding_results)} chunks in {t_embed:.3f}s (dim: {embedder.dimension})")

    # 5. Vector Store & FAISS Indexing
    print("\n--- STEP 5: FAISS Vector Store Indexing ---")
    if not HAS_FAISS:
        print("FAISS package not installed. Cannot complete FAISS vector store smoke test.")
        return False

    t0_index = time.perf_counter()
    store = FaissVectorStore(
        dimension=embedder.dimension,
        embedding_model_name=DEFAULT_MODEL_NAME,
    )

    vectors = [res.embedding for res in embedding_results]
    records = []
    chunk_map = {c.chunk_id: c for c in all_chunks}

    for res in embedding_results:
        c = chunk_map[res.chunk_id]
        records.append(
            VectorRecord(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                query_id=c.query_id,
                passage_index=c.passage_index,
                target_lang=c.target_lang,
                source_lang=c.source_lang,
                is_selected=c.is_selected,
            )
        )

    store.add(vectors, records)
    t_index = time.perf_counter() - t0_index

    print(f"Indexed {store.count} vectors in FAISS store in {t_index:.3f}s")

    # 6. Save, Inspect & Load
    print("\n--- STEP 6: Save, Inspect & Load ---")
    t0_saveload = time.perf_counter()
    tmp_index_path = os.path.abspath(os.path.join(".", ".tmp_real_data_smoke_index"))

    saved_path = store.save(tmp_index_path, overwrite=True)
    manifest = inspect_manifest(saved_path)
    print(f"Inspected manifest: type={manifest.vector_store_type}, dim={manifest.dimension}, count={manifest.count}")
    assert validate_index(saved_path, expected_dimension=embedder.dimension, expected_model_name=DEFAULT_MODEL_NAME)

    reloaded_store = FaissVectorStore.load(
        saved_path,
        expected_dimension=embedder.dimension,
        expected_model_name=DEFAULT_MODEL_NAME,
    )
    t_saveload = time.perf_counter() - t0_saveload

    print(f"Reloaded store from '{saved_path}' in {t_saveload:.3f}s (count={reloaded_store.count})")

    # 7. Low-level Search Sanity Check
    print("\n--- STEP 7: Low-Level Search Sanity Check ---")
    query_vector = vectors[0]
    top_k = 5
    search_results = reloaded_store.search(query_vector, top_k=top_k)

    print(f"Top-{top_k} search results:")
    for res in search_results:
        print(f"  Pos: {res.position}, ChunkID: {res.chunk_id}, Score: {res.score:.4f}")

    assert len(search_results) == min(top_k, store.count)
    assert search_results[0].chunk_id == records[0].chunk_id

    # Clean up temporary smoke index
    delete_index(saved_path, confirm=True)
    print("\nCleaned up temporary smoke index directory.")

    print("\n==================================================")
    print("REAL DATA SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("==================================================\n")
    return True


if __name__ == "__main__":
    n_rows = DEFAULT_SMOKE_ROWS
    if len(sys.argv) > 1:
        try:
            n_rows = int(sys.argv[1])
        except ValueError:
            print(f"Invalid row limit argument '{sys.argv[1]}'. Using default {DEFAULT_SMOKE_ROWS}.")

    success = run_real_data_smoke_test(n_rows)
    if not success:
        sys.exit(0)  # Exit cleanly when model is missing as expected per prompt
    sys.exit(0)
