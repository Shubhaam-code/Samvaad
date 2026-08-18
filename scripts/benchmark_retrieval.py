"""Retrieval and Reranker Performance Benchmark CLI (Phase 5.2).

Benchmarks end-to-end retrieval performance against the persisted FAISS vector
index and measures per-stage latencies (Embedding, FAISS search, Reranking, Total)
across realistic multilingual Indic and English queries.

Target: < 30ms End-to-End P50 Retrieval Latency.

Usage:
    python scripts/benchmark_retrieval.py --index-dir data/processed --queries 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure backend is on sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.chunking.models import Chunk, ChunkingStrategy
from app.embedding.huggingface import create_huggingface_embedder
from app.retrieval.models import RetrievedChunk
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval.reranker import FastReranker
from app.retrieval.resolver import DictChunkResolver
from app.vectorstore.faiss_store import FaissVectorStore

BENCHMARK_QUERIES = [
    "भारत की राजधानी क्या है?",
    "what is the capital of India?",
    "मौसम का पूर्वानुमान कैसे किया जाता है?",
    "how to calculate compound interest",
    "कंप्यूटर और इंटरनेट के मुख्य लाभ",
    "solar energy benefits and cost",
    "भारतीय संविधान की प्रमुख विशेषताएं",
    "how do airplane engines work",
    "जल संरक्षण के उपाय",
    "healthy diet and nutrition tips",
]


def run_retrieval_benchmark(
    index_dir: Path,
    num_queries: int = 50,
    top_k: int = 5,
    top_n_candidates: int = 15,
) -> None:
    print("=" * 70)
    print(" [*] SAMVAAD - RETRIEVAL & RERANKER BENCHMARK (Phase 5.2)")
    print("=" * 70)

    if not (index_dir / "index.faiss").is_file():
        print(f"Error: FAISS index not found at {index_dir / 'index.faiss'}")
        print("Please build the index first using scripts/build_full_index.py")
        sys.exit(1)

    print(f"[*] Loading FAISS vector store from: {index_dir}")
    t0 = time.perf_counter()
    store = FaissVectorStore.load(index_dir)
    print(f"    Loaded {store.count} vectors in {(time.perf_counter() - t0) * 1000:.2f}ms")

    print("[*] Loading embedding model (intfloat/multilingual-e5-small)...")
    t0 = time.perf_counter()
    embedder = create_huggingface_embedder()
    print(f"    Embedding model ready in {(time.perf_counter() - t0) * 1000:.2f}ms")

    # Build an in-memory chunk resolver from stored metadata records
    print("[*] Building chunk resolver from index metadata...")
    chunks_dict = {}
    for pos, record in enumerate(store.records):
        text = record.extra_metadata.get("chunk_text", f"Passage chunk {record.chunk_id}") if record.extra_metadata else f"Passage chunk {record.chunk_id}"
        chunk = Chunk.from_passage_segment(
            document_id=record.document_id,
            chunk_index=record.chunk_index,
            strategy=ChunkingStrategy.PASSAGE,
            chunk_text=text,
            query_id=record.query_id or 1,
            passage_index=record.passage_index or 0,
            target_lang=record.target_lang or "hi",
            source_lang=record.source_lang or "en",
            query="sample query",
            eng_query="sample query",
            query_type="general",
            answer=None,
            eng_answer=None,
            is_selected=bool(record.is_selected),
        )
        chunks_dict[record.chunk_id] = chunk
    resolver = DictChunkResolver(chunks_dict)

    reranker = FastReranker(semantic_weight=0.6, lexical_weight=0.4)
    orchestrator = RetrievalOrchestrator(
        embedder=embedder,
        vector_store=store,
        resolver=resolver,
        reranker=reranker,
        top_k=top_k,
    )

    # Warmup
    print("[*] Running 3 warmup queries...")
    for q in BENCHMARK_QUERIES[:3]:
        orchestrator.retrieve(q, top_k=top_k, top_n_candidates=top_n_candidates)

    print(f"[*] Executing {num_queries} benchmark retrieval queries...")
    embedding_latencies = []
    search_latencies = []
    rerank_latencies = []
    total_latencies = []

    for i in range(num_queries):
        query = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
        t_start = time.perf_counter()
        result = orchestrator.retrieve(query, top_k=top_k, top_n_candidates=top_n_candidates)
        t_total = (time.perf_counter() - t_start) * 1000.0

        embedding_latencies.append(result.latencies_ms.get("embedding_ms", 0.0))
        search_latencies.append(result.latencies_ms.get("search_ms", 0.0))
        rerank_latencies.append(result.latencies_ms.get("rerank_ms", 0.0))
        total_latencies.append(t_total)

    def stats(arr: List[float]):
        s = sorted(arr)
        p50 = s[len(s) // 2]
        p90 = s[int(len(s) * 0.9)]
        p99 = s[min(len(s) - 1, int(len(s) * 0.99))]
        avg = sum(s) / len(s)
        return min(s), p50, p90, p99, max(s), avg

    e_min, e_p50, e_p90, e_p99, e_max, e_avg = stats(embedding_latencies)
    s_min, s_p50, s_p90, s_p99, s_max, s_avg = stats(search_latencies)
    r_min, r_p50, r_p90, r_p99, r_max, r_avg = stats(rerank_latencies)
    t_min, t_p50, t_p90, t_p99, t_max, t_avg = stats(total_latencies)

    print("\n" + "=" * 70)
    print(f"{'STAGE':<22} | {'P50 (ms)':<10} | {'P90 (ms)':<10} | {'P99 (ms)':<10} | {'AVG (ms)':<10}")
    print("-" * 70)
    print(f"{'1. Query Embedding':<22} | {e_p50:<10.2f} | {e_p90:<10.2f} | {e_p99:<10.2f} | {e_avg:<10.2f}")
    print(f"{'2. FAISS Vector Search':<22} | {s_p50:<10.2f} | {s_p90:<10.2f} | {s_p99:<10.2f} | {s_avg:<10.2f}")
    print(f"{'3. Fast Hybrid Rerank':<22} | {r_p50:<10.2f} | {r_p90:<10.2f} | {r_p99:<10.2f} | {r_avg:<10.2f}")
    print("-" * 70)
    print(f"{'TOTAL END-TO-END':<22} | {t_p50:<10.2f} | {t_p90:<10.2f} | {t_p99:<10.2f} | {t_avg:<10.2f}")
    print("=" * 70)

    if t_p50 < 30.0:
        print(f"\n[PASS] Target ACHIEVED! End-to-End P50 Retrieval Latency = {t_p50:.2f}ms (< 30.0ms target)\n")
    else:
        print(f"\n[WARN] Latency {t_p50:.2f}ms exceeded 30.0ms target.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Samvaad Retrieval & Reranker Benchmark")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed"), help="Path to processed index directory")
    parser.add_argument("--queries", type=int, default=30, help="Number of benchmark queries to run")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final chunks to return")
    parser.add_argument("--top-n", type=int, default=15, help="Number of candidate chunks before reranking")
    args = parser.parse_args()

    run_retrieval_benchmark(
        index_dir=args.index_dir,
        num_queries=args.queries,
        top_k=args.top_k,
        top_n_candidates=args.top_n,
    )


if __name__ == "__main__":
    main()
