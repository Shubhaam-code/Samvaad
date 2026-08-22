"""CLI Latency Benchmark Runner (Phase 5.9).

Usage:
    python -m scripts.run_latency_benchmark
    python -m scripts.run_latency_benchmark --queries evaluation/sample_queries.json --mode local
"""

import argparse
import json
import os
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.api.dependencies import (
    get_grounding_verifier,
    get_guardrail_pipeline,
    get_llm,
    get_orchestrator,
    get_tts,
)
from app.guardrails.grounding_verifier import GroundingVerifier
from app.guardrails.pipeline import GuardrailPipeline
from app.indexing.loader import load_index
from app.llm.fake import FakeLLM
from app.tts.fake import FakeTTS
from evaluation.latency_benchmark import LatencyBenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Samvaad Voice RAG Latency Benchmark")
    parser.add_argument(
        "--queries",
        type=str,
        default="evaluation/sample_queries.json",
        help="Path to sample queries JSON file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "live"],
        default="local",
        help="Benchmark execution mode ('local' using offline stubs or 'live' using configured cloud APIs)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of initial warmup queries",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Benchmark only the first N queries (0 = all). Useful in --mode live, "
            "where every query spends real LLM and TTS API quota."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="evaluation/latency_report.json",
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="evaluation/latency_report.md",
        help="Output path for Markdown report",
    )
    return parser.parse_args()


def main():
    # Configure UTF-8 output if possible
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    print("=" * 70)
    print(" SAMVAAD VOICE RAG: LATENCY BENCHMARK RUNNER (Phase 5.9)")
    print("=" * 70)
    print(f"Mode:            {args.mode.upper()}")
    print(f"Queries File:    {args.queries}")
    print(f"Warmup Count:    {args.warmup}")
    print(f"Output Reports:  {args.output_md} | {args.output_json}")
    print("-" * 70)

    if not os.path.exists(args.queries):
        print(f"[-] Error: Queries file not found at '{args.queries}'")
        sys.exit(1)

    with open(args.queries, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if args.limit and args.limit > 0:
        queries = queries[: args.limit]

    print(f"Loaded {len(queries)} test queries.")

    # Instantiate pipeline components
    print("\n[*] Initializing Pipeline Components...")
    guardrail = get_guardrail_pipeline()
    grounding = get_grounding_verifier()
    orchestrator = get_orchestrator()
    if orchestrator is None:
        # Build an in-memory vector orchestrator with real Embedder, VectorStore, and FastReranker
        from app.embedding import FakeEmbedder
        from app.vectorstore import NumpyVectorStore
        from app.vectorstore.base import VectorRecord
        from app.chunking.models import Chunk, ChunkingStrategy
        from app.retrieval.reranker import FastReranker
        from app.retrieval import DictChunkResolver, RetrievalOrchestrator

        embedder = FakeEmbedder(dimension=384)
        store = NumpyVectorStore(dimension=384)
        chunks_dict = {}

        vectors_list = []
        records_list = []
        for i, q in enumerate(queries):
            doc_id = f"doc_{i+1}"
            chunk_id = f"chunk_{i+1}_0"
            vec = embedder.encode(q["query"])
            vectors_list.append(vec)
            rec = VectorRecord(
                vector_id=i,
                chunk_id=chunk_id,
                document_id=doc_id,
                chunk_index=0,
            )
            records_list.append(rec)
            chunks_dict[chunk_id] = Chunk.from_passage_segment(
                document_id=doc_id,
                chunk_index=0,
                strategy=ChunkingStrategy.PASSAGE,
                chunk_text=f"Passage information about: {q['query']}",
                query_id=q["id"],
                passage_index=0,
                target_lang=q.get("lang", "en"),
                source_lang="en",
                query=q["query"],
                eng_query=q["query"],
                query_type="general",
                answer=None,
                eng_answer=None,
                is_selected=False,
            )

        store.add(vectors_list, records_list)

        resolver = DictChunkResolver(chunks_dict)
        reranker = FastReranker(max_latency_ms=15.0)
        orchestrator = RetrievalOrchestrator(
            embedder=embedder,
            vector_store=store,
            resolver=resolver,
            reranker=reranker,
            guardrail_pipeline=guardrail,
        )

    if args.mode == "live":
        llm = get_llm() or FakeLLM()
        tts = get_tts() or FakeTTS()
    else:
        llm = FakeLLM()
        tts = FakeTTS()

    print("   [+] Guardrail Pipeline: Ready")
    print(f"   [+] Vector Orchestrator: Ready ({type(orchestrator).__name__} + {type(orchestrator._reranker).__name__})")
    print(f"   [+] LLM Generator: Ready ({type(llm).__name__})")
    print("   [+] Grounding Verifier: Ready")
    print(f"   [+] TTS Synthesizer: Ready ({type(tts).__name__})")

    runner = LatencyBenchmarkRunner(
        guardrail_pipeline=guardrail,
        orchestrator=orchestrator,
        llm=llm,
        grounding_verifier=grounding,
        tts=tts,
        sla_target_ms=200.0,
    )

    print("\n[*] Running Benchmark Execution (with nanosecond precision)...")

    def progress_callback(current, total):
        pct = (current / total) * 100.0
        bar = "#" * int(pct // 4) + "-" * (25 - int(pct // 4))
        sys.stdout.write(f"\r   [{bar}] {current}/{total} queries ({pct:.1f}%)")
        sys.stdout.flush()

    report = runner.run_benchmark(
        queries=queries,
        warmup_count=args.warmup,
        progress_cb=progress_callback,
    )

    print("\n\n" + "=" * 70)
    print(" BENCHMARK PERCENTILE SUMMARY (ms)")
    print("=" * 70)

    stats = report.stage_statistics
    print(f"{'Pipeline Stage':<30} | {'P50':<8} | {'P70':<8} | {'P90':<8} | {'P95':<8} | {'P100':<8}")
    print("-" * 80)

    stage_display = [
        ("guardrail_ms", "1. Guardrail Safety"),
        ("embedding_ms", "2. Dense Embedding"),
        ("faiss_lookup_ms", "3. FAISS Vector Search"),
        ("rerank_ms", "4. Fast Hybrid Reranker"),
        ("retrieval_total_ms", "=> Total Retrieval"),
        ("llm_generation_ms", "5. LLM Generation (TTFT)"),
        ("grounding_ms", "6. Grounding Verifier"),
        ("tts_ms", "7. Voice TTS"),
        ("total_e2e_ms", "=> END-TO-END PIPELINE"),
    ]

    for key, label in stage_display:
        st = stats.get(key)
        if st:
            print(f"{label:<30} | {st.p50:<8.2f} | {st.p70:<8.2f} | {st.p90:<8.2f} | {st.p95:<8.2f} | {st.p100:<8.2f}")

    print("=" * 80)
    total_p50 = stats["total_e2e_ms"].p50
    if report.sla_passed:
        print(f" STATUS: SLA PASSED! (P50 = {total_p50:.2f}ms < 200.00ms Target)")
    else:
        print(f" STATUS: BUDGET EXCEEDED (P50 = {total_p50:.2f}ms >= 200.00ms Target)")
    print("=" * 80)

    # Save output artifacts
    report.save(args.output_json, args.output_md)
    print(f"\n[+] Saved JSON Report:      {args.output_json}")
    print(f"[+] Saved Markdown Report:  {args.output_md}\n")


if __name__ == "__main__":
    main()
