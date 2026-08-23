"""Latency Benchmark & Analytics Measurement Engine (Phase 5.9).

Executes realistic test queries through every stage of the Samvaad RAG pipeline,
measures high-precision nanosecond timings, calculates exact percentiles
(P50, P70, P90, P95, P100), and exports structured JSON and Markdown submission reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import os
import platform
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

TARGET_SLA_MS: float = 200.0


@dataclass
class LatencyStageMetrics:
    """Detailed stage latencies measured for a single query execution."""

    query_id: int
    query: str
    lang: str
    guardrail_ms: float
    embedding_ms: float
    faiss_lookup_ms: float
    rerank_ms: float
    retrieval_total_ms: float
    llm_generation_ms: float
    grounding_ms: float
    tts_ms: float
    total_e2e_ms: float


@dataclass
class PercentileStat:
    """Statistical distribution summary for a latency metric."""

    p50: float
    p70: float
    p90: float
    p95: float
    p100: float
    mean: float
    min_val: float
    max_val: float


def compute_percentiles(values: List[float]) -> PercentileStat:
    """Compute exact P50, P70, P90, P95, and P100 percentiles."""
    if not values:
        return PercentileStat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _get_pct(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return d0 + d1

    mean_val = sum(sorted_vals) / n
    return PercentileStat(
        p50=round(_get_pct(50.0), 3),
        p70=round(_get_pct(70.0), 3),
        p90=round(_get_pct(90.0), 3),
        p95=round(_get_pct(95.0), 3),
        p100=round(sorted_vals[-1], 3),
        mean=round(mean_val, 3),
        min_val=round(sorted_vals[0], 3),
        max_val=round(sorted_vals[-1], 3),
    )


@dataclass
class BenchmarkReport:
    """Complete summary report of latency benchmark execution."""

    total_queries: int
    timestamp_utc: str
    system_info: Dict[str, str]
    stage_statistics: Dict[str, PercentileStat]
    sla_target_ms: float
    sla_passed: bool
    runs: List[LatencyStageMetrics]
    # Which implementations actually produced these timings. Without this a
    # report generated against FakeLLM/FakeTTS is indistinguishable from one
    # measured against the real cloud providers, and sub-millisecond stub
    # timings get published as though they were achievable.
    provider_modes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize benchmark report to dictionary."""
        return {
            "total_queries": self.total_queries,
            "timestamp_utc": self.timestamp_utc,
            "system_info": self.system_info,
            "sla_target_ms": self.sla_target_ms,
            "sla_passed": self.sla_passed,
            "provider_modes": self.provider_modes,
            "stage_statistics": {
                stage: asdict(stat) for stage, stat in self.stage_statistics.items()
            },
            "runs": [asdict(r) for r in self.runs],
        }

    def to_markdown(self) -> str:
        """Generate human-readable Markdown table report for competition submission."""
        stat = self.stage_statistics
        total_p50 = stat.get("total_e2e_ms", PercentileStat(0,0,0,0,0,0,0,0)).p50
        status_badge = "✅ **PASSED (< 200ms Budget)**" if self.sla_passed else "❌ **EXCEEDED BUDGET**"

        lines = [
            "# 🚀 Samvaad Voice RAG Latency Benchmark Report",
            "",
            f"- **Execution Timestamp (UTC):** `{self.timestamp_utc}`",
            f"- **Total Benchmark Queries:** `{self.total_queries}`",
            f"- **Platform:** `{self.system_info.get('platform', 'unknown')}` ({self.system_info.get('processor', 'CPU')})",
            f"- **SLA Target:** `< {self.sla_target_ms:g}ms`",
            f"- **Benchmark Status:** {status_badge} (`P50 = {total_p50:.2f}ms`)",
            "",
            "## 📊 Per-Stage Latency Percentiles (Milliseconds)",
            "",
            "| Pipeline Stage | P50 (Median) | P70 | P90 | P95 | P100 (Max) | Mean | Target Budget |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        stage_labels = [
            ("guardrail_ms", "1. Guardrail Safety Check", "< 5ms"),
            ("embedding_ms", "2. Dense Query Embedding", "< 15ms"),
            ("faiss_lookup_ms", "3. FAISS Vector Search", "< 5ms"),
            ("rerank_ms", "4. Fast Hybrid Reranker", "< 5ms"),
            ("retrieval_total_ms", "📌 Total Retrieval Time", "< 30ms"),
            ("llm_generation_ms", "5. LLM Inference (TTFT)", "< 80ms"),
            ("grounding_ms", "6. Grounding Verification", "< 10ms"),
            ("tts_ms", "7. Voice TTS Synthesis", "< 50ms"),
            ("total_e2e_ms", "⚡ **END-TO-END PIPELINE**", "**< 200ms**"),
        ]

        for key, label, budget in stage_labels:
            s = stat.get(key, PercentileStat(0,0,0,0,0,0,0,0))
            lines.append(
                f"| {label} | `{s.p50:.2f}` | `{s.p70:.2f}` | `{s.p90:.2f}` | `{s.p95:.2f}` | `{s.p100:.2f}` | `{s.mean:.2f}` | {budget} |"
            )

        e2e = stat.get("total_e2e_ms", PercentileStat(0, 0, 0, 0, 0, 0, 0, 0))
        retrieval = stat.get("retrieval_total_ms", PercentileStat(0, 0, 0, 0, 0, 0, 0, 0))
        llm_stat = stat.get("llm_generation_ms", PercentileStat(0, 0, 0, 0, 0, 0, 0, 0))
        tts_stat = stat.get("tts_ms", PercentileStat(0, 0, 0, 0, 0, 0, 0, 0))

        guardrail_stat = stat.get("guardrail_ms", PercentileStat(0, 0, 0, 0, 0, 0, 0, 0))
        grounding_stat = stat.get("grounding_ms", PercentileStat(0, 0, 0, 0, 0, 0, 0, 0))

        network_p50 = llm_stat.p50 + tts_stat.p50
        network_share = (network_p50 / e2e.p50 * 100.0) if e2e.p50 else 0.0
        llm_variance = (llm_stat.p100 / llm_stat.p50) if llm_stat.p50 else 0.0
        local_p50 = retrieval.p50 + guardrail_stat.p50 + grounding_stat.p50

        if self.provider_modes:
            lines.extend([
                "",
                "## ⚙️ Measured Components",
                "",
                "| Stage | Implementation |",
                "| :--- | :--- |",
            ])
            for role, impl in self.provider_modes.items():
                lines.append(f"| {role} | `{impl}` |")
            if any("STUB" in v for v in self.provider_modes.values()):
                lines.extend([
                    "",
                    "> ⚠️ One or more stages were measured against in-process stubs. "
                    "Those rows reflect harness overhead only and must not be read as "
                    "achievable end-to-end performance.",
                ])

        lines.extend([
            "",
            "---",
            "### 🎯 Findings",
            "",
            f"1. **End-to-end P50 is `{e2e.p50:.2f}ms`** against a `{self.sla_target_ms:g}ms` target "
            f"({'within' if self.sla_passed else 'over'} budget). "
            f"P95 is `{e2e.p95:.2f}ms`, P100 `{e2e.p100:.2f}ms`.",
            f"2. **Local compute is not the bottleneck.** Retrieval P50 is `{retrieval.p50:.2f}ms` "
            f"(embedding + FAISS + rerank).",
            f"3. **Third-party API calls dominate.** LLM P50 `{llm_stat.p50:.2f}ms` + TTS P50 "
            f"`{tts_stat.p50:.2f}ms` = `{network_p50:.2f}ms`, about "
            f"{network_share:.0f}% of end-to-end latency. These are network-bound and "
            f"not reducible by local optimization.",
            f"4. **Tail latency is driven by LLM generation variance.** LLM P100 "
            f"`{llm_stat.p100:.2f}ms` is {llm_variance:.0f}x its P50 of `{llm_stat.p50:.2f}ms`. "
            f"On a reasoning-class model most completion tokens are spent deliberating "
            f"rather than on the answer text, so a short reply can still be slow and the "
            f"spread between a fast and a slow response is large.",
            f"5. **The `{self.sla_target_ms:g}ms` target is not reachable while the LLM and TTS "
            f"are remote API calls.** Retrieval, guardrail, and grounding together are "
            f"`{local_p50:.2f}ms` at P50, so the budget is only meaningful as a local-compute "
            f"target. Meeting it end to end would require on-device or streamed generation.",
            "",
        ])

        return "\n".join(lines)

    def save(self, json_path: str, md_path: str) -> None:
        """Save report to JSON and Markdown paths."""
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())


class LatencyBenchmarkRunner:
    """Harness that coordinates executing queries and profiling timings."""

    def __init__(
        self,
        guardrail_pipeline: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        llm: Optional[Any] = None,
        grounding_verifier: Optional[Any] = None,
        tts: Optional[Any] = None,
        sla_target_ms: float = TARGET_SLA_MS,
    ) -> None:
        self.guardrail = guardrail_pipeline
        self.orchestrator = orchestrator
        self.llm = llm
        self.grounding = grounding_verifier
        self.tts = tts
        self.sla_target_ms = sla_target_ms

    def run_query(self, query_item: Dict[str, Any]) -> LatencyStageMetrics:
        """Run a single query through the full pipeline measuring each stage."""
        query_id = query_item.get("id", 1)
        query_text = query_item.get("query", "")
        lang = query_item.get("lang", "en")

        t_total_start = time.perf_counter()

        # 1. Guardrail
        t0 = time.perf_counter()
        if self.guardrail is not None:
            self.guardrail.check_input(query_text)
        guardrail_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Retrieval (Embedding + FAISS + Reranking)
        embedding_ms = 0.0
        faiss_lookup_ms = 0.0
        rerank_ms = 0.0
        retrieval_total_ms = 0.0
        retrieved_chunks = []
        ret_res = None

        if self.orchestrator is not None:
            t0 = time.perf_counter()
            ret_res = self.orchestrator.retrieve(query_text)
            retrieval_total_ms = (time.perf_counter() - t0) * 1000.0
            embedding_ms = ret_res.latencies_ms.get("embedding_ms", 0.0)
            faiss_lookup_ms = ret_res.latencies_ms.get("search_ms", 0.0)
            rerank_ms = ret_res.latencies_ms.get("rerank_ms", 0.0)
            retrieved_chunks = [item.chunk for item in getattr(ret_res, "retrieved_chunks", [])]

        # 3. LLM Generation
        #
        # Uses the same grounded prompt builder as /api/chat. Sending the bare
        # query instead (no system prompt, no evidence) measures a different
        # operation entirely: the model answers open-domain from pretraining and
        # emits far more tokens, so the stage timing does not describe the
        # pipeline being shipped.
        llm_ms = 0.0
        answer_text = "Sample response"
        if self.llm is not None:
            from app.llm.models import LLMRequest
            from app.llm.prompt_engine import build_grounded_rag_prompt

            retrieved_items = list(getattr(ret_res, "retrieved_chunks", [])) if ret_res else []
            sys_prompt, user_prompt = build_grounded_rag_prompt(
                query=query_text,
                retrieved_chunks=retrieved_items,
            )
            t0 = time.perf_counter()
            llm_res = self.llm.generate(
                LLMRequest(prompt=user_prompt, system_prompt=sys_prompt)
            )
            llm_ms = (time.perf_counter() - t0) * 1000.0
            answer_text = getattr(llm_res, "text", answer_text)

        # 4. Grounding Verification
        grounding_ms = 0.0
        if self.grounding is not None:
            t0 = time.perf_counter()
            self.grounding.verify(answer_text, retrieved_chunks, query=query_text)
            grounding_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Voice TTS Synthesis (optional simulation or live)
        #
        # Synthesizes the whole answer, matching /api/voice-query. Truncating to
        # 100 characters understates the stage, since TTS cost scales with text
        # length.
        tts_ms = 0.0
        if self.tts is not None:
            from app.tts.models import TTSRequest
            t0 = time.perf_counter()
            self.tts.synthesize(TTSRequest(text=answer_text, language=lang))
            tts_ms = (time.perf_counter() - t0) * 1000.0

        total_e2e_ms = (time.perf_counter() - t_total_start) * 1000.0

        return LatencyStageMetrics(
            query_id=query_id,
            query=query_text,
            lang=lang,
            guardrail_ms=round(guardrail_ms, 3),
            embedding_ms=round(embedding_ms, 3),
            faiss_lookup_ms=round(faiss_lookup_ms, 3),
            rerank_ms=round(rerank_ms, 3),
            retrieval_total_ms=round(retrieval_total_ms, 3),
            llm_generation_ms=round(llm_ms, 3),
            grounding_ms=round(grounding_ms, 3),
            tts_ms=round(tts_ms, 3),
            total_e2e_ms=round(total_e2e_ms, 3),
        )

    def run_benchmark(
        self,
        queries: List[Dict[str, Any]],
        warmup_count: int = 5,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        delay_seconds: float = 0.0,
    ) -> BenchmarkReport:
        """Run warmups followed by full query benchmark, calculating percentiles.

        Args:
            queries: Query items to benchmark.
            warmup_count: Leading queries used only to prime caches.
            progress_cb: Optional progress callback.
            delay_seconds: Pause between queries. Firing cloud LLM/TTS requests
                back to back trips provider rate limits, and the retry plus
                backoff that follows is charged to the stage timing. That
                measures burst behaviour, not the latency a single user sees,
                so pace the run when reporting interactive numbers.
        """
        logger.info("Starting latency benchmark on %d queries (warmup: %d)...", len(queries), warmup_count)

        # Warm-up runs to ensure model caches & JIT compilation are primed
        for i in range(min(warmup_count, len(queries))):
            self.run_query(queries[i])
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        runs: List[LatencyStageMetrics] = []
        for idx, q in enumerate(queries):
            metric = self.run_query(q)
            runs.append(metric)
            if progress_cb is not None:
                progress_cb(idx + 1, len(queries))
            if delay_seconds > 0 and idx < len(queries) - 1:
                time.sleep(delay_seconds)

        # Compute percentiles for all stages
        stage_names = [
            "guardrail_ms",
            "embedding_ms",
            "faiss_lookup_ms",
            "rerank_ms",
            "retrieval_total_ms",
            "llm_generation_ms",
            "grounding_ms",
            "tts_ms",
            "total_e2e_ms",
        ]

        stage_stats: Dict[str, PercentileStat] = {}
        for stage in stage_names:
            vals = [getattr(r, stage) for r in runs]
            stage_stats[stage] = compute_percentiles(vals)

        total_p50 = stage_stats["total_e2e_ms"].p50
        sla_passed = total_p50 <= self.sla_target_ms

        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor() or "CPU",
        }

        return BenchmarkReport(
            total_queries=len(runs),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            system_info=system_info,
            stage_statistics=stage_stats,
            sla_target_ms=self.sla_target_ms,
            sla_passed=sla_passed,
            runs=runs,
            provider_modes=self._provider_modes(),
        )

    def _provider_modes(self) -> Dict[str, str]:
        """Record the concrete implementation behind each measured stage.

        Derived from the live objects rather than a caller-supplied flag, so a
        report cannot claim real providers while timing stubs.
        """

        def describe(component: Optional[Any]) -> str:
            if component is None:
                return "not configured"
            name = type(component).__name__
            return f"{name} (STUB - not representative)" if "Fake" in name else name

        return {
            "embedder": describe(getattr(self.orchestrator, "embedder", None)),
            "llm": describe(self.llm),
            "tts": describe(self.tts),
        }
