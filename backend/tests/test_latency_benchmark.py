"""Unit tests for the Latency Benchmark & Measurement Engine (Phase 5.9).

All tests run 100% offline with zero external network or model downloads.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock
import pytest

# Ensure repository root is on sys.path for evaluation package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.guardrails.pipeline import GuardrailPipeline
from app.llm.fake import FakeLLM
from app.tts.fake import FakeTTS
from evaluation.latency_benchmark import (
    BenchmarkReport,
    LatencyBenchmarkRunner,
    LatencyStageMetrics,
    PercentileStat,
    compute_percentiles,
)


def test_compute_percentiles_empty():
    stat = compute_percentiles([])
    assert stat.p50 == 0.0
    assert stat.p100 == 0.0
    assert stat.mean == 0.0


def test_compute_percentiles_single():
    stat = compute_percentiles([42.5])
    assert stat.p50 == 42.5
    assert stat.p70 == 42.5
    assert stat.p90 == 42.5
    assert stat.p100 == 42.5
    assert stat.mean == 42.5
    assert stat.min_val == 42.5
    assert stat.max_val == 42.5


def test_compute_percentiles_series():
    # 10 values from 10 to 100
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    stat = compute_percentiles(values)

    assert stat.p50 == 55.0  # Median of 1..10
    assert stat.p70 == 73.0
    assert stat.p100 == 100.0
    assert stat.min_val == 10.0
    assert stat.max_val == 100.0
    assert stat.mean == 55.0


def test_benchmark_runner_single_query():
    guardrail_mock = MagicMock(spec=GuardrailPipeline)
    guardrail_mock.check_input.return_value = MagicMock()

    orchestrator_mock = MagicMock()
    mock_ret_res = MagicMock()
    mock_ret_res.latencies_ms = {
        "embedding_ms": 5.2,
        "search_ms": 1.1,
        "rerank_ms": 0.4,
    }
    mock_ret_res.retrieved_chunks = []
    orchestrator_mock.retrieve.return_value = mock_ret_res

    llm = FakeLLM()
    grounding_mock = MagicMock()
    tts = FakeTTS()

    runner = LatencyBenchmarkRunner(
        guardrail_pipeline=guardrail_mock,
        orchestrator=orchestrator_mock,
        llm=llm,
        grounding_verifier=grounding_mock,
        tts=tts,
        sla_target_ms=200.0,
    )

    query_item = {"id": 1, "query": "What is the capital of Goa?", "lang": "en"}
    metrics = runner.run_query(query_item)

    assert isinstance(metrics, LatencyStageMetrics)
    assert metrics.query_id == 1
    assert metrics.query == "What is the capital of Goa?"
    assert metrics.embedding_ms == 5.2
    assert metrics.faiss_lookup_ms == 1.1
    assert metrics.rerank_ms == 0.4
    assert metrics.total_e2e_ms > 0.0


def test_benchmark_runner_full_suite():
    llm = FakeLLM()
    tts = FakeTTS()

    runner = LatencyBenchmarkRunner(
        guardrail_pipeline=None,
        orchestrator=None,
        llm=llm,
        grounding_verifier=None,
        tts=tts,
        sla_target_ms=200.0,
    )

    queries = [
        {"id": 1, "query": "Query 1", "lang": "en"},
        {"id": 2, "query": "Query 2", "lang": "hi"},
        {"id": 3, "query": "Query 3", "lang": "en"},
    ]

    report = runner.run_benchmark(queries, warmup_count=1)

    assert isinstance(report, BenchmarkReport)
    assert report.total_queries == 3
    assert "total_e2e_ms" in report.stage_statistics
    assert report.sla_passed is True

    # Test Markdown and JSON exports
    md_content = report.to_markdown()
    assert "# 🚀 Samvaad Voice RAG Latency Benchmark Report" in md_content
    assert "P50 (Median)" in md_content

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "report.json")
        md_path = os.path.join(tmpdir, "report.md")

        report.save(json_path, md_path)

        assert os.path.exists(json_path)
        assert os.path.exists(md_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_queries"] == 3
            assert data["sla_passed"] is True
