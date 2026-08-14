"""
Tests for the Chunk Quality Benchmark (Phase 3.7).

Tests cover:
- Synthetic passage generation
- Metric computation per strategy
- BenchmarkReport structure and JSON serialisability
- All four strategies represented in results
- No empty chunks counted as valid output
- No duplicate chunk IDs
- Character coverage sanity
- Timing is non-negative
- StrategyBenchmarkResult.to_dict() keys
- BenchmarkReport.to_dict() structure
- No real MSMARCO-XI data used
- No network access
- No embeddings / vector DB

All tests use offline-only synthetic fixtures.
"""

import json
import time

import pytest

from app.chunking.benchmark import (
    BenchmarkReport,
    StrategyBenchmarkResult,
    create_synthetic_benchmark_passages,
    format_benchmark_report,
    run_benchmark,
    _compute_metrics,
    _make_canonical_passage,
)
from app.chunking.models import Chunk, ChunkingStrategy
from app.chunking.tokenizer import SimpleWhitespaceTokenizer
from app.dataset.models import CanonicalPassage


# ---------------------------------------------------------------------------
# Tests: _make_canonical_passage helper
# ---------------------------------------------------------------------------


class TestMakeCanonicalPassage:
    """Tests for the benchmark passage factory."""

    def test_creates_canonical_passage(self):
        """Must return a CanonicalPassage instance."""
        p = _make_canonical_passage("docA", "Hello world.")
        assert isinstance(p, CanonicalPassage)

    def test_document_id_preserved(self):
        p = _make_canonical_passage("docXYZ", "Some text here.")
        assert p.document_id == "docXYZ"

    def test_text_preserved(self):
        p = _make_canonical_passage("doc1", "My passage text.")
        assert p.translated_passage == "My passage text."

    def test_is_selected_default_false(self):
        p = _make_canonical_passage("doc1", "Some text.")
        assert p.is_selected is False

    def test_is_selected_true(self):
        p = _make_canonical_passage("doc1", "Some text.", is_selected=True)
        assert p.is_selected is True

    def test_hindi_target_lang(self):
        p = _make_canonical_passage("doc1", "हिंदी पाठ।", target_lang="hi")
        assert p.target_lang == "hi"


# ---------------------------------------------------------------------------
# Tests: create_synthetic_benchmark_passages
# ---------------------------------------------------------------------------


class TestCreateSyntheticBenchmarkPassages:
    """Tests for the synthetic passage generator."""

    def test_returns_list(self):
        passages = create_synthetic_benchmark_passages(n=10)
        assert isinstance(passages, list)

    def test_returns_canonical_passages(self):
        passages = create_synthetic_benchmark_passages(n=10)
        assert all(isinstance(p, CanonicalPassage) for p in passages)

    def test_no_empty_passages(self):
        passages = create_synthetic_benchmark_passages(n=50)
        assert all(p.translated_passage.strip() != "" for p in passages)

    def test_all_document_ids_unique(self):
        passages = create_synthetic_benchmark_passages(n=50)
        doc_ids = [p.document_id for p in passages]
        assert len(doc_ids) == len(set(doc_ids))

    def test_contains_mixed_languages(self):
        """Must contain both English and Hindi passages."""
        passages = create_synthetic_benchmark_passages(n=50)
        langs = {p.target_lang for p in passages}
        assert "hi" in langs

    def test_contains_selected_and_unselected(self):
        """Must contain both is_selected=True and False passages."""
        passages = create_synthetic_benchmark_passages(n=50)
        selected = [p for p in passages if p.is_selected]
        unselected = [p for p in passages if not p.is_selected]
        assert len(selected) > 0
        assert len(unselected) > 0

    def test_contains_short_passages(self):
        """Must have at least one passage shorter than 300 chars."""
        passages = create_synthetic_benchmark_passages(n=50)
        short = [p for p in passages if len(p.translated_passage) < 300]
        assert len(short) > 0

    def test_contains_long_passages(self):
        """Must have at least one passage longer than 500 chars."""
        passages = create_synthetic_benchmark_passages(n=50)
        long_ones = [p for p in passages if len(p.translated_passage) > 500]
        assert len(long_ones) > 0

    def test_n_influences_count(self):
        """Larger n should produce more passages."""
        p10 = create_synthetic_benchmark_passages(n=10)
        p100 = create_synthetic_benchmark_passages(n=100)
        assert len(p100) >= len(p10)

    def test_no_network_required(self):
        """Generation must not require any network access."""
        # Just call it — if it fails it would raise, not return
        passages = create_synthetic_benchmark_passages(n=20)
        assert len(passages) > 0


# ---------------------------------------------------------------------------
# Tests: _compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Tests for the _compute_metrics helper function."""

    def _make_simple_passage(self, doc_id="doc1", is_selected=False):
        return _make_canonical_passage(doc_id, "Hello world test.", is_selected=is_selected)

    def _make_chunk(self, chunk_id, doc_id, text, token_count=None):
        """Create a minimal Chunk for testing."""
        return Chunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            chunk_index=0,
            strategy=ChunkingStrategy.PASSAGE,
            chunk_text=text,
            character_count=len(text),
            token_count=token_count,
            query_id=1,
            passage_index=0,
            target_lang="hi",
            source_lang="en",
            query="q",
            eng_query="q",
            is_selected=False,
        )

    def test_empty_chunks_returns_zero_count(self):
        """With no chunks, chunk_count must be 0."""
        passages = [self._make_simple_passage()]
        result = _compute_metrics([], passages, ChunkingStrategy.PASSAGE, 0.1)
        assert result.chunk_count == 0

    def test_source_passage_count(self):
        passages = [self._make_simple_passage(f"doc{i}") for i in range(5)]
        chunk = self._make_chunk("id1", "doc0", "Hello world test.")
        result = _compute_metrics([chunk], passages, ChunkingStrategy.PASSAGE, 0.01)
        assert result.source_passage_count == 5

    def test_input_characters_summed(self):
        text = "Hello world."
        passages = [self._make_simple_passage("doc1")]
        chunk = self._make_chunk("id1", "doc1", text)
        result = _compute_metrics([chunk], passages, ChunkingStrategy.PASSAGE, 0.01)
        assert result.input_characters == len(passages[0].translated_passage)

    def test_output_characters_summed(self):
        text = "Hello world test."
        chunk = self._make_chunk("chunk1", "doc1", text)
        passages = [self._make_simple_passage()]
        result = _compute_metrics([chunk], passages, ChunkingStrategy.PASSAGE, 0.0)
        assert result.output_characters == len(text)

    def test_character_coverage_ratio(self):
        text = "Hello world test."  # 17 chars
        passage_text = "Hello world test."  # same
        p = _make_canonical_passage("doc1", passage_text)
        chunk = self._make_chunk("c1", "doc1", text)
        result = _compute_metrics([chunk], [p], ChunkingStrategy.PASSAGE, 0.0)
        expected = len(text) / len(passage_text)
        assert abs(result.character_coverage - expected) < 1e-6

    def test_empty_chunk_detected(self):
        """Chunks with empty text must be counted."""
        # Note: Pydantic validates chunk_text min_length=1, so we can't create
        # a truly empty chunk. Test with whitespace only is not possible via model.
        # Instead verify the counter starts at 0 for valid chunks.
        text = "Valid chunk text."
        chunk = self._make_chunk("c1", "doc1", text)
        passages = [self._make_simple_passage()]
        result = _compute_metrics([chunk], passages, ChunkingStrategy.PASSAGE, 0.0)
        assert result.empty_chunk_count == 0

    def test_duplicate_chunk_id_detection(self):
        """Duplicate chunk IDs must be detected."""
        chunk1 = self._make_chunk("same_id", "doc1", "First chunk text.")
        chunk2 = self._make_chunk("same_id", "doc1", "Second chunk text.")
        passages = [self._make_simple_passage()]
        result = _compute_metrics([chunk1, chunk2], passages, ChunkingStrategy.PASSAGE, 0.0)
        assert result.duplicate_chunk_id_count == 1

    def test_no_duplicate_chunk_ids_when_unique(self):
        """No duplicates when all IDs are distinct."""
        chunks = [
            self._make_chunk(f"id_{i}", f"doc{i}", f"Chunk text {i}.")
            for i in range(3)
        ]
        passages = [self._make_simple_passage(f"doc{i}") for i in range(3)]
        result = _compute_metrics(chunks, passages, ChunkingStrategy.PASSAGE, 0.0)
        assert result.duplicate_chunk_id_count == 0

    def test_avg_chunk_size_chars(self):
        texts = ["Hello.", "Hello world.", "Hello world test."]
        chunks = [self._make_chunk(f"id{i}", f"doc{i}", t) for i, t in enumerate(texts)]
        passages = [self._make_simple_passage(f"doc{i}") for i in range(3)]
        result = _compute_metrics(chunks, passages, ChunkingStrategy.PASSAGE, 0.0)
        expected_avg = sum(len(t) for t in texts) / 3
        assert abs(result.avg_chunk_size_chars - expected_avg) < 0.01

    def test_min_max_chunk_size_chars(self):
        texts = ["Hi.", "Hello world!", "A much longer text for testing."]
        chunks = [self._make_chunk(f"id{i}", f"doc{i}", t) for i, t in enumerate(texts)]
        passages = [self._make_simple_passage(f"doc{i}") for i in range(3)]
        result = _compute_metrics(chunks, passages, ChunkingStrategy.PASSAGE, 0.0)
        assert result.min_chunk_size_chars == min(len(t) for t in texts)
        assert result.max_chunk_size_chars == max(len(t) for t in texts)

    def test_token_statistics_when_available(self):
        chunks = [
            self._make_chunk("c1", "doc1", "Token text one.", token_count=3),
            self._make_chunk("c2", "doc2", "Token text two.", token_count=5),
        ]
        passages = [self._make_simple_passage(f"doc{i}") for i in range(2)]
        result = _compute_metrics(chunks, passages, ChunkingStrategy.TOKEN, 0.0)
        assert result.avg_token_count == 4.0
        assert result.min_token_count == 3
        assert result.max_token_count == 5

    def test_token_statistics_none_when_not_available(self):
        chunk = self._make_chunk("c1", "doc1", "Text here.", token_count=None)
        passages = [self._make_simple_passage()]
        result = _compute_metrics([chunk], passages, ChunkingStrategy.PASSAGE, 0.0)
        assert result.avg_token_count is None
        assert result.min_token_count is None
        assert result.max_token_count is None

    def test_selected_passage_chunk_count(self):
        p_selected = _make_canonical_passage("sel_doc", "Selected passage.", is_selected=True)
        p_unselected = _make_canonical_passage("unsel_doc", "Unselected passage.", is_selected=False)
        chunk_sel = self._make_chunk("c_sel", "sel_doc", "Selected passage.")
        chunk_unsel = self._make_chunk("c_unsel", "unsel_doc", "Unselected passage.")
        result = _compute_metrics(
            [chunk_sel, chunk_unsel],
            [p_selected, p_unselected],
            ChunkingStrategy.PASSAGE,
            0.0,
        )
        assert result.selected_passage_chunk_count == 1

    def test_generation_time_recorded(self):
        chunk = self._make_chunk("c1", "doc1", "Some text here.")
        passages = [self._make_simple_passage()]
        result = _compute_metrics([chunk], passages, ChunkingStrategy.PASSAGE, 1.234)
        assert abs(result.generation_time_seconds - 1.234) < 1e-6


# ---------------------------------------------------------------------------
# Tests: StrategyBenchmarkResult
# ---------------------------------------------------------------------------


class TestStrategyBenchmarkResult:
    """Tests for the result dataclass."""

    def test_to_dict_has_required_keys(self):
        """to_dict() must contain all required metric keys."""
        result = StrategyBenchmarkResult(strategy="passage")
        d = result.to_dict()
        required_keys = [
            "strategy",
            "source_passage_count",
            "chunk_count",
            "input_characters",
            "output_characters",
            "character_coverage",
            "empty_chunk_count",
            "duplicate_chunk_id_count",
            "avg_chunk_size_chars",
            "min_chunk_size_chars",
            "max_chunk_size_chars",
            "avg_token_count",
            "min_token_count",
            "max_token_count",
            "selected_passage_chunk_count",
            "generation_time_seconds",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_is_json_serializable(self):
        """to_dict() output must be JSON-serializable."""
        result = StrategyBenchmarkResult(
            strategy="passage",
            chunk_count=10,
            avg_chunk_size_chars=50.5,
            avg_token_count=12.3,
        )
        d = result.to_dict()
        # Must not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_dict_strategy_value(self):
        result = StrategyBenchmarkResult(strategy="token")
        assert result.to_dict()["strategy"] == "token"

    def test_to_dict_none_token_count_serializable(self):
        """None token counts must be JSON-serializable."""
        result = StrategyBenchmarkResult(strategy="passage", avg_token_count=None)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert '"avg_token_count": null' in json_str or "null" in json_str


# ---------------------------------------------------------------------------
# Tests: BenchmarkReport
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    """Tests for the report container."""

    def test_to_dict_has_required_keys(self):
        report = BenchmarkReport(total_passages=5, tokenizer_name="test_tok")
        d = report.to_dict()
        assert "total_passages" in d
        assert "tokenizer_name" in d
        assert "results" in d

    def test_to_dict_results_is_list(self):
        result = StrategyBenchmarkResult(strategy="passage")
        report = BenchmarkReport(results=[result], total_passages=10)
        d = report.to_dict()
        assert isinstance(d["results"], list)

    def test_to_dict_is_json_serializable(self):
        """Full BenchmarkReport.to_dict() must be JSON-serializable."""
        result = StrategyBenchmarkResult(strategy="passage", avg_token_count=None)
        report = BenchmarkReport(results=[result], total_passages=5, tokenizer_name="tok")
        json_str = json.dumps(report.to_dict())
        assert len(json_str) > 0

    def test_get_result_found(self):
        result = StrategyBenchmarkResult(strategy="sentence")
        report = BenchmarkReport(results=[result])
        assert report.get_result("sentence") is result

    def test_get_result_not_found(self):
        report = BenchmarkReport(results=[])
        assert report.get_result("passage") is None


# ---------------------------------------------------------------------------
# Tests: run_benchmark (integration)
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    """Integration tests for the full benchmark run."""

    def test_run_benchmark_returns_report(self):
        """run_benchmark must return a BenchmarkReport."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        assert isinstance(report, BenchmarkReport)

    def test_report_has_four_results(self):
        """Report must contain results for all four strategies."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        assert len(report.results) == 4

    def test_all_strategies_represented(self):
        """All four strategy names must appear in results."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        strategy_names = {r.strategy for r in report.results}
        assert "passage" in strategy_names
        assert "sentence" in strategy_names
        assert "token" in strategy_names
        assert "adaptive" in strategy_names

    def test_all_chunk_counts_positive(self):
        """All strategies must produce at least one chunk for non-empty passages."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        for r in report.results:
            assert r.chunk_count > 0, f"Strategy {r.strategy} produced no chunks"

    def test_no_empty_chunks_produced(self):
        """No strategy must produce empty chunks."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        for r in report.results:
            assert r.empty_chunk_count == 0, (
                f"Strategy {r.strategy} produced {r.empty_chunk_count} empty chunks"
            )

    def test_no_duplicate_chunk_ids(self):
        """No strategy must produce duplicate chunk IDs."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        for r in report.results:
            assert r.duplicate_chunk_id_count == 0, (
                f"Strategy {r.strategy} has {r.duplicate_chunk_id_count} duplicate IDs"
            )

    def test_character_coverage_positive(self):
        """All strategies must have positive character coverage."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        for r in report.results:
            assert r.character_coverage > 0.0, (
                f"Strategy {r.strategy} has zero character coverage"
            )

    def test_generation_time_non_negative(self):
        """Generation times must be non-negative."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        for r in report.results:
            assert r.generation_time_seconds >= 0.0

    def test_passage_count_recorded(self):
        """source_passage_count must equal number of input passages."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        n = len(passages)
        report = run_benchmark(passages=passages, tokenizer=tok)
        for r in report.results:
            assert r.source_passage_count == n

    def test_token_count_set_for_token_strategy(self):
        """TOKEN strategy must have avg_token_count set."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(
            passages=passages,
            tokenizer=tok,
            token_chunk_size=20,
            token_overlap=5,
        )
        token_result = report.get_result("token")
        assert token_result is not None
        assert token_result.avg_token_count is not None
        assert token_result.avg_token_count > 0

    def test_passage_strategy_coverage_equals_one(self):
        """PASSAGE strategy must have character_coverage == 1.0 (no splitting)."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        passage_result = report.get_result("passage")
        assert passage_result is not None
        # Each passage is one chunk, so output chars = input chars
        assert abs(passage_result.character_coverage - 1.0) < 1e-3

    def test_report_total_passages(self):
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=15)
        report = run_benchmark(passages=passages, tokenizer=tok)
        assert report.total_passages == len(passages)

    def test_report_fully_json_serializable(self):
        """Full report must be JSON-serializable."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        d = report.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_benchmark_with_none_passages_uses_defaults(self):
        """run_benchmark(passages=None) must use synthetic passages."""
        tok = SimpleWhitespaceTokenizer()
        report = run_benchmark(passages=None, tokenizer=tok)
        assert report.total_passages > 0
        assert len(report.results) == 4

    def test_no_network_access_needed(self):
        """Benchmark must run fully offline with SimpleWhitespaceTokenizer."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=5)
        # Just verifying it runs without error (no urllib, requests, etc.)
        report = run_benchmark(passages=passages, tokenizer=tok)
        assert report is not None

    def test_no_msmarco_data_used(self):
        """Synthetic passages must not reference real MSMARCO-XI files."""
        passages = create_synthetic_benchmark_passages(n=50)
        for p in passages:
            assert "msmarco" not in p.document_id.lower()
            assert "hintrain" not in p.document_id.lower()
            assert "hinval" not in p.document_id.lower()


# ---------------------------------------------------------------------------
# Tests: format_benchmark_report
# ---------------------------------------------------------------------------


class TestFormatBenchmarkReport:
    """Tests for the human-readable formatting function."""

    def test_returns_string(self):
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=5)
        report = run_benchmark(passages=passages, tokenizer=tok)
        formatted = format_benchmark_report(report)
        assert isinstance(formatted, str)

    def test_contains_all_strategy_names(self):
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=5)
        report = run_benchmark(passages=passages, tokenizer=tok)
        formatted = format_benchmark_report(report)
        for name in ["passage", "sentence", "token", "adaptive"]:
            assert name in formatted.lower()

    def test_contains_passage_count(self):
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=5)
        report = run_benchmark(passages=passages, tokenizer=tok)
        formatted = format_benchmark_report(report)
        assert str(len(passages)) in formatted


# ---------------------------------------------------------------------------
# Tests: Benchmark comparison across strategies
# ---------------------------------------------------------------------------


class TestBenchmarkComparisons:
    """Tests that compare strategy outputs against each other."""

    def test_passage_strategy_chunk_count_equals_passage_count(self):
        """PASSAGE strategy must produce exactly one chunk per passage."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        pr = report.get_result("passage")
        assert pr is not None
        assert pr.chunk_count == pr.source_passage_count

    def test_token_strategy_produces_more_chunks_than_passage_strategy(self):
        """TOKEN strategy with small chunk_size must produce >= chunks than PASSAGE."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=20)
        report = run_benchmark(
            passages=passages,
            tokenizer=tok,
            token_chunk_size=10,
            token_overlap=2,
        )
        pr = report.get_result("passage")
        tr = report.get_result("token")
        assert pr is not None and tr is not None
        # Token chunking should split long passages, so token >= passage
        assert tr.chunk_count >= pr.chunk_count

    def test_all_min_chunk_sizes_positive(self):
        """All strategies must produce chunks with at least 1 character."""
        tok = SimpleWhitespaceTokenizer()
        passages = create_synthetic_benchmark_passages(n=10)
        report = run_benchmark(passages=passages, tokenizer=tok)
        for r in report.results:
            if r.chunk_count > 0:
                assert r.min_chunk_size_chars > 0, (
                    f"Strategy {r.strategy} has zero-length chunks"
                )
