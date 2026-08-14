"""
Chunk Quality Benchmark (Phase 3.7).

Offline synthetic benchmark comparing all four chunking strategies on
measurable, intrinsic metrics. Does NOT use real MSMARCO-XI data.
Does NOT require network access, embeddings, or vector databases.
Does NOT claim retrieval quality from these metrics.

Metrics measured per strategy:
- source_passage_count: Number of input passages
- chunk_count: Total chunks produced
- input_characters: Total characters in source passages
- output_characters: Total characters in all chunks
- character_coverage: output_characters / input_characters
- empty_chunk_count: Chunks with empty or whitespace-only text
- duplicate_chunk_id_count: Number of duplicate chunk IDs
- avg_chunk_size_chars: Mean chunk character count
- min_chunk_size_chars: Minimum chunk character count
- max_chunk_size_chars: Maximum chunk character count
- avg_token_count: Mean token_count (when available)
- min_token_count: Minimum token_count (when available)
- max_token_count: Maximum token_count (when available)
- selected_passage_chunk_count: Chunks from is_selected=True passages
- generation_time_seconds: Wall-clock time for chunking
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.chunking.engine import ChunkingEngine
from app.chunking.models import Chunk, ChunkingStrategy
from app.chunking.tokenizer import TokenizerProtocol, create_default_tokenizer
from app.dataset.models import CanonicalPassage


# ---------------------------------------------------------------------------
# Benchmark result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StrategyBenchmarkResult:
    """Benchmark metrics for a single chunking strategy.

    All metrics are intrinsic (no retrieval quality claims).
    JSON-serialisable via to_dict().
    """

    strategy: str

    # Passage / chunk counts
    source_passage_count: int = 0
    chunk_count: int = 0

    # Character statistics
    input_characters: int = 0
    output_characters: int = 0
    character_coverage: float = 0.0

    # Quality flags
    empty_chunk_count: int = 0
    duplicate_chunk_id_count: int = 0

    # Chunk size distribution (characters)
    avg_chunk_size_chars: float = 0.0
    min_chunk_size_chars: int = 0
    max_chunk_size_chars: int = 0

    # Token statistics (when available)
    avg_token_count: float | None = None
    min_token_count: int | None = None
    max_token_count: int | None = None

    # Relevance
    selected_passage_chunk_count: int = 0

    # Timing
    generation_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "strategy": self.strategy,
            "source_passage_count": self.source_passage_count,
            "chunk_count": self.chunk_count,
            "input_characters": self.input_characters,
            "output_characters": self.output_characters,
            "character_coverage": round(self.character_coverage, 4),
            "empty_chunk_count": self.empty_chunk_count,
            "duplicate_chunk_id_count": self.duplicate_chunk_id_count,
            "avg_chunk_size_chars": round(self.avg_chunk_size_chars, 2),
            "min_chunk_size_chars": self.min_chunk_size_chars,
            "max_chunk_size_chars": self.max_chunk_size_chars,
            "avg_token_count": (
                round(self.avg_token_count, 2)
                if self.avg_token_count is not None
                else None
            ),
            "min_token_count": self.min_token_count,
            "max_token_count": self.max_token_count,
            "selected_passage_chunk_count": self.selected_passage_chunk_count,
            "generation_time_seconds": round(self.generation_time_seconds, 6),
        }


@dataclass
class BenchmarkReport:
    """Full benchmark report across all four strategies.

    JSON-serialisable via to_dict().
    """

    results: list[StrategyBenchmarkResult] = field(default_factory=list)
    total_passages: int = 0
    tokenizer_name: str = "none"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "total_passages": self.total_passages,
            "tokenizer_name": self.tokenizer_name,
            "results": [r.to_dict() for r in self.results],
        }

    def get_result(self, strategy: str) -> StrategyBenchmarkResult | None:
        """Get result for a specific strategy by name."""
        for r in self.results:
            if r.strategy == strategy:
                return r
        return None


# ---------------------------------------------------------------------------
# Synthetic fixture generators
# ---------------------------------------------------------------------------


def _make_canonical_passage(
    document_id: str,
    text: str,
    query_id: int = 1,
    passage_index: int = 0,
    target_lang: str = "hi",
    is_selected: bool = False,
    query: str = "benchmark query",
) -> CanonicalPassage:
    """Create a CanonicalPassage from a text string."""
    return CanonicalPassage(
        document_id=document_id,
        translated_passage=text,
        english_passage=text,
        query_id=query_id,
        passage_index=passage_index,
        target_lang=target_lang,
        source_lang="en",
        query=query,
        eng_query=query,
        is_selected=is_selected,
        query_type=None,
        answer=None,
        eng_answer=None,
    )


def create_synthetic_benchmark_passages(n: int = 50) -> list[CanonicalPassage]:
    """Create a diverse set of synthetic passages for benchmarking.

    Generates passages with varying lengths and languages to exercise
    all strategy branches of the adaptive chunker. All passages are
    synthetic — no real MSMARCO-XI data is used.

    Args:
        n: Total number of passages to generate (split across categories)

    Returns:
        List of CanonicalPassage instances
    """
    passages: list[CanonicalPassage] = []
    idx = 0

    # ---- Category 1: Short English passages (< 200 chars) ----
    short_texts = [
        "The Eiffel Tower is located in Paris, France.",
        "Python is a high-level programming language.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Sun is approximately 93 million miles from Earth.",
        "Photosynthesis converts sunlight into chemical energy.",
    ]
    for i, text in enumerate(short_texts[: max(1, n // 10)]):
        passages.append(
            _make_canonical_passage(
                document_id=f"short_{i}",
                text=text,
                query_id=1000 + i,
                passage_index=i,
                is_selected=(i % 2 == 0),
            )
        )
        idx += 1

    # ---- Category 2: Medium English passages (200–600 chars) ----
    medium_template = (
        "The {topic} is a significant subject in modern research. "
        "Scientists have studied it extensively over the past decade. "
        "Recent findings suggest that {topic} plays a key role in "
        "understanding natural phenomena. Further investigation is needed "
        "to fully characterise its implications."
    )
    topics = ["climate change", "machine learning", "quantum physics",
               "genomics", "renewable energy", "materials science"]
    for i, topic in enumerate(topics[: max(1, n // 5)]):
        text = medium_template.format(topic=topic)
        passages.append(
            _make_canonical_passage(
                document_id=f"medium_{i}",
                text=text,
                query_id=2000 + i,
                passage_index=i,
                is_selected=(i % 3 == 0),
            )
        )
        idx += 1

    # ---- Category 3: Long English passages (600+ chars) ----
    long_template = (
        "This passage discusses {topic} in considerable depth. "
        "{topic} has a rich history dating back to early scientific inquiry. "
        "Modern approaches to {topic} leverage advanced computational methods. "
        "Researchers have found that {topic} interacts with several related domains. "
        "The practical applications of {topic} span industry and academia alike. "
        "Future directions in {topic} research include interdisciplinary collaboration. "
        "In summary, {topic} represents a frontier of ongoing investigation. "
    ) * 3  # ~1400 chars
    long_topics = ["artificial intelligence", "bioinformatics", "astrophysics",
                   "materials science", "computational linguistics"]
    for i, topic in enumerate(long_topics[: max(1, n // 5)]):
        text = long_template.format(topic=topic)
        passages.append(
            _make_canonical_passage(
                document_id=f"long_{i}",
                text=text,
                query_id=3000 + i,
                passage_index=i,
                is_selected=(i % 2 == 0),
            )
        )
        idx += 1

    # ---- Category 4: Hindi passages ----
    hindi_texts = [
        "भारत एशिया का एक प्रमुख देश है। यह विश्व का सातवाँ सबसे बड़ा देश है।",
        (
            "हिंदी भारत की राजभाषा है और इसे करोड़ों लोग बोलते हैं। "
            "यह इंडो-आर्यन भाषा परिवार से संबंधित है और देवनागरी लिपि में लिखी जाती है। "
            "हिंदी साहित्य की परंपरा अत्यंत समृद्ध है।"
        ),
        (
            "विज्ञान और प्रौद्योगिकी के क्षेत्र में भारत ने उल्लेखनीय प्रगति की है। "
            "अंतरिक्ष अनुसंधान, सूचना प्रौद्योगिकी, और जैव-प्रौद्योगिकी में "
            "भारतीय वैज्ञानिकों का महत्वपूर्ण योगदान रहा है। "
            "चंद्रयान और मंगलयान जैसे अभियान इसके प्रमाण हैं।"
        ),
    ]
    for i, text in enumerate(hindi_texts[: max(1, n // 10)]):
        passages.append(
            _make_canonical_passage(
                document_id=f"hindi_{i}",
                text=text,
                query_id=4000 + i,
                passage_index=i,
                target_lang="hi",
                is_selected=(i == 0),
                query="यह क्या है?",
            )
        )
        idx += 1

    # ---- Category 5: Mixed-language passages ----
    mixed_texts = [
        "The concept of artificial intelligence (AI) — or कृत्रिम बुद्धिमत्ता — "
        "has transformed modern technology. From voice assistants to autonomous vehicles, "
        "AI applications are becoming increasingly prevalent in our daily lives.",
        "Machine learning algorithms process vast amounts of data. "
        "डेटा विज्ञान और मशीन लर्निंग आज के युग में अत्यंत महत्वपूर्ण हैं। "
        "These techniques power recommendation systems, fraud detection, and more.",
    ]
    for i, text in enumerate(mixed_texts[: max(1, n // 10)]):
        passages.append(
            _make_canonical_passage(
                document_id=f"mixed_{i}",
                text=text,
                query_id=5000 + i,
                passage_index=i,
                is_selected=(i % 2 == 0),
            )
        )
        idx += 1

    # ---- Category 6: Very long passages for token chunking stress ----
    very_long_text = (
        "In the field of natural language processing, researchers have made "
        "substantial progress over the past two decades. Early approaches relied "
        "on hand-crafted rules and statistical models. The advent of neural networks "
        "and particularly transformer architectures has revolutionised the field. "
        "Models such as BERT, GPT, and their successors have achieved state-of-the-art "
        "performance on a wide range of language understanding tasks. These advances "
        "have enabled practical applications including machine translation, question "
        "answering, text summarisation, and sentiment analysis. Despite these successes, "
        "challenges remain in areas such as multilingual understanding, low-resource "
        "languages, and interpretability of model decisions. "
    ) * 4
    for i in range(min(3, max(1, n // 15))):
        passages.append(
            _make_canonical_passage(
                document_id=f"vlong_{i}",
                text=very_long_text,
                query_id=6000 + i,
                passage_index=i,
                is_selected=(i == 0),
            )
        )
        idx += 1

    return passages


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _compute_metrics(
    chunks: list[Chunk],
    passages: list[CanonicalPassage],
    strategy: ChunkingStrategy,
    elapsed: float,
) -> StrategyBenchmarkResult:
    """Compute intrinsic metrics from a list of chunks.

    Args:
        chunks: All chunks produced by the strategy
        passages: The source passages
        strategy: The chunking strategy used
        elapsed: Wall-clock time in seconds

    Returns:
        StrategyBenchmarkResult with all metrics populated
    """
    result = StrategyBenchmarkResult(strategy=strategy.value)
    result.source_passage_count = len(passages)
    result.chunk_count = len(chunks)
    result.generation_time_seconds = elapsed

    # Input characters
    result.input_characters = sum(len(p.translated_passage) for p in passages)

    if not chunks:
        return result

    # Output characters
    char_sizes = [len(c.chunk_text) for c in chunks]
    result.output_characters = sum(char_sizes)
    result.avg_chunk_size_chars = result.output_characters / len(chunks)
    result.min_chunk_size_chars = min(char_sizes)
    result.max_chunk_size_chars = max(char_sizes)

    # Character coverage
    if result.input_characters > 0:
        result.character_coverage = result.output_characters / result.input_characters
    else:
        result.character_coverage = 0.0

    # Empty chunks
    result.empty_chunk_count = sum(1 for c in chunks if not c.chunk_text.strip())

    # Duplicate chunk IDs
    all_ids = [c.chunk_id for c in chunks]
    result.duplicate_chunk_id_count = len(all_ids) - len(set(all_ids))

    # Token statistics (only where token_count is set)
    token_counts = [c.token_count for c in chunks if c.token_count is not None]
    if token_counts:
        result.avg_token_count = sum(token_counts) / len(token_counts)
        result.min_token_count = min(token_counts)
        result.max_token_count = max(token_counts)

    # Selected passage chunks
    selected_doc_ids = {p.document_id for p in passages if p.is_selected}
    result.selected_passage_chunk_count = sum(
        1 for c in chunks if c.document_id in selected_doc_ids
    )

    return result


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    passages: list[CanonicalPassage] | None = None,
    tokenizer: TokenizerProtocol | None = None,
    token_chunk_size: int = 64,
    token_overlap: int = 16,
    sentence_chunk_size: int = 3,
    sentence_overlap: int = 1,
    adaptive_short_max: int = 500,
    adaptive_medium_max: int = 2000,
) -> BenchmarkReport:
    """Run the chunk quality benchmark across all four strategies.

    Uses only offline synthetic passages (or provided passages).
    No network access. No real MSMARCO-XI data.

    Args:
        passages: Passages to benchmark on. If None, uses
                  create_synthetic_benchmark_passages().
        tokenizer: Optional tokenizer for TOKEN and ADAPTIVE strategies.
                   If None, create_default_tokenizer() is used (which may
                   fall back to SimpleWhitespaceTokenizer offline).
        token_chunk_size: Token window size for TOKEN strategy.
        token_overlap: Token overlap for TOKEN strategy.
        sentence_chunk_size: Sentences per chunk for SENTENCE strategy.
        sentence_overlap: Sentence overlap for SENTENCE strategy.
        adaptive_short_max: Short passage threshold for ADAPTIVE strategy.
        adaptive_medium_max: Medium passage threshold for ADAPTIVE strategy.

    Returns:
        BenchmarkReport with results for all four strategies.
    """
    if passages is None:
        passages = create_synthetic_benchmark_passages()

    if tokenizer is None:
        tokenizer = create_default_tokenizer()

    tokenizer_name = repr(tokenizer)

    # Build engine
    engine = ChunkingEngine(
        tokenizer=tokenizer,
        sentence_chunk_size=sentence_chunk_size,
        sentence_overlap=sentence_overlap,
        token_chunk_size=token_chunk_size,
        token_overlap=token_overlap,
        adaptive_short_max=adaptive_short_max,
        adaptive_medium_max=adaptive_medium_max,
    )

    report = BenchmarkReport(
        total_passages=len(passages),
        tokenizer_name=tokenizer_name,
    )

    # Run each strategy
    for strategy in [
        ChunkingStrategy.PASSAGE,
        ChunkingStrategy.SENTENCE,
        ChunkingStrategy.TOKEN,
        ChunkingStrategy.ADAPTIVE,
    ]:
        t0 = time.perf_counter()
        try:
            chunks = engine.chunk_batch(passages, strategy)
        except Exception as exc:
            # Record the error but continue benchmarking other strategies
            chunks = []
            result = StrategyBenchmarkResult(strategy=strategy.value)
            result.source_passage_count = len(passages)
            result.generation_time_seconds = time.perf_counter() - t0
            report.results.append(result)
            continue
        elapsed = time.perf_counter() - t0

        result = _compute_metrics(chunks, passages, strategy, elapsed)
        report.results.append(result)

    return report


# ---------------------------------------------------------------------------
# Human-readable formatting
# ---------------------------------------------------------------------------


def format_benchmark_report(report: BenchmarkReport) -> str:
    """Format a BenchmarkReport as a human-readable table.

    Args:
        report: The benchmark report to format

    Returns:
        Multi-line string table
    """
    lines = [
        "=" * 70,
        "Chunk Quality Benchmark Report",
        f"  Passages: {report.total_passages}",
        f"  Tokenizer: {report.tokenizer_name}",
        "=" * 70,
    ]

    header = (
        f"{'Strategy':<12} {'Chunks':>7} {'Cov%':>6} "
        f"{'AvgChars':>9} {'Empty':>6} {'DupIDs':>7} "
        f"{'AvgTok':>8} {'TimeSec':>9}"
    )
    lines.append(header)
    lines.append("-" * 70)

    for r in report.results:
        cov_pct = round(r.character_coverage * 100, 1)
        avg_tok = f"{r.avg_token_count:.1f}" if r.avg_token_count is not None else "N/A"
        row = (
            f"{r.strategy:<12} {r.chunk_count:>7} {cov_pct:>5.1f}% "
            f"{r.avg_chunk_size_chars:>9.1f} {r.empty_chunk_count:>6} "
            f"{r.duplicate_chunk_id_count:>7} {avg_tok:>8} "
            f"{r.generation_time_seconds:>9.4f}"
        )
        lines.append(row)

    lines.append("=" * 70)
    return "\n".join(lines)
