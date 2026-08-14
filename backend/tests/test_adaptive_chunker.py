"""
Tests for AdaptiveChunker (Phase 3.5).

Tests the rule-based adaptive strategy selection:
- short passages → PassageChunker
- medium passages with sentence structure → SentenceChunker
- long passages or very long sentences → TokenChunker

All tests use tiny synthetic CanonicalPassage fixtures.
No real MSMARCO-XI data. No network access. No LLM/embeddings.
"""

import pytest

from app.chunking.adaptive_chunker import AdaptiveChunker
from app.chunking.models import Chunk, ChunkingStrategy
from app.chunking.tokenizer import SimpleWhitespaceTokenizer
from app.dataset.models import CanonicalPassage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def create_test_passage(
    document_id: str = "doc1",
    translated_passage: str = "This is a test passage.",
    english_passage: str = "This is a test passage.",
    query_id: int = 1,
    passage_index: int = 0,
    target_lang: str = "hi",
    source_lang: str = "en",
    query: str = "test query",
    eng_query: str = "test query",
    is_selected: bool = True,
    query_type: str | None = None,
    answer: str | None = None,
    eng_answer: str | None = None,
) -> CanonicalPassage:
    """Helper to create synthetic CanonicalPassage for tests."""
    return CanonicalPassage(
        document_id=document_id,
        translated_passage=translated_passage,
        english_passage=english_passage,
        query_id=query_id,
        passage_index=passage_index,
        target_lang=target_lang,
        source_lang=source_lang,
        query=query,
        eng_query=eng_query,
        is_selected=is_selected,
        query_type=query_type,
        answer=answer,
        eng_answer=eng_answer,
    )


def make_short_passage(document_id: str = "doc1") -> CanonicalPassage:
    """Passage clearly below short_passage_max_chars=500."""
    return create_test_passage(
        document_id=document_id,
        translated_passage="This is a short sentence.",
    )


def make_medium_passage(document_id: str = "doc1") -> CanonicalPassage:
    """Passage between short (500) and medium (2000) with multiple sentences."""
    # ~700 chars, 7 sentences
    sentences = ["This is sentence number {i}. ".format(i=i) * 5 for i in range(7)]
    text = " ".join(sentences)
    # Trim to a known range
    text = "Sentence one is here. Sentence two is here as well. Sentence three completes this. " * 4
    return create_test_passage(document_id=document_id, translated_passage=text)


def make_long_passage(document_id: str = "doc1") -> CanonicalPassage:
    """Passage clearly above medium_passage_max_chars=2000."""
    text = ("This is a long sentence that contributes to the passage length. " * 40)
    return create_test_passage(document_id=document_id, translated_passage=text)


def make_passage_with_very_long_sentence(document_id: str = "doc1") -> CanonicalPassage:
    """Passage containing a single very long sentence (no periods)."""
    # A single sentence longer than long_sentence_threshold=500 chars
    text = "a " * 300  # 600 words-ish, no periods
    return create_test_passage(document_id=document_id, translated_passage=text)


# ---------------------------------------------------------------------------
# Tests: Configuration validation
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerConfiguration:
    """Test AdaptiveChunker configuration validation."""

    def test_default_configuration(self):
        """Default parameters must be accepted."""
        chunker = AdaptiveChunker()
        assert chunker.short_passage_max_chars == 500
        assert chunker.medium_passage_max_chars == 2000
        assert chunker.long_sentence_threshold == 500

    def test_custom_valid_configuration(self):
        """Custom valid configuration must be accepted."""
        chunker = AdaptiveChunker(
            short_passage_max_chars=300,
            medium_passage_max_chars=1000,
            long_sentence_threshold=200,
        )
        assert chunker.short_passage_max_chars == 300
        assert chunker.medium_passage_max_chars == 1000

    def test_short_max_zero_raises(self):
        """short_passage_max_chars=0 must raise ValueError."""
        with pytest.raises(ValueError, match="short_passage_max_chars must be positive"):
            AdaptiveChunker(short_passage_max_chars=0)

    def test_short_max_negative_raises(self):
        """Negative short_passage_max_chars must raise ValueError."""
        with pytest.raises(ValueError, match="short_passage_max_chars must be positive"):
            AdaptiveChunker(short_passage_max_chars=-1)

    def test_medium_max_less_than_short_raises(self):
        """medium_passage_max_chars <= short_passage_max_chars must raise."""
        with pytest.raises(ValueError, match="must be greater than short_passage_max_chars"):
            AdaptiveChunker(short_passage_max_chars=500, medium_passage_max_chars=300)

    def test_medium_max_equal_short_raises(self):
        """medium_passage_max_chars == short_passage_max_chars must raise."""
        with pytest.raises(ValueError, match="must be greater than short_passage_max_chars"):
            AdaptiveChunker(short_passage_max_chars=500, medium_passage_max_chars=500)

    def test_long_sentence_threshold_zero_raises(self):
        """long_sentence_threshold=0 must raise ValueError."""
        with pytest.raises(ValueError, match="long_sentence_threshold must be positive"):
            AdaptiveChunker(long_sentence_threshold=0)

    def test_long_sentence_threshold_negative_raises(self):
        """Negative long_sentence_threshold must raise ValueError."""
        with pytest.raises(ValueError, match="long_sentence_threshold must be positive"):
            AdaptiveChunker(long_sentence_threshold=-100)


# ---------------------------------------------------------------------------
# Tests: Strategy selection — short passages
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerShortPassages:
    """Test strategy selection for short passages."""

    def test_short_passage_returns_passage_strategy(self):
        """Short passage must be kept whole (passage strategy)."""
        chunker = AdaptiveChunker(
            short_passage_max_chars=500,
            medium_passage_max_chars=2000,
        )
        passage = make_short_passage()
        strategy = chunker._select_strategy(passage)
        assert strategy == "passage"

    def test_short_passage_produces_single_chunk(self):
        """Short passage must produce exactly one chunk."""
        chunker = AdaptiveChunker(
            short_passage_max_chars=500,
            medium_passage_max_chars=2000,
        )
        passage = make_short_passage()
        chunks = chunker.chunk(passage)
        assert len(chunks) == 1

    def test_short_chunk_has_adaptive_strategy(self):
        """Chunk from short passage must have strategy=ADAPTIVE."""
        chunker = AdaptiveChunker()
        passage = make_short_passage()
        chunks = chunker.chunk(passage)
        assert chunks[0].strategy == ChunkingStrategy.ADAPTIVE

    def test_single_char_below_threshold(self):
        """Very short text (1 char below threshold) must use passage strategy."""
        chunker = AdaptiveChunker(short_passage_max_chars=500)
        # 10 chars < 500
        passage = create_test_passage(translated_passage="Short text.")
        strategy = chunker._select_strategy(passage)
        assert strategy == "passage"


# ---------------------------------------------------------------------------
# Tests: Strategy selection — medium passages
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerMediumPassages:
    """Test strategy selection for medium passages with good sentence structure."""

    def test_medium_passage_with_sentences_uses_sentence_strategy(self):
        """Medium passage with multiple sentences must use sentence strategy."""
        chunker = AdaptiveChunker(
            short_passage_max_chars=50,  # very low threshold
            medium_passage_max_chars=2000,
        )
        # 7 short sentences totalling more than 50 chars
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six. Sentence seven."
        passage = create_test_passage(translated_passage=text)
        strategy = chunker._select_strategy(passage)
        assert strategy == "sentence"

    def test_medium_passage_chunks_have_adaptive_strategy(self):
        """All chunks from medium passage must have strategy=ADAPTIVE."""
        chunker = AdaptiveChunker(
            short_passage_max_chars=50,
            medium_passage_max_chars=2000,
        )
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        passage = create_test_passage(translated_passage=text)
        chunks = chunker.chunk(passage)
        assert all(c.strategy == ChunkingStrategy.ADAPTIVE for c in chunks)


# ---------------------------------------------------------------------------
# Tests: Strategy selection — long passages
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerLongPassages:
    """Test strategy selection for long passages."""

    def test_long_passage_with_tokenizer_uses_token_strategy(self):
        """Long passage with tokenizer available must use token strategy."""
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        # Create a passage well above 200 chars
        text = "Word " * 60  # 300 chars
        passage = create_test_passage(translated_passage=text)
        strategy = chunker._select_strategy(passage)
        assert strategy == "token"

    def test_long_passage_with_tokenizer_produces_multiple_chunks(self):
        """Long passage with tokenizer must produce multiple chunks."""
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        text = "Word " * 60
        passage = create_test_passage(translated_passage=text)
        chunks = chunker.chunk(passage)
        assert len(chunks) > 1

    def test_long_passage_without_tokenizer_falls_back_to_sentence(self):
        """Long passage without tokenizer must fall back to sentence strategy."""
        chunker = AdaptiveChunker(
            tokenizer=None,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
        )
        text = "Sentence one is here. Sentence two follows. " * 10  # > 200 chars
        passage = create_test_passage(translated_passage=text)
        strategy = chunker._select_strategy(passage)
        assert strategy == "sentence"


# ---------------------------------------------------------------------------
# Tests: Long sentence detection
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerLongSentenceDetection:
    """Test detection and handling of very long sentences."""

    def test_very_long_sentence_detected(self):
        """_has_very_long_sentence must return True for text without sentence boundaries."""
        chunker = AdaptiveChunker(long_sentence_threshold=50)
        long_text = "a " * 50  # 100 chars, no periods
        assert chunker._has_very_long_sentence(long_text) is True

    def test_short_sentences_not_detected_as_long(self):
        """Short sentences must not trigger long-sentence detection."""
        chunker = AdaptiveChunker(long_sentence_threshold=500)
        text = "Short. Very short. Also short."
        assert chunker._has_very_long_sentence(text) is False

    def test_very_long_sentence_with_tokenizer_uses_token_strategy(self):
        """A medium-length text with very long sentence must use token strategy when tokenizer present."""
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=2000,
            long_sentence_threshold=100,
            token_chunk_size=10,
            token_overlap=2,
        )
        # One very long "sentence" (no punctuation), total chars in range (50, 2000)
        text = "word " * 40  # 200 chars, no periods, each "sentence" is 200 chars
        passage = create_test_passage(translated_passage=text)
        strategy = chunker._select_strategy(passage)
        assert strategy == "token"

    def test_very_long_sentence_without_tokenizer_uses_sentence_fallback(self):
        """Long sentence without tokenizer must fall back to sentence strategy."""
        chunker = AdaptiveChunker(
            tokenizer=None,
            short_passage_max_chars=50,
            medium_passage_max_chars=2000,
            long_sentence_threshold=100,
        )
        text = "word " * 40  # no periods, > 100 chars per "sentence"
        passage = create_test_passage(translated_passage=text)
        strategy = chunker._select_strategy(passage)
        assert strategy == "sentence"


# ---------------------------------------------------------------------------
# Tests: Empty text handling
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerEmptyText:
    """Test empty/whitespace text handling."""

    def test_empty_batch_returns_empty(self):
        """chunk_batch([]) must return []."""
        chunker = AdaptiveChunker()
        assert chunker.chunk_batch([]) == []

    def test_all_chunks_have_adaptive_strategy(self):
        """All output chunks must have strategy=ADAPTIVE."""
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        passages = [
            make_short_passage("short"),
            make_long_passage("long"),
        ]
        chunks = chunker.chunk_batch(passages)
        assert all(c.strategy == ChunkingStrategy.ADAPTIVE for c in chunks)


# ---------------------------------------------------------------------------
# Tests: Metadata preservation
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerMetadata:
    """Test metadata preservation through adaptive dispatch."""

    def test_metadata_preserved_short_passage(self):
        """Metadata must be preserved when using passage strategy."""
        chunker = AdaptiveChunker(short_passage_max_chars=500)
        passage = create_test_passage(
            document_id="doc_meta",
            query_id=99,
            passage_index=3,
            target_lang="hi",
            source_lang="en",
            query="प्रश्न",
            eng_query="question",
            query_type="entity",
            answer="उत्तर",
            eng_answer="answer",
            is_selected=False,
            translated_passage="Short metadata test passage.",
        )
        chunks = chunker.chunk(passage)
        chunk = chunks[0]
        assert chunk.document_id == "doc_meta"
        assert chunk.query_id == 99
        assert chunk.passage_index == 3
        assert chunk.target_lang == "hi"
        assert chunk.query_type == "entity"
        assert chunk.answer == "उत्तर"
        assert chunk.is_selected is False

    def test_metadata_preserved_long_passage(self):
        """Metadata must be preserved when using token strategy."""
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        text = "word " * 60
        passage = create_test_passage(
            document_id="doc_long_meta",
            query_id=77,
            passage_index=2,
            translated_passage=text,
        )
        chunks = chunker.chunk(passage)
        for chunk in chunks:
            assert chunk.document_id == "doc_long_meta"
            assert chunk.query_id == 77
            assert chunk.passage_index == 2


# ---------------------------------------------------------------------------
# Tests: Input immutability
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerImmutability:
    """Test that input passages are not mutated."""

    def test_chunk_does_not_mutate_passage(self):
        """chunk() must not mutate the input passage."""
        chunker = AdaptiveChunker()
        passage = make_short_passage()
        original = passage.model_dump()
        chunker.chunk(passage)
        assert passage.model_dump() == original

    def test_chunk_batch_does_not_mutate_passages(self):
        """chunk_batch() must not mutate any input passage."""
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        passages = [
            make_short_passage("doc1"),
            make_long_passage("doc2"),
        ]
        originals = [p.model_dump() for p in passages]
        chunker.chunk_batch(passages)
        for p, original in zip(passages, originals):
            assert p.model_dump() == original


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerDeterminism:
    """Test deterministic output."""

    def test_deterministic_short_passage(self):
        """Short passage chunks must be identical on repeated calls."""
        chunker = AdaptiveChunker()
        passage = make_short_passage()
        chunks1 = chunker.chunk(passage)
        chunks2 = chunker.chunk(passage)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_deterministic_long_passage(self):
        """Long passage chunks must be identical on repeated calls."""
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        passage = make_long_passage()
        chunks1 = chunker.chunk(passage)
        chunks2 = chunker.chunk(passage)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


# ---------------------------------------------------------------------------
# Tests: Hindi and multilingual
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerMultilingual:
    """Test multilingual and Hindi text handling."""

    def test_hindi_short_passage(self):
        """Short Hindi passage must be handled correctly."""
        chunker = AdaptiveChunker(short_passage_max_chars=500)
        hindi = "यह एक छोटा परीक्षण पैसेज है।"
        passage = create_test_passage(translated_passage=hindi)
        chunks = chunker.chunk(passage)
        assert len(chunks) >= 1
        assert all(c.strategy == ChunkingStrategy.ADAPTIVE for c in chunks)

    def test_hindi_long_passage(self):
        """Long Hindi passage must be chunked without errors."""
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        hindi = ("यह एक लंबा परीक्षण पैसेज है। " * 20)
        passage = create_test_passage(translated_passage=hindi)
        chunks = chunker.chunk(passage)
        assert len(chunks) >= 1

    def test_mixed_language_passage(self):
        """Mixed Hindi/English passage must be chunked correctly."""
        chunker = AdaptiveChunker(short_passage_max_chars=500)
        mixed = "This is English. यह हिंदी है। Mixed content here."
        passage = create_test_passage(translated_passage=mixed)
        chunks = chunker.chunk(passage)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Tests: No empty chunks
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerNoEmptyChunks:
    """Verify no empty chunks are produced."""

    def test_no_empty_chunks_short(self):
        chunker = AdaptiveChunker()
        chunks = chunker.chunk(make_short_passage())
        assert all(c.chunk_text.strip() != "" for c in chunks)

    def test_no_empty_chunks_long(self):
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        chunks = chunker.chunk(make_long_passage())
        assert all(c.chunk_text.strip() != "" for c in chunks)


# ---------------------------------------------------------------------------
# Tests: Batch processing
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerBatch:
    """Test batch processing."""

    def test_batch_empty_input_returns_empty(self):
        chunker = AdaptiveChunker()
        assert chunker.chunk_batch([]) == []

    def test_batch_ordering_preserved(self):
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        p1 = make_short_passage("doc1")
        p2 = make_long_passage("doc2")
        all_chunks = chunker.chunk_batch([p1, p2])
        p1_chunks = chunker.chunk(p1)
        p2_chunks = chunker.chunk(p2)
        assert all_chunks == p1_chunks + p2_chunks

    def test_batch_each_passage_different_strategy(self):
        """Batch must handle passages using different strategies."""
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        passages = [
            make_short_passage("short"),
            make_long_passage("long"),
        ]
        chunks = chunker.chunk_batch(passages)
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Tests: Sentence count heuristic
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerSentenceCount:
    """Test _count_sentences_rough heuristic."""

    def test_count_no_punctuation_returns_one(self):
        """Text with no sentence punctuation must return at least 1."""
        chunker = AdaptiveChunker()
        assert chunker._count_sentences_rough("hello world") >= 1

    def test_count_multiple_punctuation(self):
        """Multiple periods must be counted."""
        chunker = AdaptiveChunker()
        count = chunker._count_sentences_rough("S1. S2. S3. S4.")
        assert count == 4

    def test_count_devanagari_danda(self):
        """Hindi दंड (।) must be counted as sentence boundary."""
        chunker = AdaptiveChunker()
        count = chunker._count_sentences_rough("वाक्य एक।वाक्य दो।वाक्य तीन।")
        assert count == 3

    def test_count_empty_text(self):
        """Empty text must return 0."""
        chunker = AdaptiveChunker()
        assert chunker._count_sentences_rough("") == 0


# ---------------------------------------------------------------------------
# Tests: repr and protocol conformance
# ---------------------------------------------------------------------------


class TestAdaptiveChunkerMisc:
    """Miscellaneous tests."""

    def test_repr_contains_class_name(self):
        chunker = AdaptiveChunker()
        assert "AdaptiveChunker" in repr(chunker)

    def test_conforms_to_base_chunker(self):
        from app.chunking.base import BaseChunker
        chunker = AdaptiveChunker()
        assert isinstance(chunker, BaseChunker)

    def test_no_infinite_loop_edge_case(self):
        """Must terminate for pathological but valid inputs."""
        tok = SimpleWhitespaceTokenizer()
        chunker = AdaptiveChunker(
            tokenizer=tok,
            short_passage_max_chars=50,
            medium_passage_max_chars=200,
            token_chunk_size=3,
            token_overlap=2,
        )
        text = "word " * 200
        passage = create_test_passage(translated_passage=text)
        chunks = chunker.chunk(passage)
        assert len(chunks) > 0
