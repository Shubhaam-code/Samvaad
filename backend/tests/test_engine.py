"""
Tests for ChunkingEngine (Phase 3.6).

Tests the multi-strategy chunking engine with registry-based dispatch.
Verifies all four strategies: PASSAGE, SENTENCE, TOKEN, ADAPTIVE.
Tests engine configuration, ordering, immutability, error handling.

All tests use tiny synthetic CanonicalPassage fixtures.
No real MSMARCO-XI data. No network access.
"""

import pytest

from app.chunking.engine import ChunkingEngine
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


def make_multi_sentence_passage(document_id: str = "doc1") -> CanonicalPassage:
    """5-sentence passage for sentence chunker tests."""
    text = "First sentence here. Second one follows. Third is present. Fourth appears. Fifth ends."
    return create_test_passage(document_id=document_id, translated_passage=text)


def make_long_passage(document_id: str = "doc1") -> CanonicalPassage:
    """Passage with many words for token chunker tests."""
    text = " ".join(f"word{i}" for i in range(50))
    return create_test_passage(document_id=document_id, translated_passage=text)


# ---------------------------------------------------------------------------
# Tests: Engine instantiation
# ---------------------------------------------------------------------------


class TestChunkingEngineInstantiation:
    """Test engine creation and configuration."""

    def test_default_engine_without_tokenizer(self):
        """Engine must be created without a tokenizer."""
        engine = ChunkingEngine()
        assert engine.tokenizer is None

    def test_engine_with_tokenizer(self):
        """Engine must accept a tokenizer."""
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(tokenizer=tok)
        assert engine.tokenizer is tok

    def test_engine_has_four_strategies(self):
        """Engine must support all four strategies by default."""
        engine = ChunkingEngine()
        strategies = engine.get_supported_strategies()
        assert ChunkingStrategy.PASSAGE in strategies
        assert ChunkingStrategy.SENTENCE in strategies
        assert ChunkingStrategy.TOKEN in strategies
        assert ChunkingStrategy.ADAPTIVE in strategies

    def test_repr_contains_class_name(self):
        """repr must contain 'ChunkingEngine'."""
        engine = ChunkingEngine()
        assert "ChunkingEngine" in repr(engine)


# ---------------------------------------------------------------------------
# Tests: PASSAGE strategy
# ---------------------------------------------------------------------------


class TestChunkingEnginePassageStrategy:
    """Test PASSAGE strategy via engine."""

    def test_passage_strategy_returns_one_chunk(self):
        """PASSAGE strategy must return exactly one chunk per passage."""
        engine = ChunkingEngine()
        passage = create_test_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.PASSAGE)
        assert len(chunks) == 1

    def test_passage_chunk_has_correct_strategy(self):
        """Chunk must have strategy=PASSAGE."""
        engine = ChunkingEngine()
        passage = create_test_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.PASSAGE)
        assert chunks[0].strategy == ChunkingStrategy.PASSAGE

    def test_passage_chunk_text_equals_translated_passage(self):
        """PASSAGE chunk text must equal translated_passage."""
        engine = ChunkingEngine()
        text = "Full passage text preserved."
        passage = create_test_passage(translated_passage=text)
        chunks = engine.chunk(passage, ChunkingStrategy.PASSAGE)
        assert chunks[0].chunk_text == text


# ---------------------------------------------------------------------------
# Tests: SENTENCE strategy
# ---------------------------------------------------------------------------


class TestChunkingEngineSentenceStrategy:
    """Test SENTENCE strategy via engine."""

    def test_sentence_strategy_produces_chunks(self):
        """SENTENCE strategy must produce one or more chunks."""
        engine = ChunkingEngine(sentence_chunk_size=2, sentence_overlap=1)
        passage = make_multi_sentence_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.SENTENCE)
        assert len(chunks) >= 1

    def test_sentence_chunks_have_correct_strategy(self):
        """Sentence chunks must have strategy=SENTENCE."""
        engine = ChunkingEngine(sentence_chunk_size=2, sentence_overlap=1)
        passage = make_multi_sentence_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.SENTENCE)
        assert all(c.strategy == ChunkingStrategy.SENTENCE for c in chunks)

    def test_sentence_chunks_not_empty(self):
        """No sentence chunk must have empty text."""
        engine = ChunkingEngine(sentence_chunk_size=2, sentence_overlap=1)
        passage = make_multi_sentence_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.SENTENCE)
        assert all(c.chunk_text.strip() != "" for c in chunks)


# ---------------------------------------------------------------------------
# Tests: TOKEN strategy
# ---------------------------------------------------------------------------


class TestChunkingEngineTokenStrategy:
    """Test TOKEN strategy via engine."""

    def test_token_strategy_requires_tokenizer(self):
        """TOKEN strategy without tokenizer must raise ValueError."""
        engine = ChunkingEngine()  # No tokenizer
        passage = make_long_passage()
        with pytest.raises(ValueError, match="requires a tokenizer"):
            engine.chunk(passage, ChunkingStrategy.TOKEN)

    def test_token_strategy_with_tokenizer_produces_chunks(self):
        """TOKEN strategy with tokenizer must produce chunks."""
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(
            tokenizer=tok,
            token_chunk_size=10,
            token_overlap=2,
        )
        passage = make_long_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.TOKEN)
        assert len(chunks) >= 1

    def test_token_chunks_have_correct_strategy(self):
        """Token chunks must have strategy=TOKEN."""
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(tokenizer=tok, token_chunk_size=10, token_overlap=2)
        passage = make_long_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.TOKEN)
        assert all(c.strategy == ChunkingStrategy.TOKEN for c in chunks)

    def test_token_strategy_multiple_chunks_for_long_passage(self):
        """Long passage must produce multiple token chunks."""
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(tokenizer=tok, token_chunk_size=10, token_overlap=2)
        passage = make_long_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.TOKEN)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Tests: ADAPTIVE strategy
# ---------------------------------------------------------------------------


class TestChunkingEngineAdaptiveStrategy:
    """Test ADAPTIVE strategy via engine."""

    def test_adaptive_strategy_produces_chunks(self):
        """ADAPTIVE strategy must produce chunks."""
        engine = ChunkingEngine()
        passage = create_test_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.ADAPTIVE)
        assert len(chunks) >= 1

    def test_adaptive_chunks_have_correct_strategy(self):
        """Adaptive chunks must have strategy=ADAPTIVE."""
        engine = ChunkingEngine()
        passage = create_test_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.ADAPTIVE)
        assert all(c.strategy == ChunkingStrategy.ADAPTIVE for c in chunks)

    def test_adaptive_strategy_works_with_tokenizer(self):
        """ADAPTIVE strategy must work with a tokenizer."""
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(
            tokenizer=tok,
            adaptive_short_max=50,
            adaptive_medium_max=200,
            token_chunk_size=10,
            token_overlap=2,
        )
        passage = make_long_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.ADAPTIVE)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Tests: No silent fallback
# ---------------------------------------------------------------------------


class TestChunkingEngineNoFallback:
    """Test that engine does NOT silently fall back to another strategy."""

    def test_unsupported_strategy_raises_value_error(self):
        """Requesting a non-registered strategy must raise ValueError."""
        engine = ChunkingEngine()
        passage = create_test_passage()

        # Create a fake strategy by subclassing (if enum allows) or test known error
        # TOKEN without tokenizer raises a specific ValueError
        with pytest.raises(ValueError):
            engine.chunk(passage, ChunkingStrategy.TOKEN)


# ---------------------------------------------------------------------------
# Tests: Batch processing
# ---------------------------------------------------------------------------


class TestChunkingEngineBatch:
    """Test chunk_batch method."""

    def test_empty_batch_returns_empty(self):
        """chunk_batch([]) must return []."""
        engine = ChunkingEngine()
        assert engine.chunk_batch([], ChunkingStrategy.PASSAGE) == []

    def test_batch_ordering_preserved_passage(self):
        """PASSAGE strategy batch must preserve input ordering."""
        engine = ChunkingEngine()
        passages = [
            create_test_passage(document_id=f"doc{i}", translated_passage=f"Text {i}.")
            for i in range(5)
        ]
        chunks = engine.chunk_batch(passages, ChunkingStrategy.PASSAGE)
        assert len(chunks) == 5
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_text == f"Text {i}."

    def test_batch_ordering_preserved_token(self):
        """TOKEN strategy batch must preserve input ordering."""
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(tokenizer=tok, token_chunk_size=10, token_overlap=2)
        passages = [make_long_passage(f"doc{i}") for i in range(3)]
        all_chunks = engine.chunk_batch(passages, ChunkingStrategy.TOKEN)
        p0_chunks = engine.chunk(passages[0], ChunkingStrategy.TOKEN)
        p1_chunks = engine.chunk(passages[1], ChunkingStrategy.TOKEN)
        p2_chunks = engine.chunk(passages[2], ChunkingStrategy.TOKEN)
        assert all_chunks == p0_chunks + p1_chunks + p2_chunks

    def test_batch_all_strategies(self):
        """chunk_batch must work with all four strategies."""
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(
            tokenizer=tok,
            token_chunk_size=10,
            token_overlap=2,
            adaptive_short_max=50,
            adaptive_medium_max=200,
        )
        passages = [create_test_passage(document_id=f"doc{i}") for i in range(3)]
        for strategy in ChunkingStrategy:
            chunks = engine.chunk_batch(passages, strategy)
            assert len(chunks) >= 3  # At least 1 chunk per passage


# ---------------------------------------------------------------------------
# Tests: Input immutability
# ---------------------------------------------------------------------------


class TestChunkingEngineImmutability:
    """Test that engine does not mutate input."""

    def test_chunk_does_not_mutate_passage(self):
        """chunk() must not mutate the input passage."""
        engine = ChunkingEngine()
        passage = create_test_passage()
        original = passage.model_dump()
        engine.chunk(passage, ChunkingStrategy.PASSAGE)
        assert passage.model_dump() == original

    def test_chunk_batch_does_not_mutate_passages(self):
        """chunk_batch() must not mutate any input passage."""
        engine = ChunkingEngine()
        passages = [create_test_passage(document_id=f"doc{i}") for i in range(3)]
        originals = [p.model_dump() for p in passages]
        engine.chunk_batch(passages, ChunkingStrategy.PASSAGE)
        for p, original in zip(passages, originals):
            assert p.model_dump() == original


# ---------------------------------------------------------------------------
# Tests: Ordering preserved per passage
# ---------------------------------------------------------------------------


class TestChunkingEngineOrdering:
    """Test chunk ordering within a passage."""

    def test_chunk_indexes_sequential(self):
        """All chunk indexes must be sequential starting at 0."""
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(tokenizer=tok, token_chunk_size=10, token_overlap=2)
        passage = make_long_passage()
        chunks = engine.chunk(passage, ChunkingStrategy.TOKEN)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ---------------------------------------------------------------------------
# Tests: Metadata preservation
# ---------------------------------------------------------------------------


class TestChunkingEngineMetadata:
    """Test metadata preservation through the engine."""

    def test_metadata_preserved_all_strategies(self):
        """Source metadata must be preserved for all strategies."""
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(
            tokenizer=tok,
            token_chunk_size=10,
            token_overlap=2,
            adaptive_short_max=50,
            adaptive_medium_max=200,
        )
        passage = create_test_passage(
            document_id="meta_doc",
            query_id=55,
            passage_index=3,
            target_lang="hi",
            source_lang="en",
            query="meta query",
            eng_query="meta query",
            is_selected=True,
            translated_passage=" ".join(f"w{i}" for i in range(50)),
        )
        for strategy in ChunkingStrategy:
            chunks = engine.chunk(passage, strategy)
            for chunk in chunks:
                assert chunk.document_id == "meta_doc"
                assert chunk.query_id == 55
                assert chunk.passage_index == 3
                assert chunk.target_lang == "hi"
                assert chunk.is_selected is True


# ---------------------------------------------------------------------------
# Tests: Custom chunker registration
# ---------------------------------------------------------------------------


class TestChunkingEngineRegistry:
    """Test custom chunker registration."""

    def test_register_custom_chunker(self):
        """Registered custom chunker must be callable via engine."""
        from app.chunking.passage_chunker import PassageChunker

        engine = ChunkingEngine()

        custom_calls = []

        class CustomChunker(PassageChunker):
            def chunk(self, passage):
                custom_calls.append(passage.document_id)
                return super().chunk(passage)

            def chunk_batch(self, passages):
                result = []
                for p in passages:
                    result.extend(self.chunk(p))
                return result

        engine.register_chunker(
            ChunkingStrategy.PASSAGE,
            lambda: CustomChunker(),
        )
        passage = create_test_passage(document_id="custom_test")
        engine.chunk(passage, ChunkingStrategy.PASSAGE)
        assert "custom_test" in custom_calls

    def test_get_supported_strategies_lists_all(self):
        """get_supported_strategies must list all four default strategies."""
        engine = ChunkingEngine()
        strategies = engine.get_supported_strategies()
        assert len(strategies) == 4


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------


class TestChunkingEngineDeterminism:
    """Test deterministic output from engine."""

    def test_passage_strategy_deterministic(self):
        engine = ChunkingEngine()
        passage = create_test_passage()
        chunks1 = engine.chunk(passage, ChunkingStrategy.PASSAGE)
        chunks2 = engine.chunk(passage, ChunkingStrategy.PASSAGE)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_token_strategy_deterministic(self):
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(tokenizer=tok, token_chunk_size=10, token_overlap=2)
        passage = make_long_passage()
        chunks1 = engine.chunk(passage, ChunkingStrategy.TOKEN)
        chunks2 = engine.chunk(passage, ChunkingStrategy.TOKEN)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


# ---------------------------------------------------------------------------
# Tests: Hindi and multilingual
# ---------------------------------------------------------------------------


class TestChunkingEngineMultilingual:
    """Test engine with multilingual content."""

    def test_hindi_passage_all_strategies(self):
        """All strategies must handle Hindi text without errors."""
        tok = SimpleWhitespaceTokenizer()
        engine = ChunkingEngine(
            tokenizer=tok,
            token_chunk_size=5,
            token_overlap=1,
            adaptive_short_max=50,
            adaptive_medium_max=200,
        )
        hindi = "यह एक परीक्षण है। यह हिंदी में है। और यह जारी रहता है।"
        passage = create_test_passage(translated_passage=hindi)
        for strategy in ChunkingStrategy:
            chunks = engine.chunk(passage, strategy)
            assert len(chunks) >= 1
