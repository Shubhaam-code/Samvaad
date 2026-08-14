"""
Tests for TokenChunker (Phase 3.4).

Tests the token-aware chunker with sliding window overlap, the
TokenizerProtocol abstraction, SimpleWhitespaceTokenizer, and
HuggingFaceTokenizerAdapter (using only locally-cached models).

All tests use tiny synthetic CanonicalPassage fixtures.
No real MSMARCO-XI data. No network access.
"""

import pytest

from app.chunking.models import Chunk, ChunkingStrategy
from app.chunking.token_chunker import TokenChunker
from app.chunking.tokenizer import (
    HuggingFaceTokenizerAdapter,
    SimpleWhitespaceTokenizer,
    TokenizerProtocol,
    create_default_tokenizer,
)
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


def make_passage_with_words(n_words: int, document_id: str = "doc1") -> CanonicalPassage:
    """Create a passage with exactly n_words whitespace-separated words."""
    text = " ".join(f"word{i}" for i in range(n_words))
    return create_test_passage(document_id=document_id, translated_passage=text)


# ---------------------------------------------------------------------------
# Tests: SimpleWhitespaceTokenizer
# ---------------------------------------------------------------------------


class TestSimpleWhitespaceTokenizer:
    """Tests for the SimpleWhitespaceTokenizer fallback."""

    def test_encode_returns_list(self):
        tok = SimpleWhitespaceTokenizer()
        result = tok.encode("hello world")
        assert len(result) == 2

    def test_encode_word_count_matches(self):
        tok = SimpleWhitespaceTokenizer()
        text = "one two three four five"
        assert len(tok.encode(text)) == 5

    def test_encode_empty_string(self):
        tok = SimpleWhitespaceTokenizer()
        assert list(tok.encode("")) == []

    def test_encode_whitespace_only(self):
        tok = SimpleWhitespaceTokenizer()
        assert list(tok.encode("   ")) == []

    def test_encode_single_word(self):
        tok = SimpleWhitespaceTokenizer()
        assert len(tok.encode("hello")) == 1

    def test_encode_returns_integers(self):
        tok = SimpleWhitespaceTokenizer()
        ids = tok.encode("foo bar baz")
        assert all(isinstance(i, int) for i in ids)

    def test_count_tokens_empty(self):
        tok = SimpleWhitespaceTokenizer()
        assert tok.count_tokens("") == 0

    def test_count_tokens_whitespace_only(self):
        tok = SimpleWhitespaceTokenizer()
        assert tok.count_tokens("   \t\n") == 0

    def test_count_tokens_matches_encode_length(self):
        tok = SimpleWhitespaceTokenizer()
        for text in ["hello", "hello world", "a b c d e f"]:
            assert tok.count_tokens(text) == len(tok.encode(text))

    def test_decode_empty_returns_empty_string(self):
        tok = SimpleWhitespaceTokenizer()
        assert tok.decode([]) == ""

    def test_decode_returns_string(self):
        tok = SimpleWhitespaceTokenizer()
        result = tok.decode([1, 2, 3])
        assert isinstance(result, str)

    def test_encode_hindi_text(self):
        tok = SimpleWhitespaceTokenizer()
        hindi = "यह परीक्षण है"
        ids = tok.encode(hindi)
        assert len(ids) == 3

    def test_encode_mixed_language(self):
        tok = SimpleWhitespaceTokenizer()
        mixed = "hello दुनिया world"
        ids = tok.encode(mixed)
        assert len(ids) == 3

    def test_encode_punctuation(self):
        tok = SimpleWhitespaceTokenizer()
        text = "Hello, world! How are you?"
        assert len(tok.encode(text)) == 5

    def test_encode_urls(self):
        tok = SimpleWhitespaceTokenizer()
        text = "Visit https://example.com for more info"
        ids = tok.encode(text)
        assert len(ids) == 5

    def test_encode_numbers(self):
        tok = SimpleWhitespaceTokenizer()
        text = "123 456 789"
        assert len(tok.encode(text)) == 3

    def test_repr(self):
        tok = SimpleWhitespaceTokenizer()
        assert "SimpleWhitespaceTokenizer" in repr(tok)

    def test_encode_deterministic(self):
        tok = SimpleWhitespaceTokenizer()
        text = "deterministic test"
        assert list(tok.encode(text)) == list(tok.encode(text))


# ---------------------------------------------------------------------------
# Tests: create_default_tokenizer
# ---------------------------------------------------------------------------


class TestCreateDefaultTokenizer:
    """Tests for the create_default_tokenizer factory."""

    def test_returns_tokenizer_with_required_methods(self):
        tok = create_default_tokenizer()
        assert hasattr(tok, "encode")
        assert hasattr(tok, "decode")
        assert hasattr(tok, "count_tokens")
        assert callable(tok.encode)
        assert callable(tok.decode)
        assert callable(tok.count_tokens)

    def test_tokenizer_can_encode(self):
        tok = create_default_tokenizer()
        result = tok.encode("hello world")
        assert len(result) >= 1

    def test_tokenizer_count_tokens_positive(self):
        tok = create_default_tokenizer()
        assert tok.count_tokens("hello world") > 0

    def test_tokenizer_count_empty_is_zero(self):
        tok = create_default_tokenizer()
        assert tok.count_tokens("") == 0


# ---------------------------------------------------------------------------
# Tests: TokenChunker configuration validation
# ---------------------------------------------------------------------------


class TestTokenChunkerConfiguration:
    """Test TokenChunker configuration validation."""

    def setup_method(self):
        self.tok = SimpleWhitespaceTokenizer()

    def test_default_configuration(self):
        chunker = TokenChunker(self.tok)
        assert chunker.chunk_size == 256
        assert chunker.token_overlap == 32

    def test_custom_configuration(self):
        chunker = TokenChunker(self.tok, chunk_size=128, token_overlap=16)
        assert chunker.chunk_size == 128
        assert chunker.token_overlap == 16

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TokenChunker(self.tok, chunk_size=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TokenChunker(self.tok, chunk_size=-10)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="token_overlap must be non-negative"):
            TokenChunker(self.tok, chunk_size=10, token_overlap=-1)

    def test_overlap_equal_chunk_size_raises(self):
        with pytest.raises(ValueError, match="must be less than"):
            TokenChunker(self.tok, chunk_size=10, token_overlap=10)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="must be less than"):
            TokenChunker(self.tok, chunk_size=10, token_overlap=15)

    def test_zero_overlap_allowed(self):
        chunker = TokenChunker(self.tok, chunk_size=10, token_overlap=0)
        assert chunker.token_overlap == 0

    def test_chunk_size_one_zero_overlap_allowed(self):
        chunker = TokenChunker(self.tok, chunk_size=1, token_overlap=0)
        assert chunker.chunk_size == 1


# ---------------------------------------------------------------------------
# Tests: Short passages (single chunk)
# ---------------------------------------------------------------------------


class TestTokenChunkerShortPassages:
    """Test passages shorter than chunk_size produce exactly one chunk."""

    def setup_method(self):
        self.tok = SimpleWhitespaceTokenizer()

    def test_short_passage_single_chunk(self):
        chunker = TokenChunker(self.tok, chunk_size=100, token_overlap=10)
        passage = make_passage_with_words(5)
        chunks = chunker.chunk(passage)
        assert len(chunks) == 1

    def test_short_passage_chunk_text_nonempty(self):
        chunker = TokenChunker(self.tok, chunk_size=100, token_overlap=10)
        passage = make_passage_with_words(5)
        chunks = chunker.chunk(passage)
        assert chunks[0].chunk_text.strip() != ""

    def test_exact_chunk_size_single_chunk(self):
        chunk_size = 10
        chunker = TokenChunker(self.tok, chunk_size=chunk_size, token_overlap=2)
        passage = make_passage_with_words(chunk_size)
        chunks = chunker.chunk(passage)
        assert len(chunks) == 1

    def test_one_word_passage(self):
        chunker = TokenChunker(self.tok, chunk_size=10, token_overlap=2)
        passage = create_test_passage(translated_passage="hello")
        chunks = chunker.chunk(passage)
        assert len(chunks) == 1

    def test_strategy_is_token(self):
        chunker = TokenChunker(self.tok, chunk_size=10, token_overlap=2)
        passage = make_passage_with_words(5)
        chunks = chunker.chunk(passage)
        assert all(c.strategy == ChunkingStrategy.TOKEN for c in chunks)


# ---------------------------------------------------------------------------
# Tests: Long passages (multiple chunks)
# ---------------------------------------------------------------------------


class TestTokenChunkerLongPassages:
    """Test passages longer than chunk_size produce multiple chunks."""

    def setup_method(self):
        self.tok = SimpleWhitespaceTokenizer()

    def test_long_passage_multiple_chunks(self):
        chunk_size = 10
        overlap = 2
        chunker = TokenChunker(self.tok, chunk_size=chunk_size, token_overlap=overlap)
        passage = make_passage_with_words(25)
        chunks = chunker.chunk(passage)
        assert len(chunks) > 1

    def test_sequential_chunk_indexes(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = make_passage_with_words(20)
        chunks = chunker.chunk(passage)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_no_empty_chunks(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = make_passage_with_words(30)
        chunks = chunker.chunk(passage)
        assert all(c.chunk_text.strip() != "" for c in chunks)

    def test_final_partial_chunk_preserved(self):
        chunker = TokenChunker(self.tok, chunk_size=7, token_overlap=2)
        passage = make_passage_with_words(20)
        chunks = chunker.chunk(passage)
        assert chunks[-1].chunk_text.strip() != ""

    def test_token_count_is_set(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = make_passage_with_words(20)
        chunks = chunker.chunk(passage)
        assert all(c.token_count is not None for c in chunks)
        assert all(c.token_count > 0 for c in chunks)

    def test_token_count_at_most_chunk_size(self):
        chunk_size = 7
        chunker = TokenChunker(self.tok, chunk_size=chunk_size, token_overlap=2)
        passage = make_passage_with_words(30)
        chunks = chunker.chunk(passage)
        assert all(c.token_count <= chunk_size for c in chunks)

    def test_character_count_equals_len_chunk_text(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = make_passage_with_words(20)
        chunks = chunker.chunk(passage)
        assert all(c.character_count == len(c.chunk_text) for c in chunks)


# ---------------------------------------------------------------------------
# Tests: Token window internals
# ---------------------------------------------------------------------------


class TestTokenChunkerWindowInternals:
    """Test _chunk_tokens window splitting logic."""

    def setup_method(self):
        self.tok = SimpleWhitespaceTokenizer()

    def test_empty_token_list_returns_empty(self):
        chunker = TokenChunker(self.tok, chunk_size=10, token_overlap=2)
        assert chunker._chunk_tokens([]) == []

    def test_single_token_list_returns_one_window(self):
        chunker = TokenChunker(self.tok, chunk_size=10, token_overlap=2)
        result = chunker._chunk_tokens([42])
        assert len(result) == 1
        assert list(result[0]) == [42]

    def test_window_count_formula(self):
        import math
        chunk_size = 5
        overlap = 2
        stride = chunk_size - overlap  # 3
        n = 14

        chunker = TokenChunker(self.tok, chunk_size=chunk_size, token_overlap=overlap)
        token_ids = list(range(n))
        windows = chunker._chunk_tokens(token_ids)

        expected = math.ceil((n - chunk_size) / stride) + 1
        assert len(windows) == expected

    def test_overlap_near_chunk_size(self):
        # stride=1, chunk_size=5, tokens=10 => 10-5+1=6 windows
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=4)
        passage = make_passage_with_words(10)
        chunks = chunker.chunk(passage)
        assert len(chunks) == 6

    def test_no_infinite_loop(self):
        chunker = TokenChunker(self.tok, chunk_size=3, token_overlap=2)
        passage = make_passage_with_words(100)
        chunks = chunker.chunk(passage)
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------


class TestTokenChunkerDeterminism:
    """Test deterministic behaviour."""

    def setup_method(self):
        self.tok = SimpleWhitespaceTokenizer()

    def test_deterministic_chunk_ids(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = make_passage_with_words(20)
        chunks1 = chunker.chunk(passage)
        chunks2 = chunker.chunk(passage)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_deterministic_chunk_texts(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = make_passage_with_words(20)
        chunks1 = chunker.chunk(passage)
        chunks2 = chunker.chunk(passage)
        assert [c.chunk_text for c in chunks1] == [c.chunk_text for c in chunks2]

    def test_different_passages_different_ids(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        p1 = make_passage_with_words(20, document_id="doc1")
        p2 = make_passage_with_words(20, document_id="doc2")
        ids1 = {c.chunk_id for c in chunker.chunk(p1)}
        ids2 = {c.chunk_id for c in chunker.chunk(p2)}
        assert ids1.isdisjoint(ids2)

    def test_unique_chunk_ids_within_passage(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = make_passage_with_words(30)
        chunks = chunker.chunk(passage)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Tests: Metadata preservation
# ---------------------------------------------------------------------------


class TestTokenChunkerMetadata:
    """Test source passage metadata preservation."""

    def setup_method(self):
        self.tok = SimpleWhitespaceTokenizer()

    def test_metadata_preserved(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = create_test_passage(
            document_id="doc999",
            query_id=42,
            passage_index=7,
            target_lang="hi",
            source_lang="en",
            query="test query",
            eng_query="test query",
            query_type="factoid",
            answer="ans",
            eng_answer="answer",
            is_selected=True,
            translated_passage="one two three four five six seven eight nine ten",
        )
        chunks = chunker.chunk(passage)
        for chunk in chunks:
            assert chunk.document_id == "doc999"
            assert chunk.query_id == 42
            assert chunk.passage_index == 7
            assert chunk.target_lang == "hi"
            assert chunk.source_lang == "en"
            assert chunk.query_type == "factoid"
            assert chunk.answer == "ans"
            assert chunk.eng_answer == "answer"
            assert chunk.is_selected is True

    def test_optional_metadata_none_preserved(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = create_test_passage(
            query_type=None,
            answer=None,
            eng_answer=None,
            translated_passage="one two three four five six seven eight",
        )
        chunks = chunker.chunk(passage)
        for chunk in chunks:
            assert chunk.query_type is None
            assert chunk.answer is None
            assert chunk.eng_answer is None


# ---------------------------------------------------------------------------
# Tests: No input mutation
# ---------------------------------------------------------------------------


class TestTokenChunkerImmutability:
    """Test that input passages are not mutated."""

    def setup_method(self):
        self.tok = SimpleWhitespaceTokenizer()

    def test_chunk_does_not_mutate_passage(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = make_passage_with_words(20)
        original = passage.model_dump()
        chunker.chunk(passage)
        assert passage.model_dump() == original

    def test_chunk_batch_does_not_mutate_passages(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passages = [make_passage_with_words(20, document_id=f"doc{i}") for i in range(3)]
        originals = [p.model_dump() for p in passages]
        chunker.chunk_batch(passages)
        for p, original in zip(passages, originals):
            assert p.model_dump() == original


# ---------------------------------------------------------------------------
# Tests: Batch processing
# ---------------------------------------------------------------------------


class TestTokenChunkerBatch:
    """Test batch processing."""

    def setup_method(self):
        self.tok = SimpleWhitespaceTokenizer()

    def test_empty_batch_returns_empty(self):
        chunker = TokenChunker(self.tok, chunk_size=10, token_overlap=2)
        assert chunker.chunk_batch([]) == []

    def test_batch_ordering_preserved(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        p1 = make_passage_with_words(20, document_id="doc1")
        p2 = make_passage_with_words(20, document_id="doc2")
        all_chunks = chunker.chunk_batch([p1, p2])
        p1_chunks = chunker.chunk(p1)
        p2_chunks = chunker.chunk(p2)
        assert all_chunks == p1_chunks + p2_chunks

    def test_batch_single_passage(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passage = make_passage_with_words(20)
        assert chunker.chunk_batch([passage]) == chunker.chunk(passage)

    def test_batch_multiple_passages_chunk_count(self):
        chunker = TokenChunker(self.tok, chunk_size=5, token_overlap=1)
        passages = [make_passage_with_words(20, document_id=f"doc{i}") for i in range(4)]
        all_chunks = chunker.chunk_batch(passages)
        expected = sum(len(chunker.chunk(p)) for p in passages)
        assert len(all_chunks) == expected


# ---------------------------------------------------------------------------
# Tests: Multilingual text
# ---------------------------------------------------------------------------


class TestTokenChunkerMultilingual:
    """Test multilingual and Unicode text handling."""

    def setup_method(self):
        self.tok = SimpleWhitespaceTokenizer()

    def test_hindi_text_chunked(self):
        chunker = TokenChunker(self.tok, chunk_size=3, token_overlap=1)
        hindi = "यह एक परीक्षण है जो हिंदी भाषा में लिखा गया है और इसमें कई शब्द हैं।"
        passage = create_test_passage(translated_passage=hindi)
        chunks = chunker.chunk(passage)
        assert len(chunks) >= 1
        assert all(c.chunk_text.strip() != "" for c in chunks)

    def test_mixed_language_text_chunked(self):
        chunker = TokenChunker(self.tok, chunk_size=4, token_overlap=1)
        mixed = "hello दुनिया world परीक्षण test हिंदी english मिश्रित"
        passage = create_test_passage(translated_passage=mixed)
        chunks = chunker.chunk(passage)
        assert len(chunks) >= 1

    def test_unicode_special_chars(self):
        chunker = TokenChunker(self.tok, chunk_size=3, token_overlap=1)
        text = "café résumé naïve über ñoño fête"
        passage = create_test_passage(translated_passage=text)
        chunks = chunker.chunk(passage)
        assert len(chunks) >= 1

    def test_numbers_in_text(self):
        chunker = TokenChunker(self.tok, chunk_size=3, token_overlap=1)
        text = "year 2024 2025 2026 quarter 1 2 3 4 value 99 100 200"
        passage = create_test_passage(translated_passage=text)
        chunks = chunker.chunk(passage)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Tests: repr and protocol conformance
# ---------------------------------------------------------------------------


class TestTokenChunkerMisc:
    """Miscellaneous tests."""

    def test_repr_contains_class_name(self):
        tok = SimpleWhitespaceTokenizer()
        chunker = TokenChunker(tok, chunk_size=64, token_overlap=8)
        assert "TokenChunker" in repr(chunker)

    def test_repr_contains_chunk_size(self):
        tok = SimpleWhitespaceTokenizer()
        chunker = TokenChunker(tok, chunk_size=64, token_overlap=8)
        assert "64" in repr(chunker)

    def test_conforms_to_base_chunker(self):
        from app.chunking.base import BaseChunker
        tok = SimpleWhitespaceTokenizer()
        chunker = TokenChunker(tok)
        assert isinstance(chunker, BaseChunker)

    def test_simple_whitespace_has_protocol_methods(self):
        tok = SimpleWhitespaceTokenizer()
        assert callable(tok.encode)
        assert callable(tok.decode)
        assert callable(tok.count_tokens)

    def test_simple_whitespace_decode_reconstruction(self):
        tok = SimpleWhitespaceTokenizer()
        text = "hello world testing decode"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_simple_whitespace_unknown_token_id(self):
        tok = SimpleWhitespaceTokenizer()
        decoded = tok.decode([9999999])
        assert "[9999999]" in decoded


class DummyHuggingFaceTokenizer:
    """Mock HuggingFace tokenizer for unit testing without downloading models."""

    def __init__(self):
        self.vocab = {"hello": 101, "world": 102, "foo": 103, "bar": 104}
        self.id_to_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text or not text.strip():
            return []
        return [self.vocab.get(w.lower(), 999) for w in text.split()]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        if not token_ids:
            return ""
        return " ".join([self.id_to_vocab.get(tid, "<unk>") for tid in token_ids])


class TestHuggingFaceTokenizerAdapterMocked:
    """Tests for HuggingFaceTokenizerAdapter using a mock tokenizer."""

    def test_adapter_requires_tokenizer(self):
        with pytest.raises(ValueError, match="requires a valid tokenizer instance"):
            HuggingFaceTokenizerAdapter(None)

    def test_encode_delegation(self):
        mock_hf = DummyHuggingFaceTokenizer()
        adapter = HuggingFaceTokenizerAdapter(mock_hf)
        ids = adapter.encode("hello world")
        assert list(ids) == [101, 102]

    def test_decode_delegation(self):
        mock_hf = DummyHuggingFaceTokenizer()
        adapter = HuggingFaceTokenizerAdapter(mock_hf)
        text = adapter.decode([101, 102])
        assert text == "hello world"

    def test_count_tokens_delegation(self):
        mock_hf = DummyHuggingFaceTokenizer()
        adapter = HuggingFaceTokenizerAdapter(mock_hf)
        assert adapter.count_tokens("hello world foo") == 3

    def test_empty_input_handling(self):
        mock_hf = DummyHuggingFaceTokenizer()
        adapter = HuggingFaceTokenizerAdapter(mock_hf)
        assert list(adapter.encode("")) == []
        assert adapter.decode([]) == ""
        assert adapter.count_tokens("") == 0

    def test_repr(self):
        mock_hf = DummyHuggingFaceTokenizer()
        adapter = HuggingFaceTokenizerAdapter(mock_hf)
        assert "HuggingFaceTokenizerAdapter" in repr(adapter)


class TestCreateHuggingFaceTokenizer:
    """Tests for create_huggingface_tokenizer factory."""

    def test_raises_runtime_error_when_model_not_cached_locally(self):
        from app.chunking.tokenizer import create_huggingface_tokenizer
        with pytest.raises(RuntimeError, match="Failed to load HuggingFace tokenizer"):
            create_huggingface_tokenizer("non-existent-local-model-id-12345", local_files_only=True)


class TestTokenChunkerValidation:
    """Tests for TokenChunker initialization validation."""

    def test_none_tokenizer_raises(self):
        with pytest.raises(ValueError, match="requires a valid tokenizer"):
            TokenChunker(tokenizer=None)  # type: ignore
