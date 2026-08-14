"""Tests for Chunk data model.

Phase 3.1: Chunk schema testing (no chunking algorithms).
"""

import pytest
from pydantic import ValidationError

from app.chunking.models import Chunk, ChunkingStrategy


def create_valid_chunk(
    chunk_index=0,
    strategy=ChunkingStrategy.PASSAGE,
    chunk_text="Test chunk text",
):
    """Helper to create a valid Chunk for testing."""
    return Chunk.from_passage_segment(
        document_id="test_doc_123",
        chunk_index=chunk_index,
        strategy=strategy,
        chunk_text=chunk_text,
        query_id=100,
        passage_index=0,
        target_lang="hi",
        source_lang="en",
        query="Test query",
        eng_query="Test query",
        query_type="TEST",
        answer="Test answer",
        eng_answer="Test answer",
        is_selected=True,
    )


def test_chunk_valid_creation():
    """Test creating a valid Chunk."""
    chunk = create_valid_chunk()
    
    assert chunk.chunk_id is not None
    assert len(chunk.chunk_id) == 64  # SHA-256 hex
    assert chunk.document_id == "test_doc_123"
    assert chunk.chunk_index == 0
    assert chunk.strategy == ChunkingStrategy.PASSAGE
    assert chunk.chunk_text == "Test chunk text"
    assert chunk.query_id == 100
    assert chunk.passage_index == 0
    assert chunk.target_lang == "hi"
    assert chunk.is_selected is True


def test_chunk_required_fields():
    """Test that required fields are enforced."""
    # Missing chunk_text
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="test",
            document_id="doc",
            chunk_index=0,
            strategy=ChunkingStrategy.PASSAGE,
            chunk_text="",  # Empty not allowed
            query_id=1,
            passage_index=0,
            target_lang="hi",
            source_lang="en",
            query="query",
            eng_query="query",
            is_selected=True,
        )


def test_chunk_empty_text_rejection():
    """Test that empty chunk_text is rejected."""
    with pytest.raises(ValidationError):  # Remove match constraint
        Chunk.from_passage_segment(
            document_id="doc",
            chunk_index=0,
            strategy=ChunkingStrategy.SENTENCE,
            chunk_text="",  # Empty
            query_id=1,
            passage_index=0,
            target_lang="hi",
            source_lang="en",
            query="query",
            eng_query="query",
            query_type=None,
            answer=None,
            eng_answer=None,
            is_selected=False,
        )


def test_chunk_whitespace_only_text_rejection():
    """Test that whitespace-only chunk_text is rejected."""
    with pytest.raises(ValidationError):  # Catches both min_length and custom validator
        Chunk.from_passage_segment(
            document_id="doc",
            chunk_index=0,
            strategy=ChunkingStrategy.TOKEN,
            chunk_text="   ",  # Whitespace only
            query_id=1,
            passage_index=0,
            target_lang="hi",
            source_lang="en",
            query="query",
            eng_query="query",
            query_type=None,
            answer=None,
            eng_answer=None,
            is_selected=False,
        )


def test_chunk_negative_passage_index_rejection():
    """Test that negative passage_index is rejected."""
    with pytest.raises(ValidationError):
        Chunk.from_passage_segment(
            document_id="doc",
            chunk_index=0,
            strategy=ChunkingStrategy.PASSAGE,
            chunk_text="text",
            query_id=1,
            passage_index=-1,  # Negative
            target_lang="hi",
            source_lang="en",
            query="query",
            eng_query="query",
            query_type=None,
            answer=None,
            eng_answer=None,
            is_selected=False,
        )


def test_chunk_negative_chunk_index_rejection():
    """Test that negative chunk_index is rejected."""
    with pytest.raises(ValidationError):
        Chunk.from_passage_segment(
            document_id="doc",
            chunk_index=-1,  # Negative
            strategy=ChunkingStrategy.SENTENCE,
            chunk_text="text",
            query_id=1,
            passage_index=0,
            target_lang="hi",
            source_lang="en",
            query="query",
            eng_query="query",
            query_type=None,
            answer=None,
            eng_answer=None,
            is_selected=False,
        )


def test_chunk_deterministic_id():
    """Test that chunk_id is deterministic."""
    chunk1 = Chunk.from_passage_segment(
        document_id="doc123",
        chunk_index=0,
        strategy=ChunkingStrategy.PASSAGE,
        chunk_text="text",
        query_id=1,
        passage_index=0,
        target_lang="hi",
        source_lang="en",
        query="query",
        eng_query="query",
        query_type=None,
        answer=None,
        eng_answer=None,
        is_selected=False,
    )
    
    chunk2 = Chunk.from_passage_segment(
        document_id="doc123",
        chunk_index=0,
        strategy=ChunkingStrategy.PASSAGE,
        chunk_text="text",
        query_id=1,
        passage_index=0,
        target_lang="hi",
        source_lang="en",
        query="query",
        eng_query="query",
        query_type=None,
        answer=None,
        eng_answer=None,
        is_selected=False,
    )
    
    # Same inputs should produce same chunk_id
    assert chunk1.chunk_id == chunk2.chunk_id


def test_chunk_different_index_different_id():
    """Test that different chunk_index produces different chunk_id."""
    chunk0 = create_valid_chunk(chunk_index=0)
    chunk1 = create_valid_chunk(chunk_index=1)
    chunk2 = create_valid_chunk(chunk_index=2)
    
    assert chunk0.chunk_id != chunk1.chunk_id
    assert chunk0.chunk_id != chunk2.chunk_id
    assert chunk1.chunk_id != chunk2.chunk_id


def test_chunk_different_strategy_different_id():
    """Test that different strategy produces different chunk_id."""
    chunk_passage = create_valid_chunk(strategy=ChunkingStrategy.PASSAGE)
    chunk_sentence = create_valid_chunk(strategy=ChunkingStrategy.SENTENCE)
    chunk_token = create_valid_chunk(strategy=ChunkingStrategy.TOKEN)
    
    assert chunk_passage.chunk_id != chunk_sentence.chunk_id
    assert chunk_passage.chunk_id != chunk_token.chunk_id
    assert chunk_sentence.chunk_id != chunk_token.chunk_id


def test_chunk_hindi_unicode_preservation():
    """Test that Hindi Unicode text is preserved."""
    chunk = Chunk.from_passage_segment(
        document_id="doc",
        chunk_index=0,
        strategy=ChunkingStrategy.SENTENCE,
        chunk_text="भारत की राजधानी नई दिल्ली है।",
        query_id=1,
        passage_index=0,
        target_lang="hi",
        source_lang="en",
        query="भारत की राजधानी?",
        eng_query="What is India's capital?",
        query_type="LOCATION",
        answer="नई दिल्ली",
        eng_answer="New Delhi",
        is_selected=True,
    )
    
    assert "भारत" in chunk.chunk_text
    assert "नई दिल्ली" in chunk.chunk_text
    assert "भारत" in chunk.query
    assert "नई दिल्ली" in chunk.answer


def test_chunk_metadata_preservation():
    """Test that source passage metadata is preserved."""
    chunk = Chunk.from_passage_segment(
        document_id="doc123",
        chunk_index=0,
        strategy=ChunkingStrategy.PASSAGE,
        chunk_text="chunk text",
        query_id=456,
        passage_index=2,
        target_lang="hi",
        source_lang="en",
        query="query text",
        eng_query="eng query text",
        query_type="TEST_TYPE",
        answer="answer text",
        eng_answer="eng answer text",
        is_selected=True,
    )
    
    # Verify all metadata preserved
    assert chunk.document_id == "doc123"
    assert chunk.query_id == 456
    assert chunk.passage_index == 2
    assert chunk.target_lang == "hi"
    assert chunk.source_lang == "en"
    assert chunk.query == "query text"
    assert chunk.eng_query == "eng query text"
    assert chunk.query_type == "TEST_TYPE"
    assert chunk.answer == "answer text"
    assert chunk.eng_answer == "eng answer text"
    assert chunk.is_selected is True


def test_chunk_optional_metadata_handling():
    """Test that optional metadata fields work correctly."""
    # Without optional fields
    chunk1 = Chunk.from_passage_segment(
        document_id="doc",
        chunk_index=0,
        strategy=ChunkingStrategy.PASSAGE,
        chunk_text="text",
        query_id=1,
        passage_index=0,
        target_lang="hi",
        source_lang="en",
        query="query",
        eng_query="query",
        query_type=None,
        answer=None,
        eng_answer=None,
        is_selected=False,
    )
    
    assert chunk1.query_type is None
    assert chunk1.answer is None
    assert chunk1.eng_answer is None
    assert chunk1.character_count is None
    assert chunk1.token_count is None
    
    # With optional fields
    chunk2 = Chunk.from_passage_segment(
        document_id="doc",
        chunk_index=0,
        strategy=ChunkingStrategy.TOKEN,
        chunk_text="text",
        query_id=1,
        passage_index=0,
        target_lang="hi",
        source_lang="en",
        query="query",
        eng_query="query",
        query_type="TEST",
        answer="answer",
        eng_answer="answer",
        is_selected=False,
        character_count=100,
        token_count=20,
        start_offset=0,
        end_offset=100,
        overlap_before=5,
        overlap_after=5,
    )
    
    assert chunk2.character_count == 100
    assert chunk2.token_count == 20
    assert chunk2.start_offset == 0
    assert chunk2.end_offset == 100
    assert chunk2.overlap_before == 5
    assert chunk2.overlap_after == 5


def test_chunk_strategy_values():
    """Test that all strategy enum values are valid."""
    # Test all strategies
    for strategy in ChunkingStrategy:
        chunk = create_valid_chunk(strategy=strategy)
        assert chunk.strategy == strategy


def test_chunk_character_count_validation():
    """Test that character_count must be non-negative."""
    with pytest.raises(ValidationError):
        Chunk.from_passage_segment(
            document_id="doc",
            chunk_index=0,
            strategy=ChunkingStrategy.PASSAGE,
            chunk_text="text",
            query_id=1,
            passage_index=0,
            target_lang="hi",
            source_lang="en",
            query="query",
            eng_query="query",
            query_type=None,
            answer=None,
            eng_answer=None,
            is_selected=False,
            character_count=-1,  # Negative
        )


def test_chunk_token_count_validation():
    """Test that token_count must be non-negative."""
    with pytest.raises(ValidationError):
        Chunk.from_passage_segment(
            document_id="doc",
            chunk_index=0,
            strategy=ChunkingStrategy.TOKEN,
            chunk_text="text",
            query_id=1,
            passage_index=0,
            target_lang="hi",
            source_lang="en",
            query="query",
            eng_query="query",
            query_type=None,
            answer=None,
            eng_answer=None,
            is_selected=False,
            token_count=-1,  # Negative
        )


def test_chunk_offset_validation():
    """Test that end_offset must be >= start_offset."""
    # Valid offsets
    chunk_valid = Chunk.from_passage_segment(
        document_id="doc",
        chunk_index=0,
        strategy=ChunkingStrategy.SENTENCE,
        chunk_text="text",
        query_id=1,
        passage_index=0,
        target_lang="hi",
        source_lang="en",
        query="query",
        eng_query="query",
        query_type=None,
        answer=None,
        eng_answer=None,
        is_selected=False,
        start_offset=10,
        end_offset=20,
    )
    assert chunk_valid.start_offset == 10
    assert chunk_valid.end_offset == 20
    
    # Invalid: end < start
    with pytest.raises(ValidationError, match="end_offset.*must be"):
        Chunk.from_passage_segment(
            document_id="doc",
            chunk_index=0,
            strategy=ChunkingStrategy.SENTENCE,
            chunk_text="text",
            query_id=1,
            passage_index=0,
            target_lang="hi",
            source_lang="en",
            query="query",
            eng_query="query",
            query_type=None,
            answer=None,
            eng_answer=None,
            is_selected=False,
            start_offset=20,
            end_offset=10,  # Less than start
        )


def test_chunk_to_dict():
    """Test serialization to dict."""
    chunk = create_valid_chunk()
    chunk_dict = chunk.to_dict()
    
    assert isinstance(chunk_dict, dict)
    assert chunk_dict["chunk_id"] == chunk.chunk_id
    assert chunk_dict["document_id"] == chunk.document_id
    assert chunk_dict["chunk_index"] == chunk.chunk_index
    assert chunk_dict["strategy"] == chunk.strategy.value
    assert chunk_dict["chunk_text"] == chunk.chunk_text


def test_chunk_repr():
    """Test string representation."""
    chunk = create_valid_chunk()
    repr_str = repr(chunk)
    
    assert "Chunk" in repr_str
    assert "chunk_index=0" in repr_str
    assert "strategy=passage" in repr_str
    assert "lang=hi" in repr_str


def test_chunk_generate_id_directly():
    """Test direct chunk ID generation."""
    id1 = Chunk.generate_chunk_id("doc123", ChunkingStrategy.PASSAGE, 0)
    id2 = Chunk.generate_chunk_id("doc123", ChunkingStrategy.PASSAGE, 0)
    id3 = Chunk.generate_chunk_id("doc123", ChunkingStrategy.SENTENCE, 0)
    
    # Same inputs = same ID
    assert id1 == id2
    
    # Different strategy = different ID
    assert id1 != id3
    
    # All should be 64-char hex strings
    assert len(id1) == 64
    assert all(c in "0123456789abcdef" for c in id1)
