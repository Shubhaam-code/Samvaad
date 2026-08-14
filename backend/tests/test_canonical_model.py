"""Tests for the canonical passage data model.

Phase 2.2: Dataset preprocessing model validation.
"""

import pytest
from pydantic import ValidationError

from app.dataset.models import CanonicalPassage


def test_generate_document_id_is_deterministic():
    """Document ID generation must be deterministic for the same inputs."""
    id1 = CanonicalPassage.generate_document_id("hi", 123, 0)
    id2 = CanonicalPassage.generate_document_id("hi", 123, 0)
    assert id1 == id2
    assert len(id1) == 64  # SHA-256 hex string length


def test_generate_document_id_is_unique():
    """Different inputs must produce different document IDs."""
    id1 = CanonicalPassage.generate_document_id("hi", 123, 0)
    id2 = CanonicalPassage.generate_document_id("hi", 123, 1)
    id3 = CanonicalPassage.generate_document_id("hi", 124, 0)
    
    assert id1 != id2
    assert id1 != id3
    assert id2 != id3


def test_generate_document_id_is_unique_across_languages():
    """Same query_id and passage_index with different target_lang must produce different IDs."""
    # Same query_id and passage_index but different languages
    id_hindi = CanonicalPassage.generate_document_id("hi", 123, 0)
    id_tamil = CanonicalPassage.generate_document_id("ta", 123, 0)
    id_bengali = CanonicalPassage.generate_document_id("bn", 123, 0)
    id_telugu = CanonicalPassage.generate_document_id("te", 123, 0)
    
    # All IDs must be different
    ids = [id_hindi, id_tamil, id_bengali, id_telugu]
    assert len(ids) == len(set(ids)), "IDs for different languages must be unique"
    
    # Verify each is a valid 64-character hex string
    for doc_id in ids:
        assert len(doc_id) == 64
        assert all(c in "0123456789abcdef" for c in doc_id)


def test_from_msmarco_record_creates_valid_passage():
    """Factory method should create valid passage with auto-generated document_id."""
    passage = CanonicalPassage.from_msmarco_record(
        query_id=12345,
        query="भारत की राजधानी क्या है?",
        query_type="LOCATION",
        answer="नई दिल्ली",
        source_lang="en",
        target_lang="hi",
        eng_query="What is the capital of India?",
        eng_answer="New Delhi",
        passage_index=0,
        translated_passage="भारत की राजधानी नई दिल्ली है।",
        english_passage="The capital of India is New Delhi.",
        is_selected=True,
    )
    
    assert passage.query_id == 12345
    assert passage.passage_index == 0
    assert passage.is_selected is True
    assert passage.target_lang == "hi"
    assert len(passage.document_id) == 64
    
    # Verify document_id is deterministic and includes language
    expected_id = CanonicalPassage.generate_document_id("hi", 12345, 0)
    assert passage.document_id == expected_id


def test_canonical_passage_with_minimal_optional_fields():
    """Test passage with None for optional fields (answer, query_type)."""
    passage = CanonicalPassage.from_msmarco_record(
        query_id=999,
        query="Test query",
        query_type=None,
        answer=None,
        source_lang="en",
        target_lang="hi",
        eng_query="Test query",
        eng_answer=None,
        passage_index=0,
        translated_passage="Test passage in Hindi",
        english_passage="Test passage in English",
        is_selected=False,
    )
    
    assert passage.query_type is None
    assert passage.answer is None
    assert passage.eng_answer is None
    assert passage.is_selected is False


def test_canonical_passage_validates_required_fields():
    """Test that required text fields cannot be empty."""
    with pytest.raises(ValidationError) as exc_info:
        CanonicalPassage(
            document_id="test_id",
            query_id=1,
            passage_index=0,
            query="",  # Empty query should fail
            query_type=None,
            answer=None,
            source_lang="en",
            target_lang="hi",
            eng_query="Valid query",
            eng_answer=None,
            translated_passage="Valid passage",
            english_passage="Valid passage",
            is_selected=True,
        )
    
    errors = exc_info.value.errors()
    assert any("query" in str(e) for e in errors)


def test_canonical_passage_validates_non_empty_text():
    """Test validator rejects whitespace-only strings."""
    with pytest.raises(ValidationError):
        CanonicalPassage.from_msmarco_record(
            query_id=1,
            query="   ",  # Whitespace-only
            query_type=None,
            answer=None,
            source_lang="en",
            target_lang="hi",
            eng_query="Valid",
            eng_answer=None,
            passage_index=0,
            translated_passage="Valid",
            english_passage="Valid",
            is_selected=True,
        )


def test_canonical_passage_validates_passage_index_non_negative():
    """Passage index must be >= 0."""
    with pytest.raises(ValidationError):
        CanonicalPassage.from_msmarco_record(
            query_id=1,
            query="Valid query",
            query_type=None,
            answer=None,
            source_lang="en",
            target_lang="hi",
            eng_query="Valid query",
            eng_answer=None,
            passage_index=-1,  # Negative index should fail
            translated_passage="Valid passage",
            english_passage="Valid passage",
            is_selected=True,
        )


def test_canonical_passage_validates_language_codes():
    """Language codes must be 2-10 characters."""
    # Too short
    with pytest.raises(ValidationError):
        CanonicalPassage.from_msmarco_record(
            query_id=1,
            query="Valid",
            query_type=None,
            answer=None,
            source_lang="e",  # Too short
            target_lang="hi",
            eng_query="Valid",
            eng_answer=None,
            passage_index=0,
            translated_passage="Valid",
            english_passage="Valid",
            is_selected=True,
        )
    
    # Too long
    with pytest.raises(ValidationError):
        CanonicalPassage.from_msmarco_record(
            query_id=1,
            query="Valid",
            query_type=None,
            answer=None,
            source_lang="en",
            target_lang="this_is_too_long",  # Too long
            eng_query="Valid",
            eng_answer=None,
            passage_index=0,
            translated_passage="Valid",
            english_passage="Valid",
            is_selected=True,
        )


def test_canonical_passage_to_dict():
    """Test serialization to dictionary."""
    passage = CanonicalPassage.from_msmarco_record(
        query_id=100,
        query="Test query",
        query_type="TEST",
        answer="Test answer",
        source_lang="en",
        target_lang="hi",
        eng_query="Test query",
        eng_answer="Test answer",
        passage_index=0,
        translated_passage="Test passage",
        english_passage="Test passage",
        is_selected=True,
    )
    
    passage_dict = passage.to_dict()
    
    assert isinstance(passage_dict, dict)
    assert passage_dict["query_id"] == 100
    assert passage_dict["passage_index"] == 0
    assert passage_dict["is_selected"] is True
    assert "document_id" in passage_dict


def test_canonical_passage_repr():
    """Test string representation."""
    passage = CanonicalPassage.from_msmarco_record(
        query_id=555,
        query="Test",
        query_type=None,
        answer=None,
        source_lang="en",
        target_lang="hi",
        eng_query="Test",
        eng_answer=None,
        passage_index=2,
        translated_passage="Test",
        english_passage="Test",
        is_selected=False,
    )
    
    repr_str = repr(passage)
    
    assert "CanonicalPassage" in repr_str
    assert "query_id=555" in repr_str
    assert "passage_index=2" in repr_str
    assert "is_selected=False" in repr_str
    assert "lang=hi" in repr_str


def test_multiple_passages_from_same_query():
    """Test creating multiple passages from the same query with different indices."""
    passages = []
    for i in range(3):
        passage = CanonicalPassage.from_msmarco_record(
            query_id=777,
            query="Common query",
            query_type=None,
            answer=None,
            source_lang="en",
            target_lang="hi",
            eng_query="Common query",
            eng_answer=None,
            passage_index=i,
            translated_passage=f"Passage {i}",
            english_passage=f"Passage {i}",
            is_selected=(i == 0),  # Only first is selected
        )
        passages.append(passage)
    
    # All should have same query_id but different document_ids
    assert all(p.query_id == 777 for p in passages)
    assert len(set(p.document_id for p in passages)) == 3
    assert len(set(p.passage_index for p in passages)) == 3
    
    # Only first should be selected
    assert passages[0].is_selected is True
    assert passages[1].is_selected is False
    assert passages[2].is_selected is False


def test_canonical_passage_preserves_unicode():
    """Test that Unicode text in various languages is preserved correctly."""
    passage = CanonicalPassage.from_msmarco_record(
        query_id=1,
        query="भारत की राजधानी क्या है?",  # Hindi
        query_type=None,
        answer="नई दिल्ली",
        source_lang="en",
        target_lang="hi",
        eng_query="What is the capital of India?",
        eng_answer="New Delhi",
        passage_index=0,
        translated_passage="भारत की राजधानी नई दिल्ली है। यह देश का राजनीतिक केंद्र है।",
        english_passage="The capital of India is New Delhi. It is the political center of the country.",
        is_selected=True,
    )
    
    assert "भारत" in passage.query
    assert "नई दिल्ली" in passage.answer
    assert "राजधानी" in passage.translated_passage
    
    # Verify serialization preserves Unicode
    passage_dict = passage.to_dict()
    assert "भारत" in passage_dict["query"]


def test_same_query_different_languages_have_unique_ids():
    """Test that the same query in different languages produces unique document IDs."""
    # Create the same passage (same query_id and passage_index) in different languages
    passage_hindi = CanonicalPassage.from_msmarco_record(
        query_id=999,
        query="भारत की राजधानी क्या है?",
        query_type=None,
        answer=None,
        source_lang="en",
        target_lang="hi",
        eng_query="What is the capital of India?",
        eng_answer=None,
        passage_index=0,
        translated_passage="Test passage in Hindi",
        english_passage="Test passage in English",
        is_selected=True,
    )
    
    passage_tamil = CanonicalPassage.from_msmarco_record(
        query_id=999,  # Same query_id
        query="இந்தியாவின் தலைநகர் என்ன?",
        query_type=None,
        answer=None,
        source_lang="en",
        target_lang="ta",  # Different language
        eng_query="What is the capital of India?",
        eng_answer=None,
        passage_index=0,  # Same passage_index
        translated_passage="Test passage in Tamil",
        english_passage="Test passage in English",
        is_selected=True,
    )
    
    passage_bengali = CanonicalPassage.from_msmarco_record(
        query_id=999,  # Same query_id
        query="ভারতের রাজধানী কী?",
        query_type=None,
        answer=None,
        source_lang="en",
        target_lang="bn",  # Different language
        eng_query="What is the capital of India?",
        eng_answer=None,
        passage_index=0,  # Same passage_index
        translated_passage="Test passage in Bengali",
        english_passage="Test passage in English",
        is_selected=True,
    )
    
    # All should have same query_id and passage_index but different document_ids
    assert passage_hindi.query_id == passage_tamil.query_id == passage_bengali.query_id == 999
    assert passage_hindi.passage_index == passage_tamil.passage_index == passage_bengali.passage_index == 0
    
    # Document IDs must all be different
    assert passage_hindi.document_id != passage_tamil.document_id
    assert passage_hindi.document_id != passage_bengali.document_id
    assert passage_tamil.document_id != passage_bengali.document_id
    
    # Verify each matches its expected deterministic ID
    assert passage_hindi.document_id == CanonicalPassage.generate_document_id("hi", 999, 0)
    assert passage_tamil.document_id == CanonicalPassage.generate_document_id("ta", 999, 0)
    assert passage_bengali.document_id == CanonicalPassage.generate_document_id("bn", 999, 0)
