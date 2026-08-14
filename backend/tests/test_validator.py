"""Tests for validation utilities.

Phase 2.2.6: Validation testing.
"""

import pytest

from app.dataset.models import CanonicalPassage
from app.dataset.validator import validate_batch, validate_passage


def create_valid_passage():
    """Helper to create a valid CanonicalPassage."""
    return CanonicalPassage.from_msmarco_record(
        query_id=123,
        query="भारत की राजधानी?",
        query_type="LOCATION",
        answer="नई दिल्ली",
        source_lang="en",
        target_lang="hi",
        eng_query="What is India's capital?",
        eng_answer="New Delhi",
        passage_index=0,
        translated_passage="भारत की राजधानी नई दिल्ली है।",
        english_passage="The capital of India is New Delhi.",
        is_selected=True,
    )


def test_validate_valid_record():
    """Test validation of a valid record."""
    passage = create_valid_passage()
    result = validate_passage(passage)
    
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.record == passage


def test_validate_empty_query():
    """Test validation fails for empty query."""
    passage = create_valid_passage()
    # Bypass Pydantic validation for testing
    passage.__dict__["query"] = ""
    
    result = validate_passage(passage)
    
    assert result.is_valid is False
    assert any(e.error_type == "empty_query" for e in result.errors)


def test_validate_empty_translated_passage():
    """Test validation fails for empty translated passage."""
    passage = create_valid_passage()
    passage.__dict__["translated_passage"] = ""
    
    result = validate_passage(passage)
    
    assert result.is_valid is False
    assert any(e.error_type == "empty_translated_passage" for e in result.errors)


def test_validate_empty_english_passage():
    """Test validation fails for empty English passage."""
    passage = create_valid_passage()
    passage.__dict__["english_passage"] = ""
    
    result = validate_passage(passage)
    
    assert result.is_valid is False
    assert any(e.error_type == "empty_english_passage" for e in result.errors)


def test_validate_whitespace_only_required_field():
    """Test validation fails for whitespace-only required field."""
    passage = create_valid_passage()
    passage.__dict__["query"] = "   "
    
    result = validate_passage(passage)
    
    assert result.is_valid is False
    assert any(e.error_type == "empty_query" for e in result.errors)


def test_validate_missing_optional_answer():
    """Test validation passes when optional answer is None."""
    passage = CanonicalPassage.from_msmarco_record(
        query_id=123,
        query="Test query",
        query_type=None,
        answer=None,  # Optional
        source_lang="en",
        target_lang="hi",
        eng_query="Test query",
        eng_answer=None,  # Optional
        passage_index=0,
        translated_passage="Test passage",
        english_passage="Test passage",
        is_selected=True,
    )
    
    result = validate_passage(passage)
    
    assert result.is_valid is True


def test_validate_control_characters():
    """Test validation fails for invalid control characters."""
    passage = create_valid_passage()
    passage.__dict__["query"] = "Test\x00query"  # Null byte
    
    result = validate_passage(passage)
    
    assert result.is_valid is False
    assert any(e.error_type == "invalid_control_chars" for e in result.errors)


def test_validate_multiple_errors():
    """Test validation reports multiple errors."""
    passage = create_valid_passage()
    passage.__dict__["query"] = ""
    passage.__dict__["translated_passage"] = ""
    passage.__dict__["english_passage"] = ""
    
    result = validate_passage(passage)
    
    assert result.is_valid is False
    assert len(result.errors) >= 3


def test_validate_batch():
    """Test batch validation."""
    valid1 = create_valid_passage()
    valid2 = create_valid_passage()
    
    result = validate_batch([valid1, valid2])
    
    assert result.total_count == 2
    assert result.valid_count == 2
    assert result.invalid_count == 0
    assert len(result.valid_records) == 2


def test_validate_batch_with_invalid_records():
    """Test batch validation with mixed valid/invalid records."""
    valid = create_valid_passage()
    invalid = create_valid_passage()
    invalid.__dict__["query"] = ""
    
    result = validate_batch([valid, invalid])
    
    assert result.total_count == 2
    assert result.valid_count == 1
    assert result.invalid_count == 1
    assert len(result.valid_records) == 1
    assert len(result.invalid_records) == 1


def test_validate_batch_empty():
    """Test batch validation with empty list."""
    result = validate_batch([])
    
    assert result.total_count == 0
    assert result.valid_count == 0
    assert result.invalid_count == 0


def test_validate_deterministic():
    """Test that validation is deterministic."""
    passage = create_valid_passage()
    
    result1 = validate_passage(passage)
    result2 = validate_passage(passage)
    
    assert result1.is_valid == result2.is_valid
    assert len(result1.errors) == len(result2.errors)


def test_validate_hindi_unicode():
    """Test validation with Hindi Unicode text."""
    passage = create_valid_passage()
    
    result = validate_passage(passage)
    
    assert result.is_valid is True
    assert "भारत" in passage.query


def test_validate_document_id_mismatch():
    """Test validation detects document_id mismatch."""
    passage = create_valid_passage()
    # Manually set incorrect document_id
    passage.__dict__["document_id"] = "wrong_id"
    
    result = validate_passage(passage)
    
    assert result.is_valid is False
    assert any(e.error_type == "inconsistent_document_id" for e in result.errors)


def test_validate_error_counts():
    """Test error count aggregation in batch validation."""
    passages = []
    
    # 2 with empty query
    for _ in range(2):
        p = create_valid_passage()
        p.__dict__["query"] = ""
        passages.append(p)
    
    # 1 with empty passage
    p = create_valid_passage()
    p.__dict__["translated_passage"] = ""
    passages.append(p)
    
    result = validate_batch(passages)
    
    assert result.error_counts.get("empty_query", 0) == 2
    assert result.error_counts.get("empty_translated_passage", 0) == 1
