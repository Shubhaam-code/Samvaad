"""Tests for passage flattening utilities.

Phase 2.2.3: Passage flattening testing.
"""

import pytest

from app.dataset.models import CanonicalPassage
from app.dataset.passage_flattener import (
    MalformedRecordError,
    _parse_is_selected,
    flatten_msmarco_batch,
    flatten_msmarco_record,
)


def create_test_record(
    query_id=123,
    num_passages=3,
    query="Test query",
    eng_query="Test query",
):
    """Helper to create a test MSMARCO-XI style record."""
    return {
        "query_id": query_id,
        "Query": query,
        "Eng_Query": eng_query,
        "Answer": "Test answer",
        "Eng_Answer": "Test answer",
        "query_type": "TEST",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "Translated_passages": [f"अनुच्छेद {i}" for i in range(num_passages)],
            "English_passages": [f"Passage {i}" for i in range(num_passages)],
            "is_selected": [1 if i == 0 else 0 for i in range(num_passages)],
        },
    }


def test_flatten_normal_record_with_three_passages():
    """Test flattening a normal record with 3 passages."""
    record = create_test_record(query_id=123, num_passages=3)
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    assert len(passages) == 3
    assert all(isinstance(p, CanonicalPassage) for p in passages)
    assert all(p.query_id == 123 for p in passages)
    assert [p.passage_index for p in passages] == [0, 1, 2]


def test_flatten_preserves_is_selected():
    """Test that is_selected values are correctly converted to boolean."""
    record = create_test_record(query_id=123, num_passages=3)
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    assert passages[0].is_selected is True  # is_selected was 1
    assert passages[1].is_selected is False  # is_selected was 0
    assert passages[2].is_selected is False  # is_selected was 0


def test_flatten_preserves_english_hindi_alignment():
    """Test that English and Hindi passages are correctly aligned."""
    record = create_test_record(query_id=123, num_passages=3)
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    # Check alignment by index
    assert passages[0].translated_passage == "अनुच्छेद 0"
    assert passages[0].english_passage == "Passage 0"
    
    assert passages[1].translated_passage == "अनुच्छेद 1"
    assert passages[1].english_passage == "Passage 1"
    
    assert passages[2].translated_passage == "अनुच्छेद 2"
    assert passages[2].english_passage == "Passage 2"


def test_flatten_preserves_query_metadata():
    """Test that query-level metadata is preserved in all passages."""
    record = create_test_record(
        query_id=456,
        num_passages=2,
        query="भारत की राजधानी?",
        eng_query="What is India's capital?",
    )
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    for passage in passages:
        assert passage.query_id == 456
        assert passage.query == "भारत की राजधानी?"
        assert passage.eng_query == "What is India's capital?"
        assert passage.answer == "Test answer"
        assert passage.query_type == "TEST"
        assert passage.source_lang == "en"
        assert passage.target_lang == "hi"


def test_flatten_multiple_records():
    """Test flattening multiple query records."""
    record1 = create_test_record(query_id=1, num_passages=2)
    record2 = create_test_record(query_id=2, num_passages=3)
    
    passages1 = flatten_msmarco_record(record1, normalize=False)
    passages2 = flatten_msmarco_record(record2, normalize=False)
    
    assert len(passages1) == 2
    assert len(passages2) == 3
    
    # Check query_ids are different
    assert all(p.query_id == 1 for p in passages1)
    assert all(p.query_id == 2 for p in passages2)


def test_flatten_handles_empty_passages():
    """Test handling of record with empty passage lists."""
    record = create_test_record(query_id=123, num_passages=3)
    record["passages"]["Translated_passages"] = []
    record["passages"]["English_passages"] = []
    record["passages"]["is_selected"] = []
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    assert passages == []


def test_flatten_handles_missing_passages_field():
    """Test handling of record without passages field."""
    record = create_test_record(query_id=123, num_passages=3)
    del record["passages"]
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    assert passages == []


def test_flatten_handles_none_passages_field():
    """Test handling of record with None passages field."""
    record = create_test_record(query_id=123, num_passages=3)
    record["passages"] = None
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    assert passages == []


def test_flatten_handles_unequal_passage_list_lengths():
    """Test handling of unequal list lengths (uses minimum length)."""
    record = create_test_record(query_id=123, num_passages=3)
    # Make lists different lengths
    record["passages"]["Translated_passages"] = ["अनुच्छेद 0", "अनुच्छेद 1"]  # 2 items
    record["passages"]["English_passages"] = ["Passage 0", "Passage 1", "Passage 2"]  # 3 items
    record["passages"]["is_selected"] = [1, 0, 0, 0]  # 4 items
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    # Should use minimum length (2) to preserve alignment
    assert len(passages) == 2
    assert passages[0].translated_passage == "अनुच्छेद 0"
    assert passages[0].english_passage == "Passage 0"
    assert passages[1].translated_passage == "अनुच्छेद 1"
    assert passages[1].english_passage == "Passage 1"


def test_flatten_handles_invalid_is_selected_values():
    """Test handling of invalid is_selected values."""
    record = create_test_record(query_id=123, num_passages=2)
    record["passages"]["is_selected"] = ["invalid", 0]
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    # First passage should default to False (invalid value)
    # Second passage should be False (0 value)
    assert len(passages) == 2
    assert passages[0].is_selected is False
    assert passages[1].is_selected is False


def test_flatten_handles_null_passage_text():
    """Test handling of null passage text (skips that passage)."""
    record = create_test_record(query_id=123, num_passages=3)
    record["passages"]["Translated_passages"][1] = None  # Null middle passage
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    # Should skip passage with index 1
    assert len(passages) == 2
    assert passages[0].passage_index == 0
    assert passages[1].passage_index == 2  # Skipped 1


def test_flatten_with_unicode_hindi_text():
    """Test flattening with real Hindi Unicode text."""
    record = {
        "query_id": 999,
        "Query": "भारत की राजधानी क्या है?",
        "Eng_Query": "What is the capital of India?",
        "Answer": "नई दिल्ली",
        "Eng_Answer": "New Delhi",
        "query_type": "LOCATION",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "Translated_passages": [
                "भारत की राजधानी नई दिल्ली है।",
                "नई दिल्ली भारत का राजनीतिक केंद्र है।",
            ],
            "English_passages": [
                "The capital of India is New Delhi.",
                "New Delhi is the political center of India.",
            ],
            "is_selected": [1, 0],
        },
    }
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    assert len(passages) == 2
    assert "भारत" in passages[0].query
    assert "नई दिल्ली" in passages[0].answer
    assert "राजधानी" in passages[0].translated_passage


def test_flatten_raises_on_missing_query_id():
    """Test that missing query_id raises MalformedRecordError."""
    record = create_test_record(query_id=123, num_passages=2)
    del record["query_id"]
    
    with pytest.raises(MalformedRecordError, match="query_id"):
        flatten_msmarco_record(record, normalize=False)


def test_flatten_raises_on_invalid_query_id():
    """Test that invalid query_id raises MalformedRecordError."""
    record = create_test_record(query_id=123, num_passages=2)
    record["query_id"] = "not_a_number"
    
    with pytest.raises(MalformedRecordError, match="query_id"):
        flatten_msmarco_record(record, normalize=False)


def test_flatten_with_normalization_enabled():
    """Test that normalization is applied when enabled."""
    record = {
        "query_id": 123,
        "Query": "  Test   query  \n with  whitespace  ",
        "Eng_Query": "  Test   query  \n with  whitespace  ",
        "Answer": "  Test   answer  ",
        "Eng_Answer": "  Test   answer  ",
        "query_type": "TEST",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "Translated_passages": ["  Passage   with\nwhitespace  "],
            "English_passages": ["  Passage   with\nwhitespace  "],
            "is_selected": [1],
        },
    }
    
    passages = flatten_msmarco_record(record, normalize=True)
    
    assert len(passages) == 1
    passage = passages[0]
    
    # Check normalization applied
    assert passage.query == "Test query with whitespace"
    assert passage.eng_query == "Test query with whitespace"
    assert passage.answer == "Test answer"
    assert passage.translated_passage == "Passage with whitespace"
    assert passage.english_passage == "Passage with whitespace"
    
    # No double spaces
    assert "  " not in passage.query
    assert "  " not in passage.translated_passage


def test_flatten_with_normalization_disabled():
    """Test that normalization is not applied when disabled."""
    record = {
        "query_id": 123,
        "Query": "  Test   query  ",
        "Eng_Query": "  Test   query  ",
        "Answer": None,
        "Eng_Answer": None,
        "query_type": "TEST",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "Translated_passages": ["  Passage  "],
            "English_passages": ["  Passage  "],
            "is_selected": [1],
        },
    }
    
    passages = flatten_msmarco_record(record, normalize=False)
    
    assert len(passages) == 1
    passage = passages[0]
    
    # Whitespace should be preserved
    assert passage.query == "  Test   query  "
    assert passage.translated_passage == "  Passage  "


def test_flatten_batch():
    """Test batch flattening of multiple records."""
    records = [
        create_test_record(query_id=1, num_passages=2),
        create_test_record(query_id=2, num_passages=3),
        create_test_record(query_id=3, num_passages=1),
    ]
    
    all_passages = flatten_msmarco_batch(records, normalize=False)
    
    # Should have 2 + 3 + 1 = 6 total passages
    assert len(all_passages) == 6
    
    # Check query_ids
    query_ids = [p.query_id for p in all_passages]
    assert query_ids == [1, 1, 2, 2, 2, 3]


def test_flatten_batch_skips_malformed_records():
    """Test that malformed records are skipped in batch processing."""
    records = [
        create_test_record(query_id=1, num_passages=2),
        {"invalid": "record"},  # Malformed
        create_test_record(query_id=3, num_passages=1),
    ]
    
    all_passages = flatten_msmarco_batch(records, normalize=False)
    
    # Should have 2 + 0 + 1 = 3 passages (malformed skipped)
    assert len(all_passages) == 3
    assert [p.query_id for p in all_passages] == [1, 1, 3]


def test_parse_is_selected_bool():
    """Test parsing boolean is_selected values."""
    assert _parse_is_selected(True) is True
    assert _parse_is_selected(False) is False


def test_parse_is_selected_int():
    """Test parsing integer is_selected values."""
    assert _parse_is_selected(0) is False
    assert _parse_is_selected(1) is True


def test_parse_is_selected_string():
    """Test parsing string is_selected values."""
    assert _parse_is_selected("0") is False
    assert _parse_is_selected("1") is True
    assert _parse_is_selected("false") is False
    assert _parse_is_selected("true") is True
    assert _parse_is_selected("False") is False
    assert _parse_is_selected("True") is True


def test_parse_is_selected_rejects_invalid_int():
    """Test that invalid integer values are rejected."""
    with pytest.raises(ValueError, match="Invalid int value"):
        _parse_is_selected(2)
    
    with pytest.raises(ValueError, match="Invalid int value"):
        _parse_is_selected(-1)


def test_parse_is_selected_rejects_invalid_string():
    """Test that invalid string values are rejected."""
    with pytest.raises(ValueError, match="Invalid string value"):
        _parse_is_selected("yes")
    
    with pytest.raises(ValueError, match="Invalid string value"):
        _parse_is_selected("invalid")


def test_parse_is_selected_rejects_invalid_type():
    """Test that unsupported types are rejected."""
    with pytest.raises(ValueError, match="Unsupported type"):
        _parse_is_selected([1])
    
    with pytest.raises(ValueError, match="Unsupported type"):
        _parse_is_selected({"value": 1})


def test_flatten_generates_deterministic_document_ids():
    """Test that document IDs are deterministic and unique."""
    record = create_test_record(query_id=123, num_passages=3)
    
    passages1 = flatten_msmarco_record(record, normalize=False)
    passages2 = flatten_msmarco_record(record, normalize=False)
    
    # Same query_id and passage_index should produce same document_id
    for p1, p2 in zip(passages1, passages2):
        assert p1.document_id == p2.document_id
    
    # Different passage_index should produce different document_id
    assert len(set(p.document_id for p in passages1)) == 3


def test_flatten_document_ids_include_language():
    """Test that document IDs are unique across languages."""
    # Same query_id and passage_index but different languages
    record_hi = create_test_record(query_id=123, num_passages=1)
    record_hi["target_lang"] = "hi"
    
    record_ta = create_test_record(query_id=123, num_passages=1)
    record_ta["target_lang"] = "ta"
    
    passages_hi = flatten_msmarco_record(record_hi, normalize=False)
    passages_ta = flatten_msmarco_record(record_ta, normalize=False)
    
    # Different languages should produce different document_ids
    assert passages_hi[0].document_id != passages_ta[0].document_id
