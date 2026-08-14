"""Tests for deduplication utilities.

Phase 2.2.5: Deduplication testing.
"""

import pytest

from app.dataset.deduplicator import (
    IncrementalDeduplicator,
    deduplicate_passages,
)
from app.dataset.models import CanonicalPassage


def create_test_passage(
    query_id=1,
    passage_index=0,
    query="Test query",
    translated=None,  # Will be auto-generated if None
    english=None,  # Will be auto-generated if None
    target_lang="hi",
    is_selected=True,
):
    """Helper to create test CanonicalPassage with unique content by default."""
    # Auto-generate unique content based on query_id and passage_index if not provided
    if translated is None:
        translated = f"Test passage q{query_id}p{passage_index}"
    if english is None:
        english = f"Test passage q{query_id}p{passage_index}"
    
    return CanonicalPassage.from_msmarco_record(
        query_id=query_id,
        query=query,
        query_type="TEST",
        answer="Test answer",
        source_lang="en",
        target_lang=target_lang,
        eng_query="Test query",
        eng_answer="Test answer",
        passage_index=passage_index,
        translated_passage=translated,
        english_passage=english,
        is_selected=is_selected,
    )


def test_deduplicate_no_duplicates():
    """Test deduplication with no duplicates."""
    passages = [
        create_test_passage(query_id=1, passage_index=0),
        create_test_passage(query_id=1, passage_index=1),
        create_test_passage(query_id=2, passage_index=0),
    ]
    
    result = deduplicate_passages(passages)
    
    assert result.total_input == 3
    assert result.total_output == 3
    assert result.identity_duplicates_removed == 0
    assert result.content_duplicates_removed == 0
    assert len(result.unique_records) == 3


def test_deduplicate_same_document_id():
    """Test identity duplicate detection (same document_id)."""
    passage1 = create_test_passage(query_id=1, passage_index=0)
    passage2 = create_test_passage(query_id=1, passage_index=0)  # Same document_id
    
    result = deduplicate_passages([passage1, passage2])
    
    assert result.total_input == 2
    assert result.total_output == 1
    assert result.identity_duplicates_removed == 1
    assert result.content_duplicates_removed == 0


def test_deduplicate_exact_content_duplicate():
    """Test content duplicate detection."""
    passages = [
        create_test_passage(
            query_id=1,
            passage_index=0,
            translated="Same passage",
            english="Same passage"
        ),
        create_test_passage(
            query_id=2,  # Different query_id
            passage_index=0,
            translated="Same passage",  # Same content
            english="Same passage"
        ),
    ]
    
    result = deduplicate_passages(passages)
    
    assert result.total_input == 2
    assert result.total_output == 1
    assert result.identity_duplicates_removed == 0
    assert result.content_duplicates_removed == 1


def test_deduplicate_same_content_different_languages():
    """Test that same content in different languages is NOT deduplicated."""
    passages = [
        create_test_passage(
            query_id=1,
            passage_index=0,
            translated="Same passage",
            english="Same passage",
            target_lang="hi"
        ),
        create_test_passage(
            query_id=2,
            passage_index=0,
            translated="Same passage",
            english="Same passage",
            target_lang="ta"  # Different language
        ),
    ]
    
    result = deduplicate_passages(passages)
    
    # Should NOT be deduplicated (different languages)
    assert result.total_output == 2
    assert result.content_duplicates_removed == 0


def test_deduplicate_conflicting_is_selected():
    """Test handling of conflicting is_selected values."""
    passages = [
        create_test_passage(
            query_id=1,
            passage_index=0,
            translated="Passage",
            english="Passage",
            is_selected=False
        ),
        create_test_passage(
            query_id=2,
            passage_index=0,
            translated="Passage",
            english="Passage",
            is_selected=True  # Conflict!
        ),
    ]
    
    result = deduplicate_passages(passages, keep_relevance_priority=True)
    
    assert result.total_output == 1
    assert result.relevance_conflicts == 1
    
    # Should keep the one with is_selected=True
    kept_record = result.unique_records[0]
    assert kept_record.is_selected is True


def test_deduplicate_identical_is_selected():
    """Test deduplication with identical is_selected values."""
    passages = [
        create_test_passage(
            query_id=1,
            passage_index=0,
            translated="Passage",
            english="Passage",
            is_selected=True
        ),
        create_test_passage(
            query_id=2,
            passage_index=0,
            translated="Passage",
            english="Passage",
            is_selected=True  # Same
        ),
    ]
    
    result = deduplicate_passages(passages)
    
    assert result.total_output == 1
    assert result.relevance_conflicts == 0
    assert result.content_duplicates_removed == 1


def test_deduplicate_multiple_duplicate_groups():
    """Test multiple separate duplicate groups."""
    passages = [
        # Group 1: passage A
        create_test_passage(query_id=1, passage_index=0, translated="A", english="A"),
        create_test_passage(query_id=2, passage_index=0, translated="A", english="A"),
        create_test_passage(query_id=3, passage_index=0, translated="A", english="A"),
        # Group 2: passage B
        create_test_passage(query_id=4, passage_index=0, translated="B", english="B"),
        create_test_passage(query_id=5, passage_index=0, translated="B", english="B"),
        # Unique passage C
        create_test_passage(query_id=6, passage_index=0, translated="C", english="C"),
    ]
    
    result = deduplicate_passages(passages)
    
    assert result.total_input == 6
    assert result.total_output == 3  # A, B, C
    assert result.content_duplicates_removed == 3  # 2 from A, 1 from B


def test_deduplicate_is_deterministic():
    """Test that deduplication produces deterministic output."""
    passages = [
        create_test_passage(query_id=i, passage_index=0, translated="Same", english="Same")
        for i in range(5)
    ]
    
    result1 = deduplicate_passages(passages.copy())
    result2 = deduplicate_passages(passages.copy())
    
    assert result1.total_output == result2.total_output
    assert result1.content_duplicates_removed == result2.content_duplicates_removed


def test_deduplicate_empty_input():
    """Test deduplication with empty input."""
    result = deduplicate_passages([])
    
    assert result.total_input == 0
    assert result.total_output == 0
    assert result.identity_duplicates_removed == 0
    assert result.content_duplicates_removed == 0
    assert len(result.unique_records) == 0


def test_deduplicate_unicode_hindi_content():
    """Test deduplication with Hindi Unicode content."""
    passages = [
        create_test_passage(
            query_id=1,
            passage_index=0,
            translated="भारत की राजधानी नई दिल्ली है।",
            english="The capital of India is New Delhi."
        ),
        create_test_passage(
            query_id=2,
            passage_index=0,
            translated="भारत की राजधानी नई दिल्ली है।",  # Same Hindi
            english="The capital of India is New Delhi."
        ),
    ]
    
    result = deduplicate_passages(passages)
    
    assert result.total_output == 1
    assert result.content_duplicates_removed == 1
    assert "भारत" in result.unique_records[0].translated_passage


def test_deduplicate_preserves_original_records():
    """Test that original passage objects are not mutated."""
    original = create_test_passage(query_id=1, passage_index=0)
    original_doc_id = original.document_id
    original_query = original.query
    
    result = deduplicate_passages([original])
    
    # Original should be unchanged
    assert original.document_id == original_doc_id
    assert original.query == original_query


def test_incremental_deduplicator_initialization():
    """Test IncrementalDeduplicator initialization."""
    deduper = IncrementalDeduplicator()
    
    stats = deduper.get_statistics()
    assert stats["total_processed"] == 0
    assert stats["unique_records"] == 0


def test_incremental_deduplicator_single_batch():
    """Test incremental deduplication with single batch."""
    deduper = IncrementalDeduplicator()
    
    passages = [
        create_test_passage(query_id=1, passage_index=0),
        create_test_passage(query_id=2, passage_index=0),
    ]
    
    unique = deduper.process_batch(passages)
    
    assert len(unique) == 2
    stats = deduper.get_statistics()
    assert stats["total_processed"] == 2
    assert stats["unique_records"] == 2


def test_incremental_deduplicator_multiple_batches():
    """Test incremental deduplication across multiple batches."""
    deduper = IncrementalDeduplicator()
    
    # Batch 1
    batch1 = [
        create_test_passage(query_id=1, passage_index=0, translated="A", english="A"),
        create_test_passage(query_id=2, passage_index=0, translated="B", english="B"),
    ]
    unique1 = deduper.process_batch(batch1)
    assert len(unique1) == 2
    
    # Batch 2 with duplicate from batch 1
    batch2 = [
        create_test_passage(query_id=3, passage_index=0, translated="A", english="A"),  # Duplicate!
        create_test_passage(query_id=4, passage_index=0, translated="C", english="C"),
    ]
    unique2 = deduper.process_batch(batch2)
    assert len(unique2) == 1  # Only C is new
    
    stats = deduper.get_statistics()
    assert stats["total_processed"] == 4
    assert stats["unique_records"] == 3  # A, B, C
    assert stats["content_duplicates_removed"] == 1


def test_incremental_deduplicator_reset():
    """Test resetting incremental deduplicator."""
    deduper = IncrementalDeduplicator()
    
    passages = [create_test_passage(query_id=1, passage_index=0)]
    deduper.process_batch(passages)
    
    stats_before = deduper.get_statistics()
    assert stats_before["total_processed"] > 0
    
    deduper.reset()
    
    stats_after = deduper.get_statistics()
    assert stats_after["total_processed"] == 0
    assert stats_after["unique_records"] == 0
