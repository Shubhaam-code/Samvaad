"""Tests for processed dataset writer.

Phase 2.2.7: Processed dataset writer testing.
"""

import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from app.dataset.models import CanonicalPassage
from app.dataset.processed_writer import (
    CANONICAL_PASSAGE_SCHEMA,
    ProcessedDatasetWriter,
)


def create_test_passages(count=5):
    """Helper to create test CanonicalPassage records."""
    passages = []
    for i in range(count):
        passage = CanonicalPassage.from_msmarco_record(
            query_id=100 + i,
            query=f"Query {i}",
            query_type="TEST",
            answer=f"Answer {i}",
            source_lang="en",
            target_lang="hi",
            eng_query=f"Query {i}",
            eng_answer=f"Answer {i}",
            passage_index=0,
            translated_passage=f"Passage {i}",
            english_passage=f"Passage {i}",
            is_selected=(i % 2 == 0),
        )
        passages.append(passage)
    return passages


def test_writer_initialization(tmp_path):
    """Test writer initialization."""
    output_path = tmp_path / "test_output.parquet"
    
    writer = ProcessedDatasetWriter(output_path, overwrite=False)
    
    assert writer.output_path == output_path
    assert writer.records_written == 0
    assert writer.batches_written == 0
    assert not writer.is_closed


def test_writer_single_batch(tmp_path):
    """Test writing a single batch."""
    output_path = tmp_path / "single_batch.parquet"
    passages = create_test_passages(5)
    
    with ProcessedDatasetWriter(output_path) as writer:
        writer.write_batch(passages)
    
    assert output_path.exists()
    assert writer.records_written == 5
    assert writer.batches_written == 1


def test_writer_multiple_batches(tmp_path):
    """Test writing multiple batches."""
    output_path = tmp_path / "multiple_batches.parquet"
    
    with ProcessedDatasetWriter(output_path) as writer:
        writer.write_batch(create_test_passages(5))
        writer.write_batch(create_test_passages(3))
        writer.write_batch(create_test_passages(7))
    
    assert output_path.exists()
    assert writer.records_written == 15
    assert writer.batches_written == 3


def test_writer_schema():
    """Test that schema is correctly defined."""
    assert "document_id" in CANONICAL_PASSAGE_SCHEMA.names
    assert "query_id" in CANONICAL_PASSAGE_SCHEMA.names
    assert "passage_index" in CANONICAL_PASSAGE_SCHEMA.names
    assert "query" in CANONICAL_PASSAGE_SCHEMA.names
    assert "translated_passage" in CANONICAL_PASSAGE_SCHEMA.names
    assert "english_passage" in CANONICAL_PASSAGE_SCHEMA.names
    assert "is_selected" in CANONICAL_PASSAGE_SCHEMA.names


def test_writer_unicode_preservation(tmp_path):
    """Test that Unicode text (Hindi) is preserved correctly."""
    output_path = tmp_path / "unicode_test.parquet"
    
    passage = CanonicalPassage.from_msmarco_record(
        query_id=1,
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
    
    with ProcessedDatasetWriter(output_path) as writer:
        writer.write_batch([passage])
    
    # Read back and verify
    table = pq.read_table(output_path)
    record = table.to_pylist()[0]
    
    assert "भारत" in record["query"]
    assert "नई दिल्ली" in record["answer"]
    assert "राजधानी" in record["translated_passage"]


def test_writer_all_fields_roundtrip(tmp_path):
    """Test that all fields survive roundtrip."""
    output_path = tmp_path / "roundtrip.parquet"
    original = create_test_passages(1)[0]
    
    with ProcessedDatasetWriter(output_path) as writer:
        writer.write_batch([original])
    
    # Read back
    table = pq.read_table(output_path)
    record = table.to_pylist()[0]
    
    # Verify all fields
    assert record["document_id"] == original.document_id
    assert record["query_id"] == original.query_id
    assert record["passage_index"] == original.passage_index
    assert record["query"] == original.query
    assert record["query_type"] == original.query_type
    assert record["answer"] == original.answer
    assert record["source_lang"] == original.source_lang
    assert record["target_lang"] == original.target_lang
    assert record["eng_query"] == original.eng_query
    assert record["eng_answer"] == original.eng_answer
    assert record["translated_passage"] == original.translated_passage
    assert record["english_passage"] == original.english_passage
    assert record["is_selected"] == original.is_selected


def test_writer_empty_batch(tmp_path):
    """Test writing an empty batch."""
    output_path = tmp_path / "empty_batch.parquet"
    
    with ProcessedDatasetWriter(output_path) as writer:
        writer.write_batch([])  # Empty batch
        writer.write_batch(create_test_passages(5))  # Then real data
    
    assert writer.records_written == 5
    assert writer.batches_written == 1  # Empty batch shouldn't count


def test_writer_output_directory_creation(tmp_path):
    """Test that output directory is created if missing."""
    nested_path = tmp_path / "nested" / "dir" / "output.parquet"
    
    with ProcessedDatasetWriter(nested_path) as writer:
        writer.write_batch(create_test_passages(3))
    
    assert nested_path.exists()
    assert nested_path.parent.exists()


def test_writer_overwrite_protection(tmp_path):
    """Test that existing files are protected by default."""
    output_path = tmp_path / "existing.parquet"
    
    # Create existing file
    with ProcessedDatasetWriter(output_path, overwrite=True) as writer:
        writer.write_batch(create_test_passages(2))
    
    # Try to write again without overwrite
    with pytest.raises(FileExistsError, match="already exists"):
        ProcessedDatasetWriter(output_path, overwrite=False)


def test_writer_explicit_overwrite(tmp_path):
    """Test that overwrite=True allows replacing existing file."""
    output_path = tmp_path / "overwrite.parquet"
    
    # First write
    with ProcessedDatasetWriter(output_path, overwrite=True) as writer:
        writer.write_batch(create_test_passages(5))
    
    # Overwrite
    with ProcessedDatasetWriter(output_path, overwrite=True) as writer:
        writer.write_batch(create_test_passages(3))
    
    # Verify only latest data
    table = pq.read_table(output_path)
    assert len(table) == 3


def test_writer_close():
    """Test that writer can be closed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "close_test.parquet"
        
        writer = ProcessedDatasetWriter(output_path)
        writer.write_batch(create_test_passages(2))
        
        assert not writer.is_closed
        writer.close()
        assert writer.is_closed
        
        # Multiple closes should be safe
        writer.close()
        assert writer.is_closed


def test_writer_write_after_close_raises_error():
    """Test that writing after close raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "closed.parquet"
        
        writer = ProcessedDatasetWriter(output_path)
        writer.write_batch(create_test_passages(2))
        writer.close()
        
        with pytest.raises(RuntimeError, match="Cannot write to closed writer"):
            writer.write_batch(create_test_passages(1))


def test_writer_context_manager_failure_cleanup(tmp_path):
    """Test that failed writes clean up incomplete output."""
    output_path = tmp_path / "failed.parquet"
    
    try:
        with ProcessedDatasetWriter(output_path) as writer:
            writer.write_batch(create_test_passages(2))
            raise RuntimeError("Simulated failure")
    except RuntimeError:
        pass
    
    # File should be cleaned up after failure
    assert not output_path.exists()


def test_writer_correct_record_count(tmp_path):
    """Test that record count is accurate."""
    output_path = tmp_path / "count_test.parquet"
    
    writer = ProcessedDatasetWriter(output_path)
    
    assert writer.records_written == 0
    
    writer.write_batch(create_test_passages(5))
    assert writer.records_written == 5
    
    writer.write_batch(create_test_passages(10))
    assert writer.records_written == 15
    
    writer.close()


def test_writer_correct_batch_count(tmp_path):
    """Test that batch count is accurate."""
    output_path = tmp_path / "batch_count.parquet"
    
    writer = ProcessedDatasetWriter(output_path)
    
    assert writer.batches_written == 0
    
    writer.write_batch(create_test_passages(5))
    assert writer.batches_written == 1
    
    writer.write_batch([])  # Empty batch shouldn't count
    assert writer.batches_written == 1
    
    writer.write_batch(create_test_passages(3))
    assert writer.batches_written == 2
    
    writer.close()


def test_writer_repr():
    """Test string representation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "repr_test.parquet"
        
        writer = ProcessedDatasetWriter(output_path)
        repr_str = repr(writer)
        
        assert "ProcessedDatasetWriter" in repr_str
        assert "open" in repr_str
        
        writer.close()
        repr_str_closed = repr(writer)
        assert "closed" in repr_str_closed
