"""Tests for integrated preprocessing pipeline.

Phase 2.2: Integration testing (Tasks 2.2.5-2.2.7).
"""

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.dataset.preprocessing_pipeline import (
    preprocess_dataset,
    preprocess_dataset_dry_run,
)


def create_test_msmarco_parquet(output_path, num_records=10):
    """Create a tiny synthetic MSMARCO-style Parquet file for testing."""
    schema = pa.schema([
        ("query_id", pa.int64()),
        ("Query", pa.string()),
        ("Eng_Query", pa.string()),
        ("Answer", pa.string()),
        ("Eng_Answer", pa.string()),
        ("query_type", pa.string()),
        ("source_lang", pa.string()),
        ("target_lang", pa.string()),
        ("passages", pa.struct([
            ("Translated_passages", pa.list_(pa.string())),
            ("English_passages", pa.list_(pa.string())),
            ("is_selected", pa.list_(pa.int32())),
        ])),
    ])
    
    # Create test data
    data = {
        "query_id": list(range(1, num_records + 1)),
        "Query": [f"Query {i}" for i in range(1, num_records + 1)],
        "Eng_Query": [f"Query {i}" for i in range(1, num_records + 1)],
        "Answer": [f"Answer {i}" for i in range(1, num_records + 1)],
        "Eng_Answer": [f"Answer {i}" for i in range(1, num_records + 1)],
        "query_type": ["TEST"] * num_records,
        "source_lang": ["en"] * num_records,
        "target_lang": ["hi"] * num_records,
        "passages": [
            {
                "Translated_passages": [f"Passage {i}a", f"Passage {i}b"],
                "English_passages": [f"Passage {i}a", f"Passage {i}b"],
                "is_selected": [1, 0],
            }
            for i in range(1, num_records + 1)
        ],
    }
    
    table = pa.Table.from_pydict(data, schema=schema)
    pq.write_table(table, output_path)
    return output_path


def test_preprocess_dataset_tiny_synthetic():
    """Test end-to-end preprocessing with tiny synthetic dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create input
        input_path = create_test_msmarco_parquet(tmpdir / "input.parquet", num_records=5)
        output_path = tmpdir / "output.parquet"
        
        # Run preprocessing
        stats = preprocess_dataset(
            input_path=input_path,
            output_path=output_path,
            batch_size=10,
            overwrite=False
        )
        
        # Verify statistics
        assert stats.input_records == 5
        assert stats.flattened_passages == 10  # 5 records * 2 passages each
        assert stats.records_written > 0
        assert stats.batches_processed == 1
        
        # Verify output file exists
        assert output_path.exists()
        
        # Read back and verify
        table = pq.read_table(output_path)
        assert len(table) > 0
        
        # Verify schema
        assert "document_id" in table.column_names
        assert "query_id" in table.column_names
        assert "passage_index" in table.column_names
        assert "is_selected" in table.column_names


def test_preprocess_dataset_multiple_batches():
    """Test preprocessing with multiple batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create larger input
        input_path = create_test_msmarco_parquet(tmpdir / "input.parquet", num_records=15)
        output_path = tmpdir / "output.parquet"
        
        # Run with small batch size to force multiple batches
        stats = preprocess_dataset(
            input_path=input_path,
            output_path=output_path,
            batch_size=5,  # Force 3 batches
            overwrite=False
        )
        
        assert stats.batches_processed == 3
        assert output_path.exists()


def test_preprocess_dataset_deduplication():
    """Test that deduplication works in preprocessing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create input with duplicates
        schema = pa.schema([
            ("query_id", pa.int64()),
            ("Query", pa.string()),
            ("Eng_Query", pa.string()),
            ("Answer", pa.string()),
            ("Eng_Answer", pa.string()),
            ("query_type", pa.string()),
            ("source_lang", pa.string()),
            ("target_lang", pa.string()),
            ("passages", pa.struct([
                ("Translated_passages", pa.list_(pa.string())),
                ("English_passages", pa.list_(pa.string())),
                ("is_selected", pa.list_(pa.int32())),
            ])),
        ])
        
        # Create records with duplicate content
        data = {
            "query_id": [1, 2],
            "Query": ["Query 1", "Query 2"],
            "Eng_Query": ["Query 1", "Query 2"],
            "Answer": ["Answer 1", "Answer 2"],
            "Eng_Answer": ["Answer 1", "Answer 2"],
            "query_type": ["TEST", "TEST"],
            "source_lang": ["en", "en"],
            "target_lang": ["hi", "hi"],
            "passages": [
                {
                    "Translated_passages": ["Same passage", "Different passage"],
                    "English_passages": ["Same passage", "Different passage"],
                    "is_selected": [1, 0],
                },
                {
                    "Translated_passages": ["Same passage", "Another passage"],
                    "English_passages": ["Same passage", "Another passage"],
                    "is_selected": [1, 0],
                },
            ],
        }
        
        input_path = tmpdir / "input.parquet"
        table = pa.Table.from_pydict(data, schema=schema)
        pq.write_table(table, input_path)
        
        output_path = tmpdir / "output.parquet"
        
        # Run preprocessing
        stats = preprocess_dataset(
            input_path=input_path,
            output_path=output_path,
            batch_size=10,
            overwrite=False
        )
        
        # Should have deduplicated
        assert stats.duplicates_removed > 0


def test_preprocess_dataset_validation():
    """Test that validation catches invalid records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create input with some invalid data (empty passages)
        schema = pa.schema([
            ("query_id", pa.int64()),
            ("Query", pa.string()),
            ("Eng_Query", pa.string()),
            ("Answer", pa.string()),
            ("Eng_Answer", pa.string()),
            ("query_type", pa.string()),
            ("source_lang", pa.string()),
            ("target_lang", pa.string()),
            ("passages", pa.struct([
                ("Translated_passages", pa.list_(pa.string())),
                ("English_passages", pa.list_(pa.string())),
                ("is_selected", pa.list_(pa.int32())),
            ])),
        ])
        
        data = {
            "query_id": [1, 2],
            "Query": ["Query 1", "Query 2"],
            "Eng_Query": ["Query 1", "Query 2"],
            "Answer": ["Answer 1", "Answer 2"],
            "Eng_Answer": ["Answer 1", "Answer 2"],
            "query_type": ["TEST", "TEST"],
            "source_lang": ["en", "en"],
            "target_lang": ["hi", "hi"],
            "passages": [
                {
                    "Translated_passages": ["Valid passage"],
                    "English_passages": ["Valid passage"],
                    "is_selected": [1],
                },
                {
                    "Translated_passages": [],  # Empty - will be filtered
                    "English_passages": [],
                    "is_selected": [],
                },
            ],
        }
        
        input_path = tmpdir / "input.parquet"
        table = pa.Table.from_pydict(data, schema=schema)
        pq.write_table(table, input_path)
        
        output_path = tmpdir / "output.parquet"
        
        # Run preprocessing
        stats = preprocess_dataset(
            input_path=input_path,
            output_path=output_path,
            batch_size=10,
            overwrite=False
        )
        
        # Second record had no passages, so only 1 passage should be written
        assert stats.records_written >= 1


def test_preprocess_dataset_overwrite():
    """Test overwrite behavior."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        input_path = create_test_msmarco_parquet(tmpdir / "input.parquet", num_records=3)
        output_path = tmpdir / "output.parquet"
        
        # First run
        preprocess_dataset(input_path, output_path, overwrite=True)
        assert output_path.exists()
        
        # Second run without overwrite should fail
        with pytest.raises(FileExistsError):
            preprocess_dataset(input_path, output_path, overwrite=False)
        
        # With overwrite should succeed
        preprocess_dataset(input_path, output_path, overwrite=True)
        assert output_path.exists()


def test_preprocess_dataset_dry_run():
    """Test dry run functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        input_path = create_test_msmarco_parquet(tmpdir / "input.parquet", num_records=5)
        
        # Run dry run
        stats = preprocess_dataset_dry_run(
            input_path=input_path,
            num_batches=1,
            batch_size=10
        )
        
        # Verify statistics returned
        assert stats["batches_processed"] == 1
        assert stats["input_records"] > 0
        assert stats["flattened_passages"] > 0
        assert "sample_valid_records" in stats


def test_preprocess_dataset_unicode_preservation():
    """Test that Hindi Unicode text is preserved through pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create input with Hindi text
        schema = pa.schema([
            ("query_id", pa.int64()),
            ("Query", pa.string()),
            ("Eng_Query", pa.string()),
            ("Answer", pa.string()),
            ("Eng_Answer", pa.string()),
            ("query_type", pa.string()),
            ("source_lang", pa.string()),
            ("target_lang", pa.string()),
            ("passages", pa.struct([
                ("Translated_passages", pa.list_(pa.string())),
                ("English_passages", pa.list_(pa.string())),
                ("is_selected", pa.list_(pa.int32())),
            ])),
        ])
        
        data = {
            "query_id": [1],
            "Query": ["भारत की राजधानी क्या है?"],
            "Eng_Query": ["What is the capital of India?"],
            "Answer": ["नई दिल्ली"],
            "Eng_Answer": ["New Delhi"],
            "query_type": ["LOCATION"],
            "source_lang": ["en"],
            "target_lang": ["hi"],
            "passages": [
                {
                    "Translated_passages": ["भारत की राजधानी नई दिल्ली है।"],
                    "English_passages": ["The capital of India is New Delhi."],
                    "is_selected": [1],
                },
            ],
        }
        
        input_path = tmpdir / "input.parquet"
        table = pa.Table.from_pydict(data, schema=schema)
        pq.write_table(table, input_path)
        
        output_path = tmpdir / "output.parquet"
        
        # Run preprocessing
        preprocess_dataset(input_path, output_path, batch_size=10, overwrite=False)
        
        # Read back and verify Hindi text
        output_table = pq.read_table(output_path)
        record = output_table.to_pylist()[0]
        
        assert "भारत" in record["query"]
        assert "नई दिल्ली" in record["answer"]
        assert "राजधानी" in record["translated_passage"]


def test_preprocess_dataset_statistics_accuracy():
    """Test that preprocessing statistics are accurate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        input_path = create_test_msmarco_parquet(tmpdir / "input.parquet", num_records=10)
        output_path = tmpdir / "output.parquet"
        
        stats = preprocess_dataset(input_path, output_path, batch_size=5, overwrite=False)
        
        # Verify statistics make sense
        assert stats.input_records == 10
        assert stats.flattened_passages == 20  # 10 records * 2 passages each
        assert stats.records_written > 0
        assert stats.records_written <= stats.flattened_passages
        assert stats.batches_processed == 2  # 10 records / batch_size 5
