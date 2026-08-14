"""Tests for the Parquet batch reader.

Phase 2.2.2: Dataset preprocessing infrastructure.
"""

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.dataset.parquet_reader import (
    DEFAULT_BATCH_SIZE,
    ParquetBatchReader,
    read_parquet_batches,
)


@pytest.fixture
def tiny_parquet_file(tmp_path):
    """Create a tiny Parquet file with nested structure similar to MSMARCO-XI.
    
    Creates a file with 10 rows to test batching without processing large datasets.
    """
    # Define schema with nested structure like MSMARCO-XI
    schema = pa.schema([
        ("query_id", pa.int64()),
        ("query", pa.string()),
        ("answer", pa.string()),
        ("query_type", pa.string()),
        ("passages", pa.struct([
            ("translated_passages", pa.list_(pa.string())),
            ("english_passages", pa.list_(pa.string())),
            ("is_selected", pa.list_(pa.bool_())),
        ])),
        ("source_lang", pa.string()),
        ("target_lang", pa.string()),
    ])
    
    # Create 10 sample rows
    data = {
        "query_id": list(range(1, 11)),
        "query": [f"Query {i} in Hindi" for i in range(1, 11)],
        "answer": [f"Answer {i}" for i in range(1, 11)],
        "query_type": ["DESCRIPTION"] * 10,
        "passages": [
            {
                "translated_passages": [f"Passage {i}a", f"Passage {i}b"],
                "english_passages": [f"English passage {i}a", f"English passage {i}b"],
                "is_selected": [True, False],
            }
            for i in range(1, 11)
        ],
        "source_lang": ["en"] * 10,
        "target_lang": ["hi"] * 10,
    }
    
    table = pa.Table.from_pydict(data, schema=schema)
    
    # Write to temporary file
    file_path = tmp_path / "test_data.parquet"
    pq.write_table(table, file_path)
    
    return file_path


@pytest.fixture
def small_parquet_file(tmp_path):
    """Create a small Parquet file with 25 rows for testing batch boundaries."""
    schema = pa.schema([
        ("id", pa.int64()),
        ("value", pa.string()),
    ])
    
    data = {
        "id": list(range(1, 26)),
        "value": [f"Value {i}" for i in range(1, 26)],
    }
    
    table = pa.Table.from_pydict(data, schema=schema)
    file_path = tmp_path / "small_data.parquet"
    pq.write_table(table, file_path)
    
    return file_path


def test_parquet_batch_reader_initialization(tiny_parquet_file):
    """Test that ParquetBatchReader initializes correctly."""
    reader = ParquetBatchReader(tiny_parquet_file, batch_size=5)
    
    assert reader.path == str(tiny_parquet_file)
    assert reader.batch_size == 5
    assert reader.total_rows == 10
    assert reader.num_row_groups >= 1


def test_parquet_batch_reader_default_batch_size(tiny_parquet_file):
    """Test that default batch size is used when not specified."""
    reader = ParquetBatchReader(tiny_parquet_file)
    assert reader.batch_size == DEFAULT_BATCH_SIZE


def test_parquet_batch_reader_rejects_invalid_batch_size(tiny_parquet_file):
    """Test that invalid batch sizes are rejected."""
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        ParquetBatchReader(tiny_parquet_file, batch_size=0)
    
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        ParquetBatchReader(tiny_parquet_file, batch_size=-1)


def test_parquet_batch_reader_rejects_missing_file():
    """Test that missing files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Parquet file not found"):
        ParquetBatchReader("/nonexistent/path/file.parquet")


def test_parquet_batch_reader_rejects_directory(tmp_path):
    """Test that directories are rejected."""
    with pytest.raises(ValueError, match="Path is not a file"):
        ParquetBatchReader(tmp_path)


def test_parquet_batch_reader_produces_batches(tiny_parquet_file):
    """Test that reader produces batches as PyArrow Tables."""
    reader = ParquetBatchReader(tiny_parquet_file, batch_size=3)
    
    batches = list(reader)
    
    # With 10 rows and batch_size=3, expect 4 batches: [3, 3, 3, 1]
    assert len(batches) == 4
    assert all(isinstance(batch, pa.Table) for batch in batches)
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    assert len(batches[3]) == 1


def test_parquet_batch_reader_respects_batch_size(small_parquet_file):
    """Test that batch size is respected correctly."""
    reader = ParquetBatchReader(small_parquet_file, batch_size=10)
    
    batches = list(reader)
    
    # With 25 rows and batch_size=10, expect 3 batches: [10, 10, 5]
    assert len(batches) == 3
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5


def test_parquet_batch_reader_yields_all_rows(tiny_parquet_file):
    """Test that all rows are eventually yielded."""
    reader = ParquetBatchReader(tiny_parquet_file, batch_size=3)
    
    total_rows_yielded = 0
    for batch in reader:
        total_rows_yielded += len(batch)
    
    assert total_rows_yielded == reader.total_rows == 10


def test_parquet_batch_reader_preserves_nested_columns(tiny_parquet_file):
    """Test that nested columns are preserved correctly."""
    reader = ParquetBatchReader(tiny_parquet_file, batch_size=5)
    
    batch = next(iter(reader))
    
    # Verify nested structure exists
    assert "passages" in batch.column_names
    
    # Access nested field
    passages_column = batch.column("passages")
    first_row_passages = passages_column[0].as_py()
    
    # Verify nested lists are preserved
    assert "translated_passages" in first_row_passages
    assert "english_passages" in first_row_passages
    assert "is_selected" in first_row_passages
    assert isinstance(first_row_passages["translated_passages"], list)
    assert len(first_row_passages["translated_passages"]) == 2


def test_parquet_batch_reader_schema_access(tiny_parquet_file):
    """Test that schema can be accessed."""
    reader = ParquetBatchReader(tiny_parquet_file, batch_size=5)
    
    schema = reader.schema
    
    assert schema is not None
    assert "query_id" in schema.names
    assert "query" in schema.names
    assert "passages" in schema.names


def test_parquet_batch_reader_does_not_load_all_into_memory(tiny_parquet_file):
    """Test that reader uses iteration without loading entire dataset."""
    reader = ParquetBatchReader(tiny_parquet_file, batch_size=2)
    
    # Get first batch without consuming entire dataset
    iterator = iter(reader)
    first_batch = next(iterator)
    
    # Verify we got a batch without processing all rows
    assert len(first_batch) == 2
    # Iterator should still have more data
    second_batch = next(iterator)
    assert len(second_batch) == 2


def test_parquet_batch_reader_repr(tiny_parquet_file):
    """Test string representation."""
    reader = ParquetBatchReader(tiny_parquet_file, batch_size=5)
    
    repr_str = repr(reader)
    
    assert "ParquetBatchReader" in repr_str
    assert "batch_size=5" in repr_str
    assert "total_rows=10" in repr_str


def test_parquet_batch_reader_multiple_iterations(tiny_parquet_file):
    """Test that reader can be iterated multiple times."""
    reader = ParquetBatchReader(tiny_parquet_file, batch_size=5)
    
    # First iteration
    batches1 = list(reader)
    total_rows1 = sum(len(b) for b in batches1)
    
    # Second iteration
    batches2 = list(reader)
    total_rows2 = sum(len(b) for b in batches2)
    
    assert total_rows1 == total_rows2 == 10


def test_parquet_batch_reader_read_specific_batch(small_parquet_file):
    """Test reading a specific batch by index."""
    reader = ParquetBatchReader(small_parquet_file, batch_size=10)
    
    # Read batch 0 (rows 0-9)
    batch0 = reader.read_batch(0)
    assert len(batch0) == 10
    assert batch0.column("id")[0].as_py() == 1
    
    # Read batch 1 (rows 10-19)
    batch1 = reader.read_batch(1)
    assert len(batch1) == 10
    assert batch1.column("id")[0].as_py() == 11
    
    # Read batch 2 (rows 20-24)
    batch2 = reader.read_batch(2)
    assert len(batch2) == 5
    assert batch2.column("id")[0].as_py() == 21


def test_parquet_batch_reader_read_batch_rejects_invalid_index(small_parquet_file):
    """Test that invalid batch indices are rejected."""
    reader = ParquetBatchReader(small_parquet_file, batch_size=10)
    
    with pytest.raises(IndexError, match="batch_index must be >= 0"):
        reader.read_batch(-1)
    
    with pytest.raises(IndexError, match="out of range"):
        reader.read_batch(999)


def test_read_parquet_batches_convenience_function(tiny_parquet_file):
    """Test the convenience function for batch reading."""
    batches = list(read_parquet_batches(tiny_parquet_file, batch_size=4))
    
    # With 10 rows and batch_size=4, expect 3 batches: [4, 4, 2]
    assert len(batches) == 3
    assert all(isinstance(batch, pa.Table) for batch in batches)
    
    total_rows = sum(len(batch) for batch in batches)
    assert total_rows == 10


def test_parquet_batch_reader_with_pathlib_path(tiny_parquet_file):
    """Test that reader accepts pathlib.Path objects."""
    path_obj = Path(tiny_parquet_file)
    reader = ParquetBatchReader(path_obj, batch_size=5)
    
    assert reader.total_rows == 10


def test_parquet_batch_reader_single_batch_when_batch_size_exceeds_rows(tiny_parquet_file):
    """Test that a single batch is returned when batch_size > total_rows."""
    reader = ParquetBatchReader(tiny_parquet_file, batch_size=100)
    
    batches = list(reader)
    
    assert len(batches) == 1
    assert len(batches[0]) == 10


def test_parquet_batch_reader_preserves_column_order(tiny_parquet_file):
    """Test that column order is preserved from source file."""
    reader = ParquetBatchReader(tiny_parquet_file, batch_size=5)
    
    expected_columns = ["query_id", "query", "answer", "query_type", "passages", "source_lang", "target_lang"]
    
    batch = next(iter(reader))
    assert batch.column_names == expected_columns
