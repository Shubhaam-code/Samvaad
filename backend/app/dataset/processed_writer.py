"""Processed dataset writer for CanonicalPassage records.

Writes deduplicated, validated CanonicalPassage records to Parquet format
incrementally without loading entire dataset into memory.

Phase 2.2.7: Processed dataset writer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .models import CanonicalPassage


logger = logging.getLogger(__name__)


# Parquet schema for CanonicalPassage
CANONICAL_PASSAGE_SCHEMA = pa.schema([
    ("document_id", pa.string()),
    ("query_id", pa.int64()),
    ("passage_index", pa.int32()),
    ("query", pa.string()),
    ("query_type", pa.string()),  # Nullable
    ("answer", pa.string()),  # Nullable
    ("source_lang", pa.string()),
    ("target_lang", pa.string()),
    ("eng_query", pa.string()),
    ("eng_answer", pa.string()),  # Nullable
    ("translated_passage", pa.string()),
    ("english_passage", pa.string()),
    ("is_selected", pa.bool_()),
])


class ProcessedDatasetWriter:
    """Incremental writer for processed CanonicalPassage records.
    
    Writes batches of records to Parquet format without accumulating
    entire dataset in memory. Supports context manager protocol.
    
    Example:
        >>> with ProcessedDatasetWriter("output.parquet") as writer:
        ...     writer.write_batch(batch1)
        ...     writer.write_batch(batch2)
        >>> print(f"Wrote {writer.records_written} records")
    """
    
    def __init__(
        self,
        output_path: str | Path,
        overwrite: bool = False,
        batch_size: int = 1000,
    ):
        """Initialize processed dataset writer.
        
        Args:
            output_path: Path to output Parquet file
            overwrite: If True, overwrite existing file; if False, raise error
            batch_size: Number of records to accumulate before writing
        
        Raises:
            FileExistsError: If output_path exists and overwrite=False
            ValueError: If batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self._writer: pq.ParquetWriter | None = None
        self._records_written = 0
        self._batches_written = 0
        self._closed = False
        
        # Check if file exists
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {self.output_path}. "
                f"Use overwrite=True to replace it."
            )
        
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"ProcessedDatasetWriter initialized: {self.output_path}, "
            f"batch_size={batch_size}"
        )
    
    def __enter__(self) -> ProcessedDatasetWriter:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            # Exception occurred - try to clean up
            logger.error(f"Writer failed with exception: {exc_val}")
            self._cleanup_on_failure()
        else:
            # Normal exit - finalize
            self.close()
        return False
    
    def _ensure_writer(self):
        """Ensure ParquetWriter is initialized."""
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.output_path,
                schema=CANONICAL_PASSAGE_SCHEMA,
                compression="snappy",
            )
            logger.debug(f"ParquetWriter opened: {self.output_path}")
    
    def write_batch(self, passages: list[CanonicalPassage]) -> None:
        """Write a batch of CanonicalPassage records.
        
        Args:
            passages: List of CanonicalPassage records to write
        
        Raises:
            RuntimeError: If writer is already closed
        """
        if self._closed:
            raise RuntimeError("Cannot write to closed writer")
        
        if not passages:
            logger.debug("Empty batch, skipping write")
            return
        
        # Convert passages to PyArrow Table
        table = self._passages_to_table(passages)
        
        # Ensure writer is initialized
        self._ensure_writer()
        
        # Write table
        self._writer.write_table(table)
        
        self._records_written += len(passages)
        self._batches_written += 1
        
        if self._batches_written % 10 == 0:
            logger.info(
                f"Progress: {self._records_written:,} records in "
                f"{self._batches_written} batches"
            )
    
    def _passages_to_table(self, passages: list[CanonicalPassage]) -> pa.Table:
        """Convert CanonicalPassage records to PyArrow Table.
        
        Args:
            passages: List of CanonicalPassage records
        
        Returns:
            PyArrow Table with CANONICAL_PASSAGE_SCHEMA
        """
        # Extract fields in schema order
        data = {
            "document_id": [p.document_id for p in passages],
            "query_id": [p.query_id for p in passages],
            "passage_index": [p.passage_index for p in passages],
            "query": [p.query for p in passages],
            "query_type": [p.query_type for p in passages],
            "answer": [p.answer for p in passages],
            "source_lang": [p.source_lang for p in passages],
            "target_lang": [p.target_lang for p in passages],
            "eng_query": [p.eng_query for p in passages],
            "eng_answer": [p.eng_answer for p in passages],
            "translated_passage": [p.translated_passage for p in passages],
            "english_passage": [p.english_passage for p in passages],
            "is_selected": [p.is_selected for p in passages],
        }
        
        # Create table with explicit schema
        return pa.Table.from_pydict(data, schema=CANONICAL_PASSAGE_SCHEMA)
    
    def close(self) -> None:
        """Finalize and close the writer.
        
        Must be called to ensure all data is written and file is valid.
        """
        if self._closed:
            return
        
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        
        self._closed = True
        
        # Get file size if available
        file_size = self.output_path.stat().st_size if self.output_path.exists() else 0
        
        logger.info(
            f"ProcessedDatasetWriter closed: {self._records_written:,} records, "
            f"{self._batches_written} batches, {file_size:,} bytes written to "
            f"{self.output_path}"
        )
    
    def _cleanup_on_failure(self):
        """Clean up incomplete output after failure."""
        try:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            
            # Remove incomplete file
            if self.output_path.exists():
                self.output_path.unlink()
                logger.info(f"Cleaned up incomplete output: {self.output_path}")
        except Exception as exc:
            logger.error(f"Failed to clean up: {exc}")
        finally:
            self._closed = True
    
    @property
    def records_written(self) -> int:
        """Total number of records written."""
        return self._records_written
    
    @property
    def batches_written(self) -> int:
        """Total number of batches written."""
        return self._batches_written
    
    @property
    def is_closed(self) -> bool:
        """Whether writer is closed."""
        return self._closed
    
    def __repr__(self) -> str:
        """String representation."""
        status = "closed" if self._closed else "open"
        return (
            f"ProcessedDatasetWriter(path={self.output_path}, "
            f"records={self._records_written}, batches={self._batches_written}, "
            f"status={status})"
        )
