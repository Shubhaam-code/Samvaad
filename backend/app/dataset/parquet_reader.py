"""Batched Parquet reader for large MSMARCO-XI dataset files.

This module provides memory-efficient reading of large Parquet files by yielding
batches of rows rather than loading the entire dataset into memory.

Phase 2.2.2: Dataset preprocessing infrastructure (no transformation yet).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)

# Default batch size for reading Parquet files
DEFAULT_BATCH_SIZE = 500


class ParquetBatchReader:
    """Memory-efficient batched reader for Parquet files.
    
    Supports both local file paths and remote HTTP/HTTPS URLs (if PyArrow
    supports them). Yields batches of rows as PyArrow Tables without loading
    the entire dataset into memory.
    
    Example:
        >>> reader = ParquetBatchReader("data/train.parquet", batch_size=1000)
        >>> for batch in reader:
        ...     print(f"Processing {len(batch)} rows")
        ...     # Process batch as PyArrow Table
    """
    
    def __init__(
        self,
        path: str | Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Initialize the Parquet batch reader.
        
        Args:
            path: Local file path or remote URL to Parquet file
            batch_size: Number of rows per batch (must be > 0)
        
        Raises:
            ValueError: If batch_size is invalid (<= 0)
            FileNotFoundError: If local path doesn't exist
            RuntimeError: If Parquet file cannot be opened
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        
        self.path = str(path)
        self.batch_size = batch_size
        self._parquet_file = None
        self._total_rows = None
        
        # Validate and open the Parquet file
        self._open_file()
        
        logger.info(
            f"ParquetBatchReader initialized: path={self.path}, "
            f"batch_size={self.batch_size}, total_rows={self._total_rows}"
        )
    
    def _open_file(self) -> None:
        """Open and validate the Parquet file.
        
        Raises:
            FileNotFoundError: If local path doesn't exist
            RuntimeError: If file cannot be opened or read
        """
        # Check if it's a local path (not a URL)
        if not self.path.startswith(("http://", "https://")):
            local_path = Path(self.path)
            if not local_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {self.path}")
            if not local_path.is_file():
                raise ValueError(f"Path is not a file: {self.path}")
        
        try:
            # Open the Parquet file (works with both local and HTTP paths)
            self._parquet_file = pq.ParquetFile(self.path)
            
            # Get total row count from metadata
            metadata = self._parquet_file.metadata
            self._total_rows = metadata.num_rows
            
            logger.debug(
                f"Opened Parquet file: {metadata.num_row_groups} row groups, "
                f"{self._total_rows} total rows"
            )
            
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open Parquet file '{self.path}': {exc}"
            ) from exc
    
    @property
    def total_rows(self) -> int:
        """Total number of rows in the Parquet file."""
        return self._total_rows
    
    @property
    def num_row_groups(self) -> int:
        """Number of row groups in the Parquet file."""
        return self._parquet_file.metadata.num_row_groups
    
    @property
    def schema(self):
        """PyArrow schema of the Parquet file."""
        return self._parquet_file.schema_arrow
    
    def __iter__(self) -> Iterator[pq.Table]:
        """Iterate over batches of rows as PyArrow Tables.
        
        Yields:
            PyArrow Table containing up to batch_size rows
        
        The reader processes row groups sequentially and yields batches of the
        specified size. The last batch may contain fewer rows than batch_size.
        
        Nested columns are preserved exactly as they appear in the source Parquet.
        """
        rows_processed = 0
        batch_count = 0
        
        # Read row groups and accumulate into batches
        accumulated_tables = []
        accumulated_rows = 0
        
        for row_group_idx in range(self.num_row_groups):
            # Read one row group at a time
            table = self._parquet_file.read_row_group(row_group_idx)
            accumulated_tables.append(table)
            accumulated_rows += len(table)
            
            # Yield batches when we have enough rows
            while accumulated_rows >= self.batch_size:
                # Concatenate accumulated tables
                combined = pa.concat_tables(accumulated_tables)
                
                # Split off a batch
                batch = combined.slice(0, self.batch_size)
                remaining = combined.slice(self.batch_size)
                
                yield batch
                
                batch_count += 1
                rows_processed += len(batch)
                
                # Keep remaining rows for next batch
                if len(remaining) > 0:
                    accumulated_tables = [remaining]
                    accumulated_rows = len(remaining)
                else:
                    accumulated_tables = []
                    accumulated_rows = 0
                
                # Log progress periodically
                if batch_count % 10 == 0:
                    progress_pct = (rows_processed / self._total_rows) * 100
                    logger.info(
                        f"Progress: {rows_processed:,} / {self._total_rows:,} rows "
                        f"({progress_pct:.1f}%) in {batch_count} batches"
                    )
        
        # Yield any remaining rows as final batch
        if accumulated_rows > 0:
            combined = pa.concat_tables(accumulated_tables)
            yield combined
            batch_count += 1
            rows_processed += len(combined)
        
        logger.info(
            f"Completed: {rows_processed:,} rows in {batch_count} batches from {self.path}"
        )
    
    def read_batch(self, batch_index: int) -> pq.Table:
        """Read a specific batch by index.
        
        Args:
            batch_index: Zero-based batch index
        
        Returns:
            PyArrow Table containing the specified batch
        
        Raises:
            IndexError: If batch_index is out of range
        """
        if batch_index < 0:
            raise IndexError(f"batch_index must be >= 0, got {batch_index}")
        
        start_row = batch_index * self.batch_size
        if start_row >= self._total_rows:
            raise IndexError(
                f"batch_index {batch_index} out of range "
                f"(file has {self._total_rows} rows, batch size {self.batch_size})"
            )
        
        # Calculate how many rows to read
        num_rows = min(self.batch_size, self._total_rows - start_row)
        
        # Read the slice
        table = self._parquet_file.read(columns=None)
        return table.slice(start_row, num_rows)
    
    def __repr__(self) -> str:
        """String representation of the reader."""
        return (
            f"ParquetBatchReader(path={self.path!r}, batch_size={self.batch_size}, "
            f"total_rows={self._total_rows})"
        )


def read_parquet_batches(
    path: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[pq.Table]:
    """Convenience function to iterate over Parquet file batches.
    
    Args:
        path: Local file path or remote URL to Parquet file
        batch_size: Number of rows per batch
    
    Yields:
        PyArrow Table containing up to batch_size rows
    
    Example:
        >>> for batch in read_parquet_batches("data/train.parquet", batch_size=1000):
        ...     print(f"Batch has {len(batch)} rows")
    """
    reader = ParquetBatchReader(path, batch_size=batch_size)
    yield from reader
