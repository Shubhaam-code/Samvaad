"""Integrated preprocessing pipeline for MSMARCO-XI dataset.

Orchestrates: read → flatten/normalize → deduplicate → validate → write

Phase 2.2: Dataset preprocessing pipeline (not auto-executed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .deduplicator import IncrementalDeduplicator
from .parquet_reader import ParquetBatchReader
from .passage_flattener import flatten_msmarco_batch
from .processed_writer import ProcessedDatasetWriter
from .validator import validate_batch


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingStatistics:
    """Statistics from preprocessing pipeline.
    
    Attributes:
        input_records: Total input records read
        flattened_passages: Total passages after flattening
        duplicates_removed: Total duplicates removed
        validation_failures: Total validation failures
        records_written: Total records written to output
        batches_processed: Number of batches processed
    """
    input_records: int = 0
    flattened_passages: int = 0
    duplicates_removed: int = 0
    validation_failures: int = 0
    records_written: int = 0
    batches_processed: int = 0


def preprocess_dataset(
    input_path: str | Path,
    output_path: str | Path,
    batch_size: int = 500,
    overwrite: bool = False,
) -> PreprocessingStatistics:
    """Preprocess MSMARCO-XI dataset through complete pipeline.
    
    Pipeline steps:
    1. Read batches from input Parquet
    2. Flatten nested passages
    3. Normalize text
    4. Deduplicate (identity + content)
    5. Validate
    6. Write to output Parquet
    
    All steps are incremental to avoid loading entire dataset into memory.
    
    Args:
        input_path: Path to input MSMARCO-XI Parquet file
        output_path: Path to output processed Parquet file
        batch_size: Records to process per batch
        overwrite: Whether to overwrite existing output file
    
    Returns:
        PreprocessingStatistics with pipeline metrics
    
    Example:
        >>> stats = preprocess_dataset(
        ...     "data/train/hintrain.parquet",
        ...     "data/processed/train_processed.parquet",
        ...     batch_size=500,
        ...     overwrite=False
        ... )
        >>> print(f"Processed {stats.records_written} records")
    """
    logger.info(f"Starting preprocessing pipeline: {input_path} -> {output_path}")
    
    stats = PreprocessingStatistics()
    
    # Initialize components
    reader = ParquetBatchReader(input_path, batch_size=batch_size)
    deduplicator = IncrementalDeduplicator(keep_relevance_priority=True)
    
    # Process batches
    with ProcessedDatasetWriter(output_path, overwrite=overwrite) as writer:
        for batch_table in reader:
            stats.batches_processed += 1
            
            # Convert PyArrow table to list of dicts
            batch_records = batch_table.to_pylist()
            stats.input_records += len(batch_records)
            
            logger.debug(f"Batch {stats.batches_processed}: {len(batch_records)} input records")
            
            # Step 1: Flatten and normalize
            flattened_passages = flatten_msmarco_batch(
                batch_records,
                normalize=True
            )
            stats.flattened_passages += len(flattened_passages)
            
            if not flattened_passages:
                logger.warning(f"Batch {stats.batches_processed}: No passages after flattening")
                continue
            
            # Step 2: Deduplicate
            unique_passages = deduplicator.process_batch(flattened_passages)
            removed_in_batch = len(flattened_passages) - len(unique_passages)
            stats.duplicates_removed += removed_in_batch
            
            if not unique_passages:
                logger.warning(f"Batch {stats.batches_processed}: No passages after deduplication")
                continue
            
            # Step 3: Validate
            validation_result = validate_batch(unique_passages)
            stats.validation_failures += validation_result.invalid_count
            
            if not validation_result.valid_records:
                logger.warning(f"Batch {stats.batches_processed}: No valid passages after validation")
                continue
            
            # Step 4: Write
            writer.write_batch(validation_result.valid_records)
            stats.records_written += len(validation_result.valid_records)
            
            logger.debug(
                f"Batch {stats.batches_processed}: "
                f"{len(batch_records)} input → "
                f"{len(flattened_passages)} flattened → "
                f"{len(unique_passages)} unique → "
                f"{len(validation_result.valid_records)} valid"
            )
    
    # Get final deduplication stats
    dedup_stats = deduplicator.get_statistics()
    
    logger.info(
        f"Preprocessing complete:\n"
        f"  Input records: {stats.input_records:,}\n"
        f"  Flattened passages: {stats.flattened_passages:,}\n"
        f"  Identity duplicates: {dedup_stats['identity_duplicates_removed']:,}\n"
        f"  Content duplicates: {dedup_stats['content_duplicates_removed']:,}\n"
        f"  Validation failures: {stats.validation_failures:,}\n"
        f"  Records written: {stats.records_written:,}\n"
        f"  Batches processed: {stats.batches_processed}"
    )
    
    return stats


def preprocess_dataset_dry_run(
    input_path: str | Path,
    num_batches: int = 1,
    batch_size: int = 500,
) -> dict[str, Any]:
    """Dry run of preprocessing pipeline for testing/diagnostics.
    
    Processes a limited number of batches without writing output.
    Useful for validating pipeline on sample data.
    
    Args:
        input_path: Path to input MSMARCO-XI Parquet file
        num_batches: Number of batches to process
        batch_size: Records per batch
    
    Returns:
        Dict with dry run statistics and sample records
    """
    logger.info(f"Starting dry run: {input_path} ({num_batches} batches)")
    
    reader = ParquetBatchReader(input_path, batch_size=batch_size)
    deduplicator = IncrementalDeduplicator()
    
    stats = {
        "batches_processed": 0,
        "input_records": 0,
        "flattened_passages": 0,
        "unique_passages": 0,
        "valid_passages": 0,
        "sample_valid_records": [],
    }
    
    for batch_idx, batch_table in enumerate(reader):
        if batch_idx >= num_batches:
            break
        
        batch_records = batch_table.to_pylist()
        stats["input_records"] += len(batch_records)
        stats["batches_processed"] += 1
        
        # Flatten and normalize
        flattened = flatten_msmarco_batch(batch_records, normalize=True)
        stats["flattened_passages"] += len(flattened)
        
        # Deduplicate
        unique = deduplicator.process_batch(flattened)
        stats["unique_passages"] += len(unique)
        
        # Validate
        validation_result = validate_batch(unique)
        stats["valid_passages"] += len(validation_result.valid_records)
        
        # Save first few valid records as samples
        if len(stats["sample_valid_records"]) < 3:
            for record in validation_result.valid_records[:3]:
                stats["sample_valid_records"].append({
                    "document_id": record.document_id[:32] + "...",
                    "query_id": record.query_id,
                    "passage_index": record.passage_index,
                    "query": record.query[:50] + "..." if len(record.query) > 50 else record.query,
                    "is_selected": record.is_selected,
                })
    
    logger.info(f"Dry run complete: {stats}")
    return stats
