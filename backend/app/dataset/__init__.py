"""Dataset ingestion package.

Phase 2.1: Lightweight remote inspection of MSMARCO-XI via HTTP/Parquet metadata.
Phase 2.2: Complete preprocessing pipeline (canonical model, batched reading, 
           flattening, normalization, deduplication, validation, writing).
No full dataset download. Schema discovery, statistical helpers for bounded samples.
"""

from .deduplicator import (
    DeduplicationResult,
    IncrementalDeduplicator,
    deduplicate_passages,
)
from .loader import load_split, list_splits
from .models import CanonicalPassage
from .parquet_reader import (
    DEFAULT_BATCH_SIZE,
    ParquetBatchReader,
    read_parquet_batches,
)
from .passage_flattener import (
    MalformedRecordError,
    flatten_msmarco_batch,
    flatten_msmarco_record,
)
from .preprocessing_pipeline import (
    PreprocessingStatistics,
    preprocess_dataset,
    preprocess_dataset_dry_run,
)
from .processed_writer import (
    CANONICAL_PASSAGE_SCHEMA,
    ProcessedDatasetWriter,
)
from .remote_inspector import (
    DATASET_REPO,
    analyze_sample_rows,
    inspect_parquet_metadata,
    list_repository_files,
)
from .schema import TEXT_COLUMN_HINTS, discover_schema, infer_field_roles
from .stats import (
    compute_text_length_stats,
    count_missing,
    detect_duplicate_query_ids,
    split_health_check,
)
from .text_normalizer import (
    is_whitespace_only,
    normalize_optional_text,
    normalize_text,
    normalize_text_batch,
)
from .validator import (
    BatchValidationResult,
    RecordValidationResult,
    ValidationError,
    validate_batch,
    validate_passage,
)

__all__ = [
    # Loader (for future phases)
    "load_split",
    "list_splits",
    # Models (Phase 2.2.1)
    "CanonicalPassage",
    # Parquet reader (Phase 2.2.2)
    "ParquetBatchReader",
    "read_parquet_batches",
    "DEFAULT_BATCH_SIZE",
    # Passage flattener (Phase 2.2.3)
    "flatten_msmarco_record",
    "flatten_msmarco_batch",
    "MalformedRecordError",
    # Text normalizer (Phase 2.2.4)
    "normalize_text",
    "normalize_optional_text",
    "normalize_text_batch",
    "is_whitespace_only",
    # Deduplicator (Phase 2.2.5)
    "deduplicate_passages",
    "IncrementalDeduplicator",
    "DeduplicationResult",
    # Validator (Phase 2.2.6)
    "validate_passage",
    "validate_batch",
    "ValidationError",
    "RecordValidationResult",
    "BatchValidationResult",
    # Processed writer (Phase 2.2.7)
    "ProcessedDatasetWriter",
    "CANONICAL_PASSAGE_SCHEMA",
    # Preprocessing pipeline
    "preprocess_dataset",
    "preprocess_dataset_dry_run",
    "PreprocessingStatistics",
    # Remote inspector (Phase 2.1 primary approach)
    "DATASET_REPO",
    "list_repository_files",
    "inspect_parquet_metadata",
    "analyze_sample_rows",
    # Schema
    "discover_schema",
    "infer_field_roles",
    "TEXT_COLUMN_HINTS",
    # Stats
    "compute_text_length_stats",
    "count_missing",
    "detect_duplicate_query_ids",
    "split_health_check",
]
