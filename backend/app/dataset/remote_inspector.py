"""Lightweight remote inspection of MSMARCO-XI without downloading full dataset.

This module uses huggingface_hub API to inspect repository structure and
pyarrow to read Parquet metadata/samples via HTTP range requests.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url


logger = logging.getLogger(__name__)

DATASET_REPO = "ai4bharat/MSMARCO-XI"


def list_repository_files(repo_id: str = DATASET_REPO) -> dict[str, Any]:
    """List all files in the dataset repository.
    
    Returns a dict with:
        - files: list of file info dicts
        - train_files: list of train parquet files
        - validation_files: list of validation parquet files
        - languages: set of discovered language codes
    """
    api = HfApi()
    
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        raise RuntimeError(f"Failed to list files in {repo_id}: {exc}") from exc
    
    train_files = []
    validation_files = []
    languages = set()
    
    for file_path in files:
        if file_path.endswith(".parquet"):
            # Parse language from filename (e.g., "train/hi-train.parquet" or similar)
            parts = file_path.split("/")
            filename = parts[-1]
            
            if "train" in parts[0].lower() or "train" in filename.lower():
                train_files.append(file_path)
                # Extract language code (e.g., "hi-train.parquet" -> "hi")
                lang = filename.split("-")[0] if "-" in filename else filename.split(".")[0]
                if lang != "train":
                    languages.add(lang)
            elif "validation" in parts[0].lower() or "validation" in filename.lower() or "val" in filename.lower():
                validation_files.append(file_path)
                lang = filename.split("-")[0] if "-" in filename else filename.split(".")[0]
                if lang not in ("validation", "val"):
                    languages.add(lang)
    
    logger.info(f"Repository {repo_id}: {len(train_files)} train files, {len(validation_files)} validation files")
    logger.info(f"Languages discovered: {sorted(languages)}")
    
    return {
        "files": list(files),
        "train_files": train_files,
        "validation_files": validation_files,
        "languages": sorted(languages),
    }


def inspect_parquet_metadata(
    repo_id: str,
    file_path: str,
    max_sample_rows: int = 20,
) -> dict[str, Any]:
    """Inspect Parquet file metadata and read a small sample via HTTP.
    
    Uses PyArrow's ability to read Parquet files over HTTP with range requests,
    which allows us to read metadata and a few rows without downloading the entire file.
    
    Args:
        repo_id: HuggingFace dataset repo ID
        file_path: Path to parquet file within repo
        max_sample_rows: Maximum rows to sample (default: 20)
    
    Returns:
        Dict with schema, column names, row count (if available), and sample rows
    """
    # Construct HuggingFace URL for the file
    url = hf_hub_url(repo_id=repo_id, filename=file_path, repo_type="dataset")
    
    try:
        # Open parquet file via HTTP (PyArrow supports range requests)
        parquet_file = pq.ParquetFile(url)
        
        # Get schema
        schema = parquet_file.schema_arrow
        column_names = schema.names
        
        # Get metadata
        metadata = parquet_file.metadata
        num_row_groups = metadata.num_row_groups
        total_rows = metadata.num_rows if hasattr(metadata, 'num_rows') else None
        
        # Read a small sample from the first row group
        # This reads only the first row group, not the entire file
        sample_size = min(max_sample_rows, total_rows if total_rows else max_sample_rows)
        
        # Read just the first row group's data
        table_sample = parquet_file.read_row_group(0)
        
        # Convert to list of dicts (limit to sample_size)
        sample_rows = []
        for i in range(min(sample_size, len(table_sample))):
            row_dict = {}
            for col_name in column_names:
                value = table_sample[col_name][i].as_py()
                row_dict[col_name] = value
            sample_rows.append(row_dict)
        
        logger.info(f"Inspected {file_path}: {len(column_names)} columns, {total_rows} rows (metadata), {len(sample_rows)} sampled")
        
        return {
            "file_path": file_path,
            "schema": {name: str(schema.field(name).type) for name in column_names},
            "column_names": column_names,
            "num_row_groups": num_row_groups,
            "total_rows": total_rows,
            "sample_rows": sample_rows,
            "sample_size": len(sample_rows),
        }
    
    except Exception as exc:
        logger.warning(f"Failed to inspect {file_path} via HTTP: {exc}")
        return {
            "file_path": file_path,
            "error": str(exc),
            "schema": None,
            "column_names": None,
            "total_rows": None,
            "sample_rows": [],
        }


def analyze_sample_rows(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze a small sample of rows for basic statistics.
    
    Args:
        sample_rows: List of row dictionaries
    
    Returns:
        Dict with missing value counts, text length stats, field role inferences
    """
    if not sample_rows:
        return {
            "missing_values": {},
            "text_length_samples": {},
            "field_roles": {},
            "duplicate_ids": 0,
        }
    
    # Count missing values per field
    missing_values = {}
    text_lengths = {}
    
    for field in sample_rows[0].keys():
        missing_count = 0
        lengths = []
        
        for row in sample_rows:
            value = row.get(field)
            
            if value is None or value == "" or value == []:
                missing_count += 1
            
            # Collect text lengths
            if isinstance(value, str):
                lengths.append(len(value))
            elif isinstance(value, list):
                lengths.append(len(value))
        
        missing_values[field] = missing_count
        
        if lengths:
            text_lengths[field] = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": round(sum(lengths) / len(lengths), 2),
                "sample_size": len(lengths),
            }
    
    # Infer field roles based on naming
    query_fields = [f for f in sample_rows[0].keys() if "query" in f.lower()]
    document_fields = [f for f in sample_rows[0].keys() if any(x in f.lower() for x in ["passage", "document", "answer"])]
    metadata_fields = [f for f in sample_rows[0].keys() if any(x in f.lower() for x in ["id", "type", "lang", "meta"])]
    
    # Check for duplicate IDs
    if "query_id" in sample_rows[0]:
        ids = [row.get("query_id") for row in sample_rows if row.get("query_id")]
        duplicate_count = len(ids) - len(set(ids))
    else:
        duplicate_count = 0
    
    return {
        "missing_values": missing_values,
        "text_length_samples": text_lengths,
        "field_roles": {
            "potential_query_fields": query_fields,
            "potential_document_fields": document_fields,
            "potential_metadata_fields": metadata_fields,
        },
        "duplicate_ids": duplicate_count,
    }
