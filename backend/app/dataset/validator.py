"""Validation utilities for CanonicalPassage records.

Provides dataset-level validation beyond Pydantic model validation.

Phase 2.2.6: Canonical data validation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .models import CanonicalPassage


logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Single validation error for a record.
    
    Attributes:
        error_type: Category of error (e.g., "empty_query", "invalid_id")
        message: Human-readable error message
        field: Field name that failed validation
    """
    error_type: str
    message: str
    field: str


@dataclass
class RecordValidationResult:
    """Validation result for a single record.
    
    Attributes:
        record: The CanonicalPassage being validated
        is_valid: Whether the record passed all validations
        errors: List of validation errors
    """
    record: CanonicalPassage
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)


@dataclass
class BatchValidationResult:
    """Validation result for a batch of records.
    
    Attributes:
        valid_records: List of valid CanonicalPassage records
        invalid_records: List of RecordValidationResult for invalid records
        total_count: Total records validated
        valid_count: Number of valid records
        invalid_count: Number of invalid records
        error_counts: Count of each error type
    """
    valid_records: list[CanonicalPassage]
    invalid_records: list[RecordValidationResult]
    total_count: int
    valid_count: int
    invalid_count: int
    error_counts: dict[str, int]


# Control character pattern (exclude normal whitespace like space, tab, newline)
CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')


def validate_passage(passage: CanonicalPassage) -> RecordValidationResult:
    """Validate a single CanonicalPassage record.
    
    Performs dataset-level validation beyond Pydantic model validation:
    - Non-empty required text fields
    - No whitespace-only required fields
    - No invalid control characters
    - Document ID consistency
    - Valid language codes
    
    Args:
        passage: CanonicalPassage to validate
    
    Returns:
        RecordValidationResult with validation status and errors
    """
    errors = []
    
    # Validate document_id
    if not passage.document_id:
        errors.append(ValidationError(
            error_type="empty_document_id",
            message="document_id is empty",
            field="document_id"
        ))
    elif not passage.document_id.strip():
        errors.append(ValidationError(
            error_type="whitespace_document_id",
            message="document_id contains only whitespace",
            field="document_id"
        ))
    else:
        # Validate document_id matches expected format
        expected_id = CanonicalPassage.generate_document_id(
            passage.target_lang,
            passage.query_id,
            passage.passage_index
        )
        if passage.document_id != expected_id:
            errors.append(ValidationError(
                error_type="inconsistent_document_id",
                message=f"document_id does not match expected deterministic ID",
                field="document_id"
            ))
    
    # Validate query_id (already validated by Pydantic as int, but check range)
    if not isinstance(passage.query_id, int):
        errors.append(ValidationError(
            error_type="invalid_query_id_type",
            message="query_id is not an integer",
            field="query_id"
        ))
    
    # Validate passage_index (already validated by Pydantic as >= 0)
    if passage.passage_index < 0:
        errors.append(ValidationError(
            error_type="negative_passage_index",
            message="passage_index is negative",
            field="passage_index"
        ))
    
    # Validate query
    if not passage.query or not passage.query.strip():
        errors.append(ValidationError(
            error_type="empty_query",
            message="query is empty or whitespace-only",
            field="query"
        ))
    elif CONTROL_CHAR_PATTERN.search(passage.query):
        errors.append(ValidationError(
            error_type="invalid_control_chars",
            message="query contains invalid control characters",
            field="query"
        ))
    
    # Validate eng_query
    if not passage.eng_query or not passage.eng_query.strip():
        errors.append(ValidationError(
            error_type="empty_eng_query",
            message="eng_query is empty or whitespace-only",
            field="eng_query"
        ))
    elif CONTROL_CHAR_PATTERN.search(passage.eng_query):
        errors.append(ValidationError(
            error_type="invalid_control_chars",
            message="eng_query contains invalid control characters",
            field="eng_query"
        ))
    
    # Validate translated_passage
    if not passage.translated_passage or not passage.translated_passage.strip():
        errors.append(ValidationError(
            error_type="empty_translated_passage",
            message="translated_passage is empty or whitespace-only",
            field="translated_passage"
        ))
    elif CONTROL_CHAR_PATTERN.search(passage.translated_passage):
        errors.append(ValidationError(
            error_type="invalid_control_chars",
            message="translated_passage contains invalid control characters",
            field="translated_passage"
        ))
    
    # Validate english_passage
    if not passage.english_passage or not passage.english_passage.strip():
        errors.append(ValidationError(
            error_type="empty_english_passage",
            message="english_passage is empty or whitespace-only",
            field="english_passage"
        ))
    elif CONTROL_CHAR_PATTERN.search(passage.english_passage):
        errors.append(ValidationError(
            error_type="invalid_control_chars",
            message="english_passage contains invalid control characters",
            field="english_passage"
        ))
    
    # Validate source_lang
    if not passage.source_lang or not passage.source_lang.strip():
        errors.append(ValidationError(
            error_type="empty_source_lang",
            message="source_lang is empty or whitespace-only",
            field="source_lang"
        ))
    
    # Validate target_lang
    if not passage.target_lang or not passage.target_lang.strip():
        errors.append(ValidationError(
            error_type="empty_target_lang",
            message="target_lang is empty or whitespace-only",
            field="target_lang"
        ))
    
    # Validate is_selected (already validated by Pydantic as bool)
    if not isinstance(passage.is_selected, bool):
        errors.append(ValidationError(
            error_type="invalid_is_selected_type",
            message="is_selected is not a boolean",
            field="is_selected"
        ))
    
    # Validate optional fields (answer, eng_answer, query_type)
    # These are allowed to be None, but if present, should not be whitespace-only
    if passage.answer is not None:
        if not passage.answer.strip():
            errors.append(ValidationError(
                error_type="whitespace_answer",
                message="answer is whitespace-only (should be None if empty)",
                field="answer"
            ))
        elif CONTROL_CHAR_PATTERN.search(passage.answer):
            errors.append(ValidationError(
                error_type="invalid_control_chars",
                message="answer contains invalid control characters",
                field="answer"
            ))
    
    if passage.eng_answer is not None:
        if not passage.eng_answer.strip():
            errors.append(ValidationError(
                error_type="whitespace_eng_answer",
                message="eng_answer is whitespace-only (should be None if empty)",
                field="eng_answer"
            ))
        elif CONTROL_CHAR_PATTERN.search(passage.eng_answer):
            errors.append(ValidationError(
                error_type="invalid_control_chars",
                message="eng_answer contains invalid control characters",
                field="eng_answer"
            ))
    
    if passage.query_type is not None:
        if not passage.query_type.strip():
            errors.append(ValidationError(
                error_type="whitespace_query_type",
                message="query_type is whitespace-only (should be None if empty)",
                field="query_type"
            ))
    
    is_valid = len(errors) == 0
    
    return RecordValidationResult(
        record=passage,
        is_valid=is_valid,
        errors=errors
    )


def validate_batch(passages: list[CanonicalPassage]) -> BatchValidationResult:
    """Validate a batch of CanonicalPassage records.
    
    Args:
        passages: List of CanonicalPassage records to validate
    
    Returns:
        BatchValidationResult with valid/invalid records and statistics
    """
    if not passages:
        return BatchValidationResult(
            valid_records=[],
            invalid_records=[],
            total_count=0,
            valid_count=0,
            invalid_count=0,
            error_counts={}
        )
    
    valid_records = []
    invalid_records = []
    error_counts: dict[str, int] = {}
    
    for passage in passages:
        result = validate_passage(passage)
        
        if result.is_valid:
            valid_records.append(passage)
        else:
            invalid_records.append(result)
            
            # Count error types
            for error in result.errors:
                error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
    
    total_count = len(passages)
    valid_count = len(valid_records)
    invalid_count = len(invalid_records)
    
    if invalid_count > 0:
        logger.warning(
            f"Validation: {invalid_count}/{total_count} invalid records found. "
            f"Error types: {error_counts}"
        )
    else:
        logger.info(f"Validation: All {total_count} records valid")
    
    return BatchValidationResult(
        valid_records=valid_records,
        invalid_records=invalid_records,
        total_count=total_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        error_counts=error_counts
    )
