"""Passage flattening utilities for MSMARCO-XI dataset.

Converts MSMARCO-XI records with nested passage lists into individual
CanonicalPassage instances while preserving positional alignment.

Phase 2.2.3: Passage flattening (no deduplication, no chunking).
"""

from __future__ import annotations

import logging
from typing import Any

from .models import CanonicalPassage
from .text_normalizer import normalize_optional_text, normalize_text


logger = logging.getLogger(__name__)


class MalformedRecordError(Exception):
    """Raised when a record cannot be safely processed."""
    pass


def flatten_msmarco_record(
    record: dict[str, Any],
    normalize: bool = True,
) -> list[CanonicalPassage]:
    """Flatten a single MSMARCO-XI record into CanonicalPassage instances.
    
    Takes a record with nested passages and creates one CanonicalPassage per
    passage, preserving positional alignment:
    - English_passages[i] ↔ Translated_passages[i] ↔ is_selected[i]
    
    Args:
        record: MSMARCO-XI record dict with nested passages
        normalize: Whether to apply text normalization (default: True)
    
    Returns:
        List of CanonicalPassage instances (one per passage)
        Empty list if record has no valid passages
    
    Raises:
        MalformedRecordError: If record structure is critically malformed
    
    Example:
        >>> record = {
        ...     "query_id": 123,
        ...     "Query": "भारत की राजधानी?",
        ...     "passages": {
        ...         "Translated_passages": ["अनुच्छेद 1", "अनुच्छेद 2"],
        ...         "English_passages": ["Passage 1", "Passage 2"],
        ...         "is_selected": [1, 0]
        ...     },
        ...     ...
        ... }
        >>> passages = flatten_msmarco_record(record)
        >>> len(passages)
        2
    """
    # Extract query-level fields
    try:
        query_id = int(record["query_id"])
    except (KeyError, ValueError, TypeError) as exc:
        raise MalformedRecordError(f"Invalid or missing query_id: {exc}") from exc
    
    # Extract text fields (use various possible field names from MSMARCO-XI)
    query = record.get("Query") or record.get("query") or ""
    eng_query = record.get("Eng_Query") or record.get("eng_query") or ""
    answer = record.get("Answer") or record.get("answer")
    eng_answer = record.get("Eng_Answer") or record.get("eng_answer")
    query_type = record.get("query_type") or record.get("Query_Type")
    source_lang = record.get("source_lang", "en")
    target_lang = record.get("target_lang", "hi")
    
    # Validate required fields
    if not query and not eng_query:
        raise MalformedRecordError(
            f"Record query_id={query_id} missing both query and eng_query"
        )
    
    # Handle passages field
    passages_data = record.get("passages")
    if not passages_data or not isinstance(passages_data, dict):
        logger.warning(f"Record query_id={query_id} has no valid passages field")
        return []
    
    # Extract passage lists
    translated_passages = passages_data.get("Translated_passages") or passages_data.get("translated_passages") or []
    english_passages = passages_data.get("English_passages") or passages_data.get("english_passages") or []
    is_selected = passages_data.get("is_selected") or []
    
    # Convert to lists if not already
    if not isinstance(translated_passages, list):
        translated_passages = []
    if not isinstance(english_passages, list):
        english_passages = []
    if not isinstance(is_selected, list):
        is_selected = []
    
    # Check for empty passages
    if not translated_passages and not english_passages:
        logger.warning(f"Record query_id={query_id} has empty passage lists")
        return []
    
    # Handle length mismatches using safe strategy:
    # Use the minimum length to avoid misalignment
    num_passages = min(
        len(translated_passages) if translated_passages else 0,
        len(english_passages) if english_passages else 0,
        len(is_selected) if is_selected else 0,
    )
    
    if num_passages == 0:
        logger.warning(f"Record query_id={query_id} has no aligned passages (length mismatch)")
        return []
    
    # Check for length mismatches and log them
    lengths = [len(translated_passages), len(english_passages), len(is_selected)]
    if len(set(lengths)) > 1:
        logger.warning(
            f"Record query_id={query_id} has unequal passage list lengths: "
            f"translated={lengths[0]}, english={lengths[1]}, is_selected={lengths[2]}. "
            f"Using minimum length {num_passages} to preserve alignment."
        )
    
    # Create CanonicalPassage instances
    canonical_passages = []
    
    for passage_index in range(num_passages):
        translated_passage = translated_passages[passage_index]
        english_passage = english_passages[passage_index]
        selected_value = is_selected[passage_index]
        
        # Validate passage text
        if translated_passage is None or not str(translated_passage).strip():
            logger.warning(
                f"Record query_id={query_id} passage_index={passage_index} "
                f"has null/empty translated_passage, skipping"
            )
            continue
        
        if english_passage is None or not str(english_passage).strip():
            logger.warning(
                f"Record query_id={query_id} passage_index={passage_index} "
                f"has null/empty english_passage, skipping"
            )
            continue
        
        # Convert is_selected to boolean
        try:
            is_selected_bool = _parse_is_selected(selected_value)
        except ValueError as exc:
            logger.warning(
                f"Record query_id={query_id} passage_index={passage_index} "
                f"has invalid is_selected value {selected_value!r}, defaulting to False"
            )
            is_selected_bool = False
        
        # Apply normalization if requested
        if normalize:
            query_norm = normalize_text(query)
            eng_query_norm = normalize_text(eng_query)
            answer_norm = normalize_optional_text(answer)
            eng_answer_norm = normalize_optional_text(eng_answer)
            translated_passage_norm = normalize_text(str(translated_passage))
            english_passage_norm = normalize_text(str(english_passage))
        else:
            query_norm = query
            eng_query_norm = eng_query
            answer_norm = answer
            eng_answer_norm = eng_answer
            translated_passage_norm = str(translated_passage)
            english_passage_norm = str(english_passage)
        
        # Create CanonicalPassage
        try:
            canonical_passage = CanonicalPassage.from_msmarco_record(
                query_id=query_id,
                query=query_norm,
                query_type=query_type,
                answer=answer_norm,
                source_lang=source_lang,
                target_lang=target_lang,
                eng_query=eng_query_norm,
                eng_answer=eng_answer_norm,
                passage_index=passage_index,
                translated_passage=translated_passage_norm,
                english_passage=english_passage_norm,
                is_selected=is_selected_bool,
            )
            canonical_passages.append(canonical_passage)
        except Exception as exc:
            logger.error(
                f"Failed to create CanonicalPassage for query_id={query_id} "
                f"passage_index={passage_index}: {exc}"
            )
            continue
    
    return canonical_passages


def _parse_is_selected(value: Any) -> bool:
    """Parse is_selected value to boolean.
    
    Accepts:
    - bool: True/False
    - int: 0 (False) or 1 (True)
    - str: "0", "1", "true", "false" (case-insensitive)
    
    Args:
        value: is_selected value from dataset
    
    Returns:
        Boolean is_selected value
    
    Raises:
        ValueError: If value cannot be parsed
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, int):
        if value == 0:
            return False
        elif value == 1:
            return True
        else:
            raise ValueError(f"Invalid int value for is_selected: {value} (expected 0 or 1)")
    
    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in ("0", "false"):
            return False
        elif lower in ("1", "true"):
            return True
        else:
            raise ValueError(f"Invalid string value for is_selected: {value!r}")
    
    raise ValueError(f"Unsupported type for is_selected: {type(value).__name__}")


def flatten_msmarco_batch(
    records: list[dict[str, Any]],
    normalize: bool = True,
) -> list[CanonicalPassage]:
    """Flatten a batch of MSMARCO-XI records.
    
    Args:
        records: List of MSMARCO-XI record dicts
        normalize: Whether to apply text normalization (default: True)
    
    Returns:
        List of all CanonicalPassage instances from all records
        
    Note:
        Malformed records are logged and skipped rather than raising exceptions.
    """
    all_passages = []
    malformed_count = 0
    
    for record in records:
        try:
            passages = flatten_msmarco_record(record, normalize=normalize)
            all_passages.extend(passages)
        except MalformedRecordError as exc:
            malformed_count += 1
            logger.error(f"Malformed record skipped: {exc}")
            continue
    
    if malformed_count > 0:
        logger.warning(f"Skipped {malformed_count} malformed records in batch")
    
    return all_passages
