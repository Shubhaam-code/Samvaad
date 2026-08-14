"""Deduplication utilities for CanonicalPassage records.

Provides two-level duplicate detection:
1. Identity duplicates (same document_id)
2. Content duplicates (same translated_passage + english_passage + target_lang)

Phase 2.2.5: Safe deduplication (no fuzzy matching, no semantic similarity).
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .models import CanonicalPassage


logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of deduplication operation.
    
    Attributes:
        unique_records: List of deduplicated CanonicalPassage records
        total_input: Total input records
        total_output: Total output records (unique)
        identity_duplicates_removed: Count of identity duplicates
        content_duplicates_removed: Count of content duplicates
        relevance_conflicts: Count of conflicts where is_selected differed
        diagnostics: Detailed diagnostic information
    """
    unique_records: list[CanonicalPassage]
    total_input: int
    total_output: int
    identity_duplicates_removed: int
    content_duplicates_removed: int
    relevance_conflicts: int
    diagnostics: dict[str, Any]


def _compute_content_fingerprint(passage: CanonicalPassage) -> str:
    """Compute deterministic content fingerprint for deduplication.
    
    Uses:
    - target_lang (to avoid cross-language deduplication)
    - translated_passage (primary content)
    - english_passage (ensures alignment)
    
    Args:
        passage: CanonicalPassage to fingerprint
    
    Returns:
        SHA-256 hex fingerprint
    """
    # Create canonical string from key content fields
    canonical_string = (
        f"lang={passage.target_lang}:"
        f"trans={passage.translated_passage}:"
        f"eng={passage.english_passage}"
    )
    
    # Generate deterministic hash
    hash_bytes = hashlib.sha256(canonical_string.encode("utf-8")).digest()
    return hash_bytes.hex()


def deduplicate_passages(
    passages: list[CanonicalPassage],
    keep_relevance_priority: bool = True,
) -> DeduplicationResult:
    """Deduplicate CanonicalPassage records using two-level detection.
    
    Level 1: Identity duplicates (same document_id)
    Level 2: Content duplicates (same fingerprint)
    
    When duplicates have conflicting is_selected values and keep_relevance_priority
    is True, the record with is_selected=True is retained.
    
    Args:
        passages: List of CanonicalPassage records to deduplicate
        keep_relevance_priority: If True, prefer is_selected=True on conflicts
    
    Returns:
        DeduplicationResult with unique records and diagnostics
    """
    if not passages:
        return DeduplicationResult(
            unique_records=[],
            total_input=0,
            total_output=0,
            identity_duplicates_removed=0,
            content_duplicates_removed=0,
            relevance_conflicts=0,
            diagnostics={},
        )
    
    total_input = len(passages)
    identity_duplicates = 0
    content_duplicates = 0
    relevance_conflicts = 0
    
    # Level 1: Remove identity duplicates (same document_id)
    doc_id_map: dict[str, CanonicalPassage] = {}
    identity_dupe_groups: dict[str, int] = defaultdict(int)
    
    for passage in passages:
        doc_id = passage.document_id
        
        if doc_id in doc_id_map:
            identity_duplicates += 1
            identity_dupe_groups[doc_id] += 1
            logger.debug(f"Identity duplicate found: document_id={doc_id}")
        else:
            doc_id_map[doc_id] = passage
    
    after_identity = list(doc_id_map.values())
    
    # Level 2: Remove content duplicates (same fingerprint)
    fingerprint_map: dict[str, CanonicalPassage] = {}
    content_dupe_groups: dict[str, int] = defaultdict(int)
    relevance_conflict_details: list[dict] = []
    
    for passage in after_identity:
        fingerprint = _compute_content_fingerprint(passage)
        
        if fingerprint in fingerprint_map:
            existing = fingerprint_map[fingerprint]
            
            # Check for relevance conflict
            if existing.is_selected != passage.is_selected:
                relevance_conflicts += 1
                
                # If we prioritize relevance, keep the one with is_selected=True
                if keep_relevance_priority and passage.is_selected:
                    logger.warning(
                        f"Relevance conflict: fingerprint={fingerprint[:16]}..., "
                        f"replacing is_selected=False with is_selected=True"
                    )
                    fingerprint_map[fingerprint] = passage
                    relevance_conflict_details.append({
                        "fingerprint": fingerprint[:16] + "...",
                        "kept_is_selected": passage.is_selected,
                        "discarded_is_selected": existing.is_selected,
                        "kept_query_id": passage.query_id,
                        "discarded_query_id": existing.query_id,
                    })
                else:
                    relevance_conflict_details.append({
                        "fingerprint": fingerprint[:16] + "...",
                        "kept_is_selected": existing.is_selected,
                        "discarded_is_selected": passage.is_selected,
                        "kept_query_id": existing.query_id,
                        "discarded_query_id": passage.query_id,
                    })
            
            content_duplicates += 1
            content_dupe_groups[fingerprint] += 1
            logger.debug(f"Content duplicate found: fingerprint={fingerprint[:16]}...")
        else:
            fingerprint_map[fingerprint] = passage
    
    unique_records = list(fingerprint_map.values())
    total_output = len(unique_records)
    
    # Build diagnostics
    diagnostics = {
        "identity_duplicate_groups": len(identity_dupe_groups),
        "content_duplicate_groups": len(content_dupe_groups),
        "largest_identity_group": max(identity_dupe_groups.values()) + 1 if identity_dupe_groups else 0,
        "largest_content_group": max(content_dupe_groups.values()) + 1 if content_dupe_groups else 0,
        "relevance_conflict_details": relevance_conflict_details[:10],  # Limit to first 10
    }
    
    logger.info(
        f"Deduplication complete: {total_input} → {total_output} "
        f"(identity: -{identity_duplicates}, content: -{content_duplicates}, "
        f"relevance conflicts: {relevance_conflicts})"
    )
    
    return DeduplicationResult(
        unique_records=unique_records,
        total_input=total_input,
        total_output=total_output,
        identity_duplicates_removed=identity_duplicates,
        content_duplicates_removed=content_duplicates,
        relevance_conflicts=relevance_conflicts,
        diagnostics=diagnostics,
    )


class IncrementalDeduplicator:
    """Incremental deduplicator for processing large datasets in batches.
    
    Maintains state across batches to detect duplicates globally without
    loading the entire dataset into memory.
    
    Example:
        >>> deduper = IncrementalDeduplicator()
        >>> for batch in batches:
        ...     unique_batch = deduper.process_batch(batch)
        ...     writer.write(unique_batch)
        >>> stats = deduper.get_statistics()
    """
    
    def __init__(self, keep_relevance_priority: bool = True):
        """Initialize incremental deduplicator.
        
        Args:
            keep_relevance_priority: If True, prefer is_selected=True on conflicts
        """
        self.keep_relevance_priority = keep_relevance_priority
        self._seen_doc_ids: set[str] = set()
        self._seen_fingerprints: dict[str, bool] = {}  # fingerprint -> is_selected
        self._identity_duplicates = 0
        self._content_duplicates = 0
        self._relevance_conflicts = 0
        self._total_processed = 0
    
    def process_batch(self, passages: list[CanonicalPassage]) -> list[CanonicalPassage]:
        """Process a batch and return deduplicated records.
        
        Updates internal state to track seen records globally.
        
        Args:
            passages: Batch of CanonicalPassage records
        
        Returns:
            Deduplicated records from this batch
        """
        unique_in_batch = []
        
        for passage in passages:
            self._total_processed += 1
            
            # Level 1: Check identity duplicate
            if passage.document_id in self._seen_doc_ids:
                self._identity_duplicates += 1
                continue
            
            # Level 2: Check content duplicate
            fingerprint = _compute_content_fingerprint(passage)
            
            if fingerprint in self._seen_fingerprints:
                existing_is_selected = self._seen_fingerprints[fingerprint]
                
                # Check relevance conflict
                if existing_is_selected != passage.is_selected:
                    self._relevance_conflicts += 1
                    
                    # If prioritizing relevance and this one is selected, update state
                    if self.keep_relevance_priority and passage.is_selected:
                        self._seen_fingerprints[fingerprint] = True
                        # Note: We cannot retroactively change previously written records
                        # This is a limitation of incremental processing
                        logger.warning(
                            f"Relevance conflict in incremental mode: "
                            f"fingerprint={fingerprint[:16]}... found is_selected=True "
                            f"after is_selected=False was already processed"
                        )
                
                self._content_duplicates += 1
                continue
            
            # Not a duplicate - add to output and state
            self._seen_doc_ids.add(passage.document_id)
            self._seen_fingerprints[fingerprint] = passage.is_selected
            unique_in_batch.append(passage)
        
        return unique_in_batch
    
    def get_statistics(self) -> dict[str, int]:
        """Get deduplication statistics.
        
        Returns:
            Dict with statistics
        """
        unique_count = len(self._seen_doc_ids)
        return {
            "total_processed": self._total_processed,
            "unique_records": unique_count,
            "identity_duplicates_removed": self._identity_duplicates,
            "content_duplicates_removed": self._content_duplicates,
            "relevance_conflicts": self._relevance_conflicts,
        }
    
    def reset(self):
        """Reset internal state."""
        self._seen_doc_ids.clear()
        self._seen_fingerprints.clear()
        self._identity_duplicates = 0
        self._content_duplicates = 0
        self._relevance_conflicts = 0
        self._total_processed = 0
