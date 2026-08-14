"""Lightweight statistics for MSMARCO-XI.

All helpers operate on iterables of rows (so they work with both
``Dataset`` and ``IterableDataset``) and accept an explicit ``sample_size``
cap so expensive stats can run on a documented subset.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping


def count_missing(rows: Iterable[Mapping[str, Any]], field: str) -> int:
    """Count rows where ``field`` (dotted path) is None, "", or missing."""
    parts = field.split(".")
    missing = 0
    for row in rows:
        value: Any = row
        for part in parts:
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is None or value == "" or value == []:
            missing += 1
    return missing


def compute_text_length_stats(
    rows: Iterable[Mapping[str, Any]],
    field: str,
    sample_size: int | None = 5000,
) -> dict[str, Any]:
    """Return char-length stats for ``field`` over up to ``sample_size`` rows.

    Stats: count, mean, median, min, max, plus p50/p90/p99.
    Returns ``{"sampled": True/False, "size": ..., ...}``.
    """
    parts = field.split(".")
    lengths: list[int] = []
    total_seen = 0

    for row in rows:
        total_seen += 1
        if sample_size is not None and len(lengths) >= sample_size:
            break
        value: Any = row
        for part in parts:
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is None:
            continue
        if isinstance(value, list):
            # For passage-list fields, measure the *list size*, and skip
            # inner stats here — callers should drill into nested fields.
            lengths.append(len(value))
        elif isinstance(value, str):
            lengths.append(len(value))
        else:
            lengths.append(0)

    if not lengths:
        return {
            "sampled": sample_size is not None,
            "size": 0,
            "rows_scanned": total_seen,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }

    sorted_lengths = sorted(lengths)

    def _percentile(p: float) -> float:
        if not sorted_lengths:
            return 0.0
        idx = max(0, min(len(sorted_lengths) - 1, int(round(p * (len(sorted_lengths) - 1)))))
        return float(sorted_lengths[idx])

    return {
        "sampled": sample_size is not None,
        "size": len(lengths),
        "rows_scanned": total_seen,
        "mean": round(statistics.fmean(lengths), 2),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "p50": _percentile(0.50),
        "p90": _percentile(0.90),
        "p99": _percentile(0.99),
    }


def detect_duplicate_query_ids(
    rows: Iterable[Mapping[str, Any]],
    id_field: str = "query_id",
    sample_size: int | None = 200000,
) -> dict[str, Any]:
    """Return whether any ``query_id`` repeats within a bounded sample.

    Streaming the full ~10M-row train split would be wasteful for a
    quick structural check; a few hundred thousand is enough to flag
    duplicates if they exist.
    """
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    scanned = 0
    for row in rows:
        scanned += 1
        if sample_size is not None and scanned > sample_size:
            break
        qid = row.get(id_field)
        if qid is None:
            continue
        if qid in seen:
            duplicates.add(qid)
        else:
            seen.add(qid)
    return {
        "id_field": id_field,
        "rows_scanned": scanned,
        "sample_capped": sample_size is not None and scanned > sample_size,
        "unique_ids": len(seen),
        "duplicate_id_count": len(duplicates),
        "sample_duplicate_ids": sorted(list(duplicates))[:10],
    }


def split_health_check(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Walk a sample and report any rows where the example is empty/broken."""
    total = 0
    empty_query = 0
    empty_passages = 0
    for row in rows:
        total += 1
        if not row.get("query"):
            empty_query += 1
        passages = row.get("passages") or {}
        if not passages.get("translated_passages") and not passages.get("english_passages"):
            empty_passages += 1
        if total >= 1000:
            break
    return {
        "rows_scanned": total,
        "empty_query_rows": empty_query,
        "rows_without_passages": empty_passages,
    }
