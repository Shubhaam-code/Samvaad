"""Schema discovery + field-role inference.

The MSMARCO-XI dataset has a nested structure (``passages`` is a dict of
lists). We need to walk the schema programmatically rather than assume
column names, then classify each field as a candidate query, document,
or metadata field based on the type and name.
"""

from __future__ import annotations

from typing import Any


# Heuristic name patterns. Matched case-insensitively against field names.
QUERY_HINTS = ("query", "question", "q_")
DOCUMENT_HINTS = ("passage", "document", "doc", "context", "translated_passages", "english_passages")
METADATA_HINTS = (
    "id",
    "lang",
    "type",
    "model",
    "temperature",
    "tokens",
    "penalty",
    "is_selected",
    "meta",
)
TEXT_COLUMN_HINTS = QUERY_HINTS + DOCUMENT_HINTS + (
    "answer",
    "eng_query",
    "eng_answer",
    "english_passages",
    "translated_passages",
)


def discover_schema(example: dict[str, Any]) -> dict[str, Any]:
    """Recursively describe the schema of a single example row.

    Returns a nested mapping of ``field -> {"type": ..., "children": {...}}``
    so we can reason about nested fields (``passages.is_selected`` etc.).
    """
    return _describe(example)


def _describe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "dict",
            "children": {k: _describe(v) for k, v in value.items()},
        }
    if isinstance(value, list):
        # Lists are typically list[str] in MSMARCO-XI, but could be list[dict].
        inner_types = {type(item).__name__ for item in value[:5]}
        return {
            "type": "list",
            "element_type": (next(iter(inner_types)) if inner_types else "unknown"),
            "length_example": len(value),
        }
    if value is None:
        return {"type": "null"}
    return {"type": type(value).__name__}


def _field_path(name: str, suffix: str) -> str:
    return f"{name}.{suffix}" if suffix else name


def flatten_paths(example: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Return ``(dotted_path, sample_value)`` pairs for every leaf field.

    A "leaf" is either a scalar or a list. Dicts are walked into.
    """
    out: list[tuple[str, Any]] = []
    for key, value in example.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.extend(flatten_paths(value, prefix=path))
        else:
            out.append((path, value))
    return out


def infer_field_roles(example: dict[str, Any]) -> dict[str, list[str]]:
    """Classify field paths into query / document / metadata buckets.

    Returns ``{"potential_query_fields": [...],
              "potential_document_fields": [...],
              "potential_metadata_fields": [...]}``.
    """
    leaves = flatten_paths(example)
    queries: list[str] = []
    documents: list[str] = []
    metadata: list[str] = []

    for path, sample in leaves:
        lowered = path.lower()
        # Check metadata first so "query_id" is not classified as a query.
        if any(hint in lowered for hint in METADATA_HINTS):
            metadata.append(path)
        elif any(hint in lowered for hint in QUERY_HINTS):
            queries.append(path)
        elif any(hint in lowered for hint in DOCUMENT_HINTS):
            documents.append(path)
        else:
            # Catch-all: unknown fields are metadata until we learn otherwise.
            metadata.append(path)

    return {
        "potential_query_fields": sorted(set(queries)),
        "potential_document_fields": sorted(set(documents)),
        "potential_metadata_fields": sorted(set(metadata)),
    }
