"""Unit tests for dataset helper functions.

These tests use synthetic rows so they never touch the network or
the Hugging Face cache.
"""

from app.dataset import (
    compute_text_length_stats,
    count_missing,
    detect_duplicate_query_ids,
    discover_schema,
    infer_field_roles,
    split_health_check,
)


SAMPLE_ROWS = [
    {
        "query_id": 1,
        "query": "short",
        "eng_query": "longer english query",
        "answer": "a" * 120,
        "eng_answer": "b" * 80,
        "source_lang": "eng_Latn",
        "target_lang": "asm_Beng",
        "passages": {
            "is_selected": [1, 0, 0],
            "english_passages": ["alpha", "beta", "gamma"],
            "translated_passages": ["a-t", "b-t", "c-t"],
        },
        "meta": {"model_name": "m", "temperature": 0.2, "max_tokens": 256},
    },
    {
        "query_id": 2,
        "query": "",
        "eng_query": "",
        "answer": None,
        "eng_answer": "",
        "source_lang": "eng_Latn",
        "target_lang": "ben_Beng",
        "passages": {
            "is_selected": [0, 0],
            "english_passages": [],
            "translated_passages": [],
        },
        "meta": {"model_name": "m", "temperature": 0.1, "max_tokens": 256},
    },
    {
        "query_id": 1,  # duplicate
        "query": "another",
        "eng_query": "another en",
        "answer": "c" * 50,
        "eng_answer": "d" * 40,
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passages": {
            "is_selected": [1],
            "english_passages": ["x"],
            "translated_passages": ["x-t"],
        },
        "meta": {"model_name": "m", "temperature": 0.0, "max_tokens": 128},
    },
]


def test_discover_schema_reports_nested_types() -> None:
    schema = discover_schema(SAMPLE_ROWS[0])
    assert schema["type"] == "dict"
    assert schema["children"]["query"]["type"] == "str"
    assert schema["children"]["passages"]["type"] == "dict"
    passages_children = schema["children"]["passages"]["children"]
    assert passages_children["english_passages"]["type"] == "list"
    assert passages_children["is_selected"]["type"] == "list"
    assert schema["children"]["query_id"]["type"] == "int"


def test_infer_field_roles_classifies_known_fields() -> None:
    roles = infer_field_roles(SAMPLE_ROWS[0])
    assert "query" in roles["potential_query_fields"]
    assert "eng_query" in roles["potential_query_fields"]
    assert "passages.english_passages" in roles["potential_document_fields"]
    assert "passages.translated_passages" in roles["potential_document_fields"]
    assert "query_id" in roles["potential_metadata_fields"]
    assert "source_lang" in roles["potential_metadata_fields"]


def test_count_missing_detects_null_and_empty() -> None:
    assert count_missing(iter(SAMPLE_ROWS), "query") == 1  # row 2 has empty query
    assert count_missing(iter(SAMPLE_ROWS), "answer") == 1  # row 2 is None
    assert count_missing(iter(SAMPLE_ROWS), "eng_answer") == 1
    assert count_missing(iter(SAMPLE_ROWS), "passages.translated_passages") == 1  # empty list
    assert count_missing(iter(SAMPLE_ROWS), "query_id") == 0


def test_compute_text_length_stats_uses_sample() -> None:
    stats = compute_text_length_stats(iter(SAMPLE_ROWS), "query", sample_size=10)
    assert stats["size"] == 3
    # lengths: 5 ("short"), 0 (""), 7 ("another")
    assert stats["min"] == 0
    assert stats["max"] == 7
    assert stats["mean"] == 4.0


def test_compute_text_length_stats_handles_empty_input() -> None:
    stats = compute_text_length_stats(iter([]), "query", sample_size=10)
    assert stats["size"] == 0
    assert stats["mean"] is None


def test_detect_duplicate_query_ids_finds_repeats() -> None:
    info = detect_duplicate_query_ids(iter(SAMPLE_ROWS), id_field="query_id", sample_size=10)
    assert info["duplicate_id_count"] == 1
    assert 1 in info["sample_duplicate_ids"]


def test_split_health_check_flags_empty_rows() -> None:
    health = split_health_check(iter(SAMPLE_ROWS))
    assert health["rows_scanned"] == 3
    assert health["empty_query_rows"] == 1
    assert health["rows_without_passages"] == 1
