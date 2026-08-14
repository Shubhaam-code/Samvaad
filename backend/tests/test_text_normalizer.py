"""Tests for text normalization utilities.

Phase 2.2.4: Text normalization testing.
"""

import pytest

from app.dataset.text_normalizer import (
    is_whitespace_only,
    normalize_optional_text,
    normalize_text,
    normalize_text_batch,
)


def test_normalize_text_strips_leading_trailing_whitespace():
    """Test that leading and trailing whitespace is removed."""
    assert normalize_text("  hello  ") == "hello"
    assert normalize_text("\t\thello\t\t") == "hello"
    assert normalize_text("\n\nhello\n\n") == "hello"


def test_normalize_text_collapses_repeated_spaces():
    """Test that multiple spaces are collapsed to single space."""
    assert normalize_text("hello    world") == "hello world"
    assert normalize_text("hello     world    test") == "hello world test"


def test_normalize_text_replaces_tabs_with_spaces():
    """Test that tabs are converted to spaces."""
    assert normalize_text("hello\tworld") == "hello world"
    assert normalize_text("hello\t\tworld") == "hello world"


def test_normalize_text_replaces_newlines_with_spaces():
    """Test that newlines are converted to spaces."""
    assert normalize_text("hello\nworld") == "hello world"
    assert normalize_text("hello\r\nworld") == "hello world"
    assert normalize_text("hello\n\nworld") == "hello world"


def test_normalize_text_combined_whitespace():
    """Test combination of different whitespace types."""
    assert normalize_text("  hello \n\t  world  \t\n") == "hello world"


def test_normalize_text_preserves_hindi_characters():
    """Test that Hindi/Devanagari characters are preserved."""
    assert normalize_text("भारत की राजधानी") == "भारत की राजधानी"
    assert normalize_text("  भारत   की  \nराजधानी  ") == "भारत की राजधानी"


def test_normalize_text_preserves_hindi_matras_and_combining_marks():
    """Test that Hindi matras and combining marks are preserved."""
    # Matras and combining characters
    text_with_matras = "नमस्ते दुनिया"
    normalized = normalize_text(f"  {text_with_matras}  ")
    assert normalized == text_with_matras
    assert "नमस्ते" in normalized
    assert "दुनिया" in normalized


def test_normalize_text_preserves_english_characters():
    """Test that English characters are preserved."""
    assert normalize_text("Hello World") == "Hello World"
    assert normalize_text("  Hello   World  ") == "Hello World"


def test_normalize_text_preserves_capitalization():
    """Test that capitalization is preserved."""
    assert normalize_text("Hello World") == "Hello World"
    assert normalize_text("HELLO WORLD") == "HELLO WORLD"
    assert normalize_text("hElLo WoRlD") == "hElLo WoRlD"


def test_normalize_text_preserves_punctuation():
    """Test that punctuation is preserved."""
    assert normalize_text("Hello, World!") == "Hello, World!"
    assert normalize_text("क्या यह सही है?") == "क्या यह सही है?"
    assert normalize_text("Cost: $100.50") == "Cost: $100.50"


def test_normalize_text_preserves_numbers():
    """Test that numbers are preserved."""
    assert normalize_text("123 456") == "123 456"
    assert normalize_text("  123  456  ") == "123 456"
    assert normalize_text("Price: 1,234.56") == "Price: 1,234.56"


def test_normalize_text_handles_none():
    """Test that None input returns empty string."""
    assert normalize_text(None) == ""


def test_normalize_text_handles_empty_string():
    """Test that empty string returns empty string."""
    assert normalize_text("") == ""


def test_normalize_text_handles_whitespace_only():
    """Test that whitespace-only string returns empty string."""
    assert normalize_text("   ") == ""
    assert normalize_text("\t\n\r") == ""
    assert normalize_text("  \t  \n  ") == ""


def test_normalize_text_unicode_normalization():
    """Test that Unicode normalization (NFC) is applied."""
    # Some characters can be represented in multiple ways in Unicode
    # NFC ensures consistent representation
    text1 = "café"  # é as single character
    text2 = "café"  # é as e + combining acute accent
    
    # Both should normalize to the same result
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    # Both should use NFC form
    assert norm1 == norm2


def test_normalize_text_is_deterministic():
    """Test that normalization is deterministic."""
    text = "  यह   भारत की\nराजधानी है।  "
    result1 = normalize_text(text)
    result2 = normalize_text(text)
    result3 = normalize_text(text)
    
    assert result1 == result2 == result3


def test_normalize_text_repeated_normalization_is_idempotent():
    """Test that normalizing already-normalized text doesn't change it."""
    text = "  यह   भारत की\nराजधानी है।  "
    normalized_once = normalize_text(text)
    normalized_twice = normalize_text(normalized_once)
    normalized_thrice = normalize_text(normalized_twice)
    
    assert normalized_once == normalized_twice == normalized_thrice


def test_normalize_text_multilingual_mixed():
    """Test text with mixed Hindi and English."""
    text = "  Hello   यह  \n test   है  "
    assert normalize_text(text) == "Hello यह test है"


def test_normalize_text_preserves_semantic_meaning():
    """Test that normalization preserves semantic meaning."""
    # Hindi sentence
    hindi = "  भारत की   राजधानी \n नई दिल्ली है।  "
    normalized_hindi = normalize_text(hindi)
    assert "भारत" in normalized_hindi
    assert "राजधानी" in normalized_hindi
    assert "नई दिल्ली" in normalized_hindi
    
    # English sentence
    english = "  The   capital \n of India  is  New Delhi.  "
    normalized_english = normalize_text(english)
    assert "capital" in normalized_english
    assert "India" in normalized_english
    assert "New Delhi" in normalized_english


def test_normalize_optional_text_preserves_none():
    """Test that None input to optional normalizer returns None."""
    assert normalize_optional_text(None) is None


def test_normalize_optional_text_normalizes_valid_text():
    """Test that valid text is normalized."""
    assert normalize_optional_text("  hello  ") == "hello"
    assert normalize_optional_text("  यह  है  ") == "यह है"


def test_normalize_optional_text_converts_whitespace_only_to_none():
    """Test that whitespace-only text becomes None."""
    assert normalize_optional_text("   ") is None
    assert normalize_optional_text("\t\n") is None


def test_normalize_text_batch():
    """Test batch normalization."""
    texts = [
        "  hello  ",
        "  world  ",
        None,
        "  test\nvalue  ",
    ]
    
    normalized = normalize_text_batch(texts)
    
    assert normalized == ["hello", "world", "", "test value"]


def test_normalize_text_batch_empty():
    """Test batch normalization with empty list."""
    assert normalize_text_batch([]) == []


def test_is_whitespace_only_detects_none():
    """Test that None is detected as whitespace."""
    assert is_whitespace_only(None) is True


def test_is_whitespace_only_detects_empty():
    """Test that empty string is detected as whitespace."""
    assert is_whitespace_only("") is True


def test_is_whitespace_only_detects_whitespace():
    """Test that whitespace-only strings are detected."""
    assert is_whitespace_only("   ") is True
    assert is_whitespace_only("\t\n\r") is True
    assert is_whitespace_only("  \t  \n  ") is True


def test_is_whitespace_only_rejects_valid_text():
    """Test that valid text is not detected as whitespace."""
    assert is_whitespace_only("hello") is False
    assert is_whitespace_only("  hello  ") is False
    assert is_whitespace_only("यह") is False


def test_normalize_text_other_indic_scripts():
    """Test normalization with other Indic scripts."""
    # Tamil
    tamil = "  தமிழ்   நாடு  "
    assert "தமிழ்" in normalize_text(tamil)
    assert "நாடு" in normalize_text(tamil)
    
    # Bengali
    bengali = "  বাংলাদেশ   এর  "
    assert "বাংলাদেশ" in normalize_text(bengali)
    assert "এর" in normalize_text(bengali)


def test_normalize_text_real_world_example():
    """Test with realistic MSMARCO-XI style content."""
    query = "  भारत की   राजधानी  क्या है?  "
    passage = "  भारत की राजधानी\n  नई दिल्ली है।\t यह देश का सबसे बड़ा शहर है।  "
    
    normalized_query = normalize_text(query)
    normalized_passage = normalize_text(passage)
    
    assert normalized_query == "भारत की राजधानी क्या है?"
    assert "नई दिल्ली" in normalized_passage
    assert "सबसे बड़ा शहर" in normalized_passage
    # Should be single-spaced
    assert "  " not in normalized_passage
