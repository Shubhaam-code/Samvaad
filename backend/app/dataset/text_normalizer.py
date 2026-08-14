"""Text normalization utilities for multilingual RAG preprocessing.

Provides safe, deterministic text cleaning that preserves semantic meaning
and multilingual characters (Hindi/Indic, English, etc.) while normalizing
whitespace and Unicode representation.

Phase 2.2.4: Text normalization (no translation, no lowercasing, no stemming).
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str | None) -> str:
    """Normalize text while preserving multilingual characters and meaning.
    
    Performs:
    - Whitespace normalization (collapse, strip)
    - Unicode normalization (NFC for compatibility)
    - Safe handling of None/empty strings
    
    Preserves:
    - Hindi/Indic characters and combining marks
    - English characters
    - Capitalization
    - Punctuation
    - Numbers
    - Semantic meaning
    
    Does NOT:
    - Lowercase text
    - Remove punctuation
    - Remove stopwords
    - Stem/lemmatize
    - Transliterate
    - Translate
    
    Args:
        text: Input text (may be None or empty)
    
    Returns:
        Normalized text (empty string if input was None/empty)
    
    Example:
        >>> normalize_text("  यह   भारत की\\nराजधानी है।  ")
        'यह भारत की राजधानी है।'
        >>> normalize_text("  Hello   World!\\t\\n")
        'Hello World!'
    """
    if text is None or text == "":
        return ""
    
    # Unicode normalization (NFC - Canonical Decomposition followed by Canonical Composition)
    # This is safe for Hindi/Indic scripts and preserves combining marks
    normalized = unicodedata.normalize("NFC", text)
    
    # Replace tabs and newlines with spaces
    normalized = normalized.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    
    # Collapse multiple spaces into single space
    normalized = re.sub(r" +", " ", normalized)
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized


def normalize_optional_text(text: str | None) -> str | None:
    """Normalize optional text fields, preserving None if input is None.
    
    For optional fields like answer, eng_answer, we want to preserve the
    distinction between "field was None" and "field was empty string".
    
    Args:
        text: Input text (may be None)
    
    Returns:
        Normalized text, or None if input was None and normalization yielded empty
    
    Example:
        >>> normalize_optional_text(None)
        None
        >>> normalize_optional_text("  text  ")
        'text'
        >>> normalize_optional_text("   ")
        None
    """
    if text is None:
        return None
    
    normalized = normalize_text(text)
    
    # If normalization resulted in empty string, return None to preserve semantics
    return normalized if normalized else None


def normalize_text_batch(texts: list[str | None]) -> list[str]:
    """Normalize a batch of texts efficiently.
    
    Args:
        texts: List of text strings (may contain None values)
    
    Returns:
        List of normalized texts (empty strings for None inputs)
    """
    return [normalize_text(t) for t in texts]


def is_whitespace_only(text: str | None) -> bool:
    """Check if text is None, empty, or contains only whitespace.
    
    Args:
        text: Input text
    
    Returns:
        True if text is None/empty/whitespace-only, False otherwise
    """
    return not text or not text.strip()
