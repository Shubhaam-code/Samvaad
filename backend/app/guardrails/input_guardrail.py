"""Pre-generation input guardrail.

Checks incoming user queries before retrieval or generation.

Implements lightweight, deterministic input validation and heuristic rejection
for empty queries, prompt injection attempts, toxic/harmful content, and
gibberish/non-text noise. Preserves technical queries (e.g. programming/system terms)
and supports both English and Hindi text.

Domain & Scope Note:
- The project targets the MSMARCO-XI dataset (`ai4bharat/MSMARCO-XI`), an open-domain,
  multilingual web search Q&A corpus across English and 27 Indic languages.
- Because MSMARCO-XI is open-domain without fixed category boundaries, full semantic
  out-of-domain classification cannot be performed via regex alone without vector retrieval.
  Semantic domain relevance must be handled downstream by the vector store search.

Verdict & Score Semantics:
- The canonical verdict `GuardrailVerdict.SAFE_AND_GROUNDED` returned at this input stage
  represents a PRE-GENERATION PASS (the query passed input safety and validation checks and
  is clear to proceed to retrieval). It does NOT imply that a generated answer is grounded.
  Answer groundedness can only be verified post-generation against retrieved passages.
- The `score=1.0` field is retained for API model compatibility and represents 100% confidence
  in the input-stage guardrail check pass, not evidence/answer groundedness.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from app.dataset.text_normalizer import is_whitespace_only, normalize_text
from app.guardrails.models import GuardrailResult, GuardrailVerdict

# Pre-compiled regex patterns for deterministic, low-overhead matching (<15ms)

# Jailbreak / Prompt Injection patterns
_JAILBREAK_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"(?i)\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?"
        r"(?:previous|prior|above|system|safety|security)\s+"
        r"(?:instructions|directions|rules|prompts|guidelines|filters|restrictions)\b"
    ),
    re.compile(
        r"(?i)\b(?:you\s+are\s+now|pretend\s+(?:to\s+be|you\s+are)|act\s+as)\s+"
        r"(?:dan|jailbroken|unfiltered|god\s+mode|an?\s+unrestricted)\b"
    ),
    re.compile(
        r"(?i)\b(?:system\s+prompt|developer\s+mode|jailbreak\s+mode|override\s+your\s+rules)\b"
    ),
    re.compile(
        r"(?i)(?:\[system\s+message\]|<\|im_start\|>system|###\s*system:)"
    ),
]

# Unsafe / Harmful content patterns
_UNSAFE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"(?i)\bhow\s+to\s+(?:make|build|construct|create|manufacture|craft)\s+.*?\b"
        r"(?:bomb|explosive|weapon|malware|virus)\b"
    ),
    re.compile(
        r"(?i)\bhow\s+to\s+(?:hack|crack|breach|infiltrate)\s+.*?\b"
        r"(?:bank|server|nasa|government|database|account)\b"
    ),
    re.compile(
        r"(?i)\bhow\s+to\s+(?:commit|execute|carry\s+out|perform)\s+.*?\b"
        r"(?:suicide|mass\s+shooting|terrorist|terrorism|attack|murder)\b"
    ),
    re.compile(
        r"(?i)(?:बम\s+कैसे\s+बनाएं|हैक\s+कैसे\s+करें)"
    ),
]

# Character repetition spam pattern (e.g., "aaaaaaaaa", "zzzzzzzz")
_REPETITION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)([a-zA-Z0-9\u0900-\u0D7F])\1{7,}"
)

# Valid text character matching (English, Numbers, Devanagari & Indic scripts)
_TEXT_CHAR_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[a-zA-Z0-9\u0900-\u0D7F]"
)


class InputGuardrail:
    """Pre-generation guardrail for incoming user queries.

    Performs lightweight, deterministic pre-retrieval validation:
    1. Validates input string type (raises TypeError for non-string instances).
    2. Rejects empty or whitespace-only inputs.
    3. Normalizes text safely using NFC Unicode normalization.
    4. Detects common jailbreak and prompt-injection attempts.
    5. Detects clearly unsafe or toxic content.
    6. Detects non-text noise or gibberish.
    7. Allows safe English, Hindi, and technical queries to proceed to retrieval.

    Note on Verdict:
    - Returning `GuardrailVerdict.SAFE_AND_GROUNDED` signifies a PRE-GENERATION PASS.
      Grounding verification occurs post-generation.
    """

    def check(self, query: str) -> GuardrailResult:
        """Check a user query before retrieval.

        Args:
            query: Raw user query string.

        Returns:
            GuardrailResult with OFF_TOPIC_REJECTED for invalid/unsafe/jailbreak/gibberish queries,
            or SAFE_AND_GROUNDED (pre-generation pass state) for valid queries.

        Raises:
            TypeError: If query is not a string instance.
        """
        # Step 1: Strict type check - do not swallow programming errors
        if not isinstance(query, str):
            raise TypeError(f"query must be a string instance, got {type(query).__name__}")

        # Step 2: Empty or whitespace-only check
        if is_whitespace_only(query):
            return GuardrailResult(
                verdict=GuardrailVerdict.OFF_TOPIC_REJECTED,
                reason="Query is empty or contains only whitespace.",
            )

        # Step 3: Unicode and whitespace normalization
        normalized = normalize_text(query)
        if not normalized:
            return GuardrailResult(
                verdict=GuardrailVerdict.OFF_TOPIC_REJECTED,
                reason="Query is empty or contains only whitespace.",
            )

        # Step 4: Check for prompt injection / jailbreak attempts
        for pattern in _JAILBREAK_PATTERNS:
            if pattern.search(normalized):
                return GuardrailResult(
                    verdict=GuardrailVerdict.OFF_TOPIC_REJECTED,
                    reason="Detected potential jailbreak or prompt injection attempt.",
                )

        # Step 5: Check for unsafe / toxic content
        for pattern in _UNSAFE_PATTERNS:
            if pattern.search(normalized):
                return GuardrailResult(
                    verdict=GuardrailVerdict.OFF_TOPIC_REJECTED,
                    reason="Detected unsafe or toxic content.",
                )

        # Step 6: Check for gibberish, excessive character repetition, or non-text noise
        if _REPETITION_PATTERN.search(normalized):
            return GuardrailResult(
                verdict=GuardrailVerdict.OFF_TOPIC_REJECTED,
                reason="Query contains excessive character repetition or gibberish.",
            )

        # Ensure the query contains at least one valid alphanumeric or Indic character
        if not _TEXT_CHAR_PATTERN.search(normalized):
            return GuardrailResult(
                verdict=GuardrailVerdict.OFF_TOPIC_REJECTED,
                reason="Query contains no valid text characters.",
            )

        # Check special character symbol ratio for noise/gibberish (excluding spaces, basic punctuation)
        clean_len = len(normalized)
        if clean_len >= 5:
            # Count characters that are not letters, digits, Indic script, whitespace, or standard punctuation
            non_text_chars = sum(
                1 for c in normalized
                if not (c.isalnum() or '\u0900' <= c <= '\u0D7F' or c in " \t\n\r.,?!'\"-;:()")
            )
            if non_text_chars / clean_len > 0.4:
                return GuardrailResult(
                    verdict=GuardrailVerdict.OFF_TOPIC_REJECTED,
                    reason="Query contains excessive special characters or non-text noise.",
                )

        # Step 7: Safe query - allowed to proceed to retrieval (pre-generation pass state)
        # Note: SAFE_AND_GROUNDED at this stage indicates pre-retrieval validation pass, not answer grounding.
        # score=1.0 represents confidence in the pre-generation guardrail pass.
        return GuardrailResult(
            verdict=GuardrailVerdict.SAFE_AND_GROUNDED,
            reason="Query passed pre-retrieval safety and guardrail checks.",
            score=1.0,
        )


__all__ = [
    "InputGuardrail",
]