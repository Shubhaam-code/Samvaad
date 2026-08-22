"""Post-generation grounding verifier.

Verifies whether generated answers are factually supported by retrieved Chunk evidence.

Implements lightweight, deterministic claim extraction, token overlap matching,
numerical fact checking, and token sequence similarity against retrieved Chunk evidence text.
Operates without LLM or ML dependencies to ensure low latency (<15ms).
"""

from __future__ import annotations

import difflib
import re
from typing import Final, Sequence

from app.chunking.models import Chunk
from app.dataset.text_normalizer import is_whitespace_only, normalize_text
from app.guardrails.models import GuardrailResult, GuardrailVerdict

# Stop words to filter out when extracting substantive tokens
_ENGLISH_STOP_WORDS: Final[set[str]] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "and", "or",
    "it", "its", "this", "that", "these", "those", "has", "have", "had",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should",
    "as", "but", "not", "so", "if", "than", "then", "there", "here", "which",
}

_HINDI_STOP_WORDS: Final[set[str]] = {
    "है", "हैं", "था", "थी", "थे", "का", "की", "के", "में", "से", "ने",
    "पर", "को", "और", "या", "यह", "वह", "जो", "एक", "इन", "उन", "भी",
    "ही", "तो", "ने", "कर", "दिया", "लिए", "रहे", "रहा", "रही",
}

_ALL_STOP_WORDS: Final[set[str]] = _ENGLISH_STOP_WORDS | _HINDI_STOP_WORDS

# Sentence boundary pattern for claim splitting (. ! ? | । newline)
_SENTENCE_SPLIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?|।\n])\s+"
)

# Numeric token pattern
_NUMERIC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b\d+(?:\.\d+)?\b"
)

# Substantive word token pattern (English & Devanagari / Indic scripts).
#
# U+0964 (danda) and U+0965 (double danda) are punctuation but fall inside the
# U+0900-U+0D7F Indic block, so a naive range glues them onto the final word of
# every Hindi sentence ("है।"). Such tokens then match neither the stop-word
# set nor any evidence token, which silently depressed grounding scores for
# every Devanagari answer. Both are excluded here.
_WORD_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[a-zA-Z0-9\u0900-\u0963\u0966-\u0D7F]+"
)


# --- Refusal detection ----------------------------------------------------
#
# A refusal ("the provided context does not contain the answer") makes no
# assertion about the world, so it cannot be ungrounded. Without this, a
# correct refusal is flagged as UNGROUNDED_FLAGGED and the voice endpoint
# returns 422 for answers that are in fact exactly what we want.
#
# The test is deliberately three-part so it cannot be used to smuggle a
# fabrication past the verifier:
#   1. the claim refers to the provided context/evidence, AND
#   2. the claim contains an absence marker, AND
#   3. the claim introduces no substantive token that is not already in the
#      user's question or in the refusal frame vocabulary.
#
# Part 3 is what stops "The capital of Goa is Panduri, no other information
# is available" from being read as a refusal: "panduri" appears in neither
# the question nor the frame vocabulary.

_CONTEXT_MARKERS: Final[tuple[str, ...]] = (
    "provided", "context", "passage", "evidence", "document", "reference",
    "given", "retrieved", "source", "information", "text", "data",
    "संदर्भ", "प्रदान", "जानकारी", "पाठ", "दस्तावेज", "अंश", "सूचना",
)

_ABSENCE_MARKERS: Final[tuple[str, ...]] = (
    "not contain", "does not contain", "do not contain", "doesn't contain",
    "no information", "not mention", "does not mention", "not mentioned",
    "cannot answer", "can not answer", "can't answer", "unable to answer",
    "not provided", "not available", "no relevant", "not include",
    "does not include", "not found", "no answer", "insufficient",
    "do not know", "don't know", "not specify", "does not specify",
    "not enough", "no mention", "not present",
    # Hindi absence markers are compound on purpose: a bare "नहीं" is an
    # ordinary negation and would match assertions too.
    "जानकारी नहीं", "उल्लेख नहीं", "शामिल नहीं", "उपलब्ध नहीं",
    "मौजूद नहीं", "पता नहीं", "स्पष्ट नहीं", "उत्तर नहीं", "सूचना नहीं",
    "नहीं मिल", "नहीं दी", "नहीं दिया", "नहीं बताया", "नहीं किया गया",
)

# Words allowed to appear in a refusal without counting as new content.
_REFUSAL_FRAME_WORDS: Final[set[str]] = {
    # English frame
    "the", "provided", "context", "contexts", "passage", "passages", "evidence",
    "document", "documents", "reference", "references", "given", "retrieved",
    "source", "sources", "information", "text", "texts", "data", "answer",
    "question", "contain", "contains", "mention", "mentions", "mentioned",
    "include", "includes", "included", "specify", "specifies", "available",
    "found", "know", "about", "any", "relevant", "sufficient", "insufficient",
    "enough", "present", "unable", "cannot", "unfortunately", "sorry",
    "regarding", "related", "does", "no", "none", "nothing", "state", "stated",
    "apologize", "however", "based", "only", "asked", "requested", "unclear",
    # Hindi / Devanagari frame
    "संदर्भ", "संदर्भों", "प्रदान", "जानकारी", "पाठ", "दस्तावेज", "अंश",
    "सूचना", "उल्लेख", "शामिल", "उपलब्ध", "उत्तर", "प्रश्न", "बारे",
    "कोई", "नहीं", "नही", "मिली", "मिला", "पता", "स्पष्ट", "गई", "गए",
    "किए", "क्षमा", "करें", "दिए", "मौजूद", "आधार",
}


def _is_refusal_claim(claim: str, query_tokens: set[str]) -> bool:
    """Return True when a claim is a refusal rather than a factual assertion.

    Args:
        claim: Single normalized sentence claim.
        query_tokens: Substantive tokens from the user's question.

    Returns:
        True if the claim only declines to answer.
    """
    lowered = claim.lower()

    if not any(marker in lowered for marker in _CONTEXT_MARKERS):
        return False
    if not any(marker in lowered for marker in _ABSENCE_MARKERS):
        return False

    claim_words, claim_numbers = _extract_tokens_and_numbers(claim)

    # A refusal restates the question at most; it introduces no new numbers.
    # Any figure absent from the evidence is exactly what we must catch.
    if claim_numbers:
        return False

    novel = {
        w for w in claim_words
        if w not in _REFUSAL_FRAME_WORDS and w not in query_tokens
    }

    # One novel token is tolerated to absorb script mismatch: the user may ask
    # "Goa की राजधानी क्या है?" and be answered in Devanagari ("गोवा"), which
    # is the same topic word transliterated. Two or more novel tokens means the
    # answer is introducing content, so it gets verified normally.
    return len(novel) <= 1


def _split_into_claims(answer: str) -> list[str]:
    """Split answer string into distinct sentence claims.

    Args:
        answer: Generated answer text.

    Returns:
        List of non-empty normalized sentence claims.
    """
    raw_sentences = _SENTENCE_SPLIT_PATTERN.split(answer.strip())
    claims: list[str] = []
    for s in raw_sentences:
        norm = normalize_text(s)
        if norm:
            claims.append(norm)
    if not claims and answer.strip():
        norm = normalize_text(answer)
        if norm:
            claims.append(norm)
    return claims


def _extract_tokens_and_numbers(text: str) -> tuple[set[str], set[str]]:
    """Extract substantive word tokens and numeric strings from text.

    Args:
        text: Normalized input text.

    Returns:
        Tuple of (substantive_words_set, numeric_strings_set).
    """
    lowered = text.lower()
    numbers = set(_NUMERIC_PATTERN.findall(lowered))

    all_words = _WORD_TOKEN_PATTERN.findall(lowered)
    substantive_words = {
        w for w in all_words
        if w not in _ALL_STOP_WORDS and len(w) > 1 or w.isdigit()
    }

    return substantive_words, numbers


def _evaluate_claim_support(
    claim: str,
    evidence_chunks: list[str],
) -> float:
    """Evaluate support score for a single claim against retrieved evidence chunks.

    Args:
        claim: Single sentence claim string.
        evidence_chunks: List of normalized retrieved chunk text strings.

    Returns:
        Support score float in range [0.0, 1.0].
    """
    claim_norm = normalize_text(claim).lower()
    claim_words, claim_numbers = _extract_tokens_and_numbers(claim)

    # Combine all evidence into one text for overall numeric checking
    all_evidence_text = " ".join(evidence_chunks).lower()

    # Rule 1: Numeric accuracy check. If a claim contains a number, it MUST exist in evidence.
    if claim_numbers:
        all_evidence_numbers = set(_NUMERIC_PATTERN.findall(all_evidence_text))
        missing_numbers = claim_numbers - all_evidence_numbers
        if missing_numbers:
            return 0.0

    # Rule 2: If no substantive words left (e.g. extremely short sentence), fallback to substring match
    if not claim_words:
        return 1.0 if claim_norm in all_evidence_text else 0.0

    # Rule 3: Calculate best support score across individual evidence chunks
    best_score = 0.0
    for ev in evidence_chunks:
        ev_norm = ev.lower()
        ev_words, _ = _extract_tokens_and_numbers(ev)

        if not ev_words:
            continue

        # Lexical token overlap ratio
        overlap_count = len(claim_words.intersection(ev_words))
        token_overlap_ratio = overlap_count / len(claim_words)

        # Sequence similarity ratio (catches paraphrased phrasing)
        seq_sim = difflib.SequenceMatcher(None, claim_norm, ev_norm).ratio()

        # Combine overlap ratio and sequence similarity
        score = 0.7 * token_overlap_ratio + 0.3 * seq_sim
        if score > best_score:
            best_score = score

    return best_score


class GroundingVerifier:
    """Post-generation verifier for checking answer grounding against retrieved evidence chunks."""

    def verify(
        self,
        answer: str,
        retrieved_chunks: list[Chunk],
        query: str | None = None,
    ) -> GuardrailResult:
        """Verify whether generated answer is supported by retrieved Chunk evidence.

        Args:
            answer: Generated answer text.
            retrieved_chunks: List of retrieved Chunk instances containing evidence text.
            query: The user's question. When supplied, an answer that merely
                declines to answer is treated as grounded instead of being
                flagged, since a refusal asserts nothing about the world.

        Returns:
            GuardrailResult with verdict SAFE_AND_GROUNDED if all claims are supported,
            or UNGROUNDED_FLAGGED with flagged_claims if any claims lack evidence.

        Raises:
            TypeError: If answer is not a string or retrieved_chunks is not a list.
        """
        # Step 1: Type validation
        if not isinstance(answer, str):
            raise TypeError(f"answer must be a string instance, got {type(answer).__name__}")
        if not isinstance(retrieved_chunks, list):
            raise TypeError(f"retrieved_chunks must be a list, got {type(retrieved_chunks).__name__}")

        # Step 2: Empty / whitespace answer check
        if is_whitespace_only(answer):
            return GuardrailResult(
                verdict=GuardrailVerdict.UNGROUNDED_FLAGGED,
                reason="Generated answer is empty or contains only whitespace.",
                score=0.0,
                flagged_claims=["<empty answer>"],
            )

        # Extract claims
        claims = _split_into_claims(answer)
        if not claims:
            return GuardrailResult(
                verdict=GuardrailVerdict.UNGROUNDED_FLAGGED,
                reason="Generated answer contains no valid claims.",
                score=0.0,
                flagged_claims=[answer.strip()],
            )

        # A pure refusal is grounded by construction: it asserts nothing that
        # could contradict the evidence. Checked before the empty-evidence
        # branch, because "no evidence retrieved" is precisely the case where
        # refusing is the correct behaviour.
        if query is not None:
            query_tokens, _ = _extract_tokens_and_numbers(normalize_text(query))
            if all(_is_refusal_claim(c, query_tokens) for c in claims):
                return GuardrailResult(
                    verdict=GuardrailVerdict.SAFE_AND_GROUNDED,
                    reason="Answer declines to answer; no factual claim to verify.",
                    score=1.0,
                    flagged_claims=[],
                )

        # Extract evidence text from retrieved chunks (using Chunk.chunk_text)
        evidence_texts = [
            normalize_text(chunk.chunk_text)
            for chunk in retrieved_chunks
            if chunk and hasattr(chunk, "chunk_text") and chunk.chunk_text
        ]

        # Step 3: Empty retrieved context check
        if not evidence_texts:
            return GuardrailResult(
                verdict=GuardrailVerdict.UNGROUNDED_FLAGGED,
                reason="No retrieved evidence context was provided to verify answer.",
                score=0.0,
                flagged_claims=claims,
            )

        # Step 4: Evaluate each claim against evidence
        unsupported_claims: list[str] = []
        claim_scores: list[float] = []

        # Support score threshold: >= 0.55 required for a claim to be considered grounded
        _SUPPORT_THRESHOLD = 0.55

        for claim in claims:
            score = _evaluate_claim_support(claim, evidence_texts)
            claim_scores.append(score)
            if score < _SUPPORT_THRESHOLD:
                unsupported_claims.append(claim)

        avg_score = round(sum(claim_scores) / len(claim_scores), 4) if claim_scores else 0.0

        # Step 5: Construct verdict
        if not unsupported_claims:
            return GuardrailResult(
                verdict=GuardrailVerdict.SAFE_AND_GROUNDED,
                reason="All claims in the generated answer are supported by retrieved evidence.",
                score=avg_score,
                flagged_claims=[],
            )
        else:
            return GuardrailResult(
                verdict=GuardrailVerdict.UNGROUNDED_FLAGGED,
                reason=f"{len(unsupported_claims)} claim(s) in the generated answer lack sufficient supporting evidence.",
                score=round(min(claim_scores), 4) if claim_scores else 0.0,
                flagged_claims=unsupported_claims,
            )


__all__ = [
    "GroundingVerifier",
]
