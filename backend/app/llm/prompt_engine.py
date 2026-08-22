"""Grounded Prompt Engineering & Structured Citation Builder (Phase 5.4).

Provides prompt templates and citation formatting for the Samvaad RAG system,
ensuring strict grounding in retrieved MSMARCO-XI evidence chunks and preventing
hallucinations across Indic languages and English.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from app.api.schemas import Citation
from app.retrieval.models import RetrievedChunk

DEFAULT_GROUNDED_SYSTEM_PROMPT = (
    "You are Samvaad, an AI voice & text question-answering assistant. "
    "Your answers must be STRICTLY GROUNDED ONLY in the retrieved reference passages provided below. "
    "Do NOT assume or invent any facts not directly stated in the reference text. "
    "Do NOT fall back on your own background knowledge, even if you are confident it is correct. "
    "If the provided evidence does not contain the information needed to answer the question, "
    "state clearly and concisely that the provided context does not contain the answer. "
    "Always respond in the same language as the user's question (Hindi in Devanagari script for "
    "Hindi, English for English). "
    "Keep answers concise, direct, and factual."
)

CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are Samvaad, a voice-first assistant for a grounded question-answering "
    "system built on the MSMARCO-XI Indic corpus. "
    "The user has greeted you or made small talk rather than asking a factual question. "
    "Reply warmly in one or two short sentences and invite them to ask a question about "
    "the indexed knowledge base. "
    "Do NOT state any facts, and do NOT claim to have retrieved anything. "
    "Respond in the same language as the user (Hindi in Devanagari script for Hindi, "
    "English for English)."
)

# Short phrases that are greetings / small talk rather than information requests.
# Matching is exact against the punctuation-stripped, lowercased query so that
# "hello, what is the capital of Goa?" is still treated as a real question.
_CONVERSATIONAL_PHRASES: frozenset[str] = frozenset(
    {
        # English
        "hi", "hii", "hey", "hello", "helo", "yo",
        "good morning", "good afternoon", "good evening", "good night",
        "how are you", "how are you doing", "hows it going", "how is it going",
        "whats up", "sup", "thanks", "thank you", "thankyou", "thx",
        "ok", "okay", "cool", "nice", "great", "bye", "goodbye", "see you",
        "who are you", "what are you", "what can you do", "help",
        "test", "testing",
        # Hindi / Hinglish (Devanagari + romanized)
        "नमस्ते", "नमस्कार", "हैलो", "हेलो", "हाय",
        "शुभ प्रभात", "शुभ रात्रि", "धन्यवाद", "शुक्रिया",
        "कैसे हैं आप", "आप कैसे हैं", "क्या हाल है", "तुम कौन हो", "आप कौन हैं",
        "namaste", "namaskar", "dhanyavad", "shukriya", "kaise ho", "kya haal hai",
    }
)

# Stripped from the query edges before matching. Includes the Devanagari danda.
_PUNCT_EDGES = " \t\r\n.,!?;:\"'()[]{}-_/\\|।॥"


def is_conversational(query: str) -> bool:
    """Return True when a query is a greeting or small talk, not a question.

    Conversational turns should bypass retrieval entirely: running the vector
    search on "hello" wastes an embedding pass and attaches irrelevant
    citations, and sending the result through the grounding verifier would
    flag a friendly reply as ungrounded and refuse it.

    Matching is deliberately conservative. It is exact against the
    normalized query, so anything carrying real informational content
    ("hello, what is the capital of Goa?") is treated as a question.

    Args:
        query: Raw user query or transcript

    Returns:
        True if the query is a greeting / small talk
    """
    if not query or not query.strip():
        return False

    normalized = query.strip().strip(_PUNCT_EDGES).lower()
    if not normalized:
        return False

    # Collapse internal whitespace so "how   are  you" matches.
    normalized = " ".join(normalized.split())
    if normalized in _CONVERSATIONAL_PHRASES:
        return True

    # Allow a trailing filler token: "hi there", "thanks a lot", "hello samvaad".
    tokens = normalized.split()
    if 2 <= len(tokens) <= 3 and tokens[0] in _CONVERSATIONAL_PHRASES:
        fillers = {"there", "samvaad", "bot", "a", "lot", "you", "so", "much", "again"}
        if all(tok in fillers for tok in tokens[1:]):
            return True

    return False


def build_conversational_prompt(query: str) -> Tuple[str, str]:
    """Build prompts for a greeting / small-talk turn (no evidence involved).

    Args:
        query: Raw user query or transcript

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return CONVERSATIONAL_SYSTEM_PROMPT, f"User said: {query.strip()}"


def build_grounded_rag_prompt(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    system_prompt: Optional[str] = None,
) -> Tuple[str, str]:
    """Construct grounded system and user prompts with structured evidence context.

    Args:
        query: The user's query text
        retrieved_chunks: List of retrieved & reranked evidence chunks
        system_prompt: Optional custom system prompt override

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    sys_prompt = system_prompt.strip() if system_prompt and system_prompt.strip() else DEFAULT_GROUNDED_SYSTEM_PROMPT

    clean_query = query.strip()

    if not retrieved_chunks:
        user_prompt = (
            f"Question: {clean_query}\n\n"
            f"No background evidence available. State plainly that the provided "
            f"context does not contain the answer. Do not answer from your own knowledge."
        )
        return sys_prompt, user_prompt

    evidence_blocks = []
    for idx, item in enumerate(retrieved_chunks, start=1):
        chunk_text = getattr(item.chunk, "chunk_text", "").strip()
        doc_id = getattr(item.chunk, "document_id", "doc")
        evidence_blocks.append(f"--- [Passage {idx} (ID: {item.chunk_id}, Doc: {doc_id})] ---\n{chunk_text}")

    context_str = "\n\n".join(evidence_blocks)
    user_prompt = (
        f"Reference Evidence Passages:\n"
        f"{context_str}\n\n"
        f"--------------------------------------------------\n"
        f"User Question: {clean_query}\n"
        f"Answer using ONLY the evidence passages above. If they do not contain the "
        f"answer, say so plainly instead of answering from your own knowledge:"
    )

    return sys_prompt, user_prompt


def extract_citations(retrieved_chunks: List[RetrievedChunk]) -> List[Citation]:
    """Extract structured Citation objects from retrieved chunks.

    Args:
        retrieved_chunks: List of retrieved evidence chunks

    Returns:
        List of Citation objects with chunk_id, document_id, score, and text
    """
    citations = []
    for item in retrieved_chunks:
        chunk_text = getattr(item.chunk, "chunk_text", "")
        doc_id = getattr(item.chunk, "document_id", "doc")
        citations.append(
            Citation(
                chunk_id=item.chunk_id,
                document_id=doc_id,
                score=round(float(item.score), 4),
                text=chunk_text,
            )
        )
    return citations
