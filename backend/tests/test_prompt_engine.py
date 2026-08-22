"""Unit tests for Grounded Prompt Engine & Citation Extractor (Phase 5.4)."""

from app.api.schemas import Citation
from app.chunking.models import Chunk, ChunkingStrategy
import pytest

from app.llm.prompt_engine import (
    CONVERSATIONAL_SYSTEM_PROMPT,
    DEFAULT_GROUNDED_SYSTEM_PROMPT,
    build_conversational_prompt,
    build_grounded_rag_prompt,
    extract_citations,
    is_conversational,
)
from app.retrieval.models import RetrievedChunk


def _make_retrieved_chunk(chunk_id: str, doc_id: str, text: str, score: float = 0.95) -> RetrievedChunk:
    chunk = Chunk.from_passage_segment(
        document_id=doc_id,
        chunk_index=0,
        strategy=ChunkingStrategy.PASSAGE,
        chunk_text=text,
        query_id=1,
        passage_index=0,
        target_lang="en",
        source_lang="en",
        query="test query",
        eng_query="test query",
        query_type="general",
        answer=None,
        eng_answer=None,
        is_selected=False,
    )
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        position=0,
        chunk=chunk,
    )


def test_build_grounded_rag_prompt_empty_chunks():
    sys_prompt, user_prompt = build_grounded_rag_prompt(
        query="What is the capital of Goa?",
        retrieved_chunks=[],
    )
    assert sys_prompt == DEFAULT_GROUNDED_SYSTEM_PROMPT
    assert "Question: What is the capital of Goa?" in user_prompt
    assert "No background evidence available" in user_prompt


def test_build_grounded_rag_prompt_with_chunks():
    chunks = [
        _make_retrieved_chunk("c1", "doc_101", "Panaji is the state capital of Goa, India."),
        _make_retrieved_chunk("c2", "doc_102", "Goa is a state on the southwestern coast of India."),
    ]

    sys_prompt, user_prompt = build_grounded_rag_prompt(
        query="What is Goa's capital?",
        retrieved_chunks=chunks,
    )

    assert sys_prompt == DEFAULT_GROUNDED_SYSTEM_PROMPT
    assert "Panaji is the state capital of Goa" in user_prompt
    assert "Passage 1 (ID: c1, Doc: doc_101)" in user_prompt
    assert "Passage 2 (ID: c2, Doc: doc_102)" in user_prompt
    assert "User Question: What is Goa's capital?" in user_prompt


def test_build_grounded_rag_prompt_custom_system_prompt():
    custom_sys = "Custom strictly grounded instructions."
    sys_prompt, user_prompt = build_grounded_rag_prompt(
        query="Test query",
        retrieved_chunks=[],
        system_prompt=custom_sys,
    )
    assert sys_prompt == custom_sys


def test_extract_citations():
    chunks = [
        _make_retrieved_chunk("c1", "doc_101", "Evidence text 1", score=0.92345),
        _make_retrieved_chunk("c2", "doc_102", "Evidence text 2", score=0.81234),
    ]

    citations = extract_citations(chunks)

    assert len(citations) == 2
    assert isinstance(citations[0], Citation)
    assert citations[0].chunk_id == "c1"
    assert citations[0].document_id == "doc_101"
    assert citations[0].text == "Evidence text 1"
    assert citations[0].score == round(0.92345, 4)

    assert citations[1].chunk_id == "c2"
    assert citations[1].document_id == "doc_102"
    assert citations[1].text == "Evidence text 2"
    assert citations[1].score == round(0.81234, 4)


# --- Conversational bypass -------------------------------------------------
#
# Greetings must bypass retrieval + grounding, but anything carrying real
# informational content must NOT, or the ungrounded-answer guardrail would be
# trivially sidesteppable by prefixing a question with "hello".


@pytest.mark.parametrize(
    "query",
    [
        "hi",
        "Hello",
        "hey!",
        "  hello  ",
        "good morning",
        "how are you?",
        "thanks",
        "thank you",
        "who are you?",
        "hi there",
        "thanks a lot",
        "hello samvaad",
        "namaste",
        "नमस्ते",
        "नमस्ते।",
        "धन्यवाद",
        "आप कौन हैं?",
    ],
)
def test_is_conversational_detects_greetings(query):
    assert is_conversational(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What is the capital of Goa?",
        "hello, what is the capital of Goa?",
        "hi, tell me about the Mandovi river",
        "Goa की राजधानी क्या है?",
        "thanks, now explain quantum tunnelling",
        "who are you looking for in the 1961 census",
        "",
        "   ",
    ],
)
def test_is_conversational_rejects_real_questions(query):
    assert is_conversational(query) is False


def test_build_conversational_prompt_uses_conversational_system_prompt():
    sys_prompt, user_prompt = build_conversational_prompt("  hello  ")

    assert sys_prompt == CONVERSATIONAL_SYSTEM_PROMPT
    assert "hello" in user_prompt


def test_grounded_prompt_forbids_background_knowledge():
    """The strict grounding contract is a graded requirement; pin its wording."""
    assert "STRICTLY GROUNDED" in DEFAULT_GROUNDED_SYSTEM_PROMPT
    assert "Do NOT fall back on your own background knowledge" in DEFAULT_GROUNDED_SYSTEM_PROMPT
