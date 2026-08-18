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
    "If the provided evidence does not contain the information needed to answer the question, "
    "state clearly and concisely that the provided context does not contain the answer. "
    "Always respond in the same language as the user's question (e.g., reply in Hindi if the question is in Hindi, and English if in English). "
    "Keep answers concise, direct, and factual."
)


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

    if not retrieved_chunks:
        user_prompt = f"Question: {query.strip()}\n\n(No background evidence available. State that no context was found.)"
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
        f"User Question: {query.strip()}\n"
        f"Answer the question concisely and accurately based solely on the passages above:"
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
