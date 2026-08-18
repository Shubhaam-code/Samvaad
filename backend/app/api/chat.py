"""POST /api/chat endpoint.

Executes the full grounded-answer pipeline:

    query
      -> input guardrail        (FIRST - rejection is cheap and immediate)
      -> LLM provider check     (501 if no real provider configured)
      -> vector index check     (503 if no index available)
      -> retrieval              (guardrail -> embed -> search -> resolve)
      -> LLM generation         (real provider only)
      -> grounding verification (post-generation)
      -> final response

Guarantees:

- OFF_TOPIC_REJECTED returns HTTP 400 BEFORE any LLM configuration or
  index availability check; embedding, vector search, LLM generation,
  and grounding are NEVER called for rejected input.
- 501 is returned only for a SAFE query with no real LLM provider.
- 503 is returned only for a safe query whose retrieval index is missing.
- Citations come only from actual retrieved Chunk evidence.
- GroundingVerifier runs after LLM generation and its verdict
  (SAFE_AND_GROUNDED / UNGROUNDED_FLAGGED) is always surfaced.
- Real per-stage latencies are measured and returned.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from app.analytics import record_error, record_rejected, record_success
from app.api.dependencies import (
    get_grounding_verifier,
    get_guardrail_pipeline,
    get_llm,
    get_orchestrator,
)
from app.api.schemas import ChatRequest, ChatResponse, Citation, LatencyBreakdown
from app.guardrails.grounding_verifier import GroundingVerifier
from app.guardrails.models import GuardrailResult, GuardrailVerdict
from app.guardrails.pipeline import GuardrailPipeline
from app.llm.models import LLMRequest
from app.retrieval.orchestrator import RetrievalOrchestrator

router = APIRouter(prefix="/api", tags=["chat"])

# System prompt contract for the (future) real LLM provider: answers must
# be grounded exclusively in the retrieved evidence passed in context.
SYSTEM_PROMPT = (
    "You are a grounded question-answering assistant for the HH Goa RAG "
    "system. Answer the user's question using ONLY the retrieved evidence "
    "chunks provided. Do not invent facts, and do not cite anything that "
    "is not present in the provided evidence."
)

_MS_ROUND = 4


def _ms(start: float) -> float:
    """Elapsed time since start in milliseconds, rounded."""
    return round((time.perf_counter() - start) * 1000.0, _MS_ROUND)


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    guardrail_pipeline: GuardrailPipeline = Depends(get_guardrail_pipeline),
    grounding_verifier: GroundingVerifier = Depends(get_grounding_verifier),
    llm: object = Depends(get_llm),
    orchestrator: RetrievalOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """Answer a query with a grounded, cited response.

    - Rejected queries return HTTP 400 (QUERY_REJECTED) before any
      expensive or downstream work.
    - Safe queries without a real LLM provider return HTTP 501.
    - Safe queries without a built vector index return HTTP 503.
    """
    query = request.query
    t_total = time.perf_counter()

    # Stage 1: input guardrail - FIRST, before everything else
    t_stage = time.perf_counter()
    guardrail_result = guardrail_pipeline.check_input(query)
    guardrail_ms = _ms(t_stage)

    if guardrail_result.verdict == GuardrailVerdict.OFF_TOPIC_REJECTED:
        # Rejected input: embedding, vector search, LLM generation, and
        # grounding are NEVER called - even when no LLM/index is configured.
        record_rejected()
        raise HTTPException(
            status_code=400,
            detail={
                "code": "QUERY_REJECTED",
                "verdict": guardrail_result.verdict.value,
                "reason": guardrail_result.reason,
                "flagged_claims": guardrail_result.flagged_claims,
                "latency_ms": {"guardrail_ms": guardrail_ms},
            },
        )

    # Stage 2: real LLM provider availability (safe queries only)
    if llm is None:
        record_error()
        raise HTTPException(
            status_code=501,
            detail={
                "code": "LLM_PROVIDER_NOT_CONFIGURED",
                "message": "No real LLM provider is configured for this deployment.",
            },
        )

    # Stage 3: vector index availability (safe queries only)
    if orchestrator is None:
        record_error()
        raise HTTPException(
            status_code=503,
            detail={
                "code": "INDEX_NOT_AVAILABLE",
                "message": "No vector index is configured or built for this deployment.",
            },
        )

    # Stage 4: retrieval (guardrail -> embed -> search -> resolve)
    retrieval_result = orchestrator.retrieve(query)
    retrieval_ms = round(
        retrieval_result.latencies_ms.get("embedding_ms", 0.0)
        + retrieval_result.latencies_ms.get("search_ms", 0.0)
        + retrieval_result.latencies_ms.get("resolution_ms", 0.0),
        _MS_ROUND,
    )

    # Stage 5: LLM generation (real provider only)
    t_stage = time.perf_counter()
    llm_response = llm.generate(LLMRequest(prompt=query, system_prompt=SYSTEM_PROMPT))
    llm_ms = _ms(t_stage)

    # Stage 6: post-generation grounding verification
    t_stage = time.perf_counter()
    evidence_chunks = [item.chunk for item in retrieval_result.retrieved_chunks]
    grounding_result = grounding_verifier.verify(llm_response.text, evidence_chunks)
    grounding_ms = _ms(t_stage)

    # Stage 7: citations from actual retrieved Chunk evidence
    citations = [
        Citation(
            chunk_id=item.chunk_id,
            document_id=item.chunk.document_id,
            score=item.score,
            text=item.chunk.chunk_text,
        )
        for item in retrieval_result.retrieved_chunks
    ]

    total_ms = _ms(t_total)

    record_success(
        {
            "guardrail_ms": guardrail_ms,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "grounding_ms": grounding_ms,
            "total_ms": total_ms,
        }
    )

    return ChatResponse(
        answer=llm_response.text,
        citations=citations,
        guardrail=guardrail_result,
        grounding=grounding_result,
        latency_breakdown=LatencyBreakdown(
            guardrail_ms=guardrail_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            grounding_ms=grounding_ms,
            total_ms=total_ms,
        ),
        model=llm_response.model,
    )


__all__ = [
    "router",
    "SYSTEM_PROMPT",
]
