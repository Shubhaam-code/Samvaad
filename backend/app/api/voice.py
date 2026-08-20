"""POST /api/voice-query endpoint.

Executes the full voice-enabled grounded-answer pipeline:

    Audio upload
      -> Upload validation (bounds, MIME, container magic-bytes)
      -> STT transcription (get_stt)
      -> Input guardrail (check_input — short-circuits on OFF_TOPIC_REJECTED)
      -> LLM provider check (501 if no real LLM provider)
      -> Vector index check (503 if no index built)
      -> Retrieval (orchestrator.retrieve)
      -> LLM generation (llm.generate)
      -> Grounding verification (grounding_verifier.verify)
      -> TTS synthesis (get_tts — only for SAFE_AND_GROUNDED answers)
      -> Audio + grounded answer response

Guarantees:
- Invalid/empty/oversized audio is rejected with HTTP 400 BEFORE provider calls.
- OFF_TOPIC_REJECTED returns HTTP 400 immediately; Retrieval, LLM, Grounding,
  and TTS are NEVER executed for off-topic input.
- UNGROUNDED_FLAGGED answers return HTTP 422 immediately; TTS is NEVER called
  to speak an ungrounded or fabricated answer.
- Missing STT / LLM / TTS providers return HTTP 501.
- Missing vector index returns HTTP 503.
- Citations come only from actual retrieved Chunk evidence.
- Transient in-memory audio: audio bytes are never written to disk or logged.
- Real per-stage latencies are measured and surfaced.
"""

from __future__ import annotations

import base64
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.chat import SYSTEM_PROMPT
from app.api.dependencies import (
    get_grounding_verifier,
    get_guardrail_pipeline,
    get_llm,
    get_orchestrator,
    get_stt,
    get_tts,
)
from app.analytics import record_error, record_rejected, record_success
from app.api.schemas import Citation, VoiceLatencyBreakdown, VoiceQueryResponse
from app.guardrails.grounding_verifier import GroundingVerifier
from app.guardrails.models import GuardrailVerdict
from app.guardrails.pipeline import GuardrailPipeline
from app.llm.models import LLMRequest
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.settings import settings
from app.stt.models import STTRequest
from app.stt.validation import validate_audio
from app.tts.models import TTSRequest

router = APIRouter(prefix="/api", tags=["voice"])

_MS_ROUND = 4


def _ms(start: float) -> float:
    """Elapsed time since start in milliseconds, rounded."""
    return round((time.perf_counter() - start) * 1000.0, _MS_ROUND)


@router.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(
    audio: UploadFile = File(..., description="Uploaded audio file (wav, mp3, ogg, webm, m4a, aac)"),
    language: Optional[str] = Form(None, description="Optional language hint (e.g. 'en', 'hi')"),
    voice: Optional[str] = Form(None, description="Optional TTS voice override (e.g. 'alloy', 'nova')"),
    speed: Optional[float] = Form(None, description="Optional TTS speech speed multiplier (0.25 to 4.0)"),
    stt: object = Depends(get_stt),
    guardrail_pipeline: GuardrailPipeline = Depends(get_guardrail_pipeline),
    orchestrator: RetrievalOrchestrator = Depends(get_orchestrator),
    llm: object = Depends(get_llm),
    grounding_verifier: GroundingVerifier = Depends(get_grounding_verifier),
    tts: object = Depends(get_tts),
) -> VoiceQueryResponse:
    """Process a voice question through the full grounded voice RAG pipeline.

    Flow: Audio -> STT -> Guardrail -> Retrieval -> LLM -> Grounding -> TTS -> Response.
    """
    t_total = time.perf_counter()

    # Stage 0: Audio upload validation (size, MIME, container sniffing)
    filename = audio.filename or "recording.wav"
    try:
        audio_bytes = await audio.read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_AUDIO",
                "message": f"Failed to read uploaded audio: {exc}",
            },
        ) from exc

    max_bytes = int(settings.stt_max_audio_size_mb * 1024 * 1024)
    try:
        validated_audio = validate_audio(
            audio_bytes,
            filename=filename,
            content_type=audio.content_type,
            max_bytes=max_bytes,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_AUDIO",
                "message": str(exc),
            },
        ) from exc

    # Stage 1: STT Provider Check
    if stt is None:
        record_error()
        raise HTTPException(
            status_code=501,
            detail={
                "code": "STT_PROVIDER_NOT_CONFIGURED",
                "message": "No real STT provider is configured for this deployment.",
            },
        )

    # Stage 2: STT Transcription
    t_stage = time.perf_counter()
    stt_req = STTRequest(
        audio=audio_bytes,
        filename=filename,
        content_type=validated_audio.content_type,
        language=language,
    )
    stt_response = stt.transcribe(stt_req)  # type: ignore[union-attr]
    stt_ms = _ms(t_stage)
    transcribed_text = stt_response.text

    # Stage 3: Input Guardrail (Safety Check) — BEFORE retrieval, LLM, grounding, TTS
    t_stage = time.perf_counter()
    guardrail_result = guardrail_pipeline.check_input(transcribed_text)
    guardrail_ms = _ms(t_stage)

    if guardrail_result.verdict == GuardrailVerdict.OFF_TOPIC_REJECTED:
        record_rejected()
        raise HTTPException(
            status_code=400,
            detail={
                "code": "QUERY_REJECTED",
                "transcribed_text": transcribed_text,
                "verdict": guardrail_result.verdict.value,
                "reason": guardrail_result.reason,
                "flagged_claims": guardrail_result.flagged_claims,
                "latency_ms": {
                    "stt_ms": stt_ms,
                    "guardrail_ms": guardrail_ms,
                },
            },
        )

    # Stage 4: LLM Provider Check
    if llm is None:
        record_error()
        raise HTTPException(
            status_code=501,
            detail={
                "code": "LLM_PROVIDER_NOT_CONFIGURED",
                "message": "No real LLM provider is configured for this deployment.",
            },
        )

    # Stage 5: Vector Index Check
    if orchestrator is None:
        record_error()
        raise HTTPException(
            status_code=503,
            detail={
                "code": "INDEX_NOT_AVAILABLE",
                "message": "No vector index is configured or built for this deployment.",
            },
        )

    # Stage 6: Retrieval
    retrieval_result = orchestrator.retrieve(transcribed_text)
    retrieval_ms = round(
        retrieval_result.latencies_ms.get("embedding_ms", 0.0)
        + retrieval_result.latencies_ms.get("search_ms", 0.0)
        + retrieval_result.latencies_ms.get("resolution_ms", 0.0),
        _MS_ROUND,
    )

    citations = [
        Citation(
            chunk_id=item.chunk_id,
            document_id=item.chunk.document_id,
            score=item.score,
            text=item.chunk.chunk_text,
        )
        for item in retrieval_result.retrieved_chunks
    ]

    # Stage 7: LLM Generation
    t_stage = time.perf_counter()
    llm_response = llm.generate(LLMRequest(prompt=transcribed_text, system_prompt=SYSTEM_PROMPT))  # type: ignore[union-attr]
    llm_ms = _ms(t_stage)
    answer_text = llm_response.text

    # Stage 8: Post-generation Grounding Verification
    t_stage = time.perf_counter()
    evidence_chunks = [item.chunk for item in retrieval_result.retrieved_chunks]
    grounding_result = grounding_verifier.verify(answer_text, evidence_chunks)
    grounding_ms = _ms(t_stage)

    if grounding_result.verdict == GuardrailVerdict.UNGROUNDED_FLAGGED:
        total_ms = _ms(t_total)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "UNGROUNDED_ANSWER",
                "transcribed_text": transcribed_text,
                "answer": answer_text,
                "verdict": grounding_result.verdict.value,
                "reason": grounding_result.reason,
                "flagged_claims": grounding_result.flagged_claims,
                "citations": [c.model_dump() for c in citations],
                "latency_ms": {
                    "stt_ms": stt_ms,
                    "guardrail_ms": guardrail_ms,
                    "retrieval_ms": retrieval_ms,
                    "llm_ms": llm_ms,
                    "grounding_ms": grounding_ms,
                    "total_ms": total_ms,
                },
            },
        )

    # Stage 9: TTS Provider Check
    if tts is None:
        record_error()
        raise HTTPException(
            status_code=501,
            detail={
                "code": "TTS_PROVIDER_NOT_CONFIGURED",
                "message": "No real TTS provider is configured for this deployment.",
            },
        )

    # Stage 10: TTS Synthesis (Grounding passed)
    t_stage = time.perf_counter()
    tts_req = TTSRequest(
        text=answer_text,
        voice=voice or settings.tts_voice,
        speed=speed if speed is not None else settings.tts_speed,
        output_format=settings.tts_output_format,
        language=language,
    )
    tts_response = tts.synthesize(tts_req)  # type: ignore[union-attr]
    tts_ms = _ms(t_stage)

    audio_base64 = base64.b64encode(tts_response.audio).decode("ascii")
    total_ms = _ms(t_total)

    record_success(
        {
            "stt_ms": stt_ms,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "guardrail_ms": guardrail_ms,
            "grounding_ms": grounding_ms,
            "total_ms": total_ms,
        }
    )

    return VoiceQueryResponse(
        transcribed_text=transcribed_text,
        answer=answer_text,
        audio_base64=audio_base64,
        audio_content_type=tts_response.content_type,
        audio_format=tts_response.format,
        citations=citations,
        guardrail=guardrail_result,
        grounding=grounding_result,
        model=llm_response.model,
        stt_model=stt_response.model,
        tts_model=tts_response.model,
        latency_breakdown=VoiceLatencyBreakdown(
            stt_ms=stt_ms,
            guardrail_ms=guardrail_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            grounding_ms=grounding_ms,
            tts_ms=tts_ms,
            total_pipeline_ms=total_ms,
            total_ms=total_ms,
        ),
    )


__all__ = [
    "router",
]
