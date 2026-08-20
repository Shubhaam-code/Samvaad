"""Pydantic request/response schemas for the chat API.

These models define the public HTTP contract for ``POST /api/chat`` and
are exposed through FastAPI's OpenAPI schema (/docs).

Phase 6.2: Chat API contract (endpoint integration only).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.guardrails.models import GuardrailResult


class ChatRequest(BaseModel):
    """Request body for POST /api/chat.

    Attributes:
        query: User query to answer (must be a non-empty string)
    """

    query: str = Field(..., min_length=1, description="User query to answer")

    @field_validator("query")
    @classmethod
    def validate_non_empty_query(cls, v: str) -> str:
        """Ensure the query is not empty/whitespace-only."""
        if not v or not v.strip():
            raise ValueError("query cannot be empty or whitespace-only")
        return v


class Citation(BaseModel):
    """A single citation pointing at real retrieved Chunk evidence.

    Attributes:
        chunk_id: Chunk identifier of the evidence
        document_id: Source document identifier of the evidence
        score: Vector search similarity score for this evidence
        text: The actual chunk_text of the retrieved evidence
    """

    chunk_id: str = Field(..., min_length=1, description="Chunk identifier of the evidence")
    document_id: str = Field(..., min_length=1, description="Source document identifier")
    score: float = Field(..., description="Vector search similarity score")
    text: str = Field(..., min_length=1, description="Actual chunk_text of the retrieved evidence")


class LatencyBreakdown(BaseModel):
    """Real per-stage latencies for a single chat request, in milliseconds.

    Attributes:
        guardrail_ms: Input guardrail check latency
        retrieval_ms: Embedding + search + resolution latency
        llm_ms: LLM generation latency
        grounding_ms: Post-generation grounding verification latency
        total_ms: Total pipeline latency (first guardrail call to response build)
    """

    guardrail_ms: float = Field(0.0, ge=0.0, description="Input guardrail check latency (ms)")
    retrieval_ms: float = Field(0.0, ge=0.0, description="Retrieval latency (ms)")
    llm_ms: float = Field(0.0, ge=0.0, description="LLM generation latency (ms)")
    grounding_ms: float = Field(0.0, ge=0.0, description="Grounding verification latency (ms)")
    total_ms: float = Field(0.0, ge=0.0, description="Total pipeline latency (ms)")


class ChatResponse(BaseModel):
    """Response body for POST /api/chat.

    Attributes:
        answer: Generated answer text (empty for rejected queries)
        citations: Citations from actual retrieved Chunk evidence
        guardrail: Pre-retrieval input guardrail result
        grounding: Post-generation grounding verification result
        latency_breakdown: Real per-stage latencies in milliseconds
        model: Model identifier that produced the answer (if any)
    """

    answer: str = Field("", description="Generated answer text")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations from actual retrieved Chunk evidence",
    )
    guardrail: GuardrailResult = Field(..., description="Pre-retrieval input guardrail result")
    grounding: GuardrailResult = Field(..., description="Post-generation grounding verification result")
    latency_breakdown: LatencyBreakdown = Field(
        ...,
        description="Real per-stage latencies in milliseconds",
    )
    model: Optional[str] = Field(None, description="Model identifier that produced the answer")


class LatencyStats(BaseModel):
    """Aggregate latency statistics for a single pipeline stage.

    Attributes:
        request_count: Successful requests contributing to this stage
        sum_ms: Sum of stage latencies in milliseconds
        mean_ms: Mean stage latency (sum / count; 0.0 when count is 0)
        min_ms: Minimum stage latency in milliseconds
        max_ms: Maximum stage latency in milliseconds
        p50_ms: Median stage latency in milliseconds
        p70_ms: 70th percentile stage latency in milliseconds
        p100_ms: Maximum observed stage latency in milliseconds
    """

    request_count: int = Field(0, ge=0, description="Successful requests contributing to this stage")
    sum_ms: float = Field(0.0, ge=0.0, description="Sum of stage latencies (ms)")
    mean_ms: float = Field(0.0, ge=0.0, description="Mean stage latency (ms)")
    min_ms: float = Field(0.0, ge=0.0, description="Minimum stage latency (ms)")
    max_ms: float = Field(0.0, ge=0.0, description="Maximum stage latency (ms)")
    p50_ms: float = Field(0.0, ge=0.0, description="Median stage latency (ms)")
    p70_ms: float = Field(0.0, ge=0.0, description="70th percentile stage latency (ms)")
    p100_ms: float = Field(0.0, ge=0.0, description="Maximum observed stage latency (ms)")


class AnalyticsResponse(BaseModel):
    """Latency analytics for the chat pipeline.

    Only successful (200) completions contribute to the latency
    aggregates; guardrail rejections and errors are counted separately
    and never pollute the latency statistics.

    Attributes:
        request_count: Successful pipeline completions recorded
        rejected_count: Guardrail rejections (400 QUERY_REJECTED)
        error_count: Other non-successful outcomes (501/503/500)
        sub_200ms_achieved: True when total P50 latency is below 200 ms
        stt_ms: Aggregates for the STT transcription stage
        retrieval_ms: Aggregates for the retrieval stage
        llm_ms: Aggregates for the LLM generation stage
        tts_ms: Aggregates for the TTS synthesis stage
        guardrail_ms: Aggregates for the input guardrail stage
        grounding_ms: Aggregates for the grounding verification stage
        total_ms: Aggregates for the total pipeline
    """

    request_count: int = Field(0, ge=0, description="Successful pipeline completions recorded")
    rejected_count: int = Field(0, ge=0, description="Guardrail rejections (400 QUERY_REJECTED)")
    error_count: int = Field(0, ge=0, description="Other non-successful outcomes (501/503/500)")
    sub_200ms_achieved: bool = Field(
        False,
        description="True when total P50 latency is below the 200 ms interactive SLA",
    )
    stt_ms: LatencyStats = Field(default_factory=LatencyStats, description="STT transcription stage")
    retrieval_ms: LatencyStats = Field(default_factory=LatencyStats, description="Retrieval stage")
    llm_ms: LatencyStats = Field(default_factory=LatencyStats, description="LLM generation stage")
    tts_ms: LatencyStats = Field(default_factory=LatencyStats, description="TTS synthesis stage")
    guardrail_ms: LatencyStats = Field(default_factory=LatencyStats, description="Input guardrail stage")
    grounding_ms: LatencyStats = Field(default_factory=LatencyStats, description="Grounding verification stage")
    total_ms: LatencyStats = Field(default_factory=LatencyStats, description="Total pipeline")


class VoiceLatencyBreakdown(BaseModel):
    """Real per-stage latencies for a single voice query request, in milliseconds.

    Attributes:
        stt_ms: STT transcription latency (ms)
        guardrail_ms: Input guardrail check latency (ms)
        retrieval_ms: Embedding + search + resolution latency (ms)
        llm_ms: LLM generation latency (ms)
        grounding_ms: Grounding verification latency (ms)
        tts_ms: TTS synthesis latency (ms)
        total_pipeline_ms: Total pipeline latency (ms)
        total_ms: Alias for total_pipeline_ms for consistency with chat
    """

    stt_ms: float = Field(0.0, ge=0.0, description="STT transcription latency (ms)")
    guardrail_ms: float = Field(0.0, ge=0.0, description="Input guardrail check latency (ms)")
    retrieval_ms: float = Field(0.0, ge=0.0, description="Retrieval latency (ms)")
    llm_ms: float = Field(0.0, ge=0.0, description="LLM generation latency (ms)")
    grounding_ms: float = Field(0.0, ge=0.0, description="Grounding verification latency (ms)")
    tts_ms: float = Field(0.0, ge=0.0, description="TTS synthesis latency (ms)")
    total_pipeline_ms: float = Field(0.0, ge=0.0, description="Total pipeline latency (ms)")
    total_ms: float = Field(0.0, ge=0.0, description="Total pipeline latency alias (ms)")


class VoiceQueryResponse(BaseModel):
    """Response body for POST /api/voice-query.

    Attributes:
        transcribed_text: Transcribed query text from STT
        answer: Generated answer text
        audio_base64: Base64-encoded synthesized audio bytes
        audio_content_type: Canonical MIME type of synthesized audio
        audio_format: Format identifier of synthesized audio
        citations: Citations from actual retrieved Chunk evidence
        guardrail: Pre-retrieval input guardrail result
        grounding: Post-generation grounding verification result
        model: LLM model identifier that produced the answer
        stt_model: STT model identifier that transcribed the audio
        tts_model: TTS model identifier that synthesized the speech
        latency_breakdown: Real per-stage latencies in milliseconds
    """

    transcribed_text: str = Field(..., description="Transcribed query text from STT")
    answer: str = Field(..., description="Generated answer text")
    audio_base64: str = Field(..., description="Base64-encoded synthesized audio bytes")
    audio_content_type: str = Field(..., description="Canonical MIME type of synthesized audio")
    audio_format: str = Field(..., description="Format identifier of synthesized audio (e.g. 'mp3', 'wav')")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations from actual retrieved Chunk evidence",
    )
    guardrail: GuardrailResult = Field(..., description="Pre-retrieval input guardrail result")
    grounding: GuardrailResult = Field(..., description="Post-generation grounding verification result")
    model: Optional[str] = Field(None, description="LLM model identifier")
    stt_model: Optional[str] = Field(None, description="STT model identifier")
    tts_model: Optional[str] = Field(None, description="TTS model identifier")
    latency_breakdown: VoiceLatencyBreakdown = Field(
        ...,
        description="Real per-stage latencies in milliseconds",
    )


__all__ = [
    "ChatRequest",
    "Citation",
    "LatencyBreakdown",
    "ChatResponse",
    "LatencyStats",
    "AnalyticsResponse",
    "VoiceLatencyBreakdown",
    "VoiceQueryResponse",
]
