"""Guardrail orchestration pipeline.

Provides a unified, thin orchestration layer for pre-retrieval input safety checking
and post-generation answer grounding verification.

Key Responsibilities:
- Runs InputGuardrail before vector retrieval to allow halting rejected inputs early.
- Runs GroundingVerifier post-generation to verify answer claims against retrieved Chunk evidence.
- Maintains complete compatibility with downstream RAG pipelines without introducing artificial
  LLM or retrieval dependencies.
"""

from __future__ import annotations

from typing import Optional

from app.chunking.models import Chunk
from app.guardrails.grounding_verifier import GroundingVerifier
from app.guardrails.input_guardrail import InputGuardrail
from app.guardrails.models import GuardrailResult, GuardrailVerdict


class GuardrailPipeline:
    """Unified orchestration interface for pre-retrieval input guardrails and post-generation grounding verification."""

    def __init__(
        self,
        input_guardrail: Optional[InputGuardrail] = None,
        grounding_verifier: Optional[GroundingVerifier] = None,
    ) -> None:
        """Initialize GuardrailPipeline with optional guardrail components.

        Args:
            input_guardrail: Optional InputGuardrail instance (creates default if None).
            grounding_verifier: Optional GroundingVerifier instance (creates default if None).
        """
        self.input_guardrail = input_guardrail or InputGuardrail()
        self.grounding_verifier = grounding_verifier or GroundingVerifier()

    def check_input(self, query: str) -> GuardrailResult:
        """Run pre-generation input guardrail before retrieval.

        Args:
            query: Raw user query string.

        Returns:
            GuardrailResult. If verdict is OFF_TOPIC_REJECTED, caller must halt
            retrieval and generation immediately. Verdict SAFE_AND_GROUNDED represents
            a pre-generation pass state (cleared for retrieval).
        """
        return self.input_guardrail.check(query)

    def verify_grounding(
        self,
        answer: str,
        retrieved_chunks: list[Chunk],
    ) -> GuardrailResult:
        """Run post-generation grounding verification against retrieved Chunk evidence.

        Args:
            answer: Generated answer string.
            retrieved_chunks: List of retrieved Chunk instances containing evidence text.

        Returns:
            GuardrailResult with SAFE_AND_GROUNDED if all claims are supported,
            or UNGROUNDED_FLAGGED with flagged_claims if any claim lacks evidence.
        """
        return self.grounding_verifier.verify(answer, retrieved_chunks)


__all__ = [
    "GuardrailPipeline",
]
