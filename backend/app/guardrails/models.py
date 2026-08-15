"""Data models for the guardrail layer.

Defines the canonical guardrail verdicts and result structure used by
pre-generation input checks and post-generation grounding verification.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GuardrailVerdict(str, Enum):
    """Possible outcomes produced by the guardrail layer."""

    SAFE_AND_GROUNDED = "SAFE_AND_GROUNDED"
    OFF_TOPIC_REJECTED = "OFF_TOPIC_REJECTED"
    UNGROUNDED_FLAGGED = "UNGROUNDED_FLAGGED"


class GuardrailResult(BaseModel):
    """Result returned by a guardrail check.

    Attributes:
        verdict: Final guardrail decision.
        reason: Human-readable explanation for the decision.
        score: Optional confidence/similarity score associated with the check.
        flagged_claims: Claims that could not be sufficiently supported by
            the retrieved evidence.
    """

    model_config = ConfigDict(protected_namespaces=())

    verdict: GuardrailVerdict
    reason: str = Field(..., min_length=1)
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    flagged_claims: list[str] = Field(default_factory=list)


__all__ = [
    "GuardrailVerdict",
    "GuardrailResult",
]