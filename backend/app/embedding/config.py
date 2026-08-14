"""Embedding configuration model.

Represents the configuration for an embedding provider. The production
provider/model are intentionally NOT chosen here (Phase 4.2); this model
simply describes the shape of the final configuration.

Phase 4.1: Configuration model only (no production model selection).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingProvider(str, Enum):
    """Enumeration of supported embedding providers.

    - FAKE: Deterministic offline embedder for tests (Phase 4.1)
    - HUGGINGFACE: HuggingFace / Sentence Transformers (planned Phase 4.2)
    - LOCAL: Another local embedding model (planned Phase 4.2)
    - API: API-based embeddings (planned Phase 4.2)
    """
    FAKE = "fake"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    API = "api"


class EmbeddingConfig(BaseModel):
    """Configuration for an embedding provider.

    Attributes:
        provider: Which provider implementation to use
        model_name: Name or local path of the embedding model (None until chosen)
        dimension: Expected vector dimension (optional until model is known)
        batch_size: Maximum number of texts per encode_batch() call
        device: Inference device - "auto", "cpu", or "cuda"/"cuda:N"
        normalize: Whether returned embeddings are L2-normalized
    """

    model_config = ConfigDict(protected_namespaces=())

    provider: EmbeddingProvider = Field(
        EmbeddingProvider.FAKE,
        description="Embedding provider implementation",
    )

    model_name: Optional[str] = Field(
        None,
        description="Model name or local path (unset until Phase 4.2 model selection)",
    )

    dimension: Optional[int] = Field(
        None,
        ge=1,
        description="Expected vector dimension (optional until model is known)",
    )

    batch_size: int = Field(
        32,
        ge=1,
        description="Maximum texts per encode_batch() call",
    )

    device: str = Field(
        "auto",
        description="Inference device: 'auto', 'cpu', or 'cuda'/'cuda:N'",
    )

    normalize: bool = Field(
        True,
        description="L2-normalize embeddings before returning",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: Optional[str]) -> Optional[str]:
        """Ensure model_name is not empty/whitespace when provided."""
        if v is not None and not v.strip():
            raise ValueError("model_name cannot be empty or whitespace-only")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Ensure device is a supported value."""
        if v not in ("auto", "cpu", "cuda") and not v.startswith("cuda:"):
            raise ValueError(
                f"device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {v!r}"
            )
        return v

    def __repr__(self) -> str:
        return (
            f"EmbeddingConfig(provider={self.provider.value}, "
            f"model_name={self.model_name!r}, "
            f"dimension={self.dimension}, "
            f"batch_size={self.batch_size}, "
            f"device={self.device!r}, "
            f"normalize={self.normalize})"
        )


__all__ = [
    "EmbeddingProvider",
    "EmbeddingConfig",
]