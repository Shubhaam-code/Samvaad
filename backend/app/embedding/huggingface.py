"""Production multilingual embedding adapter (transformers + torch).

Selected model (Phase 4.2): ``intfloat/multilingual-e5-small``

Why this model:
- 100+ languages supported, including Hindi (hi) and English
- Strong multilingual retrieval quality for its size (MTEB multilingual)
- Small footprint: ~118M parameters / 384-dim embeddings -> CPU-friendly
- XLM-RoBERTa-derived architecture, loads with plain transformers
  (no sentence-transformers dependency required)
- MIT license

Documented model-card behavior (implemented here, not invented):
- Input texts must be prefixed with "query: " or "passage: "
  (E5 training scheme; skipping prefixes degrades quality)
- Embedding = masked MEAN POOLING of the last hidden state
- Embeddings are L2-normalized

This adapter implements the Phase 4.1 Embedder interface:
    encode(text: str) -> list[float]
    encode_batch(texts: list[str]) -> list[list[float]]
with identical behavior for single and batch paths, preserving input
order, returning plain Python float lists, and exposing ``dimension``.

Device handling:
- "auto" (default): CUDA if available, else CPU (CPU is always safe)
- "cpu": always supported, never requires CUDA
- "cuda"/"cuda:N": requires an available CUDA device, else ValueError

Model loading:
- The model is loaded lazily on first use and cached in a module-level
  registry keyed by (model_name, device) so repeated constructions do
  not reload it.
- ``local_files_only=True`` by default: the adapter NEVER downloads the
  model by itself. Use scripts/test_production_embedding.py with
  --allow-download to fetch the model once, explicitly.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import (
    BaseEmbedder,
    validate_batch,
    validate_batch_size,
    validate_text,
)
from .types import EmbeddingBatch, EmbeddingVector

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_MAX_LENGTH = 512
DEFAULT_QUERY_PREFIX = "query: "
DEFAULT_PASSAGE_PREFIX = "passage: "

_VALID_DEVICES = ("auto", "cpu", "cuda")

# (model_name, device) -> (model, tokenizer)
_MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}


def is_model_cached(model_name: str = DEFAULT_MODEL_NAME) -> bool:
    """Check whether a model is fully present in the local HF cache.

    Never downloads anything: only inspects the local cache.

    Args:
        model_name: HuggingFace model identifier or local path

    Returns:
        True if the model snapshot is fully cached locally, else False
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    try:
        snapshot_download(model_name, local_files_only=True)
        return True
    except Exception:
        return False


class HuggingFaceEmbedder(BaseEmbedder):
    """Production embedding adapter wrapping a transformers model.

    Args:
        model_name: HuggingFace model id or local path (default e5-small)
        device: "auto" (CUDA if available else CPU), "cpu", or "cuda"/"cuda:N"
        batch_size: Maximum texts per encode_batch() call (>= 1)
        max_length: Maximum token length per text (truncation cap)
        normalize: L2-normalize embeddings (documented E5 behavior)
        passage_prefix: Prefix applied by encode()/encode_batch() (E5 scheme)
        query_prefix: Prefix applied by encode_query()/encode_query_batch()
        local_files_only: Never download; require local cache (default True)
        model: Pre-built model to inject (tests / advanced reuse)
        tokenizer: Pre-built tokenizer to inject (tests / advanced reuse)

    Raises:
        ValueError: If device, batch_size or max_length are invalid
        RuntimeError: On first encode() if the model cannot be loaded
                      (e.g. not cached and local_files_only=True)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = DEFAULT_MAX_LENGTH,
        normalize: bool = True,
        passage_prefix: str = DEFAULT_PASSAGE_PREFIX,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        local_files_only: bool = True,
        model: Optional[object] = None,
        tokenizer: Optional[object] = None,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be an integer >= 1, got {batch_size!r}")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
            raise ValueError(f"max_length must be an integer >= 1, got {max_length!r}")
        if device not in _VALID_DEVICES and not device.startswith("cuda:"):
            raise ValueError(
                f"Invalid device {device!r}; expected one of "
                f"'auto', 'cpu', 'cuda' or 'cuda:N'"
            )

        self._model_name = model_name
        self._requested_device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._normalize = normalize
        self._passage_prefix = passage_prefix
        self._query_prefix = query_prefix
        self._local_files_only = local_files_only

        self._model = model
        self._tokenizer = tokenizer
        self._device: Optional[str] = None

        if device == "auto":
            self._resolved_device = self._resolve_auto_device()
        elif device.startswith("cuda"):
            if not self._cuda_available():
                raise ValueError(
                    f"Device '{device}' requested but CUDA is not available; "
                    f"use device='cpu' or device='auto'"
                )
            self._resolved_device = device
        else:
            self._resolved_device = "cpu"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Configured model identifier."""
        return self._model_name

    @property
    def device(self) -> str:
        """Resolved inference device ('cpu', 'cuda', ...)."""
        return self._resolved_device

    @property
    def batch_size(self) -> int:
        """Maximum texts allowed per encode_batch() call."""
        return self._batch_size

    @property
    def normalize(self) -> bool:
        """Whether returned embeddings are L2-normalized."""
        return self._normalize

    @property
    def local_files_only(self) -> bool:
        """Whether loading is restricted to the local cache (no download)."""
        return self._local_files_only

    @property
    def dimension(self) -> int:
        """Embedding dimension reported by the loaded model."""
        self._ensure_loaded()
        return int(self._model.config.hidden_size)

    # ------------------------------------------------------------------
    # Embedder interface
    # ------------------------------------------------------------------

    def encode(self, text: str) -> EmbeddingVector:
        """Embed a single text (passage/document semantics).

        Applies the E5 passage prefix, so encode()/encode_batch() are
        intended for document (chunk) content. Query encoding is
        available via encode_query()/encode_query_batch().

        Args:
            text: Non-empty text to embed

        Returns:
            Vector of plain Python floats (L2-normalized if normalize=True)

        Raises:
            ValueError: If text is empty or whitespace-only
            RuntimeError: If the model cannot be loaded from the local cache
        """
        validate_text(text)
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> EmbeddingBatch:
        """Embed a batch of texts, preserving input order.

        Uses exactly the same pooling/normalization path as encode().

        Args:
            texts: Non-empty list of texts to embed

        Returns:
            List of vectors in the same order as the input

        Raises:
            ValueError: If the batch is empty, contains empty/whitespace
                        text, or exceeds the configured batch_size
            RuntimeError: If the model cannot be loaded from the local cache
        """
        validate_batch(texts)
        validate_batch_size(texts, self._batch_size)
        self._ensure_loaded()
        prefixed = [self._passage_prefix + text for text in texts]
        return self._encode_prefixed(prefixed)

    def encode_query(self, text: str) -> EmbeddingVector:
        """Embed a single query text (E5 'query: ' prefix).

        Intended for the future retrieval phase; keeps E5's documented
        query/passage asymmetry correct from the start.

        Args:
            text: Non-empty query text to embed

        Returns:
            Vector of plain Python floats

        Raises:
            ValueError: If text is empty or whitespace-only
        """
        validate_text(text)
        return self.encode_query_batch([text])[0]

    def encode_query_batch(self, texts: list[str]) -> EmbeddingBatch:
        """Embed a batch of query texts, preserving input order.

        Args:
            texts: Non-empty list of query texts

        Returns:
            List of vectors in the same order as the input
        """
        validate_batch(texts)
        validate_batch_size(texts, self._batch_size)
        self._ensure_loaded()
        prefixed = [self._query_prefix + text for text in texts]
        return self._encode_prefixed(prefixed)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode_prefixed(self, texts: list[str]) -> EmbeddingBatch:
        """Run the shared forward path (single and batch use this)."""
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("torch is required for HuggingFace embeddings") from e

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        attention_mask = inputs["attention_mask"]

        # Masked mean pooling over the last hidden state (E5 model-card spec)
        masked = last_hidden * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

        if self._normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        # Plain Python float lists (no tensors leak to callers)
        return [vector.tolist() for vector in pooled]

    def _ensure_loaded(self) -> None:
        """Load the model lazily, once, reusing the module-level cache."""
        if self._model is not None and self._tokenizer is not None:
            return

        cache_key = (self._model_name, self._resolved_device)
        if cache_key in _MODEL_CACHE:
            self._model, self._tokenizer = _MODEL_CACHE[cache_key]
            self._device = self._resolved_device
            return

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers and torch are required to load HuggingFace "
                "embedding models"
            ) from e

        logger.info(
            "Loading embedding model '%s' on device '%s' "
            "(local_files_only=%s, normalize=%s)",
            self._model_name,
            self._resolved_device,
            self._local_files_only,
            self._normalize,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                local_files_only=self._local_files_only,
            )
            model = AutoModel.from_pretrained(
                self._model_name,
                local_files_only=self._local_files_only,
            )
            model.to(self._resolved_device)
            model.eval()
        except Exception as e:
            hint = (
                f"Model '{self._model_name}' is not available in the local "
                f"HuggingFace cache and local_files_only={self._local_files_only}. "
                f"Download it once explicitly via: "
                f"python scripts/test_production_embedding.py --allow-download"
            )
            raise RuntimeError(
                f"Failed to load embedding model '{self._model_name}': {e}. {hint}"
            ) from e

        _MODEL_CACHE[cache_key] = (model, tokenizer)
        self._model = model
        self._tokenizer = tokenizer
        self._device = self._resolved_device

        logger.info(
            "Embedding model '%s' ready (dimension=%d, device='%s')",
            self._model_name,
            int(model.config.hidden_size),
            self._resolved_device,
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _resolve_auto_device() -> str:
        return "cuda" if HuggingFaceEmbedder._cuda_available() else "cpu"

    def __repr__(self) -> str:
        return (
            f"HuggingFaceEmbedder(model_name={self._model_name!r}, "
            f"device={self._resolved_device!r}, dimension={self.dimension if self._model is not None else 'lazy'}, "
            f"batch_size={self._batch_size}, normalize={self._normalize})"
        )


def create_huggingface_embedder(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    batch_size: int = 32,
    normalize: bool = True,
    local_files_only: bool = True,
) -> HuggingFaceEmbedder:
    """Create a HuggingFaceEmbedder with defaults for the selected model.

    Args:
        model_name: HuggingFace model id or local path
        device: "auto", "cpu", or "cuda"/"cuda:N"
        batch_size: Maximum texts per encode_batch() call
        normalize: L2-normalize embeddings (documented E5 behavior)
        local_files_only: Never download; require local cache (default True)

    Returns:
        A configured HuggingFaceEmbedder instance
    """
    return HuggingFaceEmbedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
        local_files_only=local_files_only,
    )


def is_model_cached(model_name: str = DEFAULT_MODEL_NAME) -> bool:
    """Check whether a HuggingFace model is already cached locally.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        True if the model is cached locally; False otherwise.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        AutoModel.from_pretrained(model_name, local_files_only=True)
        return True
    except Exception:
        return False


__all__ = [
    "DEFAULT_MODEL_NAME",
    "HuggingFaceEmbedder",
    "create_huggingface_embedder",
    "is_model_cached",
]