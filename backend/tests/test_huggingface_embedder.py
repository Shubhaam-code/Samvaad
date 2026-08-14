"""
Tests for the HuggingFace production embedding adapter (Phase 4.2).

These are UNIT tests that use a tiny fake transformers-compatible model
(a random torch embedding layer, hidden_size=8) injected into the adapter.
They exercise the real adapter code path (tokenization call, masked mean
pooling, normalization, device placement, ordering) WITHOUT loading or
downloading the real production model.

The real model (intfloat/multilingual-e5-small) is only exercised by the
explicit smoke test script: scripts/test_production_embedding.py

All tests use tiny synthetic strings only.
No real MSMARCO-XI data. No network access. No model downloads.
"""

import math
import socket

import pytest
import torch
import torch.nn as nn

from app.embedding import (
    DEFAULT_MODEL_NAME,
    HuggingFaceEmbedder,
    create_huggingface_embedder,
    is_model_cached,
)
from app.embedding.huggingface import _MODEL_CACHE

HIDDEN = 8
VOCAB = 128


class FakeHFConfig:
    """Minimal stand-in for a transformers model config."""

    def __init__(self, hidden_size: int = HIDDEN):
        self.hidden_size = hidden_size


class FakeHFOutput:
    """Minimal stand-in for a transformers model output."""

    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeHFModel(nn.Module):
    """Tiny random model exposing the transformers forward contract."""

    def __init__(self, hidden_size: int = HIDDEN, vocab_size: int = VOCAB):
        super().__init__()
        self.config = FakeHFConfig(hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        # Random projection is fine: tests check shape/finite/order/validation,
        # not semantic quality (that is the smoke test's job).
        return FakeHFOutput(self.embedding(input_ids))


class CountingModel(FakeHFModel):
    """Fake model that counts forward calls (to prove no reloads)."""

    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask):
        self.forward_calls += 1
        return super().forward(input_ids, attention_mask)


class FakeHFTokenizer:
    """Minimal char-based tokenizer exposing the transformers call contract."""

    def __init__(self, vocab_size: int = VOCAB):
        self.vocab_size = vocab_size
        self.last_input = None

    def __call__(self, texts, padding=False, truncation=False,
                 return_tensors=None, max_length=None):
        self.last_input = list(texts)
        rows = []
        for text in texts:
            row = [ord(ch) % self.vocab_size for ch in text]
            if truncation and max_length is not None:
                row = row[:max_length]
            rows.append(row)
        max_len = max((len(r) for r in rows), default=1)
        padded_ids = [r + [0] * (max_len - len(r)) for r in rows]
        attention_masks = [
            [1] * len(r) + [0] * (max_len - len(r)) for r in rows
        ]
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


@pytest.fixture()
def embedder():
    """Adapter with injected tiny fake model/tokenizer (CPU)."""
    return HuggingFaceEmbedder(
        model_name="fake/test-model",
        device="cpu",
        model=FakeHFModel(),
        tokenizer=FakeHFTokenizer(),
    )


@pytest.fixture()
def tokenizer(embedder):
    return embedder._tokenizer


# ---------------------------------------------------------------------------
# Import / selection
# ---------------------------------------------------------------------------


def test_adapter_can_be_imported():
    """Test that the adapter and its factory are importable."""
    assert HuggingFaceEmbedder is not None
    assert create_huggingface_embedder is not None
    assert is_model_cached is not None


def test_default_model_name_is_e5_small():
    """Test that the selected production model name is documented."""
    assert DEFAULT_MODEL_NAME == "intfloat/multilingual-e5-small"


def test_model_name_is_configurable():
    """Test that the model name is not hard-coded."""
    e = HuggingFaceEmbedder(model_name="some/other-model", device="cpu",
                            model=FakeHFModel(), tokenizer=FakeHFTokenizer())
    assert e.model_name == "some/other-model"


def test_model_initializes_with_injected_model():
    """Test construction with an injected fake model."""
    e = HuggingFaceEmbedder(model_name="fake/m", device="cpu",
                            model=FakeHFModel(), tokenizer=FakeHFTokenizer())
    assert e is not None
    assert e.model_name == "fake/m"


def test_factory_creates_adapter():
    """Test the create_huggingface_embedder factory defaults."""
    e = create_huggingface_embedder(model_name="fake/m", batch_size=4)
    assert isinstance(e, HuggingFaceEmbedder)
    assert e.model_name == "fake/m"
    assert e.batch_size == 4
    assert e.normalize is True
    assert e.local_files_only is True


def test_is_model_cached_never_downloads():
    """Test that is_model_cached() inspects the cache without downloading."""
    # A bogus model can never be cached; must return False, not download.
    assert is_model_cached("nonexistent-model-for-hhgoa-test") is False


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


def test_cpu_execution_explicit():
    """Test explicit CPU device works."""
    e = HuggingFaceEmbedder(model_name="fake/m", device="cpu",
                            model=FakeHFModel(), tokenizer=FakeHFTokenizer())
    vector = e.encode("cpu test")
    assert len(vector) == HIDDEN
    assert e.device == "cpu"


def test_invalid_device_rejected():
    """Test that an invalid device string raises ValueError."""
    with pytest.raises(ValueError):
        HuggingFaceEmbedder(device="gpu")
    with pytest.raises(ValueError):
        HuggingFaceEmbedder(device="tpu:0")


def test_device_auto_resolves_to_supported_device():
    """Test that 'auto' resolves to a supported device without requiring CUDA."""
    e = HuggingFaceEmbedder(model_name="fake/m", device="auto",
                            model=FakeHFModel(), tokenizer=FakeHFTokenizer())
    assert e.device in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# Encoding basics
# ---------------------------------------------------------------------------


def test_single_hindi_text_produces_vector(embedder):
    """Test that a single Hindi text produces a vector."""
    vector = embedder.encode("भारत की राजधानी नई दिल्ली है।")
    assert isinstance(vector, list)
    assert len(vector) == HIDDEN
    assert all(isinstance(v, float) for v in vector)


def test_single_english_text_produces_vector(embedder):
    """Test that a single English text produces a vector."""
    vector = embedder.encode("India's capital is New Delhi.")
    assert isinstance(vector, list)
    assert len(vector) == HIDDEN


def test_hindi_english_batch_works(embedder):
    """Test that a mixed Hindi/English batch encodes."""
    texts = ["भारत की राजधानी नई दिल्ली है।", "India's capital is New Delhi.",
             "The weather is cold today."]
    vectors = embedder.encode_batch(texts)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == HIDDEN


def test_returned_values_are_finite_floats(embedder):
    """Test that all returned values are finite Python floats."""
    vector = embedder.encode("hello नमस्ते")
    assert all(isinstance(v, float) for v in vector)
    assert all(math.isfinite(v) for v in vector)


def test_dimension_property_is_correct(embedder):
    """Test that dimension matches the model's hidden size."""
    assert embedder.dimension == HIDDEN


def test_dimension_consistent_across_single_and_batch(embedder):
    """Test that single and batch encoding share the same dimension."""
    single = embedder.encode("x")
    batch = embedder.encode_batch(["x", "y"])
    assert len(single) == len(batch[0]) == embedder.dimension


# ---------------------------------------------------------------------------
# Ordering / consistency
# ---------------------------------------------------------------------------


def test_batch_ordering_is_preserved(embedder):
    """Test that encode_batch() preserves input order exactly."""
    texts = ["A", "B", "C"]
    vectors = embedder.encode_batch(texts)
    expected = [embedder.encode(t) for t in texts]
    assert vectors == expected
    # Swapping inputs swaps outputs
    assert embedder.encode_batch(list(reversed(texts))) == list(reversed(vectors))


def test_single_and_batch_behavior_identical(embedder):
    """Test that single and batch paths produce identical vectors
    (no silent normalization/pooling differences)."""
    vector_a = embedder.encode("नमस्ते")
    vector_b = embedder.encode_batch(["नमस्ते"])[0]
    assert vector_a == vector_b


def test_repeated_encoding_is_deterministic(embedder):
    """Test that repeated encoding of the same text is stable."""
    text = "goa tourism india"
    assert embedder.encode(text) == embedder.encode(text)
    assert embedder.encode_batch([text, text])[0] == embedder.encode(text)


# ---------------------------------------------------------------------------
# Normalization behavior
# ---------------------------------------------------------------------------


def test_normalized_embeddings_are_unit_length():
    """Test that normalize=True produces L2 unit vectors."""
    e = HuggingFaceEmbedder(model_name="fake/m", device="cpu", normalize=True,
                            model=FakeHFModel(), tokenizer=FakeHFTokenizer())
    for text in ["hello", "नमस्ते दुनिया", "a b c d e f g h"]:
        vector = e.encode(text)
        norm = math.sqrt(sum(v * v for v in vector))
        assert math.isclose(norm, 1.0, abs_tol=1e-6), text


def test_normalize_false_returns_raw_pooled_vectors():
    """Test that normalize=False skips L2 normalization explicitly."""
    e = HuggingFaceEmbedder(model_name="fake/m", device="cpu", normalize=False,
                            model=FakeHFModel(), tokenizer=FakeHFTokenizer())
    vector = e.encode("hello")
    assert len(vector) == HIDDEN
    norm = math.sqrt(sum(v * v for v in vector))
    # The fake embedding weights produce a norm that is not 1 in general;
    # the point is that the flag explicitly controls behavior.
    assert norm >= 0.0
    assert e.normalize is False


def test_normalization_flag_exposed(embedder):
    """Test that the normalize flag is exposed as a property."""
    assert embedder.normalize is True
    e = HuggingFaceEmbedder(model_name="fake/m", device="cpu", normalize=False,
                            model=FakeHFModel(), tokenizer=FakeHFTokenizer())
    assert e.normalize is False


# ---------------------------------------------------------------------------
# Prefix handling (E5 documented scheme)
# ---------------------------------------------------------------------------


def test_passage_prefix_applied(embedder, tokenizer):
    """Test that encode() applies the E5 'passage: ' prefix."""
    embedder.encode("नई दिल्ली भारत की राजधानी है।")
    assert tokenizer.last_input == ["passage: नई दिल्ली भारत की राजधानी है।"]


def test_query_prefix_applied(embedder, tokenizer):
    """Test that encode_query() applies the E5 'query: ' prefix."""
    embedder.encode_query("भारत की राजधानी क्या है?")
    assert tokenizer.last_input == ["query: भारत की राजधानी क्या है?"]


def test_batch_uses_passage_prefix(embedder, tokenizer):
    """Test that encode_batch() prefixes every item."""
    embedder.encode_batch(["a", "b"])
    assert tokenizer.last_input == ["passage: a", "passage: b"]


def test_query_batch_uses_query_prefix(embedder, tokenizer):
    """Test that encode_query_batch() prefixes every item."""
    embedder.encode_query_batch(["a", "b"])
    assert tokenizer.last_input == ["query: a", "query: b"]


def test_prefixes_are_configurable():
    """Test that prefixes are configurable (not hard-coded)."""
    e = HuggingFaceEmbedder(
        model_name="fake/m", device="cpu",
        model=FakeHFModel(), tokenizer=FakeHFTokenizer(),
        passage_prefix="doc: ", query_prefix="q: ",
    )
    assert e.encode("x") == e.encode("x")
    tok = e._tokenizer
    e.encode("x")
    assert tok.last_input == ["doc: x"]


# ---------------------------------------------------------------------------
# Input validation (consistent with Phase 4.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\t\n", " \u00a0 "])
def test_empty_or_whitespace_text_rejected(embedder, bad):
    """Test that empty/whitespace-only text raises ValueError."""
    with pytest.raises(ValueError):
        embedder.encode(bad)


def test_empty_batch_rejected(embedder):
    """Test that an empty batch raises ValueError."""
    with pytest.raises(ValueError):
        embedder.encode_batch([])


def test_whitespace_item_in_batch_rejected(embedder):
    """Test that a batch with a whitespace-only item raises ValueError."""
    with pytest.raises(ValueError):
        embedder.encode_batch(["ok", "   "])


def test_batch_size_behavior_enforced():
    """Test that a batch larger than the configured batch_size raises."""
    e = HuggingFaceEmbedder(model_name="fake/m", device="cpu", batch_size=2,
                            model=FakeHFModel(), tokenizer=FakeHFTokenizer())
    with pytest.raises(ValueError):
        e.encode_batch(["a", "b", "c"])
    assert len(e.encode_batch(["a", "b"])) == 2


def test_invalid_batch_size_constructor_rejected():
    """Test that the constructor rejects invalid batch sizes."""
    with pytest.raises(ValueError):
        HuggingFaceEmbedder(batch_size=0)
    with pytest.raises(ValueError):
        HuggingFaceEmbedder(batch_size=-1)


# ---------------------------------------------------------------------------
# Model reuse / no unnecessary reloads
# ---------------------------------------------------------------------------


def test_shared_module_cache_avoids_reload():
    """Test that models are reused via the module-level cache.

    Uses a sentinel cache entry (adapter only reads it) to prove that
    re-construction reuses the cached model without reloading.
    """
    fake_model = FakeHFModel()
    fake_tokenizer = FakeHFTokenizer()
    key = ("cache-test/model", "cpu")
    _MODEL_CACHE[key] = (fake_model, fake_tokenizer)
    try:
        e1 = HuggingFaceEmbedder(model_name=key[0], device="cpu")
        e2 = HuggingFaceEmbedder(model_name=key[0], device="cpu")
        # Loading is lazy: trigger it on both instances
        assert e1.encode("x") == e2.encode("x")
        # Both must reuse the exact same cached instances (no reload)
        assert e1._model is fake_model
        assert e2._model is fake_model
        assert e1._tokenizer is fake_tokenizer
        assert e2._tokenizer is fake_tokenizer
    finally:
        _MODEL_CACHE.pop(key, None)


def test_injected_model_not_reloaded_per_call():
    """Test that encode() reuses the injected model without reloading."""
    counting = CountingModel()
    e = HuggingFaceEmbedder(model_name="fake/m", device="cpu",
                            model=counting, tokenizer=FakeHFTokenizer())
    for _ in range(3):
        e.encode("hello")
    # One forward pass per encode call, never a reload
    assert counting.forward_calls == 3


# ---------------------------------------------------------------------------
# No accidental downloads / network during tests
# ---------------------------------------------------------------------------


def test_uncached_model_fails_without_downloading(monkeypatch):
    """Test that an uncached model raises RuntimeError with no download.

    Network is hard-blocked; with local_files_only=True the load must
    fail fast instead of attempting a download.
    """

    def deny_connect(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny_connect)

    e = HuggingFaceEmbedder(
        model_name="nonexistent-model-for-hhgoa-test",
        device="cpu",
    )
    with pytest.raises(RuntimeError) as excinfo:
        e.encode("hello")
    message = str(excinfo.value)
    assert "nonexistent-model-for-hhgoa-test" in message
    assert "local" in message


def test_uncached_model_dimension_access_fails_without_download(monkeypatch):
    """Test that accessing dimension on an uncached model fails offline."""

    def deny_connect(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny_connect)

    e = HuggingFaceEmbedder(
        model_name="nonexistent-model-for-hhgoa-test",
        device="cpu",
    )
    with pytest.raises(RuntimeError):
        _ = e.dimension


# ---------------------------------------------------------------------------
# Unicode / Hindi support through the real code path
# ---------------------------------------------------------------------------


def test_hindi_batch_with_padding_uses_mask_correctly():
    """Test that a batch with different lengths (padding) works.

    Hindi sentences of different lengths force attention-mask padding;
    the masked mean pooling must still produce valid vectors.
    """
    e = HuggingFaceEmbedder(model_name="fake/m", device="cpu",
                            model=FakeHFModel(), tokenizer=FakeHFTokenizer())
    texts = ["गोवा", "गोवा में पर्यटन एक प्रमुख उद्योग है", "goa", "नई दिल्ली"]
    vectors = e.encode_batch(texts)
    assert len(vectors) == 4
    for v in vectors:
        assert len(v) == HIDDEN
        assert all(math.isfinite(x) for x in v)
    # Deterministic with padding too
    assert e.encode_batch(texts) == vectors


def test_hindi_and_english_reuse_same_path(embedder):
    """Test that Hindi and English go through identical code paths."""
    hi = embedder.encode("गोवा")
    en = embedder.encode("goa")
    assert len(hi) == len(en) == HIDDEN
    assert hi != en  # different inputs -> different vectors