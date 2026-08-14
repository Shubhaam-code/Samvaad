"""
Tests for the embedding layer (Phase 4.1).

Covers the provider-agnostic interface, the deterministic FakeEmbedder,
configuration model, and all validation rules.

All tests use tiny synthetic strings only.
No real MSMARCO-XI data. No network access. No model downloads.
"""

import math
import socket

import pytest

from app.embedding import (
    BaseEmbedder,
    EmbedderProtocol,
    EmbeddingBatch,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingVector,
    FakeEmbedder,
    create_fake_embedder,
    validate_batch,
    validate_batch_size,
    validate_embeddings,
    validate_text,
)
from app.embedding.base import validate_text as base_validate_text


# ---------------------------------------------------------------------------
# Interface importability
# ---------------------------------------------------------------------------


def test_embedding_interface_can_be_imported():
    """Test that the embedding interface can be imported."""
    assert BaseEmbedder is not None
    assert hasattr(BaseEmbedder, "encode")
    assert hasattr(BaseEmbedder, "encode_batch")
    assert hasattr(BaseEmbedder, "dimension")


def test_embedder_protocol_can_be_imported():
    """Test that EmbedderProtocol can be imported."""
    assert EmbedderProtocol is not None


def test_type_aliases_exist():
    """Test that predictable embedding types are defined."""
    assert EmbeddingVector == list[float]
    assert EmbeddingBatch == list[list[float]]


def test_base_embedder_is_abstract():
    """Test that BaseEmbedder cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseEmbedder()


def test_base_embedder_requires_encode():
    """Test that BaseEmbedder subclass must implement encode()."""
    class IncompleteEmbedder(BaseEmbedder):
        @property
        def dimension(self):
            return None

        def encode_batch(self, texts):
            return []

    with pytest.raises(TypeError):
        IncompleteEmbedder()


def test_base_embedder_requires_encode_batch():
    """Test that BaseEmbedder subclass must implement encode_batch()."""
    class IncompleteEmbedder(BaseEmbedder):
        @property
        def dimension(self):
            return None

        def encode(self, text):
            return []

    with pytest.raises(TypeError):
        IncompleteEmbedder()


def test_base_embedder_requires_dimension():
    """Test that BaseEmbedder subclass must implement dimension."""
    class IncompleteEmbedder(BaseEmbedder):
        def encode(self, text):
            return []

        def encode_batch(self, texts):
            return []

    with pytest.raises(TypeError):
        IncompleteEmbedder()


def test_embedder_protocol_duck_typing():
    """Test that any class with the right methods satisfies EmbedderProtocol."""
    class DuckTypedEmbedder:
        """Not inheriting from BaseEmbedder, but has the right methods."""

        @property
        def dimension(self):
            return 4

        def encode(self, text: str) -> EmbeddingVector:
            return [0.0] * 4

        def encode_batch(self, texts: list[str]) -> EmbeddingBatch:
            return [[0.0] * 4 for _ in texts]

    embedder = DuckTypedEmbedder()
    assert hasattr(embedder, "encode")
    assert hasattr(embedder, "encode_batch")
    assert hasattr(embedder, "dimension")
    assert callable(embedder.encode)
    assert callable(embedder.encode_batch)


# ---------------------------------------------------------------------------
# Fake embedder basics
# ---------------------------------------------------------------------------


def test_fake_embedder_works():
    """Test that FakeEmbedder produces vectors of the configured dimension."""
    embedder = FakeEmbedder(dimension=8, batch_size=4)
    assert embedder.dimension == 8
    assert embedder.batch_size == 4

    vector = embedder.encode("test")
    assert isinstance(vector, list)
    assert len(vector) == 8
    assert all(isinstance(v, float) for v in vector)
    assert all(math.isfinite(v) for v in vector)


def test_fake_embedder_defaults():
    """Test that FakeEmbedder has sensible defaults."""
    embedder = FakeEmbedder()
    assert embedder.dimension == 16
    assert embedder.batch_size == 32


def test_fake_embedder_create_factory():
    """Test the create_fake_embedder factory."""
    embedder = create_fake_embedder(dimension=6, batch_size=2)
    assert isinstance(embedder, FakeEmbedder)
    assert embedder.dimension == 6
    assert embedder.batch_size == 2


def test_fake_embedder_is_unit_length():
    """Test that fake vectors are L2-normalized (realistic stand-ins)."""
    embedder = FakeEmbedder(dimension=10)
    vector = embedder.encode("goa tourism")
    norm = math.sqrt(sum(v * v for v in vector))
    assert math.isclose(norm, 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Single text encoding
# ---------------------------------------------------------------------------


def test_single_text_encoding():
    """Test encoding a single text."""
    embedder = FakeEmbedder(dimension=5)
    vector = embedder.encode("hello world")
    assert isinstance(vector, list)
    assert len(vector) == 5


def test_single_text_encoding_different_texts_differ():
    """Test that different texts produce different vectors."""
    embedder = FakeEmbedder(dimension=16)
    v1 = embedder.encode("alpha")
    v2 = embedder.encode("beta")
    assert v1 != v2


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------


def test_batch_encoding():
    """Test encoding a batch of texts."""
    embedder = FakeEmbedder(dimension=8, batch_size=10)
    texts = ["one", "two", "three"]
    vectors = embedder.encode_batch(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == 3
    for vector in vectors:
        assert isinstance(vector, list)
        assert len(vector) == 8


def test_batch_encoding_preserves_ordering():
    """Test that encode_batch() preserves input ordering exactly.

    Input ["A", "B", "C"] must produce [vector(A), vector(B), vector(C)].
    """
    embedder = FakeEmbedder(dimension=8)
    texts = ["A", "B", "C"]

    vectors = embedder.encode_batch(texts)
    expected = [embedder.encode(t) for t in texts]

    assert len(vectors) == len(expected)
    for produced, single in zip(vectors, expected):
        assert produced == single

    # Ordering is meaningful: swapping inputs swaps outputs
    reversed_vectors = embedder.encode_batch(list(reversed(texts)))
    assert reversed_vectors == list(reversed(vectors))


# ---------------------------------------------------------------------------
# Deterministic output
# ---------------------------------------------------------------------------


def test_deterministic_output_same_instance():
    """Test that the same text produces identical vectors on one instance."""
    embedder = FakeEmbedder(dimension=8)
    assert embedder.encode("repeat") == embedder.encode("repeat")


def test_deterministic_output_across_instances():
    """Test that the same text produces identical vectors across instances."""
    e1 = FakeEmbedder(dimension=8)
    e2 = FakeEmbedder(dimension=8)
    assert e1.encode("repeat") == e2.encode("repeat")


def test_deterministic_batch_output():
    """Test that batches are deterministic across instances."""
    texts = ["x", "y", "z"]
    e1 = FakeEmbedder(dimension=8)
    e2 = FakeEmbedder(dimension=8)
    assert e1.encode_batch(texts) == e2.encode_batch(texts)


def test_deterministic_within_single_process_is_exact():
    """Test exact float equality for determinism guarantees."""
    embedder = FakeEmbedder(dimension=4)
    a = embedder.encode("deterministic")
    b = embedder.encode("deterministic")
    assert a == b  # exact equality, not approximate


# ---------------------------------------------------------------------------
# Validation: empty / whitespace text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_text", ["", "   ", "\t\n", " \u00a0 "])
def test_encode_empty_or_whitespace_raises(bad_text):
    """Test that empty/whitespace-only text raises ValueError."""
    embedder = FakeEmbedder()
    with pytest.raises(ValueError):
        embedder.encode(bad_text)


@pytest.mark.parametrize("bad_text", ["", "   ", "\t\n"])
def test_validate_text_rejects_empty_or_whitespace(bad_text):
    """Test the shared validate_text() rule directly."""
    with pytest.raises(ValueError):
        validate_text(bad_text)


def test_validate_text_accepts_valid_text():
    """Test that validate_text() accepts non-empty text."""
    assert validate_text("  hello  ") == "  hello  "
    assert validate_text("नमस्ते") == "नमस्ते"


def test_validate_text_rejects_non_string():
    """Test that validate_text() rejects non-string inputs."""
    with pytest.raises(ValueError):
        validate_text(123)
    with pytest.raises(ValueError):
        validate_text(None)


# ---------------------------------------------------------------------------
# Validation: empty batch / invalid batch size
# ---------------------------------------------------------------------------


def test_encode_batch_empty_raises():
    """Test that encoding an empty batch raises ValueError."""
    embedder = FakeEmbedder()
    with pytest.raises(ValueError):
        embedder.encode_batch([])


def test_validate_batch_empty_raises():
    """Test the shared validate_batch() rule for empty batches."""
    with pytest.raises(ValueError):
        validate_batch([])


def test_encode_batch_rejects_whitespace_item():
    """Test that a batch containing whitespace-only text raises ValueError."""
    embedder = FakeEmbedder()
    with pytest.raises(ValueError):
        embedder.encode_batch(["ok", "   "])
    with pytest.raises(ValueError):
        embedder.encode_batch([""])


def test_encode_batch_exceeding_batch_size_raises():
    """Test that a batch larger than the configured batch_size raises."""
    embedder = FakeEmbedder(dimension=8, batch_size=2)
    with pytest.raises(ValueError):
        embedder.encode_batch(["a", "b", "c"])


def test_encode_batch_at_batch_size_is_ok():
    """Test that a batch exactly at the batch_size limit is allowed."""
    embedder = FakeEmbedder(dimension=8, batch_size=3)
    vectors = embedder.encode_batch(["a", "b", "c"])
    assert len(vectors) == 3


def test_validate_batch_size_invalid_values():
    """Test the shared validate_batch_size() rule for invalid sizes."""
    with pytest.raises(ValueError):
        validate_batch_size(["a"], 0)
    with pytest.raises(ValueError):
        validate_batch_size(["a"], -1)
    with pytest.raises(ValueError):
        validate_batch_size(["a"], 1.5)
    with pytest.raises(ValueError):
        validate_batch_size(["a"], True)


def test_validate_batch_size_ok():
    """Test that validate_batch_size() accepts valid sizes."""
    assert validate_batch_size(["a", "b"], 2) == 2
    assert validate_batch_size(["a", "b"], 10) == 10


def test_fake_embedder_invalid_dimension_raises():
    """Test that FakeEmbedder rejects invalid dimensions."""
    with pytest.raises(ValueError):
        FakeEmbedder(dimension=0)
    with pytest.raises(ValueError):
        FakeEmbedder(dimension=-4)
    with pytest.raises(ValueError):
        FakeEmbedder(dimension=4.5)


def test_fake_embedder_invalid_batch_size_raises():
    """Test that FakeEmbedder rejects invalid batch sizes."""
    with pytest.raises(ValueError):
        FakeEmbedder(batch_size=0)
    with pytest.raises(ValueError):
        FakeEmbedder(batch_size=-1)


# ---------------------------------------------------------------------------
# Validation: vector dimension consistency
# ---------------------------------------------------------------------------


def test_batch_vectors_have_consistent_dimension():
    """Test that all vectors in a batch share the same dimension."""
    embedder = FakeEmbedder(dimension=9, batch_size=20)
    vectors = embedder.encode_batch(["a", "b", "c", "d"])
    dims = {len(v) for v in vectors}
    assert dims == {9}


def test_validate_embeddings_inconsistent_dimensions_raises():
    """Test that validate_embeddings() rejects inconsistent dimensions."""
    bad_batch = [[0.1, 0.2], [0.1, 0.2, 0.3]]
    with pytest.raises(ValueError, match="Inconsistent vector dimensions"):
        validate_embeddings(bad_batch)


def test_validate_embeddings_expected_dimension_mismatch_raises():
    """Test that validate_embeddings() rejects wrong expected dimension."""
    with pytest.raises(ValueError, match="expected dimension"):
        validate_embeddings([[0.1, 0.2, 0.3]], expected_dimension=5)


def test_validate_embeddings_accepts_consistent_batch():
    """Test that validate_embeddings() accepts a consistent batch."""
    good_batch = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    result = validate_embeddings(good_batch, expected_dimension=3)
    assert result == good_batch


def test_validate_embeddings_rejects_non_list_vector():
    """Test that validate_embeddings() rejects non-list vectors."""
    with pytest.raises(ValueError):
        validate_embeddings([(0.1, 0.2)])
    with pytest.raises(ValueError):
        validate_embeddings([[0.1], "not-a-vector"])


def test_validate_embeddings_rejects_empty_batch():
    """Test that validate_embeddings() rejects an empty batch."""
    with pytest.raises(ValueError):
        validate_embeddings([])


# ---------------------------------------------------------------------------
# Validation: invalid / non-finite vector values
# ---------------------------------------------------------------------------


def test_validate_embeddings_rejects_nan():
    """Test that validate_embeddings() rejects NaN values."""
    with pytest.raises(ValueError, match="not finite"):
        validate_embeddings([[0.1, float("nan")]])


def test_validate_embeddings_rejects_infinity():
    """Test that validate_embeddings() rejects infinite values."""
    with pytest.raises(ValueError, match="not finite"):
        validate_embeddings([[0.1, float("inf")]])
    with pytest.raises(ValueError, match="not finite"):
        validate_embeddings([[0.1, float("-inf")]])


def test_validate_embeddings_rejects_non_numeric_value():
    """Test that validate_embeddings() rejects non-numeric values."""
    with pytest.raises(ValueError):
        validate_embeddings([[0.1, "0.2"]])


def test_fake_embedder_vectors_are_always_finite():
    """Test that FakeEmbedder never produces NaN/inf, even for odd inputs."""
    embedder = FakeEmbedder(dimension=16)
    odd_texts = ["\u0000", "नमस्ते दुनिया", "a" * 5000, "emoji \U0001F600"]
    for text in odd_texts:
        vector = embedder.encode(text)
        assert all(math.isfinite(v) for v in vector)


# ---------------------------------------------------------------------------
# No network / no model download
# ---------------------------------------------------------------------------


def test_fake_embedder_never_touches_network(monkeypatch):
    """Test that FakeEmbedder works with all network access blocked."""
    def deny_connect(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny_connect)

    embedder = FakeEmbedder(dimension=8)
    vector = embedder.encode("offline test")
    batch = embedder.encode_batch(["a", "b"])

    assert len(vector) == 8
    assert len(batch) == 2


def test_embedding_package_imports_no_external_ml_libraries():
    """Test that the embedding package imports without any ML frameworks."""
    import importlib

    forbidden_prefixes = (
        "transformers",
        "torch",
        "sentence_transformers",
        "numpy",
        "faiss",
    )

    # Sanity: ensure the package imports cleanly
    module_names = (
        "app.embedding",
        "app.embedding.base",
        "app.embedding.fake",
        "app.embedding.config",
        "app.embedding.types",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert module is not None

        # No name in the embedding package namespace may originate from an
        # external ML library (checked directly, immune to other test
        # modules importing those libraries first)
        for attr_name, attr in vars(module).items():
            origin = getattr(attr, "__module__", None)
            if origin is not None and origin.startswith(forbidden_prefixes):
                raise AssertionError(
                    f"{module_name} exposes {attr_name} from forbidden "
                    f"library '{origin}'"
                )


# ---------------------------------------------------------------------------
# Unicode / Hindi text support
# ---------------------------------------------------------------------------


def test_unicode_hindi_text_support():
    """Test that Hindi (Devanagari) text embeds correctly."""
    embedder = FakeEmbedder(dimension=8)
    hindi_text = "गोवा में पर्यटन एक प्रमुख उद्योग है"

    vector = embedder.encode(hindi_text)
    assert isinstance(vector, list)
    assert len(vector) == 8
    assert all(math.isfinite(v) for v in vector)


def test_unicode_hindi_deterministic():
    """Test that Hindi text embedding is deterministic."""
    embedder = FakeEmbedder(dimension=8)
    hindi_text = "भारत की राजधानी नई दिल्ली है"
    assert embedder.encode(hindi_text) == embedder.encode(hindi_text)


def test_unicode_batch_preserves_order():
    """Test that a mixed Hindi/English batch preserves order."""
    embedder = FakeEmbedder(dimension=8)
    texts = ["English one", "हिंदी पाठ", "mixed text with 123", "होटल"]
    vectors = embedder.encode_batch(texts)
    expected = [embedder.encode(t) for t in texts]
    assert vectors == expected


def test_hindi_and_english_different_vectors():
    """Test that Hindi and its English translation map to different vectors."""
    embedder = FakeEmbedder(dimension=16)
    hindi = embedder.encode("गोवा")
    english = embedder.encode("goa")
    assert hindi != english


# ---------------------------------------------------------------------------
# EmbeddingConfig model
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Test that EmbeddingConfig has safe defaults."""
    config = EmbeddingConfig()
    assert config.provider == EmbeddingProvider.FAKE
    assert config.model_name is None
    assert config.dimension is None
    assert config.batch_size == 32


def test_config_explicit_values():
    """Test that EmbeddingConfig accepts explicit values."""
    config = EmbeddingConfig(
        provider=EmbeddingProvider.HUGGINGFACE,
        model_name="some-multilingual-model",
        dimension=384,
        batch_size=64,
    )
    assert config.provider == EmbeddingProvider.HUGGINGFACE
    assert config.model_name == "some-multilingual-model"
    assert config.dimension == 384
    assert config.batch_size == 64


def test_config_rejects_invalid_dimension():
    """Test that EmbeddingConfig rejects invalid dimensions."""
    with pytest.raises(ValueError):
        EmbeddingConfig(dimension=0)
    with pytest.raises(ValueError):
        EmbeddingConfig(dimension=-5)


def test_config_rejects_invalid_batch_size():
    """Test that EmbeddingConfig rejects invalid batch sizes."""
    with pytest.raises(ValueError):
        EmbeddingConfig(batch_size=0)
    with pytest.raises(ValueError):
        EmbeddingConfig(batch_size=-10)


def test_config_rejects_empty_model_name():
    """Test that EmbeddingConfig rejects empty/whitespace model names."""
    with pytest.raises(ValueError):
        EmbeddingConfig(model_name="")
    with pytest.raises(ValueError):
        EmbeddingConfig(model_name="   ")


def test_config_fake_provider_value():
    """Test that the fake provider enum value matches the string."""
    assert EmbeddingProvider.FAKE.value == "fake"
    assert EmbeddingProvider.HUGGINGFACE.value == "huggingface"
    assert EmbeddingProvider.LOCAL.value == "local"
    assert EmbeddingProvider.API.value == "api"


def test_config_can_drive_fake_embedder_creation():
    """Test that a config with the fake provider creates a FakeEmbedder."""
    config = EmbeddingConfig(dimension=12, batch_size=5)
    embedder = FakeEmbedder(dimension=config.dimension, batch_size=config.batch_size)
    assert embedder.dimension == 12
    assert embedder.batch_size == 5


# ---------------------------------------------------------------------------
# Generic interface shape for future providers
# ---------------------------------------------------------------------------


def test_future_provider_duck_typing_compatible():
    """Test that the base interface is duck-typed for future providers.

    Simulates a future HuggingFace-style provider wrapping the interface:
    it only needs encode/encode_batch/dimension, nothing else.
    """

    class FakeHFProvider:
        """Stand-in for a future Sentence Transformers provider."""

        def __init__(self, dimension: int = 384):
            self._dimension = dimension

        @property
        def dimension(self) -> int:
            return self._dimension

        def encode(self, text: str) -> EmbeddingVector:
            return [0.5] * self._dimension

        def encode_batch(self, texts: list[str]) -> EmbeddingBatch:
            return [self.encode(t) for t in texts]

    provider = FakeHFProvider(dimension=384)
    assert isinstance(provider.encode("x"), list)
    assert len(provider.encode("x")) == 384
    assert len(provider.encode_batch(["a", "b"])) == 2
    assert provider.dimension == 384

    # Protocol-compatible shapes pass the shared validators too
    vectors = provider.encode_batch(["a", "b"])
    validated = validate_embeddings(vectors, expected_dimension=384)
    assert len(validated) == 2


def test_base_validate_text_is_reexported_consistently():
    """Test that the base module exposes the same validate_text function."""
    from app.embedding.base import validate_text as base_text
    assert base_text is base_validate_text
    assert base_text("ok") == "ok"