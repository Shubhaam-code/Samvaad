"""
Tests for the memory-safe batch embedding pipeline (Phase 4.3).

All tests use tiny synthetic Chunk objects and the FakeEmbedder or
mock embedders. The real HuggingFace model is NEVER required here.

No real MSMARCO-XI data. No network access. No model downloads.
"""

import math

import pytest

from app.chunking.models import Chunk, ChunkingStrategy
from app.embedding import (
    DEFAULT_BATCH_SIZE,
    EmbeddingFailure,
    EmbeddingPipeline,
    EmbeddingPipelineError,
    EmbeddingResult,
    FakeEmbedder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(chunk_id: str = "chunk-0", text: str = "Some chunk text.") -> Chunk:
    """Create a tiny synthetic Chunk for tests."""
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        strategy=ChunkingStrategy.PASSAGE,
        chunk_text=text,
        query_id=1,
        passage_index=0,
        target_lang="hi",
        source_lang="en",
        query="test query",
        eng_query="test query",
        is_selected=True,
    )


def make_chunks(count: int, prefix: str = "chunk-") -> list[Chunk]:
    return [make_chunk(chunk_id=f"{prefix}{i}", text=f"chunk text number {i}")
            for i in range(count)]


class MockEmbedder:
    """Mock embedder exposing the Phase 4.1 interface + optional flaws."""

    def __init__(
        self,
        dimension: int = 8,
        batch_size: int = 64,
        model_name: str | None = "mock/model",
        fail_on_batch: int | None = None,
        wrong_dim_batch: int | None = None,
        nan_batch: int | None = None,
        short_batch: int | None = None,
    ):
        self.dimension = dimension
        self.batch_size = batch_size
        self.model_name = model_name
        self.encode_calls = 0
        self.encode_batch_calls = 0
        self._fail_on_batch = fail_on_batch
        self._wrong_dim_batch = wrong_dim_batch
        self._nan_batch = nan_batch
        self._short_batch = short_batch

    def encode(self, text: str) -> list[float]:
        self.encode_calls += 1
        return [0.5] * self.dimension

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        self.encode_batch_calls += 1
        call = self.encode_batch_calls
        if self._fail_on_batch is not None and call >= self._fail_on_batch:
            raise RuntimeError("simulated provider failure")
        if self._wrong_dim_batch is not None and call >= self._wrong_dim_batch:
            return [[0.5] * (self.dimension - 1) for _ in texts]
        if self._nan_batch is not None and call >= self._nan_batch:
            return [[0.5] * (self.dimension - 1) + [float("nan")] for _ in texts]
        if self._short_batch is not None and call >= self._short_batch:
            return [[0.5] * self.dimension for _ in texts[:-1]]
        return [[0.5] * self.dimension for _ in texts]


# ---------------------------------------------------------------------------
# Import / construction
# ---------------------------------------------------------------------------


def test_pipeline_can_be_imported():
    """Test that the pipeline and its models are importable."""
    assert EmbeddingPipeline is not None
    assert EmbeddingResult is not None
    assert EmbeddingFailure is not None
    assert EmbeddingPipelineError is not None
    assert DEFAULT_BATCH_SIZE == 32


def test_default_batch_size_is_32():
    """Test the documented default batch size."""
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())
    assert pipeline.batch_size == 32


def test_pipeline_requires_embedder():
    """Test that a missing embedder is rejected."""
    with pytest.raises(ValueError):
        EmbeddingPipeline(embedder=None)


def test_pipeline_requires_encode_batch():
    """Test that an embedder without encode_batch is rejected."""

    class NotAnEmbedder:
        pass

    with pytest.raises(ValueError):
        EmbeddingPipeline(embedder=NotAnEmbedder())


def test_invalid_batch_size_rejected():
    """Test that invalid batch sizes are rejected at construction."""
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            EmbeddingPipeline(embedder=FakeEmbedder(), batch_size=bad)


def test_batch_size_exceeding_embedder_capacity_rejected():
    """Test that a pipeline batch larger than the embedder's is rejected."""
    with pytest.raises(ValueError):
        EmbeddingPipeline(embedder=FakeEmbedder(batch_size=4), batch_size=16)


# ---------------------------------------------------------------------------
# Basic batching behavior
# ---------------------------------------------------------------------------


def test_single_batch_tiny_list():
    """Test a tiny chunk list in a single batch."""
    chunks = make_chunks(3)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=32)

    batches = list(pipeline.embed_batches(chunks))
    assert len(batches) == 1
    assert len(batches[0]) == 3


def test_multiple_batches():
    """Test that 5 chunks with batch_size=2 produce 3 batches."""
    chunks = make_chunks(5)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=2)

    batches = list(pipeline.embed_batches(chunks))
    assert len(batches) == 3
    assert [len(b) for b in batches] == [2, 2, 1]


def test_batch_size_respected():
    """Test that no yielded batch exceeds the configured batch size."""
    chunks = make_chunks(11)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=4)

    batches = list(pipeline.embed_batches(chunks))
    assert all(len(b) <= 4 for b in batches)
    assert sum(len(b) for b in batches) == 11


def test_embed_batch_single_batch_api():
    """Test the simple embed_batch() API."""
    chunks = make_chunks(3)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=32)

    results = pipeline.embed_batch(chunks)
    assert len(results) == 3


def test_embed_batch_rejects_oversized_input():
    """Test that embed_batch() refuses more chunks than batch_size."""
    chunks = make_chunks(5)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=3)

    with pytest.raises(ValueError):
        pipeline.embed_batch(chunks)


def test_embed_all_convenience_api():
    """Test the embed_all() convenience API."""
    chunks = make_chunks(6)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=2)

    results = pipeline.embed_all(chunks)
    assert len(results) == 6


# ---------------------------------------------------------------------------
# Ordering guarantees
# ---------------------------------------------------------------------------


def test_batch_ordering_preserved():
    """Test that result chunk_ids follow the exact input order."""
    chunks = make_chunks(5)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=2)

    results = pipeline.embed_all(chunks)
    assert [r.chunk_id for r in results] == [c.chunk_id for c in chunks]


def test_ordering_across_batches():
    """Test that ordering holds across multiple batches."""
    chunks = make_chunks(7)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=3)

    flattened = [r for batch in pipeline.embed_batches(chunks) for r in batch]
    assert [r.chunk_id for r in flattened] == [c.chunk_id for c in chunks]


def test_chunk_id_preserved():
    """Test that each result carries its chunk's chunk_id."""
    chunks = [make_chunk(chunk_id=f"id-{i}", text=f"text {i}") for i in range(4)]
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=2)

    results = pipeline.embed_all(chunks)
    for chunk, result in zip(chunks, results):
        assert result.chunk_id == chunk.chunk_id


def test_duplicate_chunk_ids_keep_positions():
    """Test that duplicate chunk_ids are preserved positionally."""
    chunks = [make_chunk(chunk_id="same", text="first"),
              make_chunk(chunk_id="same", text="second")]
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=1)

    results = pipeline.embed_all(chunks)
    assert results[0].chunk_id == results[1].chunk_id == "same"


# ---------------------------------------------------------------------------
# Result structure / vector quality
# ---------------------------------------------------------------------------


def test_result_structure():
    """Test that results expose chunk_id, embedding, dimension, metadata."""
    chunks = make_chunks(2)
    pipeline = EmbeddingPipeline(
        embedder=MockEmbedder(dimension=8, model_name="mock/model"), batch_size=2
    )

    result = pipeline.embed_all(chunks)[0]
    assert result.chunk_id == chunks[0].chunk_id
    assert isinstance(result.embedding, list)
    assert isinstance(result.dimension, int)
    assert result.dimension == 8
    assert result.model_name == "mock/model"
    assert result.provider == "MockEmbedder"


def test_embedding_dimension_preserved():
    """Test that result.dimension matches the embedder dimension."""
    chunks = make_chunks(5)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=12), batch_size=2)

    results = pipeline.embed_all(chunks)
    for result in results:
        assert result.dimension == 12
        assert len(result.embedding) == 12


def test_values_are_finite_floats():
    """Test that all embedding values are finite floats."""
    chunks = make_chunks(4)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=2)

    for result in pipeline.embed_all(chunks):
        assert all(isinstance(v, float) for v in result.embedding)
        assert all(math.isfinite(v) for v in result.embedding)


def test_deterministic_output():
    """Test that the pipeline is deterministic for identical inputs."""
    chunks = make_chunks(4)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=2)

    first = pipeline.embed_all(chunks)
    second = pipeline.embed_all(chunks)
    assert first == second


def test_unicode_hindi_chunk_text():
    """Test that Hindi chunk text embeds through the pipeline."""
    chunks = [
        make_chunk(chunk_id="hi-1", text="भारत की राजधानी नई दिल्ली है।"),
        make_chunk(chunk_id="hi-2", text="गोवा में पर्यटन एक प्रमुख उद्योग है"),
        make_chunk(chunk_id="en-1", text="India's capital is New Delhi."),
    ]
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=2)

    results = pipeline.embed_all(chunks)
    assert [r.chunk_id for r in results] == ["hi-1", "hi-2", "en-1"]
    assert all(r.dimension == 8 for r in results)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_input_rejected_batch_api():
    """Test that embed_batch() rejects an empty list."""
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        pipeline.embed_batch([])


def test_empty_input_rejected_batches_api():
    """Test that embed_batches() rejects an empty list."""
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        list(pipeline.embed_batches([]))


def test_non_list_input_rejected():
    """Test that non-list input is rejected."""
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        pipeline.embed_batch("not-a-list")


def test_malformed_chunk_rejected():
    """Test that an object without chunk attributes is rejected."""
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        pipeline.embed_batch([object()])


def test_missing_chunk_id_rejected():
    """Test that a chunk-like object without chunk_id is rejected."""
    from types import SimpleNamespace
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())
    bad = SimpleNamespace(chunk_text="has text but no id")
    with pytest.raises(ValueError, match="chunk_id"):
        pipeline.embed_batch([bad])


def test_empty_chunk_id_rejected():
    """Test that an empty chunk_id is rejected."""
    from types import SimpleNamespace
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())
    bad = SimpleNamespace(chunk_id="", chunk_text="text")
    with pytest.raises(ValueError, match="chunk_id"):
        pipeline.embed_batch([bad])


def test_missing_chunk_text_rejected():
    """Test that a chunk-like object without chunk_text is rejected."""
    from types import SimpleNamespace
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())
    bad = SimpleNamespace(chunk_id="id-1")
    with pytest.raises(ValueError, match="chunk_text"):
        pipeline.embed_batch([bad])


def test_whitespace_chunk_text_rejected():
    """Test that whitespace-only chunk_text is rejected."""
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="chunk_text"):
        pipeline.embed_batch([make_chunk(chunk_id="x", text="   ")])


def test_invalid_chunk_in_middle_of_list_rejected():
    """Test that a malformed chunk anywhere in the list is rejected."""
    from types import SimpleNamespace
    chunks = [make_chunk(chunk_id="ok-1"), SimpleNamespace(chunk_id="bad")]
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        pipeline.embed_batch(chunks)


# ---------------------------------------------------------------------------
# Provider failures / error handling
# ---------------------------------------------------------------------------


def test_provider_failure_fail_fast_raises():
    """Test that a provider failure raises with fail_fast=True (default)."""
    embedder = MockEmbedder(fail_on_batch=2)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2)

    with pytest.raises(EmbeddingPipelineError) as excinfo:
        list(pipeline.embed_batches(make_chunks(4)))

    message = str(excinfo.value)
    assert "simulated provider failure" in message
    assert "chunk-2" in message or "chunk-3" in message
    assert len(pipeline.errors) == 1
    assert pipeline.errors[0].batch_index == 2


def test_provider_failure_reported_via_on_error():
    """Test that on_error receives a failure record before raising."""
    embedder = MockEmbedder(fail_on_batch=2)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2)
    failures = []

    def report(failure: EmbeddingFailure):
        failures.append(failure)

    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2, on_error=report)
    with pytest.raises(EmbeddingPipelineError):
        list(pipeline.embed_batches(make_chunks(4)))

    assert len(failures) == 1
    assert failures[0].batch_index == 2
    assert failures[0].chunk_ids == ["chunk-2", "chunk-3"]
    assert "simulated provider failure" in failures[0].error


def test_provider_failure_continue_with_fail_fast_false():
    """Test that fail_fast=False reports failures and continues."""
    embedder = MockEmbedder(fail_on_batch=2)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2, fail_fast=False)

    results = pipeline.embed_all(make_chunks(4))

    # Failed batch produces NO results; others are intact and ordered
    assert [r.chunk_id for r in results] == ["chunk-0", "chunk-1"]
    assert len(pipeline.errors) == 1
    assert pipeline.errors[0].batch_index == 2


def test_no_corrupted_output_on_failure():
    """Test that a failed batch never yields partial/misaligned results."""
    embedder = MockEmbedder(short_batch=1)  # returns fewer vectors than inputs
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2, fail_fast=True)

    with pytest.raises(EmbeddingPipelineError):
        list(pipeline.embed_batches(make_chunks(2)))


def test_wrong_vector_count_is_rejected():
    """Test that an embedder returning a mismatched vector count raises
    instead of silently truncating."""
    embedder = MockEmbedder(short_batch=1)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2)

    with pytest.raises(EmbeddingPipelineError):
        list(pipeline.embed_batches(make_chunks(2)))
    assert len(pipeline.errors) == 1


def test_dimension_mismatch_fail_fast():
    """Test that a dimension mismatch is detected and raised."""
    embedder = MockEmbedder(dimension=8, wrong_dim_batch=1)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2)

    with pytest.raises(EmbeddingPipelineError) as excinfo:
        list(pipeline.embed_batches(make_chunks(2)))

    assert "dimension" in str(excinfo.value)


def test_dimension_mismatch_reported_with_fail_fast_false():
    """Test that a dimension mismatch is reported when not failing fast."""
    embedder = MockEmbedder(dimension=8, wrong_dim_batch=2)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2, fail_fast=False)

    results = pipeline.embed_all(make_chunks(4))
    assert [r.chunk_id for r in results] == ["chunk-0", "chunk-1"]
    assert len(pipeline.errors) == 1
    assert "dimension" in pipeline.errors[0].error


def test_non_finite_values_fail_fast():
    """Test that non-finite embedding values are detected and raised."""
    embedder = MockEmbedder(dimension=8, nan_batch=1)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2)

    with pytest.raises(EmbeddingPipelineError) as excinfo:
        list(pipeline.embed_batches(make_chunks(2)))

    assert "not finite" in str(excinfo.value)


def test_non_finite_values_reported_with_fail_fast_false():
    """Test that non-finite values are reported when not failing fast."""
    embedder = MockEmbedder(dimension=8, nan_batch=2)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2, fail_fast=False)

    results = pipeline.embed_all(make_chunks(4))
    assert len(results) == 2
    assert len(pipeline.errors) == 1
    assert "not finite" in pipeline.errors[0].error


def test_errors_reset_between_runs():
    """Test that errors from a previous run do not leak into the next."""
    embedder = MockEmbedder(fail_on_batch=1)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2, fail_fast=False)

    with pytest.raises(ValueError):
        pipeline.embed_batch([])  # structural error, not a run

    results = pipeline.embed_all(make_chunks(2))
    assert results == []
    assert len(pipeline.errors) == 1

    # Second run: healthy embedder -> no stale errors
    healthy = MockEmbedder()
    pipeline = EmbeddingPipeline(embedder=healthy, batch_size=2)
    results = pipeline.embed_all(make_chunks(2))
    assert len(results) == 2
    assert pipeline.errors == []


# ---------------------------------------------------------------------------
# Batching discipline (no per-item encode, encode_batch per batch)
# ---------------------------------------------------------------------------


def test_no_per_item_encode_calls():
    """Test that the pipeline never calls encode() per chunk."""
    embedder = MockEmbedder()
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=3)

    list(pipeline.embed_batches(make_chunks(6)))

    assert embedder.encode_calls == 0
    assert embedder.encode_batch_calls == 2


def test_one_encode_batch_call_per_configured_batch():
    """Test that encode_batch() is called exactly once per pipeline batch."""
    embedder = MockEmbedder()
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=4)

    list(pipeline.embed_batches(make_chunks(10)))

    assert embedder.encode_batch_calls == 3


def test_generator_is_lazy():
    """Test that embed_batches() is lazy: nothing embedded before iteration."""
    embedder = MockEmbedder()
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2)

    gen = pipeline.embed_batches(make_chunks(4))
    assert embedder.encode_batch_calls == 0  # not started yet

    first = next(gen)
    assert len(first) == 2
    assert embedder.encode_batch_calls == 1
    assert [r.chunk_id for r in first] == ["chunk-0", "chunk-1"]


def test_generator_yields_incremental_results():
    """Test that the generator yields batch results incrementally."""
    embedder = MockEmbedder()
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2)

    seen = []
    for batch in pipeline.embed_batches(make_chunks(5)):
        seen.extend(r.chunk_id for r in batch)
    assert seen == ["chunk-0", "chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    assert embedder.encode_batch_calls == 3


# ---------------------------------------------------------------------------
# Provider-agnostic behavior (works with FakeEmbedder and HF-style embedder)
# ---------------------------------------------------------------------------


def test_works_with_fake_embedder():
    """Test the pipeline with FakeEmbedder (Phase 4.1 provider)."""
    chunks = make_chunks(4)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=2)

    results = pipeline.embed_all(chunks)
    assert len(results) == 4
    assert results[0].dimension == 8
    assert results[0].model_name is None  # FakeEmbedder has no model_name
    assert results[0].provider == "FakeEmbedder"


def test_works_with_hf_style_embedder():
    """Test the pipeline with an embedder exposing the HuggingFace adapter
    surface (model_name + dimension + batch_size) without any real model."""
    embedder = MockEmbedder(dimension=384, model_name="intfloat/multilingual-e5-small")
    chunks = make_chunks(3)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=2)

    results = pipeline.embed_all(chunks)
    assert all(r.dimension == 384 for r in results)
    assert all(r.model_name == "intfloat/multilingual-e5-small" for r in results)
    assert all(r.provider == "MockEmbedder" for r in results)


def test_embedder_without_dimension_attribute():
    """Test a duck-typed embedder that lacks a dimension attribute."""

    class BareEmbedder:
        def encode(self, text):
            return [1.0, 2.0, 3.0]

        def encode_batch(self, texts):
            return [[1.0, 2.0, 3.0] for _ in texts]

    chunks = make_chunks(2)
    pipeline = EmbeddingPipeline(embedder=BareEmbedder(), batch_size=2)

    results = pipeline.embed_all(chunks)
    assert results[0].dimension == 3
    assert len(results[0].embedding) == 3


# ---------------------------------------------------------------------------
# Large synthetic count (batching without large allocations)
# ---------------------------------------------------------------------------


def test_large_synthetic_count_batches_correctly():
    """Test 1000 tiny chunks with a small batch size.

    Verifies correct batching, exact ordering, and one encode_batch()
    call per configured batch without large memory use.
    """
    chunks = make_chunks(1000)
    pipeline = EmbeddingPipeline(
        embedder=FakeEmbedder(dimension=8, batch_size=1000), batch_size=7
    )

    total = 0
    batch_sizes = []
    for batch in pipeline.embed_batches(chunks):
        batch_sizes.append(len(batch))
        total += len(batch)
        assert len(batch) <= 7
        for result in batch:
            assert len(result.embedding) == 8

    assert total == 1000
    assert len(batch_sizes) == 143  # ceil(1000 / 7)
    assert batch_sizes[-1] == 1000 % 7

    # Ordering verified in one pass with a full materialization (test-only)
    all_ids = []
    embedder = FakeEmbedder(dimension=8, batch_size=1000)
    pipeline = EmbeddingPipeline(embedder=embedder, batch_size=7)
    for batch in pipeline.embed_batches(chunks):
        all_ids.extend(r.chunk_id for r in batch)
    assert all_ids == [c.chunk_id for c in chunks]


def test_pipeline_logs_batch_progress(caplog):
    """Test that progress is logged at batch level only (no vectors)."""
    import logging

    chunks = make_chunks(5)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(dimension=8), batch_size=2)

    with caplog.at_level(logging.INFO, logger="app.embedding.pipeline"):
        list(pipeline.embed_batches(chunks))

    messages = [r.message for r in caplog.records]
    assert any("Embedded batch 1/3" in m for m in messages)
    assert any("Embedded batch 3/3" in m for m in messages)
    # No vector values may appear in logs
    assert not any("0.123" in m or "[" in m and "0." in m for m in messages)