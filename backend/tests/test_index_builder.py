import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.chunking.models import Chunk, ChunkingStrategy
from app.embedding import FakeEmbedder, HuggingFaceEmbedder
from app.guardrails.grounding_verifier import GroundingVerifier
from app.guardrails.models import GuardrailVerdict
from app.indexing.chunk_store import CHUNKS_FILENAME, CHUNKS_INDEX_FILENAME
from app.indexing.loader import VECTORSTORE_DIRNAME, load_index
from app.indexing.manifest import IndexCompatibilityError, read_manifest
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.vectorstore import NumpyVectorStore
from scripts import build_index as build_index_module
from scripts.build_index import (
    IndexBuildConfig,
    IndexBuildError,
    build_index,
)


RAW_SCHEMA = pa.schema(
    [
        ("query_id", pa.int64()),
        ("Query", pa.string()),
        ("Eng_Query", pa.string()),
        ("Answer", pa.string()),
        ("Eng_Answer", pa.string()),
        ("query_type", pa.string()),
        (
            "passages",
            pa.struct(
                [
                    ("Translated_passages", pa.list_(pa.string())),
                    ("English_passages", pa.list_(pa.string())),
                    ("is_selected", pa.list_(pa.int64())),
                ]
            ),
        ),
        ("source_lang", pa.string()),
        ("target_lang", pa.string()),
    ]
)


def _write_raw_parquet(path: Path, count: int = 3, empty: bool = False) -> Path:
    rows = []
    for i in range(count):
        if empty:
            passages = {
                "Translated_passages": [],
                "English_passages": [],
                "is_selected": [],
            }
        else:
            text = f"Goa beach evidence number {i} supports tourism facts."
            passages = {
                "Translated_passages": [text],
                "English_passages": [text],
                "is_selected": [1],
            }
        rows.append(
            {
                "query_id": i + 1,
                "Query": f"goa tourism {i}",
                "Eng_Query": f"goa tourism {i}",
                "Answer": f"answer {i}",
                "Eng_Answer": f"answer {i}",
                "query_type": "DESCRIPTION",
                "passages": passages,
                "source_lang": "en",
                "target_lang": "hi",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=RAW_SCHEMA), path)
    return path


def _config(tmp_path: Path, source: Path, **overrides) -> IndexBuildConfig:
    values = {
        "source_path": source,
        "processed_output": tmp_path / "processed.parquet",
        "index_dir": tmp_path / "index",
        "vector_store": "numpy",
        "embedding_model": "test-fake-model",
        "embedding_device": "cpu",
        "embedding_batch_size": 2,
        "chunking_strategy": ChunkingStrategy.PASSAGE,
        "top_k": 2,
        "limit": None,
        "overwrite": False,
        "allow_download": False,
    }
    values.update(overrides)
    return IndexBuildConfig(**values)


def test_package_import_and_cli_help():
    assert build_index_module.main is not None
    backend_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.build_index", "--help"],
        cwd=backend_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--source-path" in result.stdout
    assert "--embedding-device" in result.stdout
    assert "--top-k" in result.stdout


def test_config_validation(tmp_path):
    source = _write_raw_parquet(tmp_path / "raw.parquet")
    config = _config(tmp_path, source, vector_store="numpy", top_k=3)
    assert config.vector_store == "numpy"
    assert config.top_k == 3
    with pytest.raises(ValueError, match="Exactly one dataset source"):
        IndexBuildConfig(index_dir=tmp_path / "idx")
    with pytest.raises(ValueError, match="vector_store_backend"):
        _config(tmp_path, source, vector_store="bad")


def test_production_embedder_factory_never_returns_fake(monkeypatch, tmp_path):
    source = _write_raw_parquet(tmp_path / "raw.parquet")
    config = _config(
        tmp_path,
        source,
        embedding_model="intfloat/multilingual-e5-small",
        embedding_batch_size=1,
    )
    monkeypatch.setattr(build_index_module, "is_model_cached", lambda model: True)
    embedder = build_index_module._create_production_embedder(config)
    assert isinstance(embedder, HuggingFaceEmbedder)
    assert not isinstance(embedder, FakeEmbedder)
    assert embedder.local_files_only is True


def test_no_network_behavior_requires_allow_download(monkeypatch, tmp_path):
    source = _write_raw_parquet(tmp_path / "raw.parquet")
    config = _config(tmp_path, source, embedding_model="missing-model")
    monkeypatch.setattr(build_index_module, "is_model_cached", lambda model: False)
    with pytest.raises(IndexBuildError, match="downloads are disabled"):
        build_index_module._create_production_embedder(config)


def test_synthetic_offline_build_persists_loadable_index(tmp_path):
    source = _write_raw_parquet(tmp_path / "raw.parquet", count=3)
    result = build_index(_config(tmp_path, source), embedder=FakeEmbedder(dimension=8, batch_size=2), log=lambda _: None)

    index_dir = result.index_dir
    assert (index_dir / "manifest.json").is_file()
    assert (index_dir / CHUNKS_FILENAME).is_file()
    assert (index_dir / CHUNKS_INDEX_FILENAME).is_file()
    assert (index_dir / VECTORSTORE_DIRNAME / "vectors.npy").is_file()

    manifest = read_manifest(index_dir)
    assert manifest.dataset.source == "local-parquet"
    assert manifest.embedding.provider == "FakeEmbedder"
    assert manifest.embedding.model == "test-fake-model"
    assert manifest.embedding.dimension == 8
    assert manifest.vector_store.backend == "numpy"
    assert manifest.chunking.strategy == "passage"
    assert manifest.counts.documents == 3
    assert manifest.counts.chunks == 3
    assert manifest.counts.vectors == 3

    store, resolver, loaded_manifest = load_index(
        index_dir,
        expected_model_name="test-fake-model",
        expected_dimension=8,
        expected_backend="numpy",
        lazy_chunks=True,
    )
    assert isinstance(store, NumpyVectorStore)
    assert store.count == resolver.count == loaded_manifest.counts.chunks == 3
    first_chunk = resolver.resolve([resolver.chunk_ids[0]])[0]
    assert isinstance(first_chunk, Chunk)
    assert first_chunk.chunk_text

    query_vector = FakeEmbedder(dimension=8, batch_size=2).encode(first_chunk.chunk_text)
    hits = store.search(query_vector, top_k=2)
    resolved = resolver.resolve([hit.chunk_id for hit in hits])
    assert all(isinstance(chunk, Chunk) and chunk.chunk_text for chunk in resolved)
    grounding = GroundingVerifier().verify(first_chunk.chunk_text, resolved)
    assert grounding.verdict == GuardrailVerdict.SAFE_AND_GROUNDED

    orchestrator = RetrievalOrchestrator(
        embedder=FakeEmbedder(dimension=8, batch_size=2),
        vector_store=store,
        resolver=resolver,
        top_k=2,
    )
    retrieval = orchestrator.retrieve(first_chunk.chunk_text)
    assert retrieval.allowed is True
    assert retrieval.retrieved_chunks
    assert retrieval.retrieved_chunks[0].chunk.chunk_text


def test_dimension_mismatch_fails_without_index(tmp_path):
    class BadEmbedder(FakeEmbedder):
        def encode_batch(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    source = _write_raw_parquet(tmp_path / "raw.parquet")
    config = _config(tmp_path, source)
    with pytest.raises(Exception, match="dimension"):
        build_index(config, embedder=BadEmbedder(dimension=2, batch_size=2), log=lambda _: None)
    assert not config.index_dir.exists()


def test_empty_dataset_fails_clearly(tmp_path):
    source = _write_raw_parquet(tmp_path / "raw.parquet", count=2, empty=True)
    config = _config(tmp_path, source)
    with pytest.raises(IndexBuildError, match="zero passages"):
        build_index(config, embedder=FakeEmbedder(dimension=8, batch_size=2), log=lambda _: None)
    assert not config.index_dir.exists()


def test_corrupted_index_rejected(tmp_path):
    source = _write_raw_parquet(tmp_path / "raw.parquet")
    result = build_index(_config(tmp_path, source), embedder=FakeEmbedder(dimension=8, batch_size=2), log=lambda _: None)
    (result.index_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexCompatibilityError):
        load_index(result.index_dir)


def test_deterministic_build_chunk_ids(tmp_path):
    source = _write_raw_parquet(tmp_path / "raw.parquet", count=2)
    first = build_index(
        _config(tmp_path / "a", source, processed_output=tmp_path / "a" / "processed.parquet", index_dir=tmp_path / "a" / "index"),
        embedder=FakeEmbedder(dimension=8, batch_size=2),
        log=lambda _: None,
    )
    second = build_index(
        _config(tmp_path / "b", source, processed_output=tmp_path / "b" / "processed.parquet", index_dir=tmp_path / "b" / "index"),
        embedder=FakeEmbedder(dimension=8, batch_size=2),
        log=lambda _: None,
    )
    ids1 = json.loads((first.index_dir / CHUNKS_INDEX_FILENAME).read_text(encoding="utf-8"))
    ids2 = json.loads((second.index_dir / CHUNKS_INDEX_FILENAME).read_text(encoding="utf-8"))
    assert list(ids1) == list(ids2)


def test_failed_build_preserves_old_index(tmp_path):
    source = _write_raw_parquet(tmp_path / "raw.parquet", count=2)
    config = _config(tmp_path, source)
    first = build_index(config, embedder=FakeEmbedder(dimension=8, batch_size=2), log=lambda _: None)
    old_ids = json.loads((first.index_dir / CHUNKS_INDEX_FILENAME).read_text(encoding="utf-8"))

    class FailingEmbedder(FakeEmbedder):
        def encode_batch(self, texts):
            raise RuntimeError("forced embedding failure")

    with pytest.raises(Exception, match="forced embedding failure"):
        build_index(
            _config(tmp_path, source, overwrite=True),
            embedder=FailingEmbedder(dimension=8, batch_size=2),
            log=lambda _: None,
        )
    assert json.loads((first.index_dir / CHUNKS_INDEX_FILENAME).read_text(encoding="utf-8")) == old_ids
    store, resolver, _ = load_index(first.index_dir, expected_model_name="test-fake-model", expected_backend="numpy")
    assert store.count == resolver.count == 2


def test_overwrite_protection_and_successful_overwrite(tmp_path):
    source_one = _write_raw_parquet(tmp_path / "one.parquet", count=1)
    source_two = _write_raw_parquet(tmp_path / "two.parquet", count=2)
    config = _config(tmp_path, source_one)
    build_index(config, embedder=FakeEmbedder(dimension=8, batch_size=2), log=lambda _: None)

    with pytest.raises(IndexBuildError, match="already exists"):
        build_index(_config(tmp_path, source_two), embedder=FakeEmbedder(dimension=8, batch_size=2), log=lambda _: None)

    result = build_index(
        _config(tmp_path, source_two, overwrite=True),
        embedder=FakeEmbedder(dimension=8, batch_size=2),
        log=lambda _: None,
    )
    assert result.manifest.counts.chunks == 2
    assert load_index(result.index_dir, expected_model_name="test-fake-model", expected_backend="numpy")[0].count == 2


def test_limit_behavior(tmp_path):
    source = _write_raw_parquet(tmp_path / "raw.parquet", count=5)
    result = build_index(
        _config(tmp_path, source, limit=2),
        embedder=FakeEmbedder(dimension=8, batch_size=2),
        log=lambda _: None,
    )
    assert result.statistics.passages == 2
    assert result.statistics.chunks == 2
    assert result.statistics.vectors_indexed == 2
