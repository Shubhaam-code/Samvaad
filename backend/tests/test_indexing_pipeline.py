"""Unit tests for MSMARCO-XI dataset indexer pipeline (Phase 5.1)."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.chunking.models import ChunkingStrategy
from app.dataset.indexer import DatasetIndexer, DatasetIndexerConfig, IndexingStatistics
from app.vectorstore.base import VectorSearchResult
from app.vectorstore.lifecycle import validate_index
from app.vectorstore.numpy_store import NumpyVectorStore


def create_synthetic_msmarco_record(query_id: int, lang: str = "hi") -> Dict[str, Any]:
    """Create a synthetic MSMARCO-XI record matching the official schema."""
    return {
        "query_id": query_id,
        "query": f"यह एक परीक्षण प्रश्न {query_id} है?",
        "query_type": "DESCRIPTION",
        "Answer": f"यह प्रश्न {query_id} का उत्तर है।",
        "source_lang": "eng_Latn",
        "target_lang": f"{lang}_Deva",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                f"This is the relevant passage for query {query_id}. It contains detailed facts.",
                f"This is a distractor passage for query {query_id} with irrelevant information.",
            ],
            "Translated_passages": [
                f"यह प्रश्न {query_id} के लिए प्रासंगिक अनुच्छेद है। इसमें विस्तृत तथ्य हैं।",
                f"यह प्रश्न {query_id} के लिए एक अप्रासंगिक अनुच्छेद है।",
            ],
        },
        "Eng_Query": f"Is this test query {query_id}?",
        "Eng_Answer": f"This is the answer for query {query_id}.",
    }


class TestDatasetIndexer:
    """Test suite for DatasetIndexer."""

    @pytest.fixture
    def temp_dir(self):
        tmp = tempfile.mkdtemp()
        yield Path(tmp)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_indexer_initialization_dry_run(self, temp_dir):
        config = DatasetIndexerConfig(
            lang="hi",
            split="validation",
            max_samples=10,
            batch_size=4,
            chunk_strategy=ChunkingStrategy.ADAPTIVE,
            store_type="numpy",
            output_dir=temp_dir,
            dry_run=True,
        )
        indexer = DatasetIndexer(config)
        assert indexer.dimension == 384
        assert isinstance(indexer.vector_store, NumpyVectorStore)
        assert indexer.config.dry_run is True

    def test_indexer_run_with_synthetic_records(self, temp_dir):
        config = DatasetIndexerConfig(
            lang="hi",
            split="validation",
            max_samples=5,
            batch_size=2,
            chunk_strategy=ChunkingStrategy.ADAPTIVE,
            store_type="numpy",
            output_dir=temp_dir,
            dry_run=True,
            overwrite=True,
        )
        indexer = DatasetIndexer(config)

        # Mock record streaming with 5 synthetic records (each has 2 passages = 10 passages)
        synthetic_records = [create_synthetic_msmarco_record(i) for i in range(1, 6)]
        with patch.object(indexer, "_iter_raw_records", return_value=iter(synthetic_records)):
            stats: IndexingStatistics = indexer.run()

        assert stats.raw_records_read == 5
        assert stats.canonical_passages_created == 10
        assert stats.chunks_created >= 10
        assert stats.vectors_indexed == stats.chunks_created
        assert stats.duration_seconds >= 0.0

        # Verify output directory contains index files
        assert (temp_dir / "vectors.npy").exists()
        assert (temp_dir / "metadata.json").exists()
        assert (temp_dir / "schema.json").exists()

        # Validate index
        is_valid = validate_index(temp_dir, expected_dimension=384)
        assert is_valid is True

    def test_indexer_persistence_and_searchability(self, temp_dir):
        config = DatasetIndexerConfig(
            lang="hi",
            split="validation",
            max_samples=3,
            batch_size=2,
            chunk_strategy=ChunkingStrategy.PASSAGE,
            store_type="numpy",
            output_dir=temp_dir,
            dry_run=True,
        )
        indexer = DatasetIndexer(config)

        synthetic_records = [create_synthetic_msmarco_record(i) for i in range(1, 4)]
        with patch.object(indexer, "_iter_raw_records", return_value=iter(synthetic_records)):
            indexer.run()

        # Reload store from disk
        loaded_store = NumpyVectorStore.load(temp_dir)
        assert loaded_store.count == 6  # 3 records * 2 passages

        # Perform test search with random query vector
        query_vector = indexer.embedder.encode("परीक्षण प्रश्न")
        results: List[VectorSearchResult] = loaded_store.search(query_vector, top_k=3)

        assert len(results) == 3
        assert results[0].record.target_lang.startswith("hi")
        assert results[0].record.extra_metadata is not None
        assert "chunk_text" in results[0].record.extra_metadata

    def test_indexer_deduplication(self, temp_dir):
        config = DatasetIndexerConfig(
            lang="hi",
            split="validation",
            max_samples=4,
            batch_size=2,
            chunk_strategy=ChunkingStrategy.PASSAGE,
            store_type="numpy",
            output_dir=temp_dir,
            dry_run=True,
        )
        indexer = DatasetIndexer(config)

        # Provide duplicate records with identical content
        rec1 = create_synthetic_msmarco_record(1)
        rec2 = create_synthetic_msmarco_record(1)  # Exact duplicate
        rec3 = create_synthetic_msmarco_record(2)

        with patch.object(indexer, "_iter_raw_records", return_value=iter([rec1, rec2, rec3])):
            stats = indexer.run()

        assert stats.raw_records_read == 3
        assert stats.duplicates_skipped == 2  # rec2's 2 passages are duplicates
        assert stats.canonical_passages_created == 4  # rec1 (2) + rec3 (2)

    def test_indexer_malformed_record_handling(self, temp_dir):
        config = DatasetIndexerConfig(
            lang="hi",
            split="validation",
            max_samples=5,
            batch_size=2,
            chunk_strategy=ChunkingStrategy.PASSAGE,
            store_type="numpy",
            output_dir=temp_dir,
            dry_run=True,
        )
        indexer = DatasetIndexer(config)

        # 1 good record, 1 malformed record (passages is None/corrupt), 1 good record
        rec_good1 = create_synthetic_msmarco_record(10)
        rec_corrupt = {"query_id": "bad", "passages": "not-a-dict"}
        rec_good2 = create_synthetic_msmarco_record(20)

        with patch.object(indexer, "_iter_raw_records", return_value=iter([rec_good1, rec_corrupt, rec_good2])):
            stats = indexer.run()

        # Corrupt record should be skipped without throwing unhandled exception
        assert stats.raw_records_read == 3
        assert stats.canonical_passages_created == 4
        assert stats.vectors_indexed == 4

    def test_indexer_progress_callback(self, temp_dir):
        config = DatasetIndexerConfig(
            lang="hi",
            split="validation",
            max_samples=2,
            batch_size=1,
            chunk_strategy=ChunkingStrategy.PASSAGE,
            store_type="numpy",
            output_dir=temp_dir,
            dry_run=True,
        )
        indexer = DatasetIndexer(config)

        progress_calls = []

        def mock_callback(indexed, total_chunks, elapsed):
            progress_calls.append((indexed, total_chunks, elapsed))

        synthetic_records = [create_synthetic_msmarco_record(1), create_synthetic_msmarco_record(2)]
        with patch.object(indexer, "_iter_raw_records", return_value=iter(synthetic_records)):
            indexer.run(progress_callback=mock_callback)

        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 4  # 4 vectors indexed
