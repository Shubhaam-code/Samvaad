"""Tests for local vector store and index architecture (Phase 4.4).

Tests cover:
- Imports and factory creation
- Initialization and dimension property
- Single and batch vector insertion
- Metadata mapping and 1:1 position stability
- Dimension mismatch rejection
- Invalid/non-finite vector rejection (NaN, Inf, boolean, strings)
- Invalid metadata/records rejection
- Duplicate chunk ID support and positional stability
- Low-level k-nearest-neighbor search
- Score finiteness
- top_k validation (positive integer, top_k > count clamping)
- Empty-index search exception handling
- Persistence (save/load) with schema & JSON metadata sidecar
- Search equivalence pre-save vs post-reload
- Overwrite protection (FileExistsError when overwrite=False)
- Explicit overwrite (overwrite=True)
- Unicode metadata preservation
- No network access / No MSMARCO-XI data

Uses tiny synthetic vectors only.
"""

import math
import tempfile
import pytest

from app.vectorstore import (
    BaseVectorStore,
    FaissVectorStore,
    NumpyVectorStore,
    VectorRecord,
    VectorSearchResult,
    VectorStoreError,
    create_vector_store,
    HAS_FAISS,
)


# Helper to generate synthetic normalized vectors
def make_vector(dim: int = 4, val: float = 1.0, norm: bool = True) -> list[float]:
    v = [val] * dim
    if norm:
        l2 = math.sqrt(sum(x * x for x in v))
        return [x / l2 for x in v]
    return v


def make_record(
    chunk_id: str = "chunk_0",
    doc_id: str = "doc_0",
    idx: int = 0,
    lang: str = "hi",
) -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        document_id=doc_id,
        chunk_index=idx,
        target_lang=lang,
        source_lang="en",
        query_id=100,
        passage_index=idx,
        is_selected=True,
        extra_metadata={"custom_key": "custom_val"},
    )


# Parametrize over available vector store classes
STORE_CLASSES = [NumpyVectorStore]
if HAS_FAISS:
    STORE_CLASSES.append(FaissVectorStore)


class TestVectorStoreImportsAndFactory:
    """Test package exports and factory function."""

    def test_imports_exist(self):
        assert BaseVectorStore is not None
        assert VectorRecord is not None
        assert VectorSearchResult is not None
        assert VectorStoreError is not None

    def test_create_vector_store_numpy(self):
        store = create_vector_store(dimension=8, store_type="numpy")
        assert isinstance(store, NumpyVectorStore)
        assert store.dimension == 8
        assert store.count == 0

    @pytest.mark.skipif(not HAS_FAISS, reason="FAISS is not installed")
    def test_create_vector_store_faiss(self):
        store = create_vector_store(dimension=8, store_type="faiss")
        assert isinstance(store, FaissVectorStore)
        assert store.dimension == 8
        assert store.count == 0

    def test_create_vector_store_invalid_type(self):
        with pytest.raises(ValueError, match="Unsupported vector store type"):
            create_vector_store(dimension=4, store_type="invalid_store_type")

    def test_create_vector_store_invalid_dim(self):
        with pytest.raises(ValueError, match="dimension must be a positive integer"):
            create_vector_store(dimension=0)


@pytest.mark.parametrize("StoreClass", STORE_CLASSES)
class TestVectorStoreOperations:
    """Core vector store functionality tested across implementations."""

    def test_initialization_and_properties(self, StoreClass):
        store = StoreClass(dimension=4, embedding_model_name="test-model")
        assert store.dimension == 4
        assert store.count == 0
        assert store.embedding_model_name == "test-model"

    def test_add_single_vector(self, StoreClass):
        store = StoreClass(dimension=4)
        v = make_vector(4, val=1.0)
        r = make_record("c1", "d1", 0)

        positions = store.add([v], [r])
        assert positions == [0]
        assert store.count == 1

    def test_add_multiple_vectors(self, StoreClass):
        store = StoreClass(dimension=4)
        vectors = [make_vector(4, val=float(i + 1)) for i in range(3)]
        records = [make_record(f"c_{i}", f"d_{i}", i) for i in range(3)]

        positions = store.add(vectors, records)
        assert positions == [0, 1, 2]
        assert store.count == 3

    def test_add_dimension_mismatch_raises(self, StoreClass):
        store = StoreClass(dimension=4)
        wrong_v = [1.0, 2.0, 3.0]  # Dim 3 instead of 4
        r = make_record("c1", "d1", 0)

        with pytest.raises(ValueError, match="dimension"):
            store.add([wrong_v], [r])

    def test_add_non_finite_vector_raises(self, StoreClass):
        store = StoreClass(dimension=4)
        r = make_record("c1", "d1", 0)

        for bad_val in [float("nan"), float("inf"), float("-inf")]:
            bad_v = [1.0, bad_val, 0.0, 0.0]
            with pytest.raises(ValueError, match="non-finite"):
                store.add([bad_v], [r])

    def test_add_invalid_vector_type_raises(self, StoreClass):
        store = StoreClass(dimension=4)
        r = make_record("c1", "d1", 0)

        with pytest.raises(ValueError):
            store.add(["not a vector list"], [r])  # type: ignore

    def test_add_empty_vector_list_raises(self, StoreClass):
        store = StoreClass(dimension=4)
        with pytest.raises(ValueError, match="cannot be empty"):
            store.add([], [])

    def test_add_mismatched_records_count_raises(self, StoreClass):
        store = StoreClass(dimension=4)
        v = make_vector(4)
        with pytest.raises(ValueError, match="Record count"):
            store.add([v], [])

    def test_duplicate_chunk_ids_allowed_with_stable_positions(self, StoreClass):
        """Duplicate chunk_ids map to distinct 0-based positions."""
        store = StoreClass(dimension=4)
        v1 = make_vector(4, val=1.0)
        v2 = make_vector(4, val=2.0)
        r1 = make_record("dup_chunk", "d1", 0)
        r2 = make_record("dup_chunk", "d1", 1)

        pos1 = store.add([v1], [r1])
        pos2 = store.add([v2], [r2])

        assert pos1 == [0]
        assert pos2 == [1]
        assert store.count == 2

    def test_search_empty_index_raises(self, StoreClass):
        store = StoreClass(dimension=4)
        query = make_vector(4)
        with pytest.raises(VectorStoreError, match="empty vector store"):
            store.search(query, top_k=2)

    def test_search_invalid_top_k_raises(self, StoreClass):
        store = StoreClass(dimension=4)
        v = make_vector(4)
        r = make_record("c1", "d1", 0)
        store.add([v], [r])

        query = make_vector(4)
        for bad_k in [0, -1, -5]:
            with pytest.raises(ValueError, match="top_k must be positive"):
                store.search(query, top_k=bad_k)

    def test_search_dimension_mismatch_raises(self, StoreClass):
        store = StoreClass(dimension=4)
        store.add([make_vector(4)], [make_record("c1", "d1", 0)])

        bad_query = [1.0, 0.0, 0.0]  # Dim 3
        with pytest.raises(ValueError, match="dimension"):
            store.search(bad_query, top_k=1)

    def test_search_non_finite_query_raises(self, StoreClass):
        store = StoreClass(dimension=4)
        store.add([make_vector(4)], [make_record("c1", "d1", 0)])

        bad_query = [1.0, float("nan"), 0.0, 0.0]
        with pytest.raises(ValueError, match="non-finite"):
            store.search(bad_query, top_k=1)

    def test_search_populated_index(self, StoreClass):
        store = StoreClass(dimension=4)
        # Vector 0: aligned with [1, 0, 0, 0]
        # Vector 1: aligned with [0, 1, 0, 0]
        v0 = [1.0, 0.0, 0.0, 0.0]
        v1 = [0.0, 1.0, 0.0, 0.0]
        r0 = make_record("c0", "d0", 0)
        r1 = make_record("c1", "d1", 1)
        store.add([v0, v1], [r0, r1])

        # Query close to v0
        query = [0.9, 0.1, 0.0, 0.0]
        results = store.search(query, top_k=2)

        assert len(results) == 2
        assert isinstance(results[0], VectorSearchResult)
        assert results[0].chunk_id == "c0"
        assert results[0].position == 0
        assert math.isfinite(results[0].score)
        assert results[0].score > results[1].score

    def test_search_top_k_larger_than_count_returns_all(self, StoreClass):
        store = StoreClass(dimension=4)
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        records = [make_record("c0", "d0", 0), make_record("c1", "d1", 1)]
        store.add(vectors, records)

        # Ask for top_k=10 on store of size 2
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10)
        assert len(results) == 2

    def test_save_and_load_persistence(self, StoreClass):
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.abspath(os.path.join(tmp_dir, "test_store"))
            store = StoreClass(dimension=4, embedding_model_name="persisted-model")

            v0 = [1.0, 0.0, 0.0, 0.0]
            v1 = [0.0, 1.0, 0.0, 0.0]
            r0 = make_record("c0_unicode_हिंदी", "d0", 0)
            r1 = make_record("c1", "d1", 1)
            store.add([v0, v1], [r0, r1])

            # Pre-save search
            query = [1.0, 0.0, 0.0, 0.0]
            pre_results = store.search(query, top_k=2)

            # Save
            out_path = store.save(save_path, overwrite=False)
            assert out_path == save_path

            # Load
            reloaded = StoreClass.load(save_path)
            assert reloaded.dimension == 4
            assert reloaded.count == 2
            assert reloaded.embedding_model_name == "persisted-model"

            # Post-load search
            post_results = reloaded.search(query, top_k=2)
            assert len(post_results) == 2

            # Compare pre vs post
            for pre, post in zip(pre_results, post_results):
                assert pre.chunk_id == post.chunk_id
                assert pre.position == post.position
                assert abs(pre.score - post.score) < 1e-5
                assert pre.record.target_lang == post.record.target_lang
                assert pre.record.chunk_id == post.record.chunk_id

    def test_mixed_path_separators_regression(self, StoreClass):
        """Regression test for mixed slash path handling."""
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            mixed_path = f"{tmp_dir}/mixed_slash_store"
            store = StoreClass(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c0", "d0", 0)])

            out_path = store.save(mixed_path, overwrite=True)

            assert out_path == os.path.abspath(mixed_path)
            assert os.path.samefile(out_path, mixed_path)

            reloaded1 = StoreClass.load(mixed_path)
            reloaded2 = StoreClass.load(out_path)
            assert reloaded1.count == 1
            assert reloaded2.count == 1

    def test_overwrite_protection(self, StoreClass):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = f"{tmp_dir}/overwrite_test"
            store = StoreClass(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c0", "d0", 0)])

            # First save succeeds
            store.save(save_path, overwrite=False)

            # Second save without overwrite fails
            with pytest.raises(FileExistsError):
                store.save(save_path, overwrite=False)

            # Second save with overwrite=True succeeds
            store.save(save_path, overwrite=True)
            assert StoreClass.load(save_path).count == 1

    def test_unicode_metadata_preservation(self, StoreClass):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = f"{tmp_dir}/unicode_test"
            store = StoreClass(dimension=4)
            r = VectorRecord(
                chunk_id="हिंदी_चंक_ID_123",
                document_id="दस्तावेज़_456",
                chunk_index=0,
                target_lang="hi",
                extra_metadata={"language_name": "हिंदी", "script": "Devanagari"},
            )
            store.add([[1.0, 0.0, 0.0, 0.0]], [r])
            store.save(save_path, overwrite=True)

            loaded = StoreClass.load(save_path)
            res = loaded.search([1.0, 0.0, 0.0, 0.0], top_k=1)
            assert res[0].chunk_id == "हिंदी_चंक_ID_123"
            assert res[0].record.document_id == "दस्तावेज़_456"
            assert res[0].record.extra_metadata["language_name"] == "हिंदी"

    def test_dimension_compatibility_rejection(self, StoreClass):
        """Loading a 4-dim index with expected_dimension=8 must raise VectorStoreError."""
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "dim_test")
            store = StoreClass(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(save_path)

            with pytest.raises(VectorStoreError, match="Dimension mismatch"):
                StoreClass.load(save_path, expected_dimension=8)

    def test_embedding_model_compatibility_rejection(self, StoreClass):
        """Loading an index built with model A expecting model B must raise VectorStoreError."""
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "model_test")
            store = StoreClass(dimension=4, embedding_model_name="intfloat/multilingual-e5-small")
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(save_path)

            with pytest.raises(VectorStoreError, match="Embedding model mismatch"):
                StoreClass.load(save_path, expected_model_name="some-other-model")

            # Passing matching model name succeeds
            reloaded = StoreClass.load(save_path, expected_model_name="intfloat/multilingual-e5-small")
            assert reloaded.count == 1

    def test_normalization_expectation_in_manifest(self, StoreClass):
        """Normalization expectation property must be preserved in schema manifest."""
        import json, os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "norm_test")
            store = StoreClass(dimension=4)
            assert store.normalization_expectation == "l2_normalized"
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(save_path)

            with open(os.path.join(save_path, "schema.json"), "r", encoding="utf-8") as f:
                schema_data = json.load(f)

            assert schema_data.get("normalization_expectation") == "l2_normalized"

    def test_missing_schema_file_raises_file_not_found(self, StoreClass):
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "missing_schema")
            store = StoreClass(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(save_path)

            os.remove(os.path.join(save_path, "schema.json"))
            with pytest.raises(FileNotFoundError):
                StoreClass.load(save_path)

    def test_missing_metadata_file_raises_file_not_found(self, StoreClass):
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "missing_meta")
            store = StoreClass(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(save_path)

            os.remove(os.path.join(save_path, "metadata.json"))
            with pytest.raises(FileNotFoundError):
                StoreClass.load(save_path)

    def test_malformed_schema_json_raises_vectorstore_error(self, StoreClass):
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "bad_schema")
            store = StoreClass(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(save_path)

            with open(os.path.join(save_path, "schema.json"), "w", encoding="utf-8") as f:
                f.write("{ invalid json content }")

            with pytest.raises(VectorStoreError, match="Malformed JSON"):
                StoreClass.load(save_path)

    def test_malformed_metadata_json_raises_vectorstore_error(self, StoreClass):
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "bad_meta")
            store = StoreClass(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(save_path)

            with open(os.path.join(save_path, "metadata.json"), "w", encoding="utf-8") as f:
                f.write("[ corrupt json content ]")

            with pytest.raises(VectorStoreError, match="Malformed JSON"):
                StoreClass.load(save_path)

    def test_invalid_schema_version_raises_vectorstore_error(self, StoreClass):
        import json, os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "bad_ver")
            store = StoreClass(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(save_path)

            schema_file = os.path.join(save_path, "schema.json")
            with open(schema_file, "r") as f:
                data = json.load(f)
            data["schema_version"] = "99.0"
            with open(schema_file, "w") as f:
                json.dump(data, f)

            with pytest.raises(VectorStoreError, match="Unsupported vector store schema version"):
                StoreClass.load(save_path)

    def test_metadata_count_mismatch_raises_vectorstore_error(self, StoreClass):
        import json, os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "count_mismatch")
            store = StoreClass(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], [make_record("c0", "d0", 0), make_record("c1", "d1", 1)])
            store.save(save_path)

            # Corrupt metadata file by deleting one record
            meta_file = os.path.join(save_path, "metadata.json")
            with open(meta_file, "r") as f:
                data = json.load(f)
            data.pop()
            with open(meta_file, "w") as f:
                json.dump(data, f)

            with pytest.raises(VectorStoreError, match="Metadata record count"):
                StoreClass.load(save_path)

    def test_invalid_record_structure_raises_vectorstore_error(self, StoreClass):
        import json, os
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "bad_record")
            store = StoreClass(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(save_path)

            meta_file = os.path.join(save_path, "metadata.json")
            with open(meta_file, "w") as f:
                json.dump([{"invalid_field": "no chunk_id or document_id"}], f)

            with pytest.raises(VectorStoreError, match="Invalid VectorRecord"):
                StoreClass.load(save_path)


class TestVectorStoreLifecycleAPI:
    """Tests for lightweight lifecycle helpers (exists, inspect_manifest, validate_index, delete_index)."""

    def test_exists_and_inspect_manifest(self):
        import os
        from app.vectorstore.lifecycle import delete_index, exists, inspect_manifest, validate_index
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = os.path.join(tmp_dir, "lifecycle_store")
            assert not exists(index_path)

            store = NumpyVectorStore(dimension=4, embedding_model_name="lifecycle-model")
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(index_path)

            assert exists(index_path)

            manifest = inspect_manifest(index_path)
            assert manifest.dimension == 4
            assert manifest.count == 1
            assert manifest.embedding_model_name == "lifecycle-model"
            assert manifest.vector_store_type == "numpy"

            assert validate_index(index_path, expected_dimension=4, expected_model_name="lifecycle-model")

    def test_validate_index_raises_on_dimension_mismatch(self):
        import os
        from app.vectorstore.lifecycle import validate_index
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = os.path.join(tmp_dir, "val_dim_test")
            store = NumpyVectorStore(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(index_path)

            with pytest.raises(VectorStoreError, match="Dimension mismatch"):
                validate_index(index_path, expected_dimension=8)

    def test_delete_index_requires_confirm(self):
        import os
        from app.vectorstore.lifecycle import delete_index, exists
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = os.path.join(tmp_dir, "del_test")
            store = NumpyVectorStore(dimension=4)
            store.add([[1.0, 0.0, 0.0, 0.0]], [make_record("c1", "d1", 0)])
            store.save(index_path)

            assert exists(index_path)

            with pytest.raises(ValueError, match="requires explicit confirm=True"):
                delete_index(index_path, confirm=False)

            assert exists(index_path)
            assert delete_index(index_path, confirm=True) is True
            assert not exists(index_path)
