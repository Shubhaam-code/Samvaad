"""Standalone smoke test script for local vector store architecture (Phase 4.4).

Demonstrates and verifies:
1. Creation of synthetic L2-normalized vectors and VectorRecords
2. Vector insertion and 1:1 metadata mapping
3. Low-level k-nearest-neighbor search
4. Atomic persistence (save/load)
5. Search consistency after reload

Does NOT access MSMARCO-XI dataset or network.
"""

import math
import os
import shutil
import sys
from typing import List

# Ensure backend app is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.vectorstore import (
    FaissVectorStore,
    NumpyVectorStore,
    VectorRecord,
    VectorStoreError,
    create_vector_store,
    inspect_manifest,
    validate_index,
    delete_index,
    HAS_FAISS,
)


def normalize_vector(v: List[float]) -> List[float]:
    """Helper to L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        return v
    return [x / norm for x in v]


def run_smoke_test(store_type: str = "faiss") -> bool:
    print(f"\n--- Running VectorStore Smoke Test ({store_type.upper()}) ---")

    dim = 4
    # 1. Create synthetic normalized vectors
    raw_vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.7071, 0.7071, 0.0, 0.0],
        [0.0, 0.7071, 0.7071, 0.0],
    ]
    vectors = [normalize_vector(v) for v in raw_vectors]

    # 2. Create synthetic metadata records
    records = [
        VectorRecord(chunk_id=f"chunk_{i}", document_id=f"doc_{i}", chunk_index=i, target_lang="hi")
        for i in range(len(vectors))
    ]

    # 3. Instantiate store
    if store_type == "faiss" and not HAS_FAISS:
        print("FAISS not installed; skipping FAISS smoke test.")
        return True

    store = create_vector_store(dimension=dim, store_type=store_type, embedding_model_name="smoke-test-model")
    print(f"Initialized {store.__class__.__name__} (count={store.count}, dim={store.dimension})")

    # 4. Add vectors
    positions = store.add(vectors, records)
    print(f"Inserted {len(positions)} vectors. New count: {store.count}")
    assert store.count == 5, f"Expected 5 vectors, got {store.count}"

    # 5. Search
    query = normalize_vector([1.0, 0.1, 0.0, 0.0])
    top_k = 3
    pre_results = store.search(query, top_k=top_k)
    print(f"Top-{top_k} search results before save:")
    for res in pre_results:
        print(f"  Pos: {res.position}, ChunkID: {res.chunk_id}, Score: {res.score:.4f}")

    assert len(pre_results) == 3, f"Expected 3 results, got {len(pre_results)}"
    assert pre_results[0].chunk_id == "chunk_0", f"Expected closest chunk_0, got {pre_results[0].chunk_id}"

    # 6. Save
    save_path = os.path.abspath(os.path.join(".", f".tmp_smoke_index_{store_type}"))
    store.save(save_path, overwrite=True)
    print(f"Saved index to '{save_path}'")

    # 7. Phase 4.5 Inspection without loading vectors
    manifest = inspect_manifest(save_path)
    print(f"Inspected manifest: type={manifest.vector_store_type}, dim={manifest.dimension}, count={manifest.count}")
    assert manifest.dimension == dim
    assert manifest.count == 5

    assert validate_index(save_path, expected_dimension=dim, expected_model_name="smoke-test-model")

    # 8. Reload
    if store_type == "faiss":
        reloaded_store = FaissVectorStore.load(save_path, expected_dimension=dim, expected_model_name="smoke-test-model")
    else:
        reloaded_store = NumpyVectorStore.load(save_path, expected_dimension=dim, expected_model_name="smoke-test-model")
    print(f"Loaded index from '{save_path}'. Count={reloaded_store.count}, Dim={reloaded_store.dimension}")

    # 9. Search again on reloaded store
    post_results = reloaded_store.search(query, top_k=top_k)
    print(f"Top-{top_k} search results after reload:")
    for res in post_results:
        print(f"  Pos: {res.position}, ChunkID: {res.chunk_id}, Score: {res.score:.4f}")

    # 10. Verify consistency
    assert len(pre_results) == len(post_results)
    for pre, post in zip(pre_results, post_results):
        assert pre.chunk_id == post.chunk_id, f"Chunk ID mismatch: {pre.chunk_id} vs {post.chunk_id}"
        assert abs(pre.score - post.score) < 1e-5, f"Score mismatch: {pre.score} vs {post.score}"
        assert pre.position == post.position, f"Position mismatch: {pre.position} vs {post.position}"

    # 11. Controlled failure test (delete metadata file and assert error)
    meta_path = os.path.join(save_path, "metadata.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)
        try:
            if store_type == "faiss":
                FaissVectorStore.load(save_path)
            else:
                NumpyVectorStore.load(save_path)
            print("ERROR: Controlled failure test did NOT raise expected exception!")
            return False
        except (FileNotFoundError, VectorStoreError) as exc:
            print(f"Controlled failure test passed safely: {exc}")

    # Clean up using lifecycle delete_index
    delete_index(save_path, confirm=True)

    print(f"VectorStore Smoke Test ({store_type.upper()}): SUCCESS!\n")
    return True


if __name__ == "__main__":
    success_faiss = run_smoke_test("faiss")
    success_np = run_smoke_test("numpy")
    if success_faiss and success_np:
        sys.exit(0)
    else:
        sys.exit(1)
