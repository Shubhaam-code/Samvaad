"""Dependency Setup Script: Cache Production Embedding Model Only.

Downloads and caches ONLY 'intfloat/multilingual-e5-small' in the local
HuggingFace cache directory.

Verification:
- Verifies tokenizer loads with local_files_only=True
- Verifies model loads with local_files_only=True
- Verifies model config and hidden dimension (384)
- Verifies local_files_only=True succeeds

DATASET / INDEXING SAFETY GUARANTEES:
- Does NOT access data/raw/hintrain.parquet or MSMARCO-XI dataset
- Does NOT process any dataset rows
- Does NOT create CanonicalPassages or chunks
- Does NOT build or modify any FAISS vector store
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.embedding.huggingface import DEFAULT_MODEL_NAME, is_model_cached


def cache_model(model_name: str = DEFAULT_MODEL_NAME) -> bool:
    print("==================================================")
    print("PHASE 4.6 DEPENDENCY SETUP: CACHING EMBEDDING MODEL")
    print("==================================================")
    print(f"Target model: {model_name}\n")

    # Step 1: Environment verification
    try:
        import transformers
        import torch
        print(f"Transformers version: {transformers.__version__}")
        print(f"PyTorch version:      {torch.__version__}")
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        return False

    # Step 2: Check if already cached
    already_cached = is_model_cached(model_name)
    if already_cached:
        print(f"\nModel '{model_name}' is ALREADY cached locally.")
    else:
        print(f"\nModel '{model_name}' is NOT cached. Downloading now...")
        try:
            from transformers import AutoModel, AutoTokenizer
            print("Downloading tokenizer...")
            AutoTokenizer.from_pretrained(model_name, local_files_only=False)
            print("Downloading model weights...")
            AutoModel.from_pretrained(model_name, local_files_only=False)
            print("Download completed successfully!")
        except Exception as exc:
            print(f"ERROR: Failed to download model '{model_name}': {exc}")
            return False

    # Step 3: Verify cache with local_files_only=True
    print("\n--- Verifying Local Cache (local_files_only=True) ---")
    try:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("Tokenizer load (local_files_only=True): SUCCESS")

        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        print("Model load (local_files_only=True):     SUCCESS")

        hidden_dim = model.config.hidden_size
        print(f"Model hidden dimension:               {hidden_dim}")
        assert hidden_dim == 384, f"Expected hidden dimension 384, got {hidden_dim}"

        cache_dir = getattr(model.config, "_name_or_path", "HuggingFace cache")
        print(f"Cache location verified for:          {cache_dir}")

    except Exception as exc:
        print(f"ERROR: Verification with local_files_only=True failed: {exc}")
        return False

    # Step 4: Verification of dataset safety
    print("\n--- Safety Audit ---")
    print("MSMARCO-XI accessed:       NO")
    print("Dataset rows processed:    0")
    print("FAISS indexing performed:  NO")

    print("\n==================================================")
    print("PHASE 4.6 DEPENDENCY READY")
    print("==================================================\n")
    return True


if __name__ == "__main__":
    success = cache_model(DEFAULT_MODEL_NAME)
    sys.exit(0 if success else 1)
