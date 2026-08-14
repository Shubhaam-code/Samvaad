"""Dataset loading utilities for MSMARCO-XI.

Wraps ``datasets.load_dataset`` so we have a single place to control:
  * which config (language code) we use
  * streaming vs. in-memory
  * error handling and split enumeration
"""

from __future__ import annotations

import logging

from datasets import DatasetDict, load_dataset


logger = logging.getLogger(__name__)


# MSMARCO-XI exposes a single builder with language-coded configs.
DATASET_NAME = "ai4bharat/MSMARCO-XI"


def list_splits(config_name: str = "default") -> list[str]:
    """Return the split names available for the given config.

    Loads only the split metadata (``streaming=True``), so this is cheap
    even on very large datasets.
    """
    try:
        # Load without specifying splits to discover them
        builder = load_dataset(
            DATASET_NAME,
            config_name,
            streaming=True,
        )
    except Exception as exc:  # pragma: no cover - network path
        raise RuntimeError(
            f"Failed to list splits for {DATASET_NAME!r} config={config_name!r}: {exc}"
        ) from exc

    if isinstance(builder, DatasetDict):
        names = list(builder.keys())
    else:
        # If we got a single IterableDataset, that means there's only one split
        # Try to get info from dataset info
        try:
            if hasattr(builder, 'info') and hasattr(builder.info, 'splits'):
                names = list(builder.info.splits.keys())
            else:
                # Fallback: assume canonical MSMARCO-XI splits
                names = ["train", "validation"]
        except Exception:
            names = ["train", "validation"]
    
    logger.info("Available splits for config=%s: %s", config_name, names)
    return names


def load_split(
    split: str,
    config_name: str = "default",
    streaming: bool = False,
) -> "datasets.Dataset | datasets.IterableDataset":
    """Load a single split of MSMARCO-XI.

    Args:
        split: Split name (e.g., ``"train"``).
        config_name: Dataset config (e.g., ``"default"``).
        streaming: If True, returns an ``IterableDataset`` to avoid pulling
            the full split into memory. Required for Phase 2.1 to keep
            RAM usage bounded on this 55 GB corpus.
    """
    try:
        ds = load_dataset(
            DATASET_NAME,
            config_name,
            split=split,
            streaming=streaming,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load split={split!r} config={config_name!r} "
            f"from {DATASET_NAME!r}: {exc}"
        ) from exc

    logger.info(
        "Loaded split=%s config=%s streaming=%s", split, config_name, streaming
    )
    return ds
