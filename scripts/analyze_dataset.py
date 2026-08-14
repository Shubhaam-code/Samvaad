"""Analyze the MSMARCO-XI dataset using lightweight remote inspection.

Run from the repository root:

    python -m scripts.analyze_dataset
    # or
    python scripts/analyze_dataset.py

This script:
  * Lists repository files via HuggingFace Hub API (no download).
  * Discovers train/validation splits and language files.
  * Inspects Parquet metadata remotely via HTTP range requests.
  * Reads a tiny sample (max 20 rows) from one representative language.
  * Does NOT download the full multi-GB dataset files.
  * Writes ``data/dataset_report.json`` and ``data/dataset_report.md``.

Phase 2.1 only. No embeddings, no vector DB, no LLM, no voice, no UI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `backend.app` importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.dataset.remote_inspector import (  # noqa: E402
    DATASET_REPO,
    analyze_sample_rows,
    inspect_parquet_metadata,
    list_repository_files,
)


REPORT_JSON = REPO_ROOT / "data" / "dataset_report.json"
REPORT_MD = REPO_ROOT / "data" / "dataset_report.md"

# Maximum sample size for remote inspection
MAX_SAMPLE_ROWS = 20

# Representative language for detailed inspection (Hindi for Indian-language RAG)
REPRESENTATIVE_LANGUAGE = "hi"

logger = logging.getLogger("analyze_dataset")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(value, limit: int = 200) -> str:
    """Truncate long values for display."""
    s = str(value)
    return s if len(s) <= limit else s[:limit] + "..."


def _format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def analyze_dataset_remote(
    repo_id: str = DATASET_REPO,
    representative_lang: str = REPRESENTATIVE_LANGUAGE,
    max_sample: int = MAX_SAMPLE_ROWS,
) -> dict:
    """Perform lightweight remote analysis of the dataset.
    
    Returns a complete analysis dict suitable for JSON/Markdown reports.
    """
    print(f"\n[analyze] Starting remote inspection of {repo_id}", flush=True)
    print(f"[analyze] Representative language: {representative_lang}", flush=True)
    print(f"[analyze] Max sample size: {max_sample} rows\n", flush=True)
    
    # Step 1: List repository files
    print("[1/4] Listing repository files...", flush=True)
    try:
        repo_info = list_repository_files(repo_id)
    except Exception as exc:
        raise RuntimeError(f"Failed to list repository files: {exc}") from exc
    
    print(f"  ✓ Found {len(repo_info['train_files'])} train files", flush=True)
    print(f"  ✓ Found {len(repo_info['validation_files'])} validation files", flush=True)
    print(f"  ✓ Languages: {', '.join(repo_info['languages'])}\n", flush=True)
    
    # Step 2: Select representative files
    print(f"[2/4] Selecting representative language: {representative_lang}...", flush=True)
    
    train_file = None
    validation_file = None
    
    for f in repo_info['train_files']:
        if representative_lang in f:
            train_file = f
            break
    
    for f in repo_info['validation_files']:
        if representative_lang in f:
            validation_file = f
            break
    
    if not train_file:
        # Fallback to first train file if representative lang not found
        train_file = repo_info['train_files'][0] if repo_info['train_files'] else None
        print(f"  ⚠ Representative language not found, using: {train_file}", flush=True)
    else:
        print(f"  ✓ Train file: {train_file}", flush=True)
    
    if validation_file:
        print(f"  ✓ Validation file: {validation_file}\n", flush=True)
    else:
        print(f"  ⚠ No validation file found for {representative_lang}\n", flush=True)
    
    # Step 3: Inspect train file metadata and sample
    print("[3/4] Inspecting train file metadata and sampling rows...", flush=True)
    train_inspection = None
    if train_file:
        try:
            train_inspection = inspect_parquet_metadata(repo_id, train_file, max_sample)
            if train_inspection.get("error"):
                print(f"  ⚠ Error: {train_inspection['error']}", flush=True)
            else:
                print(f"  ✓ Columns: {len(train_inspection['column_names'])}", flush=True)
                print(f"  ✓ Total rows (metadata): {train_inspection['total_rows']:,}", flush=True)
                print(f"  ✓ Sampled rows: {train_inspection['sample_size']}\n", flush=True)
        except Exception as exc:
            print(f"  ✗ Failed to inspect train file: {exc}\n", flush=True)
            train_inspection = {"error": str(exc)}
    
    # Step 4: Inspect validation file metadata (optional)
    print("[4/4] Inspecting validation file metadata...", flush=True)
    validation_inspection = None
    if validation_file:
        try:
            validation_inspection = inspect_parquet_metadata(repo_id, validation_file, max_sample)
            if validation_inspection.get("error"):
                print(f"  ⚠ Error: {validation_inspection['error']}", flush=True)
            else:
                print(f"  ✓ Columns: {len(validation_inspection['column_names'])}", flush=True)
                print(f"  ✓ Total rows (metadata): {validation_inspection['total_rows']:,}", flush=True)
                print(f"  ✓ Sampled rows: {validation_inspection['sample_size']}\n", flush=True)
        except Exception as exc:
            print(f"  ⚠ Failed to inspect validation file: {exc}\n", flush=True)
            validation_inspection = {"error": str(exc)}
    
    # Analyze samples
    train_analysis = None
    validation_analysis = None
    
    if train_inspection and train_inspection.get("sample_rows"):
        train_analysis = analyze_sample_rows(train_inspection["sample_rows"])
    
    if validation_inspection and validation_inspection.get("sample_rows"):
        validation_analysis = analyze_sample_rows(validation_inspection["sample_rows"])
    
    # Build comprehensive report
    warnings = []
    
    if not train_file:
        warnings.append("No train file found in repository")
    if train_inspection and train_inspection.get("error"):
        warnings.append(f"Failed to inspect train file: {train_inspection['error']}")
    if train_analysis and train_analysis.get("duplicate_ids", 0) > 0:
        warnings.append(f"Found {train_analysis['duplicate_ids']} duplicate IDs in train sample")
    
    return {
        "dataset": repo_id,
        "generated_at": _now_iso(),
        "inspection_method": "remote_http_parquet_metadata",
        "representative_language": representative_lang,
        "max_sample_rows": max_sample,
        "repository": {
            "all_files_count": len(repo_info['files']),
            "train_files": repo_info['train_files'],
            "validation_files": repo_info['validation_files'],
            "languages": repo_info['languages'],
        },
        "train": {
            "file": train_file,
            "inspection": train_inspection,
            "analysis": train_analysis,
        } if train_file else None,
        "validation": {
            "file": validation_file,
            "inspection": validation_inspection,
            "analysis": validation_analysis,
        } if validation_file else None,
        "warnings": warnings,
    }


def build_markdown_report(payload: dict) -> str:
    """Generate human-readable markdown report."""
    lines: list[str] = []
    lines.append("# MSMARCO-XI Dataset Analysis Report")
    lines.append("")
    lines.append(f"- **Generated:** `{payload['generated_at']}`")
    lines.append(f"- **Dataset:** `{payload['dataset']}`")
    lines.append(f"- **Inspection method:** `{payload['inspection_method']}`")
    lines.append(f"- **Representative language:** `{payload['representative_language']}`")
    lines.append(f"- **Sample size:** {payload['max_sample_rows']} rows (per split)")
    lines.append("")
    lines.append("**Important:** This analysis uses remote HTTP range requests to inspect Parquet metadata")
    lines.append("and read tiny samples without downloading the full multi-GB dataset files.")
    lines.append("")
    
    # Repository overview
    lines.append("## Repository Overview")
    lines.append("")
    repo = payload["repository"]
    lines.append(f"- **Total files:** {repo['all_files_count']}")
    lines.append(f"- **Train files:** {len(repo['train_files'])}")
    lines.append(f"- **Validation files:** {len(repo['validation_files'])}")
    lines.append(f"- **Languages discovered:** {', '.join(repo['languages'])}")
    lines.append("")
    
    lines.append("### Train Files")
    lines.append("")
    for f in repo['train_files']:
        lines.append(f"- `{f}`")
    lines.append("")
    
    lines.append("### Validation Files")
    lines.append("")
    for f in repo['validation_files']:
        lines.append(f"- `{f}`")
    lines.append("")
    
    # Train split analysis
    if payload.get("train"):
        lines.append("## Train Split (Representative Language)")
        lines.append("")
        train = payload["train"]
        lines.append(f"**File:** `{train['file']}`")
        lines.append("")
        
        inspection = train.get("inspection")
        if inspection and not inspection.get("error"):
            lines.append(f"**Total rows (from metadata):** {inspection['total_rows']:,}")
            lines.append(f"**Columns:** {len(inspection['column_names'])}")
            lines.append(f"**Row groups:** {inspection['num_row_groups']}")
            lines.append(f"**Sample size:** {inspection['sample_size']} rows")
            lines.append("")
            
            lines.append("### Schema")
            lines.append("")
            for col, dtype in inspection['schema'].items():
                lines.append(f"- `{col}`: {dtype}")
            lines.append("")
            
            # Sample row
            if inspection['sample_rows']:
                lines.append("### Example Record (first row, truncated)")
                lines.append("")
                lines.append("```json")
                example = {k: _truncate(v) for k, v in inspection['sample_rows'][0].items()}
                lines.append(json.dumps(example, indent=2, ensure_ascii=False))
                lines.append("```")
                lines.append("")
            
            # Analysis
            analysis = train.get("analysis")
            if analysis:
                lines.append("### Field Roles (inferred from sample)")
                lines.append("")
                roles = analysis['field_roles']
                lines.append(f"- **Query fields:** {', '.join(roles['potential_query_fields']) if roles['potential_query_fields'] else '_(none)_'}")
                lines.append(f"- **Document/passage fields:** {', '.join(roles['potential_document_fields']) if roles['potential_document_fields'] else '_(none)_'}")
                lines.append(f"- **Metadata fields:** {', '.join(roles['potential_metadata_fields']) if roles['potential_metadata_fields'] else '_(none)_'}")
                lines.append("")
                
                lines.append("### Missing Values (in sample)")
                lines.append("")
                for field, count in analysis['missing_values'].items():
                    if count > 0:
                        lines.append(f"- `{field}`: {count} / {payload['max_sample_rows']}")
                lines.append("")
                
                lines.append("### Text Length Samples")
                lines.append("")
                for field, stats in analysis['text_length_samples'].items():
                    lines.append(f"- `{field}`: min={stats['min']}, max={stats['max']}, mean={stats['mean']} (n={stats['sample_size']})")
                lines.append("")
                
                if analysis.get('duplicate_ids', 0) > 0:
                    lines.append(f"**⚠ Warning:** Found {analysis['duplicate_ids']} duplicate IDs in sample")
                    lines.append("")
        elif inspection and inspection.get("error"):
            lines.append(f"**Error:** {inspection['error']}")
            lines.append("")
    
    # Validation split analysis
    if payload.get("validation"):
        lines.append("## Validation Split (Representative Language)")
        lines.append("")
        val = payload["validation"]
        lines.append(f"**File:** `{val['file']}`")
        lines.append("")
        
        inspection = val.get("inspection")
        if inspection and not inspection.get("error"):
            lines.append(f"**Total rows (from metadata):** {inspection['total_rows']:,}")
            lines.append(f"**Columns:** {len(inspection['column_names'])}")
            lines.append(f"**Sample size:** {inspection['sample_size']} rows")
            lines.append("")
        elif inspection and inspection.get("error"):
            lines.append(f"**Error:** {inspection['error']}")
            lines.append("")
    
    # Warnings
    if payload.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for w in payload["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    
    # Notes for chunking strategy
    lines.append("## Notes for Phase 2.2 (Chunking Strategy)")
    lines.append("")
    lines.append("Based on this lightweight analysis:")
    lines.append("")
    lines.append("1. The dataset contains nested `passages` structures (if present in schema)")
    lines.append("2. Query and answer fields are separate from passage fields")
    lines.append("3. Both English and translated (target language) versions are available")
    lines.append("4. Each query may have multiple passages with relevance labels")
    lines.append("5. Chunking strategy should preserve passage boundaries and relevance labels")
    lines.append("")
    
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze MSMARCO-XI using lightweight remote inspection (no full download)."
    )
    parser.add_argument(
        "--repo",
        default=DATASET_REPO,
        help=f"HuggingFace dataset repository (default: {DATASET_REPO})",
    )
    parser.add_argument(
        "--lang",
        default=REPRESENTATIVE_LANGUAGE,
        help=f"Representative language code (default: {REPRESENTATIVE_LANGUAGE})",
    )
    parser.add_argument(
        "--max-sample",
        type=int,
        default=MAX_SAMPLE_ROWS,
        help=f"Maximum sample rows (default: {MAX_SAMPLE_ROWS})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        payload = analyze_dataset_remote(
            repo_id=args.repo,
            representative_lang=args.lang,
            max_sample=args.max_sample,
        )
    except Exception as exc:
        print(f"\n[fatal] Analysis failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Write reports
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    REPORT_MD.write_text(build_markdown_report(payload))

    # Print summary
    print("\n" + "=" * 70)
    print("Dataset analysis complete.")
    print(f"\nRepository: {payload['dataset']}")
    print(f"Files discovered: {payload['repository']['all_files_count']}")
    print(f"Languages: {', '.join(payload['repository']['languages'])}")
    print(f"Representative language: {payload['representative_language']}")
    
    if payload.get('train') and payload['train'].get('inspection'):
        train_rows = payload['train']['inspection'].get('total_rows', 'unknown')
        print(f"\nTrain split: {train_rows} rows (metadata)")
    
    if payload.get('validation') and payload['validation'].get('inspection'):
        val_rows = payload['validation']['inspection'].get('total_rows', 'unknown')
        print(f"Validation split: {val_rows} rows (metadata)")
    
    print(f"\nReports:")
    print(f"  {REPORT_JSON.relative_to(REPO_ROOT)}")
    print(f"  {REPORT_MD.relative_to(REPO_ROOT)}")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
