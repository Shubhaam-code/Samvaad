"""Example script demonstrating ParquetBatchReader usage.

Phase 2.2.2: Batched Parquet reading infrastructure.
"""

import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Add backend to path
sys.path.insert(0, "backend")

from app.dataset.parquet_reader import ParquetBatchReader, read_parquet_batches


def create_sample_parquet():
    """Create a sample Parquet file for demonstration."""
    # Create nested structure similar to MSMARCO-XI
    schema = pa.schema([
        ("query_id", pa.int64()),
        ("query", pa.string()),
        ("answer", pa.string()),
        ("passages", pa.struct([
            ("translated_passages", pa.list_(pa.string())),
            ("english_passages", pa.list_(pa.string())),
            ("is_selected", pa.list_(pa.bool_())),
        ])),
        ("target_lang", pa.string()),
    ])
    
    # Create 15 sample rows
    data = {
        "query_id": list(range(1, 16)),
        "query": [f"प्रश्न {i}" for i in range(1, 16)],
        "answer": [f"उत्तर {i}" for i in range(1, 16)],
        "passages": [
            {
                "translated_passages": [f"अनुच्छेद {i}a", f"अनुच्छेद {i}b", f"अनुच्छेद {i}c"],
                "english_passages": [f"Passage {i}a", f"Passage {i}b", f"Passage {i}c"],
                "is_selected": [True, False, False],
            }
            for i in range(1, 16)
        ],
        "target_lang": ["hi"] * 15,
    }
    
    table = pa.Table.from_pydict(data, schema=schema)
    
    # Write to temporary file
    temp_dir = Path(tempfile.gettempdir()) / "msmarco_example"
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / "sample_data.parquet"
    pq.write_table(table, file_path)
    
    return file_path


def main():
    print("="*70)
    print("ParquetBatchReader Example")
    print("="*70)
    
    # Create sample file
    print("\n[1/4] Creating sample Parquet file...")
    sample_file = create_sample_parquet()
    print(f"      Created: {sample_file}")
    print(f"      Size: {sample_file.stat().st_size:,} bytes")
    
    # Initialize reader
    print("\n[2/4] Initializing ParquetBatchReader...")
    reader = ParquetBatchReader(sample_file, batch_size=5)
    print(f"      {reader}")
    print(f"      Schema columns: {reader.schema.names}")
    
    # Iterate through batches
    print("\n[3/4] Reading batches...")
    batch_count = 0
    total_rows = 0
    
    for batch in reader:
        batch_count += 1
        total_rows += len(batch)
        print(f"      Batch {batch_count}: {len(batch)} rows")
        
        # Show first row of first batch
        if batch_count == 1:
            first_row = batch.to_pylist()[0]
            print(f"      First query_id: {first_row['query_id']}")
            print(f"      First query: {first_row['query']}")
            print(f"      Passages in first row: {len(first_row['passages']['translated_passages'])}")
    
    print(f"\n      Total: {total_rows} rows in {batch_count} batches")
    
    # Demonstrate convenience function
    print("\n[4/4] Using convenience function...")
    batch_sizes = []
    for batch in read_parquet_batches(sample_file, batch_size=7):
        batch_sizes.append(len(batch))
    
    print(f"      Batch sizes with batch_size=7: {batch_sizes}")
    print(f"      Total batches: {len(batch_sizes)}")
    
    print("\n" + "="*70)
    print("Key Features Demonstrated:")
    print("  ✓ Memory-efficient batched reading")
    print("  ✓ Configurable batch size")
    print("  ✓ Preserves nested structures (passages)")
    print("  ✓ Iterator interface for easy processing")
    print("  ✓ Does not load entire dataset into memory")
    print("="*70)


if __name__ == "__main__":
    main()
