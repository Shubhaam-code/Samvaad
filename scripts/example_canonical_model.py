"""Example script showing CanonicalPassage usage.

Phase 2.2: Dataset preprocessing model demonstration.
"""

import json
import sys

# Add backend to path
sys.path.insert(0, "backend")

from app.dataset.models import CanonicalPassage


def main():
    # Create an example canonical passage from MSMARCO-XI data
    passage = CanonicalPassage.from_msmarco_record(
        query_id=123456,
        query="भारत की राजधानी क्या है?",
        query_type="LOCATION",
        answer="नई दिल्ली",
        source_lang="en",
        target_lang="hi",
        eng_query="What is the capital of India?",
        eng_answer="New Delhi",
        passage_index=0,
        translated_passage="भारत की राजधानी नई दिल्ली है। यह देश का सबसे बड़ा शहर है और राजनीतिक केंद्र के रूप में कार्य करता है।",
        english_passage="The capital of India is New Delhi. It is the largest city in the country and serves as the political center.",
        is_selected=True,
    )

    print("Example CanonicalPassage Record:")
    print("=" * 70)
    print(json.dumps(passage.to_dict(), indent=2, ensure_ascii=False))
    print("=" * 70)
    print(f"\nString representation:\n{passage}")
    print("\nKey properties:")
    print(f"  - Document ID (first 32 chars): {passage.document_id[:32]}...")
    print(f"  - Query ID: {passage.query_id}")
    print(f"  - Passage Index: {passage.passage_index}")
    print(f"  - Is Selected: {passage.is_selected}")
    print(f"  - Language: {passage.target_lang}")
    print(f"  - Query length: {len(passage.query)} chars")
    print(f"  - Passage length: {len(passage.translated_passage)} chars")


if __name__ == "__main__":
    main()
