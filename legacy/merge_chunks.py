"""
Phase 1 (final step) -- Combine all processed chunks into one file
for Phase 2 embedding.

Usage:
    python merge_chunks.py \
        --hadith ../../data/processed/hadith_chunks.json \
        --tafsir ../../data/processed/tafseer_chunks.json \
        --output ../../data/processed/all_chunks.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hadith", required=True)
    parser.add_argument("--tafsir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hadith_chunks = json.loads(Path(args.hadith).read_text(encoding="utf-8"))
    tafsir_chunks = json.loads(Path(args.tafsir).read_text(encoding="utf-8"))
    all_chunks    = hadith_chunks + tafsir_chunks

    # ── quick summary by source/book ──────────────────────────────
    from collections import Counter
    book_counts = Counter(c["metadata"]["book"] for c in all_chunks)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Hadith chunks : {len(hadith_chunks)}")
    print(f"Tafsir chunks : {len(tafsir_chunks)}")
    print(f"Total         : {len(all_chunks)} -> {args.output}")
    print("\nBreakdown by book:")
    for book, count in sorted(book_counts.items()):
        print(f"  {count:>6}  {book}")


if __name__ == "__main__":
    main()
