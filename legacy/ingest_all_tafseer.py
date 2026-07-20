"""
Phase 1.3 -- Tafsir batch ingestion (all books, all surahs, one run).

Your tafsir data is already in the ideal format:
  [{"ayah": 1, "surah": 1, "text": "..."}, ...]

Each record is exactly one ayah's commentary -- no chunking or merging
needed. This script loops over every book in the config, finds all
surah_N.json files in that book's folder (in surah order), and writes
one combined tafsir_chunks.json ready for Phase 2 embedding.

CONFIG FILE FORMAT (data/raw_tafseer/tafseer_books_config.json):
[
  {
    "folder": "en_tafsir_maarif_ul_quran",
    "book":   "Maarif-ul-Quran",
    "language": "urdu"
  },
  {
    "folder": "ur_ibn_kathir",
    "book":   "Tafsir Ibn Kathir (Urdu)",
    "language": "urdu"
  },
  ...
]

Usage:
    python ingest_all_tafseer.py \
        --config  ../../data/raw_tafseer/tafseer_books_config.json \
        --raw-dir ../../data/raw_tafseer \
        --output  ../../data/processed/tafseer_chunks.json
"""

import argparse
import json
import re
from pathlib import Path

from schema import make_chunk, validate_chunk


def surah_sort_key(path: Path) -> int:
    """
    Sort surah_1.json, surah_2.json ... surah_114.json numerically,
    not lexicographically (which would give 1, 10, 100, 11, ... instead
    of 1, 2, 3, ... 114).
    """
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def ingest_book(folder: Path, book_name: str) -> tuple:
    """
    Read all surah_N.json files in one book folder and return
    (chunks, skipped_count, problems_list).
    """
    surah_files = sorted(folder.glob("surah_*.json"), key=surah_sort_key)

    if not surah_files:
        return [], 0, [f"No surah_*.json files found in {folder}"]

    chunks = []
    skipped = 0
    problems = []

    for surah_file in surah_files:
        try:
            records = json.loads(surah_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            problems.append(f"{surah_file.name}: failed to parse -- {e}")
            continue

        # Some books wrap the array: {"ayahs": [...], ...}
        # Others are a bare array: [...]
        # Handle both transparently.
        if isinstance(records, dict):
            records = records.get("ayahs", [])
            if not isinstance(records, list):
                problems.append(f"{surah_file.name}: found a dict but no 'ayahs' key inside")
                continue

        for i, rec in enumerate(records):
            # ── field extraction ──────────────────────────────────────
            # All 7 of your books use the same three fields.
            # If you ever add a book with different keys, add a mapping here.
            text    = rec.get("text", "")
            surah   = rec.get("surah")
            ayah    = rec.get("ayah")

            chunk = make_chunk(
                text=text,
                source="tafsir",
                book=book_name,
                surah=surah,
                ayah_range=[ayah, ayah] if ayah is not None else None,
            )

            issues = validate_chunk(chunk)
            if issues:
                problems.append(f"{surah_file.name} record #{i}: {issues}")
                skipped += 1
            else:
                chunks.append(chunk)

    return chunks, skipped, problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True, help="Path to tafseer_books_config.json")
    parser.add_argument("--raw-dir", required=True, help="Parent folder containing all book subfolders")
    parser.add_argument("--output",  required=True, help="Path to write combined tafseer_chunks.json")
    args = parser.parse_args()

    config  = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(args.raw_dir)

    all_chunks      = []
    grand_read      = 0
    grand_skipped   = 0

    for entry in config:
        folder_name = entry["folder"]
        book_name   = entry["book"]
        folder_path = raw_dir / folder_name

        if not folder_path.exists():
            print(f"WARNING: folder not found -- {folder_path} (skipping '{book_name}')")
            continue

        chunks, skipped, problems = ingest_book(folder_path, book_name)

        book_read = len(chunks) + skipped
        grand_read    += book_read
        grand_skipped += skipped
        all_chunks.extend(chunks)

        status = f"  kept {len(chunks)}, skipped {skipped}"
        print(f"{book_name}: read {book_read} ayah records -- {status}")

        if problems:
            for p in problems[:5]:           # show first 5 problems per book
                print(f"    ! {p}")
            if len(problems) > 5:
                print(f"    ... and {len(problems) - 5} more problems")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nTOTAL: read {grand_read} records across {len(config)} books")
    print(f"Wrote {len(all_chunks)} valid chunks -> {args.output}")
    print(f"Skipped {grand_skipped} invalid records")


if __name__ == "__main__":
    main()