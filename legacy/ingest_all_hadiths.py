"""
Phase 1.2 (batch) -- Ingest every hadith book in one run.

Reuses the exact same extract_record() logic from ingest_hadith.py, but
loops over multiple raw files using a small config that maps each file
to its book name (since the book name isn't inside the JSON itself).

STEP 1: create a config file, e.g. data/raw_hadiths/books_config.json:

[
  {"file": "sahih_bukhari.json",      "book": "Sahih al-Bukhari"},
  {"file": "sunan_abu_dawud.json",    "book": "Sunan Abu Dawud"},
  {"file": "jami_at_tirmidhi.json",   "book": "Jami` at-Tirmidhi"}
]

(file paths are relative to --raw-dir)

STEP 2: run:
    python ingest_all_hadiths.py \
        --config ../../data/raw_hadiths/books_config.json \
        --raw-dir ../../data/raw_hadiths \
        --output ../../data/processed/hadith_chunks.json \
        --primary-text urdu

This writes ONE combined hadith_chunks.json with chunks from every book
in the config -- ready to feed straight into merge_chunks.py alongside
your tafsir output. No need to run anything per-book by hand.
"""

import argparse
import json
from pathlib import Path

from schema import make_chunk, validate_chunk
from ingest_hadith import extract_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON file mapping each raw file to its book name")
    parser.add_argument("--raw-dir", required=True, help="Folder containing the raw hadith JSON files")
    parser.add_argument("--output", required=True, help="Path to write the combined processed chunks")
    parser.add_argument(
        "--primary-text",
        choices=["urdu", "english", "arabic"],
        default="urdu",
        help="Which translation becomes the embedded `text` field (default: urdu)",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(args.raw_dir)

    all_chunks = []
    grand_total_read = 0
    grand_total_skipped = 0

    for entry in config:
        file_name = entry["file"]
        book_name = entry["book"]
        file_path = raw_dir / file_name

        if not file_path.exists():
            print(f"WARNING: {file_path} not found -- skipping '{book_name}'")
            continue

        raw_data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, list):
            print(f"WARNING: {file_path} is not a JSON array -- skipping '{book_name}'")
            continue

        book_chunks = []
        book_problems = []

        for i, raw_record in enumerate(raw_data):
            fields = extract_record(raw_record, book_name=book_name, primary_text_field=args.primary_text)
            chunk = make_chunk(**fields)
            problems = validate_chunk(chunk)
            if problems:
                book_problems.append((i, problems))

                if len(book_problems) <= 20: 
                    # Get the actual hadith identifiers
                    h_num = raw_record.get("hadith_number")
                    sec_name = raw_record.get("section_name")
                    
                    # Fetch raw values exactly as they are in the JSON
                    u_txt = raw_record.get("urdu_translation")
                    e_txt = raw_record.get("english_translation")
                    a_txt = raw_record.get("arabic_text")

                    print(f"🔍 [DEBUG] FAILED RECORD DETAILS:")
                    print(f"   • Array Index: {i}")
                    print(f"   • Actual Hadith Number: {h_num}")
                    print(f"   • Section: {sec_name}")
                    print(f"   • Raw Urdu Type/Value: {type(u_txt).__name__} -> {repr(u_txt)}")
                    print(f"   • Raw English Type/Value: {type(e_txt).__name__} -> {repr(e_txt)}")
                    print(f"   • Raw Arabic Type/Value: {type(a_txt).__name__} -> {repr(a_txt)}")
                    print(f"   • Validation Failure Reason: {problems}\n")
            else:
                book_chunks.append(chunk)

        all_chunks.extend(book_chunks)
        grand_total_read += len(raw_data)
        grand_total_skipped += len(book_problems)

        print(f"{book_name}: read {len(raw_data)}, kept {len(book_chunks)}, skipped {len(book_problems)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nTOTAL: read {grand_total_read}, wrote {len(all_chunks)} chunks to {args.output}, skipped {grand_total_skipped}")


if __name__ == "__main__":
    main()
