"""
Common chunk schema shared by both Hadith and Tafsir ingestion.

Every chunk written to data/processed/*.json conforms to this shape:

{
    "text": str,                     # primary text used for embedding (Phase 2)
    "metadata": {
        "source": "hadith" | "tafsir",
        "book": str,                 # e.g. "Sunan Abu Dawud" or "Bayan-ul-Quran"
        "surah": int | None,
        "ayah_range": [int, int] | None,

        # hadith-specific
        "hadith_number": int | None,
        "book_number": int | None,
        "hadith_in_book": int | None,
        "section_name": str | None,
        "grade": str | None,             # one summary grading, picked from `grades`
        "grades_detail": str | None,     # full grades dict, JSON-encoded (see note below)
        "arabic_text": str | None,
        "english_translation": str | None,

        # tafsir-specific
        "volume": int | None,
        "page": int | None,
    }
}

Why `grades_detail` is a JSON STRING and not a nested dict:
ChromaDB metadata (used in Phase 2) only accepts flat scalar values --
str, int, float, bool. A nested dict like the raw `grades` object would
break ingestion later. So the full multi-scholar grading is preserved
as a JSON string here (still readable for display/citation), while
`grade` holds one picked-out summary value for quick filtering/display.

Keeping every field present (even as None) means downstream code never
has to branch on "does this key exist" -- only on "is this value set".
"""

from typing import Optional, List, Dict, Any

REQUIRED_METADATA_FIELDS = [
    "source", "book", "surah", "ayah_range",
    "hadith_number", "book_number", "hadith_in_book", "section_name",
    "grade", "grades_detail", "arabic_text", "english_translation",
    "volume", "page",
]


def make_chunk(
    text: str,
    source: str,
    book: str,
    surah: Optional[int] = None,
    ayah_range: Optional[List[int]] = None,
    hadith_number: Optional[int] = None,
    book_number: Optional[int] = None,
    hadith_in_book: Optional[int] = None,
    section_name: Optional[str] = None,
    grade: Optional[str] = None,
    grades_detail: Optional[str] = None,
    arabic_text: Optional[str] = None,
    english_translation: Optional[str] = None,
    volume: Optional[int] = None,
    page: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a chunk dict that conforms to the common schema."""
    return {
        "text": text,
        "metadata": {
            "source": source,
            "book": book,
            "surah": surah,
            "ayah_range": ayah_range,
            "hadith_number": hadith_number,
            "book_number": book_number,
            "hadith_in_book": hadith_in_book,
            "section_name": section_name,
            "grade": grade,
            "grades_detail": grades_detail,
            "arabic_text": arabic_text,
            "english_translation": english_translation,
            "volume": volume,
            "page": page,
        },
    }


def validate_chunk(chunk: Dict[str, Any]) -> List[str]:
    """
    Return a list of problems found in a chunk. Empty list = valid.
    Does not raise, so callers can collect every problem in a file
    before deciding what to do (fail loudly vs. just log and skip).
    """
    problems = []

    if "text" not in chunk:
        problems.append("missing 'text' key")
    elif not isinstance(chunk["text"], str) or not chunk["text"].strip():
        problems.append("'text' is empty or not a string")

    if "metadata" not in chunk:
        problems.append("missing 'metadata' key")
        return problems

    meta = chunk["metadata"]
    for field in REQUIRED_METADATA_FIELDS:
        if field not in meta:
            problems.append(f"metadata missing '{field}'")

    if meta.get("source") not in ("hadith", "tafsir"):
        problems.append(f"metadata.source must be 'hadith' or 'tafsir', got {meta.get('source')!r}")

    if not meta.get("book"):
        problems.append("metadata.book is empty")

    if meta.get("source") == "hadith" and meta.get("hadith_number") is None:
        problems.append("hadith chunk missing hadith_number")

    if meta.get("source") == "tafsir" and meta.get("surah") is None:
        problems.append("tafsir chunk missing surah")

    return problems
