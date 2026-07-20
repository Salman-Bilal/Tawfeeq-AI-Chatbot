"""
Auto-discovery and normalization of raw data into a common document shape.

The pipeline is designed so that "more data like this" just works: drop a
new hadith JSON array or a new tafsir book folder into DATA_DIR and it is
picked up automatically — no config file to edit.

DISCOVERY RULES
  * tafsir book  = any folder containing surah_*.json files
  * hadith book  = any *.json that is a flat array of hadith-shaped records

NORMALIZED DOCUMENT (before chunking):
  {
    "text":   <str>,            # the full text to be chunked + embedded
    "source": "hadith"|"tafsir",
    "book":   <str>,
    "metadata": { ... source-specific fields ... }
  }
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

# Pretty names for known books (substring match on file/folder name).
# Unknown books fall back to a humanized version of the file/folder name.
HADITH_BOOK_NAMES = {
    "abudawud": "Sunan Abu Dawud", "abu_dawud": "Sunan Abu Dawud",
    "bukhari": "Sahih al-Bukhari",
    "muslim": "Sahih Muslim",
    "tirmidhi": "Jami` at-Tirmidhi",
    "nasai": "Sunan an-Nasa'i", "nasa": "Sunan an-Nasa'i",
    "ibnmajah": "Sunan Ibn Majah", "ibn_majah": "Sunan Ibn Majah",
    "malik": "Muwatta Malik", "muwatta": "Muwatta Malik",
    "ahmad": "Musnad Ahmad",
}
TAFSIR_BOOK_NAMES = {
    "bayan_ul_quran": "Bayan-ul-Quran", "bayan": "Bayan-ul-Quran",
    "ibn_kathir": "Tafsir Ibn Kathir", "ibnkathir": "Tafsir Ibn Kathir",
    "maarif": "Maarif-ul-Quran",
    "tafheem": "Tafhim-ul-Quran", "tafhim": "Tafhim-ul-Quran",
    "jalalayn": "Tafsir al-Jalalayn",
}


def _match_name(name: str, mapping: Dict[str, str]):
    low = name.lower()
    for key, val in mapping.items():
        if key in low:
            return val
    return None


# Words to drop when humanizing an unknown file/folder name. Matched per
# underscore/hyphen-separated token (not via \b, since underscore counts as
# a word character in regex and would never actually split "en_tafsir_x").
_STRIP_WORDS = {"complete", "data", "tafsir", "tafseer", "ur", "en", "ar"}


def _humanize(name: str) -> str:
    s = re.sub(r"\.json$", "", name, flags=re.I)
    tokens = [t for t in re.split(r"[_\-]+", s) if t and t.lower() not in _STRIP_WORDS]
    return " ".join(tokens).title() or name


def guess_hadith_book(stem: str) -> str:
    return _match_name(stem, HADITH_BOOK_NAMES) or _humanize(stem)


def guess_tafsir_book(name: str) -> str:
    return _match_name(name, TAFSIR_BOOK_NAMES) or _humanize(name)


def _load_json_maybe(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _config_book_map(data_dir: Path, config_filename: str, key_field: str) -> Dict[str, str]:
    """
    Look for a `{file|folder: ..., book: ...}` config JSON anywhere under
    data_dir (e.g. raw_hadiths/books_config.json, raw_tafseer/tafseer_books_config.json)
    and return {stem/foldername.lower(): book_name}.

    These curated configs sit right next to the raw data and are the
    authoritative source for book names -- using them avoids heuristic
    collisions (e.g. "ibn_kathir_english" and "ur_ibn_kathir" both containing
    "ibn_kathir" and getting merged into one indistinguishable book name).
    """
    matches = list(Path(data_dir).rglob(config_filename))
    if not matches:
        return {}
    data = _load_json_maybe(matches[0])
    if not isinstance(data, list):
        return {}
    out = {}
    for entry in data:
        key = entry.get(key_field)
        book = entry.get("book")
        if key and book:
            stem = Path(key).stem if key_field == "file" else key
            out[stem.lower()] = book
    return out


def _surah_sort_key(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def _looks_like_hadith(rec: dict) -> bool:
    if not isinstance(rec, dict):
        return False
    keys = set(rec.keys())
    return "hadith_number" in keys or (
        "arabic_text" in keys and ({"urdu_translation", "english_translation"} & keys)
    )


def discover_sources(data_dir: Path) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """Return (hadith_files, tafsir_books) as lists of (path, book_name)."""
    data_dir = Path(data_dir)
    tafsir_books: List[Tuple[Path, str]] = []
    hadith_files: List[Tuple[Path, str]] = []

    hadith_cfg = _config_book_map(data_dir, "books_config.json", "file")
    tafsir_cfg = _config_book_map(data_dir, "tafseer_books_config.json", "folder")

    # tafsir: any directory (incl. data_dir itself) holding surah_*.json
    candidate_dirs = [data_dir] + [p for p in data_dir.rglob("*") if p.is_dir()]
    for d in sorted(set(candidate_dirs)):
        if any(d.glob("surah_*.json")):
            book = tafsir_cfg.get(d.name.lower()) or guess_tafsir_book(d.name)
            tafsir_books.append((d, book))

    # hadith: flat-array json files that are not surah_*.json
    for f in sorted(data_dir.rglob("*.json")):
        if f.name.lower().startswith("surah_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list) and data and _looks_like_hadith(data[0]):
            book = hadith_cfg.get(f.stem.lower()) or guess_hadith_book(f.stem)
            hadith_files.append((f, book))

    return hadith_files, tafsir_books


def _pick_grade(grades: dict, priority: List[str]) -> str:
    if not isinstance(grades, dict) or not grades:
        return ""
    for scholar in priority:
        if grades.get(scholar):
            return grades[scholar]
    # fall back to the first non-empty value
    for v in grades.values():
        if v:
            return v
    return ""


_LANG_FIELD = {"urdu": "urdu_translation", "english": "english_translation", "arabic": "arabic_text"}


def load_hadith_documents(file_path: Path, book: str, primary_lang: str, grade_priority: List[str]) -> Iterator[dict]:
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    primary_field = _LANG_FIELD.get(primary_lang, "urdu_translation")

    for rec in data:
        # primary text with a never-drop fallback chain
        text = (rec.get(primary_field) or "").strip()
        if not text:
            for fb in ("urdu_translation", "english_translation", "arabic_text"):
                text = (rec.get(fb) or "").strip()
                if text:
                    break
        if not text:
            continue

        grades = rec.get("grades") or {}
        yield {
            "text": text,
            "source": "hadith",
            "book": book,
            "metadata": {
                "hadith_number": rec.get("hadith_number"),
                "book_number": rec.get("book_number"),
                "hadith_in_book": rec.get("hadith_in_book"),
                "section_name": rec.get("section_name"),
                "grade": _pick_grade(grades, grade_priority),
                "grades_detail": json.dumps(grades, ensure_ascii=False) if grades else "",
                "arabic_text": rec.get("arabic_text"),
                "english_translation": rec.get("english_translation"),
                "urdu_translation": rec.get("urdu_translation"),
            },
        }


def load_tafsir_documents(folder: Path, book: str) -> Iterator[dict]:
    for f in sorted(Path(folder).glob("surah_*.json"), key=_surah_sort_key):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        records = data.get("ayahs", []) if isinstance(data, dict) else data
        if not isinstance(records, list):
            continue
        for rec in records:
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            yield {
                "text": text,
                "source": "tafsir",
                "book": book,
                "metadata": {
                    "surah": rec.get("surah"),
                    "ayah": rec.get("ayah"),
                },
            }
