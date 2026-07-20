"""
Central configuration, loaded once from the .env file.

Every other module imports `cfg` from here, so there is a single source
of truth and nothing is hard-coded. Relative paths are resolved against
the project root so the pipeline runs from any working directory.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # project root (this file lives in pipeline/)
load_dotenv(dotenv_path=ROOT / ".env")


def _bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _path(value: str, default: str) -> Path:
    p = Path(os.getenv(value, default))
    return p if p.is_absolute() else (ROOT / p)


def _list(value: str, default: str) -> list:
    return [s.strip() for s in os.getenv(value, default).split(",") if s.strip()]


class Config:
    # ── data ──
    DATA_DIR        = _path("DATA_DIR", "data")
    PROCESSED_DIR   = _path("PROCESSED_DIR", "data/processed")

    # ── model ──
    EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DEVICE     = os.getenv("EMBEDDING_DEVICE", "auto").lower()
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    NORMALIZE_EMBEDDINGS = _bool(os.getenv("NORMALIZE_EMBEDDINGS"), True)
    # fp16 on GPU: ~3x faster, half the VRAM, negligible effect on cosine
    # similarity retrieval quality. Ignored on CPU (no benefit there).
    EMBEDDING_FP16       = _bool(os.getenv("EMBEDDING_FP16"), True)

    # ── chunking ──
    MAX_CHUNK_TOKENS     = int(os.getenv("MAX_CHUNK_TOKENS", "400"))
    CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))
    MIN_CHUNK_CHARS      = int(os.getenv("MIN_CHUNK_CHARS", "15"))

    # ── vector store ──
    CHROMA_DB_PATH    = _path("CHROMA_DB_PATH", "data/chroma_db")
    HADITH_COLLECTION = os.getenv("HADITH_COLLECTION", "hadith_collection")
    TAFSIR_COLLECTION = os.getenv("TAFSIR_COLLECTION", "tafsir_collection")
    DISTANCE_METRIC   = os.getenv("DISTANCE_METRIC", "cosine")

    # ── retrieval quality gate ──
    # Dense hits below this cosine similarity are dropped before fusion --
    # calibrated against BAAI/bge-m3: on-topic Islamic queries scored
    # 0.54-0.68, off-topic/gibberish queries scored 0.38-0.46. Without this,
    # every query (even gibberish) returns "confident-looking" top-k chunks
    # because RRF rank fusion has no notion of absolute relevance.
    MIN_DENSE_SCORE = float(os.getenv("MIN_DENSE_SCORE", "0.35"))

    # ── hadith ──
    HADITH_PRIMARY_LANG = os.getenv("HADITH_PRIMARY_LANG", "urdu").lower()
    GRADE_PRIORITY      = _list("GRADE_PRIORITY", "Al-Albani,Zubair Ali Zai,Shuaib Al Arnaut")

    def is_e5(self) -> bool:
        """e5 models need 'query:' / 'passage:' prefixes; bge-m3 does not."""
        return "e5" in self.EMBEDDING_MODEL.lower()


cfg = Config()
