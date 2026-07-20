"""
End-to-end embedding pipeline:  raw data  ->  chunks  ->  embeddings  ->  ChromaDB

Run (after `pip install -r requirements.txt`, from the project root):

    python pipeline/build_embeddings.py                 # full build
    python pipeline/build_embeddings.py --dry-run       # chunk + report only, no model download
    python pipeline/build_embeddings.py --limit 50      # quick smoke test on 50 docs/source

What it does:
  1. Auto-discovers every hadith file and tafsir book folder under DATA_DIR.
  2. Normalizes them to a common document shape.
  3. Splits long text into token-bounded, overlapping chunks (nothing truncated).
  4. Embeds each chunk with the configured multilingual model.
  5. Stores vectors + documents + metadata in two persistent ChromaDB
     collections (hadith / tafsir), using cosine similarity.
  6. Writes data/processed/all_chunks.json as a reusable artifact.

All knobs live in .env (see config.py).
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

from config import cfg
from chunking import chunk_text
from data_loaders import (
    discover_sources,
    load_hadith_documents,
    load_tafsir_documents,
)

# Force UTF-8 console output so Urdu/Arabic never crashes on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ChromaDB rejects None and non-scalar metadata. Ints get -1, strings get "".
_INT_FIELDS = {"surah", "ayah", "hadith_number", "book_number",
               "hadith_in_book", "chunk_index", "n_chunks"}
_CHROMA_ADD_BATCH = 512   # well under ChromaDB's per-call limit


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())[:24] or "unknown"


def flatten_metadata(md: dict) -> dict:
    """Make a metadata dict ChromaDB-safe (flat scalars, no None)."""
    flat = {}
    for k, v in md.items():
        if k in _INT_FIELDS:
            flat[k] = int(v) if v is not None else -1
        elif v is None:
            flat[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = json.dumps(v, ensure_ascii=False)
    return flat


def make_id(rec: dict) -> str:
    """Stable, content-addressed id (same text -> same id across rebuilds)."""
    md = rec["metadata"]
    book = _slug(rec["book"])
    h = hashlib.sha1(rec["text"].encode("utf-8")).hexdigest()[:10]
    ci = md.get("chunk_index", 0)
    if rec["source"] == "hadith":
        base = f"hadith_{book}_{md.get('hadith_number', 'x')}"
    else:
        base = f"tafsir_{book}_s{md.get('surah', 'x')}_a{md.get('ayah', 'x')}"
    return f"{base}_c{ci}_{h}"


# ── load + chunk ─────────────────────────────────────────────────────────────

def load_documents(limit=None):
    hadith_files, tafsir_books = discover_sources(cfg.DATA_DIR)

    print("Discovered sources:")
    for p, b in hadith_files:
        print(f"  [hadith] {b:<22} <- {p.name}")
    for p, b in tafsir_books:
        print(f"  [tafsir] {b:<22} <- {p.name}/")
    if not hadith_files and not tafsir_books:
        sys.exit(f"ERROR: no data found under {cfg.DATA_DIR}")

    docs = []
    for path, book in hadith_files:
        these = list(load_hadith_documents(path, book, cfg.HADITH_PRIMARY_LANG, cfg.GRADE_PRIORITY))
        docs.extend(these[:limit] if limit else these)
    for folder, book in tafsir_books:
        these = list(load_tafsir_documents(folder, book))
        docs.extend(these[:limit] if limit else these)
    return docs


def chunk_documents(docs, count_tokens):
    chunks = []
    for doc in tqdm(docs, desc="Chunking"):
        pieces = chunk_text(
            doc["text"], count_tokens,
            max_tokens=cfg.MAX_CHUNK_TOKENS,
            overlap_tokens=cfg.CHUNK_OVERLAP_TOKENS,
            min_chars=cfg.MIN_CHUNK_CHARS,
        )
        for idx, piece in enumerate(pieces):
            md = dict(doc["metadata"])
            md["chunk_index"] = idx
            md["n_chunks"] = len(pieces)
            chunks.append({
                "text": piece,
                "source": doc["source"],
                "book": doc["book"],
                "metadata": md,
            })
    return chunks


# ── embed + store ────────────────────────────────────────────────────────────

def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL)
    return lambda t: len(tok.encode(t, add_special_tokens=False))


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    device = cfg.EMBEDDING_DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model '{cfg.EMBEDDING_MODEL}' on {device} ...")
    print("(first run downloads the model -- this can take a few minutes)")
    model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=device)
    if device == "cuda" and cfg.EMBEDDING_FP16:
        model = model.half()
        print("Model loaded (fp16).")
    else:
        print("Model loaded.")
    return model


def to_passage(text: str) -> str:
    return f"passage: {text}" if cfg.is_e5() else text


def embed_into_collection(chunks, source, collection, model):
    subset = [c for c in chunks if c["source"] == source]
    if not subset:
        print(f"  ({source}: nothing to embed)")
        return

    # assign unique, content-stable ids (dedupe exact duplicates)
    seen, ordered = set(), []
    for c in subset:
        cid = make_id(c)
        if cid in seen:
            continue
        seen.add(cid)
        c["_id"] = cid
        ordered.append(c)

    print(f"\nEmbedding {source}: {len(ordered)} chunks ({len(subset) - len(ordered)} dupes skipped)")
    for start in tqdm(range(0, len(ordered), _CHROMA_ADD_BATCH), desc=f"  {source}"):
        batch = ordered[start:start + _CHROMA_ADD_BATCH]
        embeddings = model.encode(
            [to_passage(c["text"]) for c in batch],
            batch_size=cfg.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=cfg.NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            ids=[c["_id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[flatten_metadata({**c["metadata"], "source": c["source"], "book": c["book"]})
                       for c in batch],
        )


def open_collections():
    import chromadb
    cfg.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DB_PATH))
    for name in (cfg.HADITH_COLLECTION, cfg.TAFSIR_COLLECTION):
        try:
            client.delete_collection(name)   # clean rebuild
        except Exception:
            pass
    hcol = client.get_or_create_collection(cfg.HADITH_COLLECTION, metadata={"hnsw:space": cfg.DISTANCE_METRIC})
    tcol = client.get_or_create_collection(cfg.TAFSIR_COLLECTION, metadata={"hnsw:space": cfg.DISTANCE_METRIC})
    return hcol, tcol


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build embeddings and store in ChromaDB.")
    ap.add_argument("--dry-run", action="store_true", help="Chunk and report only; no model, no DB writes.")
    ap.add_argument("--limit", type=int, default=None, help="Only process N documents per source (smoke test).")
    args = ap.parse_args()

    t0 = time.time()

    docs = load_documents(limit=args.limit)
    n_had = sum(1 for d in docs if d["source"] == "hadith")
    n_taf = sum(1 for d in docs if d["source"] == "tafsir")
    print(f"\nLoaded {len(docs)} documents  (hadith={n_had}, tafsir={n_taf})")

    count_tokens = get_tokenizer()
    chunks = chunk_documents(docs, count_tokens)
    expansion = len(chunks) / max(len(docs), 1)
    print(f"Produced {len(chunks)} chunks  ({expansion:.2f} chunks/doc avg)")

    # always save the artifact (useful for inspection / a future BM25 index)
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.PROCESSED_DIR / "all_chunks.json"
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    if args.dry_run:
        print(f"\nDry run complete in {time.time() - t0:.1f}s. No embeddings created.")
        return

    model = load_model()
    hcol, tcol = open_collections()
    embed_into_collection(chunks, "hadith", hcol, model)
    embed_into_collection(chunks, "tafsir", tcol, model)

    print("\nChromaDB stored at:", cfg.CHROMA_DB_PATH)
    print(f"  {cfg.HADITH_COLLECTION}: {hcol.count()} vectors")
    print(f"  {cfg.TAFSIR_COLLECTION}: {tcol.count()} vectors")
    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
