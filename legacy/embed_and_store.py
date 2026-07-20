"""
Phase 2.1 -- Embed all chunks and store in ChromaDB.

Reads data/processed/all_chunks.json and populates two persistent
ChromaDB collections:
  - hadith_collection
  - tafsir_collection

WHY TWO COLLECTIONS:
Keeping them separate lets you tune retrieval parameters per source
independently and makes per-source debugging much easier.

EMBEDDING MODEL:
intfloat/multilingual-e5-large is the recommended choice for Urdu +
Arabic text. It requires a specific prefix format:
  - Documents stored with prefix: "passage: <text>"
  - Queries at retrieval time with prefix: "query: <text>"
This is handled automatically here for storage. The retriever.py
script handles the query prefix at search time.

CHROMADB METADATA NOTE:
ChromaDB only accepts flat scalar metadata values (str, int, float,
bool). Two conversions happen at ingestion:
  - ayah_range [1, 7]  →  ayah_start=1, ayah_end=7  (two int fields)
  - None values        →  "" or -1  (ChromaDB rejects None)

all_chunks.json is NOT modified -- conversions happen only in memory.

Usage:
    python embed_and_store.py \\
        --chunks  ../../data/processed/all_chunks.json \\
        --db-path ../../data/chroma_db \\
        --model   intfloat/multilingual-e5-large \\
        --batch   64

    # Lighter model option if you have limited RAM:
    python embed_and_store.py \\
        --chunks  ../../data/processed/all_chunks.json \\
        --db-path ../../data/chroma_db \\
        --model   intfloat/multilingual-e5-base \\
        --batch   64
"""

import argparse
import json
import sys
from pathlib import Path


# ── metadata helpers ────────────────────────────────────────────────────────

def flatten_metadata(meta: dict) -> dict:
    """
    Convert a chunk's metadata dict into a ChromaDB-safe flat dict.

    Changes made:
      - ayah_range list  →  ayah_start + ayah_end  (two int fields)
      - None             →  "" for strings, -1 for ints
    """
    flat = {}

    # ── string fields (None → "")
    for key in ("source", "book", "section_name", "grade",
                "grades_detail", "arabic_text", "english_translation"):
        flat[key] = meta.get(key) or ""

    # ── int fields (None → -1)
    for key in ("surah", "hadith_number", "book_number",
                "hadith_in_book", "volume", "page"):
        val = meta.get(key)
        flat[key] = val if val is not None else -1

    # ── ayah_range list → two separate int fields
    ayah_range = meta.get("ayah_range")
    if ayah_range and isinstance(ayah_range, list) and len(ayah_range) == 2:
        flat["ayah_start"] = ayah_range[0]
        flat["ayah_end"]   = ayah_range[1]
    else:
        flat["ayah_start"] = -1
        flat["ayah_end"]   = -1

    return flat


def make_chunk_id(chunk: dict, index: int) -> str:
    """
    Build a stable, human-readable unique ID for a chunk.
    Format:
      hadith  →  hadith_sahihalbukhari_00402_00063
      tafsir  →  tafsir_maarifulquran_s001_a0001_00000

    `index` is the global position of the chunk across all batches.
    It is appended as a tiebreaker to guarantee uniqueness even when
    multiple chunks originate from the same hadith number or ayah.

    FIX 1: index appended to both hadith and tafsir IDs.
    FIX 2: ayah_range guarded against empty list (avoids IndexError).
    """
    meta   = chunk["metadata"]
    source = meta.get("source", "unknown")
    book   = (meta.get("book") or "unknown").lower()
    book   = "".join(c for c in book if c.isalnum())[:20]

    if source == "hadith":
        num = int(meta.get("hadith_number") or index)
        return f"hadith_{book}_{num:05d}_{index:05d}"
    else:
        surah      = int(meta.get("surah") or -1)
        ayah_range = meta.get("ayah_range") or []          # FIX 2: guard empty list
        ayah       = int(ayah_range[0]) if ayah_range else -1
        return f"tafsir_{book}_s{surah:03d}_a{ayah:04d}_{index:05d}"


# ── main ingestion ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks",  required=True,
                        help="Path to all_chunks.json")
    parser.add_argument("--db-path", required=True,
                        help="Directory where ChromaDB stores its files")
    parser.add_argument("--model",   default="intfloat/multilingual-e5-large",
                        help="SentenceTransformer model name")
    parser.add_argument("--batch",   type=int, default=64,
                        help="Embedding batch size (reduce if RAM is limited)")
    args = parser.parse_args()

    # ── 1. load chunks ───────────────────────────────────────────────
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    hadith_chunks = [c for c in chunks if c["metadata"]["source"] == "hadith"]
    tafsir_chunks = [c for c in chunks if c["metadata"]["source"] == "tafsir"]

    # Warn about chunks with unrecognised source (they are silently dropped above)
    skipped = [c for c in chunks
               if c["metadata"]["source"] not in ("hadith", "tafsir")]
    if skipped:
        print(f"  ⚠ WARNING: {len(skipped)} chunks have an unrecognised source "
              f"and will be skipped.")

    print(f"Loaded {len(chunks)} total chunks")
    print(f"  hadith : {len(hadith_chunks)}")
    print(f"  tafsir : {len(tafsir_chunks)}")

    # ── 2. load embedding model ──────────────────────────────────────
    print(f"\nLoading embedding model: {args.model}")
    print("(First run downloads the model -- may take a few minutes)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print("Model loaded.")

    # ── 3. connect to ChromaDB ───────────────────────────────────────
    import chromadb
    Path(args.db_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=args.db_path)

    # Delete existing collections so re-runs start clean.
    # Remove these two lines if you want to ADD to an existing DB.
    for name in ("hadith_collection", "tafsir_collection"):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    hadith_col = client.get_or_create_collection(
        name="hadith_collection",
        metadata={"hnsw:space": "cosine"}   # cosine similarity for embeddings
    )
    tafsir_col = client.get_or_create_collection(
        name="tafsir_collection",
        metadata={"hnsw:space": "cosine"}
    )

    # ── 4. embed and insert ──────────────────────────────────────────
    def insert_batch(collection, batch_chunks, batch_ids):
        texts = ["passage: " + c["text"] for c in batch_chunks]
        embeddings = model.encode(
            texts,
            batch_size=args.batch,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()

        collection.add(
            ids        = batch_ids,
            embeddings = embeddings,
            documents  = [c["text"] for c in batch_chunks],
            metadatas  = [flatten_metadata(c["metadata"]) for c in batch_chunks]
        )

    def embed_collection(collection, source_chunks, label):
        print(f"\nEmbedding {label} ({len(source_chunks)} chunks)...")
        BATCH = args.batch
        for start in range(0, len(source_chunks), BATCH):
            batch = source_chunks[start : start + BATCH]

            # FIX 1: use (start + i) so the index is globally unique
            # across ALL batches -- prevents duplicate IDs like
            # hadith_sahihalbukhari_00402 appearing twice in one batch.
            ids = [make_chunk_id(c, start + i) for i, c in enumerate(batch)]

            try:
                insert_batch(collection, batch, ids)
            except Exception as e:
                print(f"\n  ⚠ Batch at offset {start} failed: {e}")
                print(f"    Skipping {len(batch)} chunks and continuing...")

            done = min(start + BATCH, len(source_chunks))
            print(f"  {done}/{len(source_chunks)}", end="\r")

        print(f"  {len(source_chunks)}/{len(source_chunks)} -- done")

    embed_collection(hadith_col, hadith_chunks, "hadith")
    embed_collection(tafsir_col, tafsir_chunks, "tafsir")

    # ── 5. verify ────────────────────────────────────────────────────
    print(f"\nChromaDB counts:")
    print(f"  hadith_collection : {hadith_col.count()} documents")
    print(f"  tafsir_collection : {tafsir_col.count()} documents")
    print(f"\nChromaDB stored at: {args.db_path}")


if __name__ == "__main__":
    main()