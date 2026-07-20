"""
Phase 2.2 -- Build and save BM25 keyword index.

WHY BM25 IN ADDITION TO CHROMADB:
ChromaDB uses dense vector embeddings (semantic similarity). This works
well for meaning-based queries but can miss exact proper nouns -- narrator
names like "Abu Hurairah", Arabic terms like "Taqwa", hadith numbers, or
specific surah references. BM25 is a classical keyword-matching algorithm
that catches these exact matches that embeddings can miss.

Both indexes are used together in retriever.py via Reciprocal Rank Fusion.

WHAT THIS SCRIPT DOES:
  1. Reads all_chunks.json
  2. Tokenizes each chunk's text (whitespace split -- works for Urdu/Arabic/English)
  3. Builds a BM25 object over all chunks
  4. Saves the index + chunk list to disk as a pickle file

The saved file is loaded once at FastAPI startup and kept in memory.

Usage (from the project root):
    python pipeline/build_bm25_index.py \\
        --chunks data/processed/all_chunks.json \\
        --output data/bm25_index/bm25_index.pkl
"""

import argparse
import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> list:
    """
    Whitespace tokenizer.

    Urdu and Arabic are naturally whitespace-separated at the word level,
    so this is a reasonable baseline. If retrieval quality on exact Urdu
    terms feels poor in testing, a next step would be to try a proper
    Urdu tokenizer like urduhack or camel-tools for Arabic.
    """
    return text.lower().split()


def build_and_save(chunks: list, output_path) -> Path:
    """
    Build a BM25 index over `chunks` and save it (+ the chunks themselves,
    so the retriever can return full chunk dicts without a second lookup)
    to `output_path`. Used both by this script's CLI and by
    pipeline/app.py, which already has the chunk list in memory and would
    otherwise have to round-trip it through all_chunks.json for no reason.
    """
    output_path = Path(output_path)
    print(f"Tokenizing {len(chunks)} chunks...")
    corpus = [simple_tokenize(c["text"]) for c in chunks]

    print("Building BM25 index (this may take 1-2 minutes for large corpora)...")
    bm25 = BM25Okapi(corpus)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25":   bm25,
        "chunks": chunks,   # stored alongside so retriever can return full chunk
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    size_mb = output_path.stat().st_size / 1_000_000
    print(f"Saved BM25 index to {output_path}  ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True,
                        help="Path to all_chunks.json")
    parser.add_argument("--output", required=True,
                        help="Path to save the BM25 index pickle file")
    args = parser.parse_args()

    print("Loading chunks...")
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    print(f"  {len(chunks)} chunks loaded")

    build_and_save(chunks, args.output)
    print("Load in retriever with: pickle.load(open(path, 'rb'))")


if __name__ == "__main__":
    main()
