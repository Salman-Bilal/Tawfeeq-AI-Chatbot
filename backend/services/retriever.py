"""
backend/services/retriever.py

Hybrid retrieval service with Cross-Encoder Reranking.
Loads the ChromaDB collections, BM25 index, and a multilingual cross-encoder reranker
to maximize retrieval precision across English, Urdu, and Arabic text.
"""

import hashlib
import os
import pickle
import sys
from pathlib import Path
from typing import Literal

import chromadb
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

# Allow importing pipeline.config from the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.config import cfg

# Basic stopwords for English + Urdu query tokens.
STOPWORDS = {
    # English
    "what", "is", "the", "of", "in", "a", "an", "and", "or", "to", "for",
    "on", "at", "by", "with", "about", "are", "was", "were", "be", "been",
    "does", "do", "did", "how", "why", "when", "where", "who", "which",
    "this", "that", "these", "those", "it", "its", "as", "from",
    # Urdu
    "کیا", "کی", "کے", "کا", "میں", "سے", "اور", "یہ", "وہ", "ہے", "ہیں",
    "پر", "کو", "نے", "تک", "بھی",
}


def _clean_tokens(query: str) -> list[str]:
    """Lowercase, split, and drop stopwords before BM25 scoring."""
    raw = query.lower().split()
    cleaned = [t for t in raw if t not in STOPWORDS]
    return cleaned if cleaned else raw  # fallback: don't return empty


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion -- merges multiple ranked lists into one.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        rank = 0
        for doc_id in ranked:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class RetrieverService:

    def __init__(self):
        # ── embedding model ──────────────────────────────────────────
        print("[Retriever] Loading embedding model:", cfg.EMBEDDING_MODEL)
        self.model = SentenceTransformer(cfg.EMBEDDING_MODEL)
        self._is_e5 = cfg.is_e5()

        # ── Cross-Encoder Reranker 🧠 ────────────────────────────────
        print("[Retriever] Loading Cross-Encoder Reranker: BAAI/bge-reranker-v2-m3")
        
        # ✅ FIX #2: Explicitly set device to utilize GPU/CUDA acceleration if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Retriever] Reranker running on device: {device.upper()}")
        
        # ✅ FIX #2: Restricted max_length to 512 tokens to drop redundant compute frames
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=512)

        # ── ChromaDB ─────────────────────────────────────────────────
        db_path = str(cfg.CHROMA_DB_PATH)
        print(f"[Retriever] Connecting to ChromaDB at: {db_path}")
        client = chromadb.PersistentClient(path=db_path)
        self.hadith_col = client.get_collection(cfg.HADITH_COLLECTION)
        self.tafsir_col = client.get_collection(cfg.TAFSIR_COLLECTION)
        print(f"[Retriever] hadith_collection: {self.hadith_col.count()} vectors")
        print(f"[Retriever] tafsir_collection: {self.tafsir_col.count()} vectors")

        # ── BM25 ─────────────────────────────────────────────────────
        bm25_path = cfg.DATA_DIR / "bm25_index" / "bm25_index.pkl"
        print(f"[Retriever] Loading BM25 index from: {bm25_path}")
        payload          = pickle.load(open(bm25_path, "rb"))
        self.bm25        = payload["bm25"]
        self.bm25_chunks = payload["chunks"]
        print(f"[Retriever] BM25 ready: {len(self.bm25_chunks)} chunks indexed")

    # ── public API ───────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        source_filter: Literal["hadith", "tafsir", "both"] = "both",
        top_k: int = 5,
    ) -> list[dict]:
        """
        Hybrid search: dense + sparse fused via RRF, then re-ranked with a Cross-Encoder.
        """
        if not query or not query.strip():
            return []

        # 1. Fetch a larger pool candidate set (e.g., top 20) for the reranker to look through
        fetch_n = max(top_k * 4, 20)

        dense_results  = self._dense_search(query, source_filter, fetch_n)
        sparse_results = self._sparse_search(query, source_filter, fetch_n)

        # Apply dense quality gate threshold
        dense_results = [r for r in dense_results if r["score"] >= cfg.MIN_DENSE_SCORE]

        # Generate unique content keys to unify namespaces
        for r in dense_results + sparse_results:
            r["_key"] = hashlib.sha1(r["text"].encode("utf-8")).hexdigest()

        dense_keys  = [r["_key"] for r in dense_results]
        sparse_keys = [r["_key"] for r in sparse_results]

        # Combine matching channels using Reciprocal Rank Fusion
        fused = _rrf([dense_keys, sparse_keys])

        # Map unique hash keys to original objects (dense preferred for metadata)
        all_by_key: dict[str, dict] = {}
        for r in sparse_results + dense_results:
            all_by_key[r["_key"]] = r

        # Compile candidates passed to the CrossEncoder evaluation block
        rerank_candidates = []
        for key, rrf_score in fused[:fetch_n]:
            chunk = all_by_key.get(key)
            if chunk:
                # Save the structural RRF score to metadata to clear room for the true semantic score
                chunk["metadata"]["rrf_score"] = round(rrf_score, 6)
                rerank_candidates.append(chunk)

        # 2. Compute fine-grained cross-attention scores across language sets 📊
        if rerank_candidates:
            # ✅ FIX #2: Only pass clean text. Stripped the bloated c["metadata"] dictionary string expansion.
            pairs = [[query, c["text"]] for c in rerank_candidates]
            
            # ✅ FIX #2: Added explicit batching (batch_size=16) for uniform tensor calculation passes
            rerank_scores = self.reranker.predict(pairs, batch_size=16)
            
            for idx, score in enumerate(rerank_scores):
                rerank_candidates[idx]["score"] = round(float(score), 6)
            
            # Re-sort candidates purely on actual relevance values
            rerank_candidates.sort(key=lambda x: x["score"], reverse=True)

        # ✅ FIX #3: Gated stdout tracking behind an environmental DEBUG check to save console buffer latency
        if os.getenv("DEBUG_MODE") == "True":
            print("\n--- RERANK RESULTS ---")
            for c in rerank_candidates:
                print(
                    c["score"],
                    c["metadata"].get("book"),
                    c["metadata"].get("hadith_number"),
                    c["metadata"].get("grade")
                )

        # 3. Apply a post-rerank safety gate threshold to filter noise cliff 🚧
        MIN_RERANK_SCORE = 0.0
        filtered_candidates = [c for c in rerank_candidates if c["score"] >= MIN_RERANK_SCORE]

        # Clean and isolate the precise window slice requested from the filtered pool
        final = []
        for chunk in filtered_candidates[:top_k]:
            chunk.pop("_id", None)
            chunk.pop("_key", None)
            final.append(chunk)

        return final

    # ── dense (ChromaDB) ─────────────────────────────────────────────

    def _dense_search(self, query: str, source_filter: str, n: int) -> list[dict]:
        q = (f"query: {query}" if self._is_e5 else query)
        vec = self.model.encode(
            q,
            normalize_embeddings=cfg.NORMALIZE_EMBEDDINGS
        ).tolist()

        cols = self._collections(source_filter)
        results = []
        for col in cols:
            count = col.count()
            if count == 0:
                continue
            resp = col.query(
                query_embeddings=[vec],
                n_results=min(n, count),
                include=["documents", "metadatas", "distances"]
            )
            for doc_id, doc, meta, dist in zip(
                resp["ids"][0],
                resp["documents"][0],
                resp["metadatas"][0],
                resp["distances"][0],
            ):
                results.append({
                    "_id":     doc_id,
                    "text":    doc,
                    "metadata": self._restore_meta(meta),
                    "score":   round(1.0 - dist, 6),
                })
        return results

    # ── sparse (BM25) ────────────────────────────────────────────────

    def _sparse_search(self, query: str, source_filter: str, n: int) -> list[dict]:
        tokens = _clean_tokens(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:n]

        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score <= 0.0:
                continue

            chunk = self.bm25_chunks[idx]
            source = chunk.get("source", "")
            if source_filter != "both" and source != source_filter:
                continue
            meta = self._restore_meta(
                {**chunk["metadata"], "source": source, "book": chunk.get("book", "")}
            )
            results.append({
                "_id":      f"bm25_{idx}",
                "text":     chunk["text"],
                "metadata": meta,
                "score":    score,
                })
        return results

    # ── helpers ───────────────────────────────────────────────────────

    def _collections(self, source_filter: str):
        if source_filter == "hadith":
            return [self.hadith_col]
        if source_filter == "tafsir":
            return [self.tafsir_col]
        return [self.hadith_col, self.tafsir_col]

    def _restore_meta(self, flat: dict) -> dict:
        meta = dict(flat)
        ayah = meta.pop("ayah", None)
        meta["ayah_range"] = [ayah, ayah] if ayah not in (None, -1) else None
        for key in ("surah", "hadith_number", "book_number", "hadith_in_book"):
            if meta.get(key) == -1:
                meta[key] = None
        for key in ("section_name", "grade", "grades_detail",
                    "arabic_text", "english_translation"):
            if meta.get(key) == "":
                meta[key] = None
        return meta