"""
Retrieval-layer tests -- exercise RetrieverService (ChromaDB + BM25 + hybrid
fusion) directly. No OPENROUTER_API_KEY or running server needed; only the
already-built data/chroma_db and data/bm25_index artifacts.

Run:
    pytest tests/test_retrieval.py -v
"""

import pytest

from services.retriever import RetrieverService
from queries import EASY, MEDIUM, HARD, EXTREME


@pytest.fixture(scope="session")
def retriever():
    return RetrieverService()


# ── sanity: the artifacts this whole system depends on ──────────────────────

def test_collections_populated(retriever):
    assert retriever.hadith_col.count() > 0
    assert retriever.tafsir_col.count() > 0


def test_bm25_index_matches_corpus_size(retriever):
    # BM25 index and the two ChromaDB collections are built from the same
    # all_chunks.json -- their counts must line up exactly, or the indexes
    # are out of sync (e.g. one was rebuilt and the other wasn't).
    total_vectors = retriever.hadith_col.count() + retriever.tafsir_col.count()
    assert len(retriever.bm25_chunks) == total_vectors


# ── easy / medium / hard: should retrieve relevant, well-formed chunks ──────

@pytest.mark.parametrize("level,question,source_filter,expect", EASY + MEDIUM + HARD)
def test_relevant_queries_return_wellformed_chunks(retriever, level, question, source_filter, expect):
    results = retriever.retrieve(query=question, source_filter=source_filter, top_k=5)
    assert results, f"[{level}] expected results for {question!r}"

    seen_texts = set()
    for chunk in results:
        meta = chunk["metadata"]

        # regression: BM25-only hits used to lose source/book entirely
        assert meta.get("source"), f"missing source in {chunk}"
        assert meta.get("book"), f"missing book in {chunk}"
        if source_filter != "both":
            assert meta["source"] == source_filter

        # regression: tafsir ayah_range used to always resolve to None
        # because _restore_meta looked for ayah_start/ayah_end fields that
        # were never actually stored (real field is a single "ayah" int).
        if meta["source"] == "tafsir":
            assert meta.get("ayah_range") is not None, f"tafsir chunk missing ayah_range: {chunk}"

        # regression: dense/sparse fusion used incompatible id namespaces,
        # so the same chunk (or exact-duplicate boilerplate text) could
        # appear twice, or inflate its score by "voting" at multiple ranks
        # within a single ranked list.
        assert chunk["text"] not in seen_texts, "duplicate chunk text in fused results"
        seen_texts.add(chunk["text"])


# ── extreme: out-of-scope / gibberish must not look like confident hits ─────

@pytest.mark.parametrize("level,question,source_filter,expect",
                          [q for q in EXTREME if q[3] == "empty"])
def test_degenerate_queries_return_nothing(retriever, level, question, source_filter, expect):
    results = retriever.retrieve(query=question, source_filter=source_filter, top_k=5)
    assert results == [], f"[{level}] expected no results for {question!r}, got {len(results)}"


@pytest.mark.parametrize("level,question,source_filter,expect",
                          [q for q in EXTREME if q[3] == "guardrail"])
def test_out_of_scope_queries_dont_crash(retriever, level, question, source_filter, expect):
    # These may legitimately retrieve a few weak BM25 keyword-overlap hits
    # (e.g. the word "capital" appearing in an unrelated commentary passage)
    # -- catching that they're not *actually* answerable is the generation
    # layer's job (see test_api.py). Here we just assert the retrieval layer
    # doesn't error out and every returned chunk is still well-formed.
    results = retriever.retrieve(query=question, source_filter=source_filter, top_k=5)
    for chunk in results:
        assert chunk["metadata"].get("source")
        assert chunk["metadata"].get("book")


def test_high_relevance_beats_low_relevance(retriever):
    """The MIN_DENSE_SCORE gate should make an on-topic query outscore an
    off-topic one on the top dense hit."""
    on_topic = retriever._dense_search("inheritance share of a daughter in Islam", "both", 1)
    off_topic = retriever._dense_search("how to bake a chocolate cake", "both", 1)
    assert on_topic[0]["score"] > off_topic[0]["score"]
