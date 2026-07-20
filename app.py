"""
app.py — one entry point (at the project root, next to .env) that runs the
whole embedding pipeline step by step, and lets you actually ask it
questions once the knowledge base is built.

Each stage prints a banner so you can watch progress and see exactly where a
run stops if something goes wrong. It reuses the functions in
pipeline/build_embeddings.py and pipeline/build_bm25_index.py, and the same
backend/services (retriever, generator, guardrail) the FastAPI app uses --
no duplicated logic, and the CLI behaves identically to the API.

Checks whether embeddings + the BM25 index already exist (and are in sync)
before doing anything expensive: if they're already built, it just verifies
and answers a demo question instead of re-chunking/re-embedding the whole
corpus. Pass --force to rebuild anyway (e.g. after editing data/).

USAGE (from the project root)
    python app.py                  # build if missing, otherwise verify + demo answer
    python app.py --chat           # interactive Q&A loop against the knowledge base
    python app.py --query "تقویٰ کیا ہے"   # ask one question and exit
    python app.py --force          # rebuild everything from scratch regardless
    python app.py --dry-run        # stop after chunking (no model download, no DB writes)
    python app.py --limit 50       # smoke test on 50 docs per source
    python app.py --install        # pip install -r requirements.txt first, then run

Answering questions needs OPENROUTER_API_KEY set in .env (same requirement
as the backend). Config lives in .env (see pipeline/config.py).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# pipeline/*.py import each other with plain sibling imports (e.g.
# `from config import cfg`) so they still work when run directly
# (`python pipeline/build_embeddings.py`). Importing them as `pipeline.*`
# submodules from here needs pipeline/ itself on sys.path too, not just
# the project root, or those internal sibling imports fail.
# backend/services/*.py don't import each other, so they need no such
# hack -- `backend.services.x` resolves cleanly via the project root
# (already on sys.path since app.py itself lives there), which is also
# why static analyzers (Pylance etc.) can resolve this form but not a
# sys.path-inserted `from services.x import y`.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "pipeline"))

from pipeline.config import cfg, ROOT
import pipeline.build_embeddings as pipe
import pipeline.build_bm25_index as bm25_pipe
from backend.services.retriever import RetrieverService
from backend.services.generator import generate_answer
from backend.services.guardrail import check_groundedness, FALLBACK_MESSAGE

# Windows consoles default to cp1252, which cannot print Urdu/Arabic and
# would crash on output. Force UTF-8 so script output (and typed-in Urdu/
# Arabic questions in --chat) are always safe.
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

_STEP = 0


def banner(title: str) -> float:
    """Print a numbered step header and return a start timestamp."""
    global _STEP
    _STEP += 1
    line = "=" * 70
    print(f"\n{line}\n  STEP {_STEP}: {title}\n{line}")
    return time.time()


def done(t0: float):
    print(f"  ...done in {time.time() - t0:.1f}s")


# ── dependency handling ──────────────────────────────────────────────────────

def step_dependencies(auto_install: bool):
    t0 = banner("Check dependencies")
    if auto_install:
        print("  Installing requirements.txt ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        print("  Installing backend/requirements.txt ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "backend" / "requirements.txt")])

    missing = []
    for mod, pip_name in (("chromadb", "chromadb"),
                          ("sentence_transformers", "sentence-transformers"),
                          ("transformers", "transformers"),
                          ("torch", "torch"),
                          ("openai", "openai")):  # needed for services.generator / guardrail
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print("  MISSING packages:", ", ".join(missing))
        print("  Run:  pip install -r requirements.txt   (or: python app.py --install)")
        sys.exit(1)
    print("  All required packages present.")
    done(t0)


# ── retrieval helpers (also used by --query mode) ────────────────────────────

def _client():
    import chromadb
    return chromadb.PersistentClient(path=str(cfg.CHROMA_DB_PATH))


def bm25_path() -> Path:
    return cfg.DATA_DIR / "bm25_index" / "bm25_index.pkl"


def pipeline_status() -> dict:
    """
    Inspect what's already built: ChromaDB collection counts and whether the
    BM25 index file exists. Used to decide whether a run needs to build
    anything at all, and to catch the two collections and the BM25 index
    having drifted out of sync with each other (e.g. one was rebuilt and
    the other wasn't -- retriever.py assumes they describe the same corpus).
    """
    status = {"hadith_count": 0, "tafsir_count": 0, "bm25_exists": bm25_path().exists(), "bm25_count": 0}
    try:
        client = _client()
        status["hadith_count"] = client.get_collection(cfg.HADITH_COLLECTION).count()
        status["tafsir_count"] = client.get_collection(cfg.TAFSIR_COLLECTION).count()
    except Exception:
        pass
    if status["bm25_exists"]:
        try:
            import pickle
            payload = pickle.load(open(bm25_path(), "rb"))
            status["bm25_count"] = len(payload["chunks"])
        except Exception:
            status["bm25_exists"] = False
    return status


def is_pipeline_built(status: dict) -> bool:
    """True only if both collections are populated AND the BM25 index has
    the exact same number of chunks -- a mismatch means the artifacts were
    built at different times from different data and shouldn't be trusted."""
    total_vectors = status["hadith_count"] + status["tafsir_count"]
    return (
        status["hadith_count"] > 0
        and status["tafsir_count"] > 0
        and status["bm25_exists"]
        and status["bm25_count"] == total_vectors
    )


def _cite(chunk: dict) -> str:
    meta = chunk["metadata"]
    if meta.get("source") == "hadith":
        return f"{meta.get('book')} #{meta.get('hadith_number')} ({meta.get('grade') or 'ungraded'})"
    ar = meta.get("ayah_range") or []
    ayah = str(ar[0]) if ar else "?"
    return f"{meta.get('book')} {meta.get('surah')}:{ayah}"


def ask(retriever: RetrieverService, question: str, language: str = "both", top_k: int = 5) -> bool:
    """
    One full RAG round-trip: hybrid retrieve -> generate -> groundedness
    guardrail -> print. Same three services (and the same behavior on an
    ungrounded answer -- swap in FALLBACK_MESSAGE, hide sources) as
    POST /query in the backend, so the CLI and the API never disagree.

    Returns False if generation couldn't run at all (no OPENROUTER_API_KEY)
    so --chat can stop prompting instead of failing on every question.
    """
    chunks = retriever.retrieve(query=question, source_filter="both", top_k=top_k)
    if not chunks:
        print("\n  No relevant content found in the knowledge base for this question.")
        return True

    try:
        answer = generate_answer(question, chunks, language)
    except EnvironmentError as e:
        print(f"\n  {e}")
        return False

    passed, reason = check_groundedness(answer, chunks)
    if not passed:
        print(f"\n  [guardrail] answer was not fully grounded in the sources: {reason}")
        answer = FALLBACK_MESSAGE

    print(f"\n  Answer:\n  {answer}")
    if passed:
        print(f"\n  Sources ({len(chunks)}):")
        for i, c in enumerate(chunks, 1):
            print(f"    [{i}] {_cite(c)}  (score={c['score']:.3f})")
    return True


def chat_loop(retriever: RetrieverService, language: str):
    print("\nInteractive RAG chat -- ask about hadith/tafsir. Type 'exit' or Ctrl+C to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            return
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Exiting chat.")
            return
        if not ask(retriever, question, language):
            print("Cannot continue without OPENROUTER_API_KEY -- add it to .env and restart.")
            return


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run the embedding pipeline step by step.")
    ap.add_argument("--dry-run", action="store_true", help="Stop after chunking (no model, no DB).")
    ap.add_argument("--limit", type=int, default=None, help="Only N docs per source (smoke test).")
    ap.add_argument("--query", type=str, default=None, help="Ask one question against an already-built DB and exit.")
    ap.add_argument("--chat", action="store_true", help="Interactive Q&A loop against an already-built DB.")
    ap.add_argument("--language", choices=["urdu", "english", "both"], default="both",
                     help="Preferred answer language for --query/--chat (default: both).")
    ap.add_argument("--install", action="store_true", help="pip install -r requirements.txt first.")
    ap.add_argument("--force", action="store_true",
                     help="Rebuild embeddings + BM25 index from scratch even if already built.")
    args = ap.parse_args()

    run_start = time.time()

    # ── query / chat mode -- ask questions against an already-built DB ──
    if args.query or args.chat:
        step_dependencies(args.install)
        t0 = banner("Load retriever (embedding model + ChromaDB + BM25)")
        retriever = RetrieverService()
        done(t0)

        if args.chat:
            chat_loop(retriever, args.language)
        else:
            banner(f"Answering: {args.query!r}")
            ask(retriever, args.query, args.language)
        print(f"\nFinished in {time.time() - run_start:.1f}s.")
        return

    # ── STEP 1: dependencies ──
    step_dependencies(args.install)

    # ── check what's already built -- skip the expensive rebuild if it's
    #    already there and consistent, unless --force or --dry-run ──
    if not args.dry_run and not args.force:
        status = pipeline_status()
        if is_pipeline_built(status):
            banner("Embeddings + BM25 index already built -- skipping rebuild")
            print(f"  {cfg.HADITH_COLLECTION}: {status['hadith_count']} vectors")
            print(f"  {cfg.TAFSIR_COLLECTION}: {status['tafsir_count']} vectors")
            print(f"  BM25 index: {bm25_path()} ({status['bm25_count']} chunks, in sync)")
            print("  Use --force to rebuild from scratch (e.g. after adding new data to data/).")
            print("  Try: python app.py --chat   (or --query \"your question\")")

            t0 = banner("Load retriever (for the demo answer)")
            retriever = RetrieverService()
            done(t0)
            banner("Demo question (proving the already-built pipeline answers for real)")
            ask(retriever, "تقویٰ کیا ہے")

            print(f"\nPipeline already ready. Verified in {time.time() - run_start:.1f}s.")
            return
        elif status["hadith_count"] or status["tafsir_count"] or status["bm25_exists"]:
            banner("Partial/inconsistent build detected -- rebuilding everything")
            print(f"  hadith={status['hadith_count']}  tafsir={status['tafsir_count']}  "
                  f"bm25_exists={status['bm25_exists']}  bm25_count={status['bm25_count']}")
            print("  (collections and BM25 index must come from the same build to stay in sync)")

    # ── STEP 2: discover + load ──
    t0 = banner("Discover sources and load documents")
    docs = pipe.load_documents(limit=args.limit)
    n_had = sum(1 for d in docs if d["source"] == "hadith")
    n_taf = sum(1 for d in docs if d["source"] == "tafsir")
    print(f"  Loaded {len(docs)} documents (hadith={n_had}, tafsir={n_taf})")
    done(t0)

    # ── STEP 3: tokenizer ──
    t0 = banner("Load tokenizer (for token-accurate chunking)")
    count_tokens = pipe.get_tokenizer()
    done(t0)

    # ── STEP 4: chunk ──
    t0 = banner("Chunk documents")
    chunks = pipe.chunk_documents(docs, count_tokens)
    print(f"  {len(docs)} docs -> {len(chunks)} chunks "
          f"({len(chunks) / max(len(docs),1):.2f} chunks/doc)")
    done(t0)

    # ── STEP 5: save artifact ──
    t0 = banner("Save processed chunks artifact")
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.PROCESSED_DIR / "all_chunks.json"
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Wrote {out}")
    done(t0)

    if args.dry_run:
        print(f"\nDry run complete in {time.time() - run_start:.1f}s. No embeddings created.")
        return

    # ── STEP 6: load model ──
    t0 = banner("Load embedding model")
    model = pipe.load_model()
    done(t0)

    # ── STEP 7: open (reset) collections ──
    t0 = banner("Open ChromaDB collections (clean rebuild)")
    hcol, tcol = pipe.open_collections()
    print(f"  DB path: {cfg.CHROMA_DB_PATH}")
    done(t0)

    # ── STEP 8: embed + store hadith ──
    t0 = banner("Embed + store HADITH")
    pipe.embed_into_collection(chunks, "hadith", hcol, model)
    done(t0)

    # ── STEP 9: embed + store tafsir ──
    t0 = banner("Embed + store TAFSIR")
    pipe.embed_into_collection(chunks, "tafsir", tcol, model)
    done(t0)

    # ── STEP 10: build BM25 keyword index ──
    # Reuses the same in-memory `chunks` list just embedded above (with
    # ids now attached), instead of round-tripping through all_chunks.json --
    # this also guarantees the BM25 index and the two collections always
    # describe the exact same build, which is_pipeline_built() depends on.
    t0 = banner("Build BM25 keyword index")
    bm25_pipe.build_and_save(chunks, bm25_path())
    done(t0)

    # ── STEP 11: verify ──
    t0 = banner("Verify")
    print(f"  {cfg.HADITH_COLLECTION}: {hcol.count()} vectors")
    print(f"  {cfg.TAFSIR_COLLECTION}: {tcol.count()} vectors")
    print(f"  BM25 index: {bm25_path()}")
    done(t0)

    # ── STEP 12: demo question (retrieval + generation + guardrail, for real) ──
    t0 = banner("Load retriever (for the demo answer)")
    retriever = RetrieverService()
    done(t0)
    banner("Demo question")
    ask(retriever, "تقویٰ کیا ہے")

    print(f"\nALL DONE in {time.time() - run_start:.1f}s. "
          f"Vector DB ready at {cfg.CHROMA_DB_PATH}, BM25 index ready at {bm25_path()}. "
          f"Try: python app.py --chat")


if __name__ == "__main__":
    main()
