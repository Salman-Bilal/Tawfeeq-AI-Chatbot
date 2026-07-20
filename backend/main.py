"""
backend/main.py

FastAPI application entry point for the Islamic RAG system.

Startup loads the retriever (ChromaDB + BM25 + embedding model), generator, 
and guardrail check mechanics once and keeps them in app.state so every
request reuses the same loaded objects without reloading from disk or config.

Run with:
    cd backend
    uvicorn main:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Windows consoles default to cp1252, which cannot print Urdu/Arabic and
# crashes any print()/log line that contains it -- every answer this system
# generates is expected to contain Arabic/Urdu text. Same fix as app.py /
# build_embeddings.py, applied here too since this is a separate process.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load .env from project root before anything else
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env")

# Allow importing from project root (config.py etc.)
sys.path.insert(0, str(ROOT))

from services.retriever import RetrieverService
from services.generator import generate_answer, stream_answer
from services.guardrail import check_groundedness  # ✅ Added for state management
from routers.query import router as query_router
from models.schemas import HealthResponse


# ── lifespan: load heavy objects once at startup ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Islamic RAG backend...")

    # 1. Retriever: loads embedding model + ChromaDB + BM25
    app.state.retriever = RetrieverService()

    # 2. Generator: namespace wrapper so the router can call execution loops cleanly
    class _Generator:
        @staticmethod
        def generate_answer(question, chunks, language="both"):
            return generate_answer(question, chunks, language)

        @staticmethod
        async def stream_answer(question, chunks, language="both"):
            async for token in stream_answer(question, chunks, language):
                yield token

    app.state.generator = _Generator()

    # 3. Guardrail: ✅ Expose the groundedness engine inside the app state for unified routing access
    class _Guardrail:
        @staticmethod
        def check_groundedness(answer, chunks):
            return check_groundedness(answer, chunks)

    app.state.guardrail = _Guardrail()
    print("Backend ready.")

    yield  # app runs here

    print("Shutting down.")


# ── app ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Islamic RAG System",
    description = "Retrieval-Augmented Generation over Hadith and Tafsir collections",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allows the React frontend (localhost:3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── routes ───────────────────────────────────────────────────────────────────

app.include_router(query_router)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    retriever = request.app.state.retriever
    return HealthResponse(
        status        = "ok",
        hadith_count  = retriever.hadith_col.count(),
        tafsir_count  = retriever.tafsir_col.count(),
        bm25_ready    = retriever.bm25 is not None,
        model_loaded  = retriever.model is not None,
    )


@app.get("/")
async def root():
    return {
        "message": "Islamic RAG System API",
        "docs":    "/docs",
        "health":  "/health",
    }