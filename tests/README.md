# Tests

Two layers, easy -> extreme query difficulty (see `queries.py` for the bank):

- **`test_retrieval.py`** -- exercises `RetrieverService` (ChromaDB + BM25 +
  hybrid fusion) directly. No server, no API key. Runs against the
  already-built `data/chroma_db` and `data/bm25_index`.
- **`test_api.py`** -- exercises the live FastAPI backend end-to-end
  (retrieval + generation + groundedness guardrail). Needs the backend
  running and `OPENROUTER_API_KEY` set; tests skip cleanly (not fail) if
  either is missing, so `pytest tests/` is always safe to run.

## Run everything retrieval-layer only (no server needed)

```bash
pytest tests/test_retrieval.py -v
```

## Run the full suite (needs the backend + an OpenRouter key)

```bash
cd backend
uvicorn main:app --port 8000
# in another terminal:
pytest tests/ -v
```

## What "extreme" covers

- Cross-source, multi-constraint synthesis questions.
- Clearly out-of-scope questions (world capitals, physics, sports, recipes)
  that must be **declined**, not answered from the model's outside
  knowledge -- this is the core grounding guarantee of the system.
- A prompt-injection attempt ("ignore previous instructions...").
- Gibberish and empty/whitespace-only input.
