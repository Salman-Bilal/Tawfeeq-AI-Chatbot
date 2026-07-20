# Embedding pipeline (raw data → vectors → ChromaDB)

A robust, auto-discovering pipeline that turns the Islamic dataset (hadith +
tafsir) into a searchable vector database for the RAG chatbot.

## Quick start

```bash
pip install -r requirements.txt
```

**Recommended:** `python app.py` (project root) -- checks whether
embeddings + the BM25 index already exist and are in sync first, and only
does the expensive rebuild if something's actually missing. See the root
`README.md` for its flags (`--force`, `--dry-run`, `--limit`, `--query`).

The pieces it wraps can also be run directly, e.g. for a fast sanity check
without touching the model or the DB:

```bash
# All commands run from the project root.

# Fast sanity check — chunks the data, writes data/processed/all_chunks.json,
# downloads only the small tokenizer, NO embeddings, NO 2.3 GB model.
python pipeline/build_embeddings.py --dry-run

# Full build (always a clean rebuild, unlike app.py -- no already-built check)
python pipeline/build_embeddings.py

# Optional: smoke-test the whole path on 50 docs/source first
python pipeline/build_embeddings.py --limit 50
```

All settings live in **`.env`** (model, chunk sizes, paths, batch size…).

## What it produces

```
data/processed/all_chunks.json     # every chunk + metadata (reusable)
data/chroma_db/                    # persistent ChromaDB
   ├─ hadith_collection            # one vector per hadith chunk
   └─ tafsir_collection            # one vector per tafsir chunk
```

## How "more data like this" is handled

Drop new files into `data/` and re-run — no code edits:

| New data | Detected as | Example |
|----------|-------------|---------|
| flat JSON array of hadith records | **hadith book** | `data/sahih_bukhari.json` |
| folder of `surah_*.json` files | **tafsir book** | `data/ur_ibn_kathir/surah_*.json` |

Book names are inferred from the file/folder name (with a built-in map for
well-known books; unknown ones get a humanized name).

## Model

Default `BAAI/bge-m3` — the strongest open multilingual retrieval model for
Urdu + Arabic + English, with long-context support. Swap to a lighter model in
`.env` (`EMBEDDING_MODEL=intfloat/multilingual-e5-base`) if RAM is tight; the
required `query:`/`passage:` prefixes for e5 are applied automatically.

## Notes

- The same model must be loaded at query time in the retriever; on a small
  Render instance prefer `multilingual-e5-base` to avoid OOM.
- Re-running does a clean rebuild of both collections. IDs are content-stable.
