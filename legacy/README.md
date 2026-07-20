# Legacy scripts (not used by the active pipeline)

These are the original "Phase 1" per-book ingestion scripts, superseded by
the auto-discovering pipeline in `build_embeddings.py` / `data_loaders.py` /
`chunking.py` (orchestrated by `app.py`). Kept for reference only.

- `ingest_all_hadiths.py` / `ingest_all_tafseer.py` — batch ingestion driven
  by a `books_config.json`. `ingest_all_hadiths.py` imports
  `from ingest_hadith import extract_record`, a module that no longer
  exists in this repo, so **it will not run as-is**.
- `merge_chunks.py` — combined separately-ingested hadith/tafsir chunk files.
  The current pipeline builds `all_chunks.json` directly in one pass.
- `embed_and_store.py` — the original single-purpose embed+store script,
  replaced by `build_embeddings.py` (which also auto-discovers sources and
  chunks long text).
- `schema.py` — the chunk schema (`ayah_range` pairs, `make_chunk`,
  `validate_chunk`) these scripts used. The current pipeline uses a
  different, simpler shape (see `data_loaders.py`): tafsir chunks carry a
  single `ayah` int, not an `ayah_range` pair.

Do not wire these back into the active pipeline without updating them to
match the current chunk schema in `data_loaders.py` / `chunking.py`.
