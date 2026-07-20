"""
backend/routers/query.py

Three endpoints:
  POST /query        → Full JSON response (text input)
  POST /query/stream → Streaming token response (text input)
  POST /query/voice  → Voice input response (audio file input -> Whisper -> RAG)

The React frontend uses /query/stream for the live typing effect.
Use /query or /query/voice for testing with curl, Postman, or microphone blobs.
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from models.schemas import QueryRequest, QueryResponse, SourceChunk
from services.guardrail import FALLBACK_MESSAGE

router = APIRouter(prefix="/query", tags=["query"])


def _chunks_to_sources(chunks: list[dict]) -> list[SourceChunk]:
    sources = []
    for c in chunks:
        meta = c.get("metadata", {})
        ar   = meta.get("ayah_range")
        sources.append(SourceChunk(
            text         = c["text"],
            source       = meta.get("source", ""),
            book         = meta.get("book", ""),
            surah        = meta.get("surah"),
            ayah_range   = ar if ar and ar[0] != -1 else None,
            hadith_number= meta.get("hadith_number"),
            section_name = meta.get("section_name"),
            grade        = meta.get("grade"),
            score        = c.get("score", 0.0),
        ))
    return sources


# ── POST /query (Full JSON from Text Input) ──────────────────────────────────

@router.post("", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    retriever = request.app.state.retriever
    generator = request.app.state.generator
    guardrail = request.app.state.guardrail

    # 1. Retrieve
    chunks = retriever.retrieve(
        query         = body.question,
        source_filter = body.source_filter,
        top_k         = body.top_k,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the Islamic knowledge base for this query."
        )

    # 2. Generate
    try:
        answer = generator.generate_answer(
            question = body.question,
            chunks   = chunks,
            language = body.language,
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 3. Guardrail Check
    passed, reason = guardrail.check_groundedness(answer, chunks)
    print("=== RAW ANSWER ===")
    print(answer)
    print("=== GUARDRAIL REASON ===")
    print(reason)
    if not passed:
        answer = FALLBACK_MESSAGE

    OUT_OF_SCOPE_MARKER = "do not contain sufficient information"
    is_out_of_scope = OUT_OF_SCOPE_MARKER in answer.lower() or \
                      "ناکافی معلومات" in answer or \
                      answer.strip() == FALLBACK_MESSAGE.strip()

    returned_sources = [] if is_out_of_scope else _chunks_to_sources(chunks)
    returned_chunk_count = 0 if is_out_of_scope else len(chunks)    

    return QueryResponse(
        answer            = answer,
        sources           = returned_sources,
        source_filter_used= body.source_filter,
        chunks_retrieved  = returned_chunk_count,
        guardrail_passed  = passed,
    )


# ── POST /query/stream (Streaming Tokens from Text Input) ────────────────────

@router.post("/stream")
async def query_stream(request: Request, body: QueryRequest):
    retriever = request.app.state.retriever
    generator = request.app.state.generator
    guardrail = request.app.state.guardrail

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not set. Add it to your .env file: GROQ_API_KEY=gsk_...",
        )

    # 1. Retrieve
    chunks = retriever.retrieve(
        query         = body.question,
        source_filter = body.source_filter,
        top_k         = body.top_k,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the Islamic knowledge base."
        )

    # 2. Send sources metadata first as a JSON header line, then stream tokens
    sources_payload = json.dumps({
        "type":    "sources",
        "sources": [s.model_dump() for s in _chunks_to_sources(chunks)],
    }) + "\n"

    async def event_stream():
        yield sources_payload
        accumulated_tokens = []
        
        async for token in generator.stream_answer(
            question = body.question,
            chunks   = chunks,
            language = body.language,
        ):
            accumulated_tokens.append(token)
            yield token

        full_answer = "".join(accumulated_tokens)
        
        # Offload synchronous verification call to prevent blocking the async event loop
        passed, reason = await asyncio.to_thread(guardrail.check_groundedness, full_answer, chunks)
        
        print("=== STREAMED ANSWER ===")
        print(full_answer)
        print("=== STREAM GUARDRAIL REASON ===")
        print(reason)
        
        if not passed:
            yield f"\n\n⚠️ [SYSTEM NOTICE: {FALLBACK_MESSAGE}]"

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={"X-Chunks-Retrieved": str(len(chunks))},
    )


# ── NEW: POST /query/voice (Audio File Input Pipeline) 🎙️ ───────────────────

@router.post("/voice", response_model=QueryResponse)
async def query_voice(
    request: Request, 
    file: UploadFile, 
    source_filter: str = "both",
    top_k: int = 5,
    language: str = "both"
):
    retriever = request.app.state.retriever
    generator = request.app.state.generator
    guardrail = request.app.state.guardrail

    # 1. Enforce API Key verification 
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not set. Add it to your .env file: GROQ_API_KEY=gsk_...",
        )

    # 2. Transcribe Audio via Groq's Whisper Engine ⚡
    tmp_path = None
    try:
        # Whisper infers audio containers via file extensions, default to .wav if empty
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = Path(tmp_file.name)

        # Grab the initialized OpenAI/Groq client infrastructure wrapper
        from services.generator import _get_client
        client = _get_client()

        print(f"[Voice] Transcribing incoming audio file: {file.filename}")
        with open(tmp_path, "rb") as audio_data:
            transcription = client.audio.transcriptions.create(
                file=audio_data,
                model="whisper-large-v3",
                prompt="The query may contain Islamic terms, Surah names, Hadith citations, Arabic, or Urdu phrases.",
            )
        
        spoken_question = transcription.text
        print(f"[Voice] Transcribed Text result: '{spoken_question}'")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")
    
    finally:
        # File management safety clean-up loop
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

    if not spoken_question.strip():
        raise HTTPException(status_code=400, detail="Could not extract any clear text from the provided audio.")

    # 3. Retrieve documents using the transcribed question text 🔍
    chunks = retriever.retrieve(
        query         = spoken_question,
        source_filter = source_filter,
        top_k         = top_k,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the Islamic knowledge base for this query."
        )

    # 4. Generate grounded answer
    try:
        answer = generator.generate_answer(
            question = spoken_question,
            chunks   = chunks,
            language = language,
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 5. Guardrail check
    passed, reason = guardrail.check_groundedness(answer, chunks)
    print("=== RAW VOICE ANSWER ===")
    print(answer)
    print("=== GUARDRAIL REASON ===")
    print(reason)
    if not passed:
        answer = FALLBACK_MESSAGE

    OUT_OF_SCOPE_MARKER = "do not contain sufficient information"
    is_out_of_scope = OUT_OF_SCOPE_MARKER in answer.lower() or \
                      "ناکافy معلومات" in answer or \
                      answer.strip() == FALLBACK_MESSAGE.strip()

    returned_sources = [] if is_out_of_scope else _chunks_to_sources(chunks)
    returned_chunk_count = 0 if is_out_of_scope else len(chunks)    

    return QueryResponse(
        answer            = answer,
        sources           = returned_sources,
        source_filter_used= source_filter,
        chunks_retrieved  = returned_chunk_count,
        guardrail_passed  = passed,
    )