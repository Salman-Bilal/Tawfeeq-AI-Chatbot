"""
backend/models/schemas.py

Pydantic models for all FastAPI request and response bodies.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user's question in any language")
    source_filter: Literal["hadith", "tafsir", "both"] = Field(
        default="both",
        description="Restrict search to one source or search both"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    language: Literal["urdu", "english", "both"] = Field(
        default="both",
        description="Preferred language for the answer"
    )


class SourceChunk(BaseModel):
    text: str
    source: str          # "hadith" or "tafsir"
    book: str
    surah: Optional[int] = None
    ayah_range: Optional[List[int]] = None
    hadith_number: Optional[int] = None
    section_name: Optional[str] = None
    grade: Optional[str] = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    source_filter_used: str
    chunks_retrieved: int
    guardrail_passed: bool


class HealthResponse(BaseModel):
    status: str
    hadith_count: int
    tafsir_count: int
    bm25_ready: bool
    model_loaded: bool
