from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=10)
    retrieval_only: bool = False


class SourceResponse(BaseModel):
    source_id: int
    title: str | None
    url: str
    chunk_id: int
    document_id: int
    snippet: str
    score: float | None = None


class TimingResponse(BaseModel):
    retrieval_s: float
    generation_s: float
    total_s: float


class AnswerResponse(BaseModel):
    query: str
    mode: str
    status: Literal["answered", "insufficient_evidence", "retrieval_only"]
    answer: str
    sources: list[SourceResponse]
    timing: TimingResponse | None = None


class HealthResponse(BaseModel):
    status: str
    mode: str
    artifacts_loaded: bool
    generator_loaded: bool


class SampleQuestionsResponse(BaseModel):
    questions: list[str]
