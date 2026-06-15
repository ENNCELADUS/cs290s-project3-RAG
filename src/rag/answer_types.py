from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AnswerMode = Literal["dense", "hybrid"]
AnswerStatus = Literal["answered", "insufficient_evidence"]
GenerationPath = Literal["initial", "extractive_fallback", "repair", "insufficient"]
Device = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class AnswerSource:
    source_id: int
    title: str | None
    url: str
    chunk_id: int
    document_id: int
    trace_ref: str
    snippet: str


@dataclass(frozen=True)
class AnswerConfig:
    model_path: str
    device: str
    max_new_tokens: int
    temperature: float
    top_k: int
    answer_reranker_model: str | None = None
    answer_reranker_device: str = "cpu"


@dataclass(frozen=True)
class AnswerTiming:
    retrieval_s: float
    generation_s: float
    total_s: float


@dataclass(frozen=True)
class RagAnswerResult:
    query: str
    mode: AnswerMode
    status: AnswerStatus
    answer: str
    sources: list[AnswerSource]
    retrieval: dict[str, Any]
    timing: AnswerTiming
    config: AnswerConfig
    generation_path: GenerationPath = "initial"
    generation_rejection_reason: str | None = None
    fallback_source_rank: int | None = None
    answer_context_order: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractiveAnswer:
    answer: str
    source_rank: int


@dataclass(frozen=True)
class ExtractiveCandidate:
    text: str
    source_rank: int
    context_order: int
    score: float
