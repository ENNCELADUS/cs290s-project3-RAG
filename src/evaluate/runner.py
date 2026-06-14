from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rag.generate import RagAnswerer
from rag.index import DEFAULT_BM25, DEFAULT_CHUNK_INDEX, DEFAULT_DB, DEFAULT_FAISS, DEFAULT_REPORT
from rag.retrieve import (
    DEFAULT_DENSE_TOP_K,
    DEFAULT_FUSED_TOP_K,
    DEFAULT_SPARSE_TOP_K,
    HybridRetrievalResult,
    Retriever,
)

from .judge import JudgeResult, judge_answer
from .metrics import source_metrics
from .schema import QuestionSpec

RunnerKind = Literal["retrieve", "answer", "both"]
EvalMode = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class ArtifactPaths:
    db: Path = DEFAULT_DB
    bm25: Path = DEFAULT_BM25
    faiss: Path = DEFAULT_FAISS
    chunk_index: Path = DEFAULT_CHUNK_INDEX
    report: Path = DEFAULT_REPORT
    dense_model: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    runner: RunnerKind = "both"
    modes: tuple[EvalMode, ...] = ("dense", "hybrid")
    top_k: int = 5
    diagnostic_depth: int | None = None
    sparse_top_k: int | None = None
    dense_top_k: int | None = None
    fused_top_k: int | None = None
    rerank_top_k: int | None = None
    rerank_preserve_top_k: int | None = None
    rrf_k: int | None = None
    sparse_weight: float | None = None
    dense_weight: float | None = None
    url_cap: int | None = None
    model_path: Path | None = None
    reranker_model: Path | None = None
    reranker_device: str | None = None
    expanded_queries: tuple[str, ...] = ()
    expanded_queries_by_id: dict[str, tuple[str, ...]] | None = None
    device: str = "auto"
    max_new_tokens: int | None = None
    artifacts: ArtifactPaths = ArtifactPaths()


def run_evaluation(questions: list[QuestionSpec], config: EvaluationConfig) -> list[dict[str, Any]]:
    if config.runner in {"answer", "both"} and config.model_path is None:
        raise ValueError("--model-path is required when --runner is answer or both")
    retriever = Retriever.from_paths(
        db_path=config.artifacts.db,
        bm25_path=config.artifacts.bm25,
        faiss_path=config.artifacts.faiss,
        chunk_index_path=config.artifacts.chunk_index,
        report_path=config.artifacts.report,
        dense_model=config.artifacts.dense_model,
    )
    answerer = None
    if config.runner in {"answer", "both"}:
        answer_kwargs: dict[str, Any] = {"model_path": config.model_path, "device": config.device}
        if config.max_new_tokens is not None:
            answer_kwargs["max_new_tokens"] = config.max_new_tokens
        answerer = RagAnswerer(retriever, **answer_kwargs)

    records: list[dict[str, Any]] = []
    for question in questions:
        for mode in config.modes:
            if config.runner in {"retrieve", "both"}:
                records.append(_run_retrieve(question, mode=mode, retriever=retriever, config=config))
            if config.runner in {"answer", "both"} and mode in {"dense", "hybrid"}:
                records.append(_run_answer(question, mode=mode, answerer=answerer, config=config))
    return records


def _run_retrieve(
    question: QuestionSpec,
    *,
    mode: EvalMode,
    retriever: Retriever,
    config: EvaluationConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    base = _question_payload(question, mode=mode, runner="retrieve")
    try:
        retrieval_result = retriever.retrieve(
            question.query,
            mode=mode,
            top_k=config.top_k,
            **_hybrid_retrieve_kwargs(mode=mode, config=config, question=question),
        )
        hits = _hits_from_result(retrieval_result)
        observed_urls = [str(_hit_value(hit, "url") or "") for hit in hits]
        record = {
            **base,
            "status": "ok",
            "answer_status": None,
            "answer": None,
            "observed_source_urls": observed_urls,
            "retrieved_source_urls": observed_urls,
            "cited_source_urls": [],
            "top_titles": [str(_hit_value(hit, "title") or "") for hit in hits[: config.top_k]],
            "retrieval": _retrieval_payload(retrieval_result),
            "judge": asdict(JudgeResult(status="manual_review", is_correct=None, reason="retrieval-only run")),
            "metrics": source_metrics(observed_urls, question.acceptable_source_urls),
            "latency_s": round(time.perf_counter() - started, 6),
        }
        if config.diagnostic_depth is not None:
            diagnostic_result = retriever.retrieve(
                question.query,
                mode=mode,
                **_diagnostic_retrieve_kwargs(depth=config.diagnostic_depth),
                **_hybrid_retrieve_kwargs(
                    mode=mode,
                    config=config,
                    question=question,
                    diagnostic_depth=config.diagnostic_depth,
                ),
            )
            record["diagnostic_hits"] = [_hit_to_dict(hit) for hit in _hits_from_result(diagnostic_result)]
        return record
    except Exception as error:
        record = {
            **base,
            "status": "error",
            "answer_status": None,
            "answer": None,
            "observed_source_urls": [],
            "retrieved_source_urls": [],
            "cited_source_urls": [],
            "top_titles": [],
            "retrieval": None,
            "judge": asdict(JudgeResult(status="manual_review", is_correct=None, reason="run error")),
            "metrics": source_metrics([], question.acceptable_source_urls),
            "latency_s": round(time.perf_counter() - started, 6),
            "error": str(error),
        }
        if config.diagnostic_depth is not None:
            record["diagnostic_hits"] = []
        return record


def _run_answer(
    question: QuestionSpec,
    *,
    mode: EvalMode,
    answerer: RagAnswerer | None,
    config: EvaluationConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    base = _question_payload(question, mode=mode, runner="answer")
    try:
        if answerer is None:
            raise ValueError("answer runner is not configured")
        answer_result = answerer.answer(
            question.query,
            mode=mode,
            top_k=config.top_k,
            **_hybrid_retrieve_kwargs(mode=mode, config=config, question=question),
        )  # type: ignore[arg-type]
        retrieved_urls = [str(source.url) for source in answer_result.sources]
        cited_urls = _cited_source_urls(answer_result.answer, answer_result.sources)
        judge = judge_answer(question, answer_result.answer)
        return {
            **base,
            "status": "ok",
            "answer_status": answer_result.status,
            "answer": answer_result.answer,
            "observed_source_urls": cited_urls,
            "retrieved_source_urls": retrieved_urls,
            "cited_source_urls": cited_urls,
            "top_titles": [source.title or "" for source in answer_result.sources[: config.top_k]],
            "retrieval": answer_result.retrieval,
            "judge": asdict(judge),
            "metrics": source_metrics(cited_urls, question.acceptable_source_urls),
            "latency_s": round(time.perf_counter() - started, 6),
        }
    except Exception as error:
        return {
            **base,
            "status": "error",
            "answer_status": None,
            "answer": None,
            "observed_source_urls": [],
            "retrieved_source_urls": [],
            "cited_source_urls": [],
            "top_titles": [],
            "retrieval": None,
            "judge": asdict(JudgeResult(status="manual_review", is_correct=None, reason="run error")),
            "metrics": source_metrics([], question.acceptable_source_urls),
            "latency_s": round(time.perf_counter() - started, 6),
            "error": str(error),
        }


def _question_payload(question: QuestionSpec, *, mode: EvalMode, runner: RunnerKind) -> dict[str, Any]:
    return {
        "id": question.id,
        "category": question.category,
        "language": question.language,
        "complexity": question.complexity,
        "runner": runner,
        "mode": mode,
        "query": question.query,
        "gt_answer": question.gt_answer,
        "primary_source_url": question.primary_source_url,
        "expected_source_urls": question.acceptable_source_urls,
        "judge_type": question.judge_type,
        "grading_notes": question.grading_notes,
    }


def _hits_from_result(result: object) -> list[object]:
    if isinstance(result, HybridRetrievalResult):
        return list(result.hits)
    return list(result)  # type: ignore[arg-type]


def _retrieval_payload(result: object) -> dict[str, Any]:
    if isinstance(result, HybridRetrievalResult):
        return asdict(result)
    return {"hits": [_hit_to_dict(hit) for hit in _hits_from_result(result)]}


def _diagnostic_retrieve_kwargs(*, depth: int) -> dict[str, int]:
    return {"top_k": depth}


def _hybrid_retrieve_kwargs(
    *,
    mode: EvalMode,
    config: EvaluationConfig,
    question: QuestionSpec | None = None,
    diagnostic_depth: int | None = None,
) -> dict[str, object]:
    if mode != "hybrid":
        return {}
    kwargs: dict[str, object] = {}
    if diagnostic_depth is not None:
        kwargs["sparse_top_k"] = max(config.sparse_top_k or DEFAULT_SPARSE_TOP_K, diagnostic_depth)
        kwargs["dense_top_k"] = max(config.dense_top_k or DEFAULT_DENSE_TOP_K, diagnostic_depth)
        kwargs["fused_top_k"] = max(config.fused_top_k or DEFAULT_FUSED_TOP_K, diagnostic_depth)
    else:
        if config.sparse_top_k is not None:
            kwargs["sparse_top_k"] = config.sparse_top_k
        if config.dense_top_k is not None:
            kwargs["dense_top_k"] = config.dense_top_k
        if config.fused_top_k is not None:
            kwargs["fused_top_k"] = config.fused_top_k
    if config.rerank_top_k is not None:
        kwargs["rerank_top_k"] = config.rerank_top_k
    if config.rerank_preserve_top_k is not None:
        kwargs["rerank_preserve_top_k"] = config.rerank_preserve_top_k
    if config.rrf_k is not None:
        kwargs["rrf_k"] = config.rrf_k
    if config.sparse_weight is not None:
        kwargs["sparse_weight"] = config.sparse_weight
    if config.dense_weight is not None:
        kwargs["dense_weight"] = config.dense_weight
    if config.url_cap is not None:
        kwargs["url_cap"] = config.url_cap
    if config.reranker_model is not None:
        kwargs["reranker_model"] = str(config.reranker_model)
    if config.reranker_device is not None:
        kwargs["reranker_device"] = config.reranker_device
    expanded_queries = _expanded_queries_for_question(config, question)
    if expanded_queries:
        kwargs["expanded_queries"] = expanded_queries
    return kwargs


def _expanded_queries_for_question(config: EvaluationConfig, question: QuestionSpec | None) -> tuple[str, ...]:
    expanded_queries = list(config.expanded_queries)
    if question is not None and config.expanded_queries_by_id:
        expanded_queries.extend(config.expanded_queries_by_id.get(question.id, ()))
    return tuple(query for query in expanded_queries if query.strip())


def _hit_to_dict(hit: object) -> dict[str, Any]:
    try:
        return asdict(hit)
    except TypeError:
        return {
            "rank": _hit_value(hit, "rank"),
            "url": _hit_value(hit, "url"),
            "title": _hit_value(hit, "title"),
            "score": _hit_value(hit, "score"),
            "chunk_id": _hit_value(hit, "chunk_id"),
            "document_id": _hit_value(hit, "document_id"),
            "snippet": _hit_value(hit, "snippet"),
        }


def _hit_value(hit: object, name: str) -> object:
    if isinstance(hit, dict):
        return hit.get(name)
    return getattr(hit, name, None)


def _cited_source_urls(answer: str, sources: list[object]) -> list[str]:
    by_source_id = {int(source.source_id): str(source.url) for source in sources if getattr(source, "url", None)}
    cited_ids = [int(match.group(1)) for match in re.finditer(r"\[(\d+)\]", answer)]
    urls: list[str] = []
    seen: set[str] = set()
    for source_id in cited_ids:
        url = by_source_id.get(source_id)
        if url is None or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls
