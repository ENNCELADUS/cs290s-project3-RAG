from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from rag.generate import RagAnswerer, RagAnswerResult
from rag.index import (
    DEFAULT_BM25,
    DEFAULT_CHUNK_INDEX,
    DEFAULT_DB,
    DEFAULT_FAISS,
    DEFAULT_REPORT,
)
from rag.retrieve import Retriever

pytestmark = [pytest.mark.e2e, pytest.mark.real_data, pytest.mark.real_llm]

VALID_CITATION_RE = re.compile(r"\[(\d+)\]")
PROMPT_LEAKAGE_MARKERS = (
    "Question:",
    "Sources:",
    "TEXT:",
    "URL:",
    "trace_ref:",
    "The answer should",
    "Use only the provided",
)
DEEP_LEARNING_TEACHER_TEMPLATE = re.compile(
    r"(深度学习|Deep Learning|CS280).{0,120}(任课老师|授课教师|teacher|instructor|taught).{0,120}何旭明.{0,40}\[\d+\]",
    re.IGNORECASE | re.DOTALL,
)
ROBOTICS_FACULTY_TEMPLATE = re.compile(
    r"(Robotics|robotics).{0,160}Schwertfeger.{0,40}\[\d+\]|Schwertfeger.{0,160}(Robotics|robotics).{0,40}\[\d+\]",
    re.IGNORECASE | re.DOTALL,
)


@pytest.fixture(scope="module")
def real_llm_answerer() -> RagAnswerer:
    model_path = _required_model_path()
    device = _real_llm_device()
    _skip_if_missing_artifacts()
    retriever = Retriever.from_paths(
        db_path=DEFAULT_DB,
        bm25_path=DEFAULT_BM25,
        faiss_path=DEFAULT_FAISS,
        chunk_index_path=DEFAULT_CHUNK_INDEX,
        report_path=DEFAULT_REPORT,
    )
    return RagAnswerer(
        retriever,
        model_path=model_path,
        device=device,
        max_new_tokens=96,
        temperature=0.0,
    )


def test_real_llm_answers_deep_learning_teacher_with_citation(real_llm_answerer: RagAnswerer) -> None:
    result = real_llm_answerer.answer("深度学习这门课的任课老师是谁？", mode="hybrid", top_k=5)

    assert result.status == "answered"
    assert DEEP_LEARNING_TEACHER_TEMPLATE.search(result.answer)
    _assert_cited_template(result)
    _assert_no_prompt_leakage(result.answer)


def test_real_llm_answers_robotics_faculty_with_citation(real_llm_answerer: RagAnswerer) -> None:
    result = real_llm_answerer.answer(
        "Which SIST faculty work on robotics?",
        mode="hybrid",
        top_k=5,
    )

    assert result.status == "answered"
    assert ROBOTICS_FACULTY_TEMPLATE.search(result.answer)
    _assert_cited_template(result)
    _assert_no_prompt_leakage(result.answer)


def test_real_llm_refuses_unanswerable_cafeteria_menu(real_llm_answerer: RagAnswerer) -> None:
    result = real_llm_answerer.answer("What is the SIST cafeteria menu tomorrow?", mode="hybrid", top_k=3)

    assert result.status == "insufficient_evidence"
    assert (
        result.answer
        == "Evidence is insufficient: the retrieved official sources do not contain enough information to answer."
    )


def _required_model_path() -> Path:
    raw_model_path = os.getenv("RAG_TEST_MODEL_PATH")
    if not raw_model_path:
        pytest.skip("set RAG_TEST_MODEL_PATH to a local Qwen model snapshot")
    model_path = Path(raw_model_path)
    if not model_path.exists():
        pytest.skip(f"missing local LLM model path: {model_path}")
    return model_path


def _real_llm_device() -> str:
    device = os.getenv("RAG_TEST_DEVICE", "cuda")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("RAG_TEST_DEVICE=cuda but torch.cuda.is_available() is false")
    return device


def _skip_if_missing_artifacts() -> None:
    artifact_paths = (DEFAULT_DB, DEFAULT_BM25, DEFAULT_FAISS, DEFAULT_CHUNK_INDEX, DEFAULT_REPORT)
    missing = [str(path) for path in artifact_paths if not path.exists()]
    if missing:
        pytest.skip(f"missing real RAG artifacts: {', '.join(missing)}")


def _assert_cited_template(result: RagAnswerResult) -> None:
    citation_ids = [int(match.group(1)) for match in VALID_CITATION_RE.finditer(result.answer)]
    valid_source_ids = {source.source_id for source in result.sources}

    assert citation_ids
    assert all(citation_id in valid_source_ids for citation_id in citation_ids)


def _assert_no_prompt_leakage(answer: str) -> None:
    assert not any(marker in answer for marker in PROMPT_LEAKAGE_MARKERS)
