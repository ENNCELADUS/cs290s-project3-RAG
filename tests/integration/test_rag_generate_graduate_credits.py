from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag.generate import RagAnswerer
from rag.retrieve import (
    ContextItem,
    HybridRetrievalConfig,
    HybridRetrievalResult,
    OptimizedRetrievalHit,
    RetrievalTrace,
)


class _FakeTensor:
    shape = (1, 3)

    def to(self, device: str) -> _FakeTensor:
        return self


class _SequenceTokenizer:
    def __init__(self, generated_texts: list[str]) -> None:
        self.generated_texts = generated_texts

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, _FakeTensor]:
        assert "Sources:" in prompt
        return {"input_ids": _FakeTensor()}

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        if len(self.generated_texts) > 1:
            return self.generated_texts.pop(0)
        return self.generated_texts[0]


class _FakeModel:
    def generate(self, **kwargs: object) -> list[list[int]]:
        return [[1, 2, 3, 4]]


class _StaticHybridRetriever:
    def __init__(self, contexts: list[ContextItem]) -> None:
        self.contexts = contexts

    def retrieve(self, query: str, *, mode: str, top_k: int, **kwargs: Any) -> HybridRetrievalResult:
        assert mode == "hybrid"
        selected = self.contexts[:top_k]
        return HybridRetrievalResult(
            query=query,
            mode="hybrid",
            hits=[_hit_from_context(context) for context in selected],
            contexts=selected,
            config=HybridRetrievalConfig(final_top_k=top_k),
        )


def test_hybrid_source_derived_answer_fills_doctoral_enterprise_practice_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "基本学制5年，最长学制7年；总学分不低于42学分，课程学分不低于40学分；课程实践未提及。 [1]",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(
        _StaticHybridRetriever(
            [
                _context(
                    rank=1,
                    title="2025-电子信息-改革专项-企业联培.pdf",
                    url=(
                        "https://faculty.sist.shanghaitech.edu.cn/office/Academics/Graduate/"
                        "Degree%20Programs/2025-%E7%94%B5%E5%AD%90%E4%BF%A1%E6%81%AF-"
                        "%E6%94%B9%E9%9D%A9%E4%B8%93%E9%A1%B9-%E4%BC%81%E4%B8%9A"
                        "%E8%81%94%E5%9F%B9.pdf"
                    ),
                    text=(
                        "电子信息改革专项专业型直博项目，基本学制为 5 年，最长学制为 7 年，全日制。"
                        "专业学位硕博连读/直博研究生总学分不低于 42 个学分，其中课程学分不低于 40 学分。"
                        "专业课程中的实践教学课程实践部分不低于 8 学分，"
                        "专业实践采用课程实践、企业实践、项目研究形式。"
                    ),
                )
            ]
        ),
        model_path=model_path,
        device="cpu",
    )  # type: ignore[arg-type]

    result = answerer.answer(
        "2025级电子信息博士企业联合培养直博项目的全日制/基本修业年限、最长修业年限、"
        "总学分、课程学分和专业实践课程实践最低学分要求分别是什么？",
        mode="hybrid",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_credit_fields"
    assert "5年" in result.answer
    assert "7年" in result.answer
    assert "42" in result.answer
    assert "40" in result.answer
    assert "8" in result.answer
    assert "[1]" in result.answer


def test_hybrid_source_derived_answer_compares_master_and_direct_phd_duration_credit_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            ("硕士方案中课程学分不低于32学分，公共课不低于8学分，专业课不低于24学分；培养环节不低于1学分。 [2]"),
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(
        _StaticHybridRetriever(
            [
                _context(
                    rank=1,
                    title="2025-计算机科学与技术-硕博连读&直博.pdf",
                    url=(
                        "https://faculty.sist.shanghaitech.edu.cn/office/Academics/Graduate/"
                        "Degree%20Programs/2025-%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%91"
                        "%E5%AD%A6%E4%B8%8E%E6%8A%80%E6%9C%AF-%E7%A1%95%E5%8D%9A"
                        "%E8%BF%9E%E8%AF%BB&%E7%9B%B4%E5%8D%9A.pdf"
                    ),
                    text=(
                        "适用对象：本培养方案适用于上海科技大学 2025 级硕博连读生和直博生。"
                        "三、学制和学分 硕博连读生（含硕士阶段）和直博生基本学制为 5 年，"
                        "最长学制为 7 年。总学分不低于 42 个学分："
                        "其中课程学分不低于 40 学分。"
                    ),
                ),
                _context(
                    rank=2,
                    title="2025-计算机科学与技术-硕士.pdf",
                    url=(
                        "https://faculty.sist.shanghaitech.edu.cn/office/Academics/Graduate/"
                        "Degree%20Programs/2025-%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%91"
                        "%E5%AD%A6%E4%B8%8E%E6%8A%80%E6%9C%AF-%E7%A1%95%E5%A3%AB.pdf"
                    ),
                    text=(
                        "适用对象：本培养方案适用于上海科技大学 2025 级硕士研究生。"
                        "三、学制和学分 硕士研究生基本学制为 3 年，最长学制为 4 年。"
                        "总学分不低于 33 个学分：其中课程学分不低于 32 学分。"
                    ),
                ),
            ]
        ),
        model_path=model_path,
        device="cpu",
    )  # type: ignore[arg-type]

    result = answerer.answer(
        "对比2025级计算机科学与技术硕士培养方案和硕博连读/直博培养方案，"
        "两类学生的基本学制、最长学制和总学分最低要求分别是多少？",
        mode="hybrid",
        top_k=2,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_credit_fields"
    assert "硕士研究生基本学制3年" in result.answer
    assert "最长学制4年" in result.answer
    assert "总学分不低于33学分" in result.answer
    assert "硕博连读生" in result.answer
    assert "基本学制5年" in result.answer
    assert "最长学制7年" in result.answer
    assert "总学分不低于42学分" in result.answer
    assert "[1]" in result.answer
    assert "[2]" in result.answer


def _patch_generation_sequence(monkeypatch: pytest.MonkeyPatch, generated_texts: list[str]) -> None:
    tokenizer = _SequenceTokenizer(generated_texts.copy())

    def fake_load_model(self: RagAnswerer) -> tuple[_SequenceTokenizer, _FakeModel]:
        return tokenizer, _FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)


def _context(*, rank: int, title: str, url: str, text: str) -> ContextItem:
    return ContextItem(
        rank=rank,
        chunk_id=1000 + rank,
        document_id=2000 + rank,
        title=title,
        url=url,
        category="program_requirements",
        language="zh",
        snippet=text[:240],
        text=text,
        trace_ref=f"test:graduate-credit:{rank}",
    )


def _hit_from_context(context: ContextItem) -> OptimizedRetrievalHit:
    return OptimizedRetrievalHit(
        rank=context.rank,
        chunk_id=context.chunk_id,
        document_id=context.document_id,
        title=context.title,
        url=context.url,
        canonical_url=context.url,
        category=context.category,
        language=context.language,
        score=1.0,
        rrf_score=1.0,
        rerank_score=None,
        snippet=context.snippet,
        mode="hybrid",
        trace=RetrievalTrace(
            trace_id=context.trace_ref,
            chunk_id=context.chunk_id,
            sparse_rank=1,
            sparse_score=1.0,
            dense_rank=1,
            dense_score=1.0,
            rrf_score=1.0,
            rerank_score=None,
            final_rank=context.rank,
        ),
    )
