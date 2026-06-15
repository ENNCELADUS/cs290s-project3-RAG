from __future__ import annotations

from pathlib import Path

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

    def retrieve(self, query: str, *, mode: str, top_k: int, **kwargs: object) -> HybridRetrievalResult:
        assert mode == "hybrid"
        selected_contexts = self.contexts[:top_k]
        return HybridRetrievalResult(
            query=query,
            mode="hybrid",
            hits=[_hit_from_context(context) for context in selected_contexts],
            contexts=selected_contexts,
            config=HybridRetrievalConfig(final_top_k=top_k),
        )

    def contexts_for_hits(self, hits: list[object]) -> list[ContextItem]:
        return self.contexts[: len(hits)]


def test_multi_field_profile_query_rejects_one_slot_email_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "Evidence is insufficient.",
            '{"status":"insufficient_evidence","answer":""}',
        ],
    )
    retriever = _StaticHybridRetriever(
        [
            _context(
                rank=1,
                title="Professor Lin Faculty Profile",
                url="https://sist.shanghaitech.edu.cn/faculty/lin",
                text=(
                    "Professor Lin is a SIST faculty member. Contact email: lin@shanghaitech.edu.cn. "
                    "His desk is in SIST Building 3-402. He completed his PhD at UC Berkeley. "
                    "His research interests include machine learning systems."
                ),
            )
        ]
    )
    answerer = RagAnswerer(retriever, model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "What are Professor Lin's office, email, PhD school, and research direction?",
        mode="hybrid",
        top_k=1,
    )

    complete_profile = all(
        expected in result.answer
        for expected in (
            "lin@shanghaitech.edu.cn",
            "SIST Building 3-402",
            "UC Berkeley",
            "machine learning systems",
        )
    )
    assert result.status == "insufficient_evidence" or complete_profile
    assert not (
        result.status == "answered"
        and result.generation_path == "extractive_fallback"
        and "lin@shanghaitech.edu.cn" in result.answer
        and not complete_profile
    )


def test_multi_project_supplier_query_rejects_procurement_objection_boilerplate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status":"insufficient_evidence","answer":""}',
        ],
    )
    retriever = _StaticHybridRetriever(
        [
            _context(
                rank=1,
                title="信息学院采购结果公告",
                url="https://sist.shanghaitech.edu.cn/2025/procurement-results.htm",
                text=(
                    "超高真空采购项目由上海甲仪器有限公司承接。"
                    "高速相机采购项目由上海乙科技有限公司承接。"
                    "投标人如对询价结果有异议，请于公告发布之日起三日内以书面形式提出质疑材料。"
                    "材料递交地址为上海市浦东新区华夏中路393号信息学院1号楼1B-206室。"
                ),
            )
        ]
    )
    answerer = RagAnswerer(retriever, model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("哪些采购项目的供应商分别是谁？", mode="hybrid", top_k=1)

    complete_pairs = all(
        expected in result.answer
        for expected in (
            "超高真空采购项目",
            "上海甲仪器有限公司",
            "高速相机采购项目",
            "上海乙科技有限公司",
        )
    )
    assert result.status == "insufficient_evidence" or complete_pairs
    assert "异议" not in result.answer
    assert "质疑材料" not in result.answer
    assert "递交地址" not in result.answer


def _patch_generation_sequence(monkeypatch: pytest.MonkeyPatch, generated_texts: list[str]) -> None:
    tokenizer = _SequenceTokenizer(generated_texts.copy())

    def fake_load_model(self: RagAnswerer) -> tuple[_SequenceTokenizer, _FakeModel]:
        return tokenizer, _FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)


def _context(*, rank: int, title: str, url: str, text: str) -> ContextItem:
    return ContextItem(
        rank=rank,
        chunk_id=rank,
        document_id=rank,
        title=title,
        url=url,
        category=None,
        language="en",
        snippet=text[:240],
        text=text,
        trace_ref=f"test:{rank}",
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
