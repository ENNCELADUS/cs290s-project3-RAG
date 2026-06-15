from __future__ import annotations

import json
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
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(self, query: str, *, mode: str, top_k: int, **kwargs: object) -> HybridRetrievalResult:
        self.calls.append((query, mode, top_k))
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


def test_lab_question_falls_back_to_all_required_slots_when_model_only_answers_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "该实验室邮箱是yechf@shanghaitech.edu.cn。 [1]",
            '{"status":"answered","answer":"该实验室邮箱是yechf@shanghaitech.edu.cn。 [1]"}',
        ],
    )
    retriever = _StaticHybridRetriever(
        [
            _context(
                rank=1,
                title="精密传感与智能检测实验室招生通知",
                url="https://sist.shanghaitech.edu.cn/2025/precision-sensing-lab.htm",
                text=(
                    "精密传感与智能检测实验室由叶朝锋课题组负责。"
                    "研究方向：无损检测、电磁测量与成像、电磁场与电路系统建模。"
                    "拟招收2-3个硕士或博士名额。"
                    "联系邮箱：yechf@shanghaitech.edu.cn。"
                    "组长：叶朝锋。"
                ),
            )
        ]
    )
    answerer = RagAnswerer(retriever, model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "精密传感与智能检测实验室的研究方向、招生名额、联系邮箱和组长分别是什么？",
        mode="hybrid",
        top_k=1,
    )

    assert retriever.calls[0][1] == "hybrid"
    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert "无损检测" in result.answer
    assert "电磁测量与成像" in result.answer
    assert "电磁场与电路系统建模" in result.answer
    assert "2-3个硕士或博士名额" in result.answer
    assert "yechf@shanghaitech.edu.cn" in result.answer
    assert "叶朝锋" in result.answer
    assert "[1]" in result.answer


def test_psit_lab_profile_question_falls_back_to_all_required_slots_from_prose_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "联系邮箱是申请请联系yechf@shanghaitech.edu.cn。 [1]",
            '{"status":"answered","answer":"联系邮箱是申请请联系yechf@shanghaitech.edu.cn。 [1]"}',
        ],
    )
    retriever = _StaticHybridRetriever(
        [
            _context(
                rank=1,
                title="精密传感与智能检测实验室 | 智能医学信息研究中心",
                url="https://smirc.sist.shanghaitech.edu.cn/zh/project/psit/",
                text=(
                    "精密传感与智能检测实验室 | 智能医学信息研究中心\n\n"
                    "精密传感与智能检测实验室\n\n"
                    "PSIT课题组的研究方向包括无损检测、电磁测量与成像、电磁场与电路系统建模等。"
                    "课题组每年有2-3个硕士或博士名额，如果你对科研有浓厚的兴趣，欢迎加入我们，"
                    "同时，也欢迎本科生和访问生的加入，申请请联系yechf@shanghaitech.edu.cn。\n\n"
                    "NondestructiveTesting\nMedicalDevices\n\n"
                    "叶朝锋\n副教授\n\n"
                    "叶朝锋教授是上海科技大学精密传感与智能实验室课题组组长"
                ),
            )
        ]
    )
    answerer = RagAnswerer(retriever, model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "精密传感与智能检测实验室的研究方向、招生名额、联系邮箱和课题组组长分别是什么？",
        mode="hybrid",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert "无损检测" in result.answer
    assert "电磁测量与成像" in result.answer
    assert "电磁场与电路系统建模" in result.answer
    assert "2-3个硕士或博士名额" in result.answer
    assert "yechf@shanghaitech.edu.cn" in result.answer
    assert "叶朝锋" in result.answer
    assert "[1]" in result.answer


def test_procurement_question_falls_back_to_each_requested_project_supplier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "超高真空采购项目的供应商是上海甲仪器有限公司。 [1]",
            '{"status":"answered","answer":"超高真空采购项目的供应商是上海甲仪器有限公司。 [1]"}',
        ],
    )
    retriever = _StaticHybridRetriever(
        [
            _context(
                rank=1,
                title="超高真空采购项目询价结果公告",
                url="https://sist.shanghaitech.edu.cn/2025/procurement-a.htm",
                text="项目名称：超高真空采购项目。成交供应商：上海甲仪器有限公司。",
            ),
            _context(
                rank=2,
                title="高速相机采购项目询价结果公告",
                url="https://sist.shanghaitech.edu.cn/2025/procurement-b.htm",
                text="项目名称：高速相机采购项目。成交供应商：上海乙科技有限公司。",
            ),
        ]
    )
    answerer = RagAnswerer(retriever, model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("超高真空采购项目和高速相机采购项目的供应商分别是谁？", mode="hybrid", top_k=2)

    assert retriever.calls[0][1] == "hybrid"
    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert "超高真空采购项目" in result.answer
    assert "上海甲仪器有限公司" in result.answer
    assert "高速相机采购项目" in result.answer
    assert "上海乙科技有限公司" in result.answer
    assert "[1]" in result.answer
    assert "[2]" in result.answer


def test_procurement_result_question_assembles_each_requested_supplier_without_objection_boilerplate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            (
                "上海科技大学二氯二氢硅气体供气设备采购 项目编号： 询价日期： 2019 年 9 月 19 日 "
                "推荐成交单位：上海弗川自动化技术有限公司 投标人如对询价结果有异议，请于本公告发布之日起"
                "三日内以书面形式向上海科技大学信息学院提出异议，公示期满无质疑，不再另行公告询价结果。 [1]"
            ),
            (
                '{"status":"answered","answer":"上海科技大学二氯二氢硅气体供气设备采购 项目编号： '
                "询价日期： 2019 年 9 月 19 日 推荐成交单位：上海弗川自动化技术有限公司 "
                "投标人如对询价结果有异议，请于本公告发布之日起三日内以书面形式向上海科技大学信息学院"
                '提出异议，公示期满无质疑，不再另行公告询价结果。 [1]"}'
            ),
        ],
    )
    retriever = _StaticHybridRetriever(
        [
            _context(
                rank=1,
                title="上海科技大学信息学院二氯二氢硅气体供气设备采购询价结果公告",
                url="https://sist.shanghaitech.edu.cn/2019/0923/c5124a44919/page.htm",
                text=(
                    "上海科技大学信息学院二氯二氢硅气体供气设备采购询价结果公告\n\n"
                    "发布时间：2019-09-23\n"
                    "项目名称：\n上海科技大学二氯二氢硅气体供气设备采购\n\n"
                    "项目编号：\n\n"
                    "询价日期：\n2019\n年\n9\n月\n19\n日\n\n"
                    "推荐成交单位：上海弗川自动化技术有限公司\n\n"
                    "投标人如对询价结果有异议，请于本公告发布之日起三日内以书面形式向"
                    "上海科技大学信息学院（环科路199号信息学院1号楼1B-206）提出异议，"
                    "公示期满无质疑，不再另行公告询价结果。"
                ),
            ),
            _context(
                rank=2,
                title="上海科技大学信息学院磷烷气体供气设备采购询价结果公告",
                url="https://sist.shanghaitech.edu.cn/_t335/2019/0923/c5124a44920/page.htm",
                text=(
                    "上海科技大学信息学院磷烷气体供气设备采购询价结果公告\n\n"
                    "发布时间：2019-09-23\n"
                    "项目名称：\n上海科技大学磷烷气体供气设备采购\n\n"
                    "项目编号：\n\n"
                    "询价日期：\n2019\n年\n9\n月\n19\n日\n\n"
                    "推荐成交单位：上海弗川自动化技术有限公司\n\n"
                    "投标人如对询价结果有异议，请于本公告发布之日起三日内以书面形式向"
                    "上海科技大学信息学院（环科路199号信息学院1号楼1B-206）提出异议，"
                    "公示期满无质疑，不再另行公告询价结果。"
                ),
            ),
        ]
    )
    answerer = RagAnswerer(retriever, model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "在2019年9月发布的官方采购公告中，信息科学与技术学院推荐成交的"
        "“磷烷气体供气设备采购”项目和“二氯二氢硅气体供气设备采购”项目的最终中标成交供应商是哪家公司？",
        mode="hybrid",
        top_k=2,
    )

    assert retriever.calls[0][1] == "hybrid"
    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert "磷烷气体供气设备采购" in result.answer
    assert "二氯二氢硅气体供气设备采购" in result.answer
    assert result.answer.count("上海弗川自动化技术有限公司") >= 2
    assert "[1]" in result.answer
    assert "[2]" in result.answer
    assert "投标人如" not in result.answer
    assert "公示期满" not in result.answer


def test_q030_artifact_procurement_fallback_uses_requested_project_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            (
                "上海科技大学二氯二氢硅气体供气设备采购 项目编号： 询价日期： 2019 年 9 月 19 日 "
                "推荐成交单位：上海弗川自动化技术有限公司 投标人如对询价结果有异议，请于本公告发布之日起"
                "三日内以书面形式向上海科技大学信息学院提出异议，公示期满无质疑，不再另行公告询价结果。 [1]"
            ),
            (
                '{"status":"answered","answer":"上海科技大学二氯二氢硅气体供气设备采购 项目编号： '
                "询价日期： 2019 年 9 月 19 日 推荐成交单位：上海弗川自动化技术有限公司 "
                "投标人如对询价结果有异议，请于本公告发布之日起三日内以书面形式向上海科技大学信息学院"
                '提出异议，公示期满无质疑，不再另行公告询价结果。 [1]"}'
            ),
        ],
    )
    query, contexts = _q030_artifact_query_and_contexts()
    retriever = _StaticHybridRetriever(contexts)
    answerer = RagAnswerer(retriever, model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(query, mode="hybrid", top_k=5)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.answer.index("磷烷气体供气设备采购") < result.answer.index("二氯二氢硅气体供气设备采购")
    assert result.answer.count("上海弗川自动化技术有限公司") >= 2
    assert "[1]" in result.answer
    assert "[3]" in result.answer
    assert "项目编号" not in result.answer
    assert "询价日期" not in result.answer
    assert "投标人如" not in result.answer
    assert "公告期限" not in result.answer


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
        language="zh",
        snippet=text[:240],
        text=text,
        trace_ref=f"test:{rank}",
    )


def _q030_artifact_query_and_contexts() -> tuple[str, list[ContextItem]]:
    artifact_path = Path(
        "data/eval/generation_hybrid_qwen35_20260615T085457Z/run_generation_hybrid_qwen35_20260615T085457Z.jsonl"
    )
    for line in artifact_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["id"] != "q030":
            continue
        contexts = [
            ContextItem(
                rank=context["rank"],
                chunk_id=context["chunk_id"],
                document_id=context["document_id"],
                title=context["title"],
                url=context["url"],
                category=context["category"],
                language=context["language"],
                snippet=context["snippet"],
                text=context["text"],
                trace_ref=context["trace_ref"],
            )
            for context in record["retrieval"]["contexts"]
        ]
        return record["query"], contexts
    raise AssertionError(f"q030 not found in {artifact_path}")


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
