from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rag.generate import RagAnswerer, build_prompt
from rag.generate import main as answer_main
from rag.index import DEFAULT_MODEL, build_indexes
from rag.ingest import build_database
from rag.io import atomic_json_dump, write_jsonl
from rag.retrieve import ContextItem, Retriever


class FakeTensor:
    shape = (1, 3)

    def to(self, device: str) -> FakeTensor:
        return self


class FakeTokenizer:
    def __init__(self, generated_text: str) -> None:
        self.generated_text = generated_text

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, FakeTensor]:
        assert "Sources:" in prompt
        return {"input_ids": FakeTensor()}

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        return self.generated_text


class SequenceFakeTokenizer(FakeTokenizer):
    def __init__(self, generated_texts: list[str]) -> None:
        super().__init__(generated_texts[0])
        self.generated_texts = generated_texts

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        if len(self.generated_texts) > 1:
            return self.generated_texts.pop(0)
        return self.generated_texts[0]


class FakeChatTokenizer(FakeTokenizer):
    def __init__(self, generated_text: str) -> None:
        super().__init__(generated_text)
        self.chat_template_kwargs: dict[str, object] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        self.chat_template_kwargs = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        return "<chat prompt>"

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, FakeTensor]:
        assert prompt == "<chat prompt>"
        return {"input_ids": FakeTensor()}


class FakeModel:
    def generate(self, **kwargs: object) -> list[list[int]]:
        return [[1, 2, 3, 4]]


class EmptyRetriever:
    def retrieve(self, query: str, *, mode: str, top_k: int) -> list[object]:
        return []

    def contexts_for_hits(self, hits: list[object]) -> list[object]:
        return []


class StaticContextRetriever:
    def __init__(self, contexts: list[ContextItem]) -> None:
        self.contexts = contexts

    def retrieve(self, query: str, *, mode: str, top_k: int) -> list[object]:
        return list(range(min(top_k, len(self.contexts))))

    def contexts_for_hits(self, hits: list[object]) -> list[ContextItem]:
        return self.contexts[: len(hits)]


class StaticContextRetrieverWithSiblingChunks(StaticContextRetriever):
    def __init__(self, contexts: list[ContextItem], sibling_chunks: list[dict[str, object]]) -> None:
        super().__init__(contexts)
        self._chunks = sibling_chunks


def test_hybrid_answer_generation_returns_numbered_citations(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "The bridge answer is supported by the official source [1].")
    retriever = _retriever_from_paths(paths)
    answerer = RagAnswerer(retriever, model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="hybrid", top_k=2)

    assert result.status == "answered"
    assert result.mode == "hybrid"
    assert "[1]" in result.answer
    assert result.sources[0].source_id == 1
    assert result.sources[0].url.startswith("https://example.edu/")
    assert result.retrieval["mode"] == "hybrid"
    assert result.retrieval["contexts"][0]["text"]


def test_dense_answer_generation_uses_full_text_contexts(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "Dense evidence supports the answer [1].")
    retriever = _retriever_from_paths(paths)
    answerer = RagAnswerer(retriever, model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.mode == "dense"
    assert result.sources[0].trace_ref.startswith("dense:chunk:")
    assert "dense winner semantic source" in result.retrieval["contexts"][0]["text"]


def test_missing_generator_model_path_reports_local_path_error(
    tmp_path: Path, fake_hybrid_sentence_transformer_module
) -> None:
    paths = _build_generation_artifacts(tmp_path)

    with pytest.raises(FileNotFoundError, match="Local generator model path"):
        RagAnswerer(_retriever_from_paths(paths), model_path=tmp_path / "missing-qwen")


def test_cuda_device_request_reports_unavailable_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_hybrid_sentence_transformer_module
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cuda")


def test_no_context_returns_insufficient_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "This should not be generated [1].")
    answerer = RagAnswerer(EmptyRetriever(), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("SIST faculty robotics", mode="dense", top_k=1)

    assert result.status == "insufficient_evidence"
    assert result.sources == []
    assert result.answer.startswith("Evidence is insufficient")


def test_no_source_url_returns_insufficient_evidence(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path, chunk_url=None)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "This should not pass [1].")
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="dense", top_k=1)

    assert result.status == "insufficient_evidence"
    assert result.sources == []


def test_uncited_generation_returns_insufficient_evidence(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "The answer has no numbered citation.")
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="hybrid", top_k=2)

    assert result.status == "insufficient_evidence"
    assert result.answer.startswith("Evidence is insufficient")


def test_uncited_generation_can_be_repaired_to_cited_answer(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "The bridge answer is supported by the official source.",
            '{"status": "answered", "answer": "The bridge answer is supported by the official source [1]."}',
        ],
    )
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="hybrid", top_k=2)

    assert result.status == "answered"
    assert result.answer == "The bridge answer is supported by the official source [1]."


def test_insufficient_generation_can_be_repaired_to_cited_answer(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "Evidence is insufficient to answer this question [1].",
            '{"status": "answered", "answer": "Dense evidence supports the answer [1]."}',
        ],
    )
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.answer == "Dense evidence supports the answer [1]."


@pytest.mark.parametrize(
    "repair_text",
    [
        '{"status": "answered", "answer": "The answer cites a missing source [99]."}',
        '{"status": "answered", "answer": "[1] Dense Winner\\nURL: https://example.edu/b\\nTEXT: leaked"}',
        "not json",
    ],
)
def test_invalid_repair_output_returns_insufficient_evidence(
    tmp_path: Path,
    fake_hybrid_sentence_transformer_module,
    monkeypatch: pytest.MonkeyPatch,
    repair_text: str,
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "The bridge answer is supported by the official source.",
            repair_text,
        ],
    )
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="hybrid", top_k=2)

    assert result.status == "insufficient_evidence"
    assert result.answer.startswith("Evidence is insufficient")


def test_generation_with_unresolved_citation_returns_insufficient_evidence(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "This answer cites a missing source [99] and a real source [1].")
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="hybrid", top_k=2)

    assert result.status == "insufficient_evidence"
    assert result.answer.startswith("Evidence is insufficient")


def test_prompt_leakage_generation_returns_insufficient_evidence(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(
        monkeypatch,
        "[1] Dense Winner\nURL: https://example.edu/b\ntrace_ref: chunk:101\nTEXT:\nleaked source block",
    )
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="hybrid", top_k=2)

    assert result.status == "insufficient_evidence"
    assert result.answer.startswith("Evidence is insufficient")


def test_model_insufficient_evidence_text_returns_structural_insufficient_evidence(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "Evidence is insufficient to answer this question [1].")
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="hybrid", top_k=2)

    assert result.status == "insufficient_evidence"
    assert result.answer.startswith("Evidence is insufficient")


def test_model_refusal_falls_back_to_source_derived_office_email_answer(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(
        tmp_path,
        dense_text=(
            "SIST Teaching Affairs Office is located at Room 1A-105. "
            "Office email: teaching@sist.shanghaitech.edu.cn."
        ),
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "Evidence is insufficient.",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("What is the office and email for SIST Teaching Affairs?", mode="hybrid", top_k=2)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "model_reported_insufficient_evidence"
    assert result.fallback_source_rank == 1
    assert "Room 1A-105" in result.answer
    assert "teaching@sist.shanghaitech.edu.cn" in result.answer
    assert result.answer.endswith("[1].")
    assert "Question:" not in result.answer
    assert "TEXT:" not in result.answer


def test_rejected_generation_uses_repair_before_source_derived_fallback(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(
        tmp_path,
        dense_text=(
            "SIST Teaching Affairs Office is located at Room 1A-105. "
            "Office email: teaching@sist.shanghaitech.edu.cn."
        ),
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "Evidence is insufficient.",
            (
                '{"status": "answered", "answer": "The Teaching Affairs office is Room 1A-105; '
                'email is teaching@sist.shanghaitech.edu.cn [1]."}'
            ),
        ],
    )
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("What is the office and email for SIST Teaching Affairs?", mode="hybrid", top_k=2)

    assert result.status == "answered"
    assert result.generation_path == "repair"
    assert result.generation_rejection_reason == "model_reported_insufficient_evidence"
    assert result.fallback_source_rank is None
    assert (
        result.answer
        == "The Teaching Affairs office is Room 1A-105; email is teaching@sist.shanghaitech.edu.cn [1]."
    )


def test_model_refusal_falls_back_to_source_derived_chinese_office_email_answer(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(
        tmp_path,
        dense_text=(
            "王浩宇 副院长，正教授，博导。办公室： 信息学院3-530 "
            "邮箱： wanghy@shanghaitech.edu.cn 研究方向：电力电子。"
        ),
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("王浩宇教授的办公室具体在哪里？他的工作邮箱是什么？", mode="hybrid", top_k=2)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert "信息学院3-530" in result.answer
    assert "wanghy@shanghaitech.edu.cn" in result.answer
    assert result.answer.endswith("[1].")


def test_chinese_negative_answer_falls_back_to_source_derived_office_email_answer(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(
        tmp_path,
        dense_text=(
            "王浩宇 副院长，正教授，博导。办公室： 信息学院3-530 "
            "邮箱： wanghy@shanghaitech.edu.cn 研究方向：电力电子。"
        ),
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "根据提供的来源，无法找到王浩宇教授办公室的具体位置及其工作邮箱。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("王浩宇教授的办公室具体在哪里？他的工作邮箱是什么？", mode="hybrid", top_k=2)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "model_reported_insufficient_evidence"
    assert "信息学院3-530" in result.answer
    assert "wanghy@shanghaitech.edu.cn" in result.answer
    assert result.answer.endswith("[1].")


def test_source_derived_fallback_abstains_without_question_anchor_overlap(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(
        tmp_path,
        dense_text="Student Club Office is located at Room 2B-201. Office email: clubs@example.edu.",
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "Evidence is insufficient.",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("What is the office and email for SIST Teaching Affairs?", mode="hybrid", top_k=2)

    assert result.status == "insufficient_evidence"
    assert result.generation_path == "insufficient"
    assert result.generation_rejection_reason == "model_reported_insufficient_evidence"
    assert result.fallback_source_rank is None
    assert "clubs@example.edu" not in result.answer


def test_model_refusal_falls_back_to_explicit_course_teacher_evidence(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path, dense_text="Deep Learning 【Dr Alice】")
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "Evidence is insufficient.")
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("Who taught Deep Learning?", mode="hybrid", top_k=2)

    assert result.status == "answered"
    assert result.answer == "Deep Learning was taught by Dr Alice [1]."


def test_model_refusal_falls_back_to_explicit_robotics_faculty_evidence(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path, dense_text="Robotics Lab of Prof. Schwertfeger")
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "Evidence is insufficient.")
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("Which faculty work on robotics?", mode="hybrid", top_k=2)

    assert result.status == "answered"
    assert result.answer == "Prof. Schwertfeger works on robotics [1]."


def test_answer_context_selection_uses_2025_page_for_source_derived_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2022级计算机科学与技术本科培养方案",
            url="https://example.edu/cs/2022-degree",
            text="2022级计算机科学与技术本科培养方案要求学生修满120学分。",
        ),
        _context(
            rank=2,
            title="学院新闻",
            url="https://example.edu/news",
            text="信息学院举行学生交流活动。",
        ),
        _context(
            rank=3,
            title="2025级计算机科学与技术本科培养方案",
            url="https://example.edu/cs/2025-degree",
            text="培养方案要求学生修满140学分。",
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "证据不足：当前检索到的官方来源不足以回答这个问题。")
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("2025级计算机科学与技术本科培养方案需要修满多少学分？", mode="dense", top_k=3)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 3
    assert "2025级计算机科学与技术本科培养方案" in result.answer
    assert "140学分" in result.answer
    assert result.answer.endswith("[3].")
    assert [source.source_id for source in result.sources] == [1, 2, 3]
    assert result.answer_context_order[0]["source_id"] == 3


def test_answer_context_selection_prefers_admissions_direction_anchor_over_curriculum_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="信息学院本科招生页面 CS专业六个方向课程分类",
            url="https://sist.shanghaitech.edu.cn/2025/0310/cs-course-classification/page.htm",
            text=(
                "本科招生页面介绍CS专业六个方向课程分类。"
                "计算机科学与技术专业课程方向包含计算机视觉、计算机图形学、优化方法、软件工程、数据库和网络系统。"
            ),
        ),
        _context(
            rank=4,
            title="信息学院本科招生",
            url="https://sist.shanghaitech.edu.cn/undergraduate/admissions/page.htm",
            text=(
                "信息学院本科招生介绍计算机科学与技术专业。"
                "本专业设置六个学科方向：人工智能、数据科学、计算机系统、信息安全、机器人、生物信息学。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(
        monkeypatch,
        "六个学科方向是人工智能、数据科学、计算机系统、信息安全、机器人、生物信息学 [4].",
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("本科招生页面中，CS专业的六个学科方向是什么？", mode="dense", top_k=2)

    assert result.status == "answered"
    assert [source.source_id for source in result.sources] == [1, 4]
    assert result.answer_context_order[0]["source_id"] == 4
    assert "task_anchor:学科方向" in result.answer_context_order[0]["reasons"]


def test_source_derived_fallback_answers_admissions_discipline_direction_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=4,
            title="信息学院本科招生",
            url="https://sist.shanghaitech.edu.cn/undergraduate/admissions/page.htm",
            text="计算机科学与技术专业介绍。学科方向：人工智能、数据科学、计算机系统、信息安全、机器人、生物信息学。",
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("本科招生页面中，CS专业的学科方向是什么？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 4
    assert result.answer == (
        "CS专业的学科方向是人工智能、数据科学、计算机系统、信息安全、机器人、生物信息学。 [4]"
    )


def test_source_derived_fallback_answers_committee_table_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=5,
            title="信息学院委员会名单",
            url="https://sist.shanghaitech.edu.cn/committee",
            text="委员会 主任 副主任\n教学指导委员会 张明 李华、王强\n学术委员会 陈刚 赵敏",
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("信息学院教学指导委员会的主任和副主任分别是谁？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 5
    assert result.answer == "教学指导委员会主任是张明，副主任是李华、王强。 [5]"


def test_source_derived_fallback_answers_multiple_committee_table_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="信息学院院务委员会",
            url="https://sist.shanghaitech.edu.cn/committee",
            text=(
                "委员会 主任 副主任 委员人数 职责 "
                "学术委员会 哈亚军 何旭明 10 负责学术事务 "
                "学位委员会 寇煦丰 何旭明 14 负责学位事务"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("院务委员会表中，学术委员会和学位委员会的主任分别是谁？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert result.answer == "学术委员会主任是哈亚军；学位委员会主任是寇煦丰。 [1]"


def test_source_derived_fallback_includes_multiple_nearby_credit_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2025级计算机科学与技术本科培养方案",
            url="https://example.edu/cs/2025-degree",
            text=(
                "总学分：140学分。\n"
                "通识教育课程：42学分。\n"
                "专业必修课程：58学分。\n"
                "专业选修课程：28学分。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "证据不足：当前检索到的官方来源不足以回答这个问题。")
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "2025级计算机科学与技术本科培养方案中通识教育课程和专业必修课程分别是多少学分？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert "42学分" in result.answer
    assert "58学分" in result.answer
    assert result.answer.endswith("[1].")


def test_source_derived_fallback_answers_degree_plan_summary_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2025级本科生培养方案EE专业",
            url="https://example.edu/ee/2025-degree",
            text=(
                "2025 级本科生培养方案 电子信息工程专业。"
                "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
                "类别 必修 选修 学分 人文社科通识 30 15 45 "
                "自然科学通识 16 16 32 专业课程 32 27 59 任选课程 9 145。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "根据2025级电子信息工程专业（EE专业）本科生培养方案，学生毕业至少需要修满多少学分？其中任选课占多少学分？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.answer == "2025级电子信息工程专业毕业至少需要修满145学分，任选课程占9学分。 [1]"


def test_generation_rejects_degree_plan_answer_that_copies_total_into_free_elective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2025级本科生培养方案EE专业",
            url="https://example.edu/ee/2025-degree",
            text=(
                "2025 级本科生培养方案 电子信息工程专业。"
                "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
                "类别 必修 选修 学分 人文社科通识 30 15 45 "
                "自然科学通识 16 16 32 专业课程 32 27 59 任选课程 9 145。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "2025级电子信息工程专业毕业至少需要修满145学分，任选课程占145学分。 [1]",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "根据2025级电子信息工程专业（EE专业）本科生培养方案，学生毕业至少需要修满多少学分？其中任选课占多少学分？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "unsupported_label_value_binding"
    assert result.answer == "2025级电子信息工程专业毕业至少需要修满145学分，任选课程占9学分。 [1]"


def test_cited_degree_plan_answer_with_limited_caveat_is_not_treated_as_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2025级本科生培养方案EE专业",
            url="https://example.edu/ee/2025-degree",
            text=(
                "2025 级本科生培养方案 电子信息工程专业。"
                "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
                "类别 必修 选修 学分 人文社科通识 30 15 45 "
                "自然科学通识 16 16 32 专业课程 32 27 59 任选课程 9 145。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    generated = "2025级电子信息工程专业毕业至少需要修满145学分，任选课程占9学分；未提及其他要求。 [1]"
    _patch_generation_sequence(
        monkeypatch,
        [
            generated,
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "根据2025级电子信息工程专业（EE专业）本科生培养方案，学生毕业至少需要修满多少学分？其中任选课占多少学分？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "initial"
    assert result.answer == generated


def test_source_derived_fallback_answers_ee_professional_course_elective_credits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2025级本科生培养方案EE专业",
            url="https://example.edu/ee/2025-degree",
            text=(
                "2025 级本科生培养方案 电子信息工程专业。"
                "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
                "类别 必修 选修 学分 人文社科通识 30 15 45 "
                "自然科学通识 16 16 32 专业课程 32 27 59 任选课程 9 145。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "专业课程板块采用三选二规则。 [1]",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("2025级EE本科培养方案的专业课程板块至少需要多少选修学分？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_professional_elective_credits"
    assert result.answer == "2025级电子信息工程专业中，专业课程板块至少需要选修27学分。 [1]"


def test_generation_rejects_unsupported_retest_formula_weights_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2026年信息学院硕士研究生招生复试规程",
            url="https://example.edu/admission/2026-retest",
            text=(
                "2026年信息学院硕士研究生招生复试规程。"
                "复试内容包括综合素质考核和专业面试。"
                "复试成绩满分为100分，60分为合格。"
                "考生总成绩=初试成绩×50%+复试成绩×50%。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "2026年复试包括英语听力和专业课笔试，总成绩=初试成绩×20%+复试成绩×80%。 [1]",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "2026年信息学院硕士研究生招生复试包括哪些部分？复试成绩满分和合格线是多少？总成绩公式是什么？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "unsupported_numeric_formula"
    assert (
        result.answer
        == "2026年复试包括综合素质考核和专业面试；复试成绩满分为100分，60分为合格；"
        "考生总成绩=初试成绩×50%+复试成绩×50%。 [1]"
    )


def test_retest_formula_fallback_handles_score_normalization_formula(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2026年信息学院硕士研究生招生复试规程",
            url="https://example.edu/admission/2026-retest",
            text=(
                "2026年信息学院硕士研究生招生复试工作规程。"
                "复试内容包括综合素质考核和专业面试。"
                "复试成绩满分为100分，60分为合格。"
                "考生总成绩=50*初试成绩/初试满分+50*复试成绩/复试满分。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "2026年信息学院硕士研究生招生复试包括哪些部分？复试成绩满分和合格线是多少？总成绩如何计算？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert (
        result.answer
        == "2026年复试包括综合素质考核和专业面试；复试成绩满分为100分，60分为合格；"
        "考生总成绩=50*初试成绩/初试满分+50*复试成绩/复试满分。 [1]"
    )


def test_generation_rejects_adjacent_stats_when_requested_lab_counts_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="信息学院科研平台",
            url="https://example.edu/research/platforms",
            text=(
                "信息学院科研平台建设情况：学院下设83个课题组、4个联合实验室、"
                "7个研究中心，拥有200余名导师和30余家合作企业。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "信息学院有7个研究中心、200余名导师和30余家合作企业。 [1]",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("信息学院有多少个课题组和多少个联合实验室？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_labeled_fact"
    assert result.answer == "信息学院有83个课题组、4个联合实验室。 [1]"


def test_source_derived_fallback_compares_two_degree_plan_summary_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="25级本科生培养方案CS专业人工智能荣誉班",
            url="https://example.edu/cs-ai/2025-degree",
            text=(
                "2025 级计算机科学与技术专业 人工智能荣誉班培养方案。"
                "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
                "类别 必修 选修 学分 人文社科通识 30 15 45 "
                "自然科学通识 12 16 28 专业课程 42 26 68 任选课程 4 145。"
            ),
        ),
        _context(
            rank=2,
            title="2025级本科生培养方案CS专业",
            url="https://example.edu/cs/2025-degree",
            text=(
                "2025 级本科生培养方案 计算机科学与技术。"
                "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
                "类别 必修 选修 学分 人文社科通识 30 15 45 "
                "自然科学通识 16 16 32 专业课程 20 39 59 任选课程 9 145。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "对比2025级普通CS专业和CS专业人工智能荣誉班本科培养方案，两者在自然科学通识板块和专业课程板块的总学分要求分别是多少？",
        mode="dense",
        top_k=2,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert (
        result.answer
        == (
            "2025级普通CS专业要求自然科学通识32学分、专业课程59学分；"
            "CS专业人工智能荣誉班要求自然科学通识28学分、专业课程68学分。 [2][1]"
        )
    )


def test_source_derived_fallback_answers_course_design_pair_from_degree_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2025级本科生培养方案CS专业",
            url="https://example.edu/cs/2025-degree",
            text=(
                "2025级本科生培养方案CS专业。专业必修课程 课程代码 课程名称 学分 建议修读学期 "
                "CS110 计算机体系结构 I 4 二（2） "
                "CS110P 计算机体系结构 I 课程设计 2 二（2） "
                "CS120 操作系统 4 三（1）。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "2025级CS本科专业必修课中，与“计算机体系结构I”配套的课程设计代码是什么？理论课与课程设计合计多少学分，推荐在哪个学期修读？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert (
        result.answer
        == (
            "配套课程设计是CS110P《计算机体系结构I课程设计》。"
            "CS110理论课4学分、CS110P课程设计2学分，合计6学分，均推荐在二（2）学期修读。 [1]"
        )
    )


def test_source_derived_fallback_answers_compact_course_credit_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2025级本科生培养方案EE专业",
            url="https://example.edu/ee/2025-degree",
            text=(
                "专业必修课程 课程代码 课程名称 学分 建议修读学期 "
                "EE111 电路基础 3 二（1） "
                "EE111L 电路基础实验 1 二（1） "
                "EE112 模拟电子技术 4 二（2）。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("EE111L是什么课程？它是多少学分？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert result.answer == "EE111L是《电路基础实验》，1学分。 [1]"


def test_source_derived_fallback_answers_degree_plan_dedup_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2023级本科生培养方案",
            url="https://example.edu/cs/2023-degree",
            text=(
                "注意：同一门课程在培养方案中不重复计算学分，教务系统“计划完成情况”会同时检查每个层级每一个模块中的"
                "“课程门数要求”与“学分要求”是否同时满足，且在计算获得学分时会进行课程自动去重。"
                "举例：《计算机视觉 I 》同时出现在“专业方向必修”和“专业任选”模块中，如果修读《计算机视觉 I 》并通过，"
                "则在“专业方向必修”和“专业任选”两个模块会同时计入学分，但在上一层级“本学科选修”模块计算获得总学分时"
                "《计算机视觉 I 》的学分仅会被计算 1 次。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "假设一名学生修读并通过了同时属于“专业方向必修”和“专业任选”两个模块的交叉核心课程，教务系统在最终结算上一层级的“本学科选修”总学分时，会采用什么样的自动去重规则？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.answer == "教务系统在结算上一层级总学分时会自动去重；该课程学分最终仅计算1次，不会重复累加。 [1]"


def test_source_derived_fallback_answers_fu_minfan_power_electronics_video_course(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="傅旻帆：“大学教授”VS“百万UP主”，不设限的热爱",
            url="https://sist.shanghaitech.edu.cn/2026/0325/c2858a1120188/page.htm",
            text=(
                "傅旻帆介绍，他将面向本科生开设专业选修课《电力电子》统一录制成视频，"
                "让学生提前学习，其内容同样也能为研究生打下基础。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "傅旻帆提到的《电力电子》是哪门录制成视频、让学生提前学习的核心选修课？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert result.answer == "傅旻帆建议学生提前学习的专业选修课是《电力电子》。 [1]"


def test_source_derived_fallback_prefers_factual_source_over_navigation_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2025级计算机科学与技术本科培养方案导航",
            url="https://example.edu/cs/2025-degree-nav",
            text=(
                "首页 | 导航 | 站点地图 | 友情链接 | 2025级计算机科学与技术本科培养方案 | "
                "总学分140学分 | 版权所有"
            ),
        ),
        _context(
            rank=2,
            title="2025级计算机科学与技术本科培养方案",
            url="https://example.edu/cs/2025-degree",
            text="2025级计算机科学与技术本科培养方案要求学生修满140学分。",
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "证据不足：当前检索到的官方来源不足以回答这个问题。")
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("2025级计算机科学与技术本科培养方案需要修满多少学分？", mode="dense", top_k=2)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 2
    assert "140学分" in result.answer
    assert "首页" not in result.answer
    assert result.answer.endswith("[2].")


def test_source_derived_fallback_prefers_report_detail_page_over_homepage_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="School of Information Science and Technology",
            url="https://sist.shanghaitech.edu.cn/sist_en/main.htm",
            text=(
                "面向分布式系统的可观测性技术研究 学术报告 信息学院 2026。"
                "演讲者：李晨辉，华东师范大学。"
                "邀请人：李权。"
                "时间：2026年4月27日 上午11:00。"
                "地点：信息学院1A-200。"
            ),
        ),
        _context(
            rank=3,
            title="面向分布式系统的可观测性技术研究",
            url="https://sist.shanghaitech.edu.cn/sist_en/2026/0424/c11304a1121066/page.htm",
            text=(
                "面向分布式系统的可观测性技术研究。"
                "演讲者：陈明，清华大学。"
                "邀请人：王强。"
                "时间：2026年4月24日 10:00。"
                "地点：信息学院1A-200。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "证据不足：当前检索到的官方来源不足以回答这个问题。")
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "根据信息学院官网在2026年4月24日发布的《面向分布式系统的可观测性技术研究》报告通知，"
        "该报告的演讲者、所在单位、邀请人、时间和地点是什么？",
        mode="dense",
        top_k=2,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 3
    assert "陈明" in result.answer
    assert "清华大学" in result.answer
    assert "2026年4月24日" in result.answer
    assert "李晨辉" not in result.answer
    assert result.answer.endswith("[3].")


def test_source_derived_fallback_answers_seminar_event_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="智能感知与机器人学术报告",
            url="https://sist.shanghaitech.edu.cn/2026/0424/seminar",
            text=(
                "智能感知与机器人学术报告。\n"
                "报告人：陈明\n"
                "单位：清华大学\n"
                "时间：2026年4月24日10:00\n"
                "地点：信息学院1A-200"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("该学术报告的报告人、所在单位、时间和地点是什么？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert result.answer == "报告人是陈明，单位是清华大学，时间是2026年4月24日10:00，地点是信息学院1A-200。 [1]"


def test_source_derived_fallback_answers_seminar_institution_and_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="增益型碳化硅传感器研发",
            url="https://sist.shanghaitech.edu.cn/2026/0416/seminar",
            text=(
                "专业学术报告《增益型碳化硅传感器研发》。"
                "演讲者: 史欣，中国科学院高能物理研究所 "
                "时间: 2026年4月16日，下午14:00 "
                "地点: 信息学院 3-301"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "史欣老师来自中国科学院高能物理研究所。 [1]",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "在2026年4月16日下午举办的专业学术报告《增益型碳化硅传感器研发》中，"
        "受邀主讲人史欣老师来自哪一个科研机构？该场报告在学院大楼的哪一个房间进行？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_location_fact"
    assert result.fallback_source_rank == 1
    assert result.answer == "史欣老师来自中国科学院高能物理研究所，报告地点是信息学院3-301。 [1]"


def test_source_derived_fallback_uses_target_contact_notice_instead_of_mentioned_faculty_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="傅旻帆个人主页",
            url="https://sist.shanghaitech.edu.cn/fumf/main.htm",
            text=(
                "傅旻帆，信息学院常任副教授，本科毕业于上海交通大学密西根学院。"
                "办公室：信息学院3-534。邮箱：fumf@shanghaitech.edu.cn。"
            ),
        ),
        _context(
            rank=5,
            title="傅旻帆：“大学教授”VS“百万UP主”，不设限的热爱",
            url="https://sist.shanghaitech.edu.cn/2026/0325/c2858a1120188/page.htm",
            text=(
                "接受过《“大学教授”VS“百万UP主”》专访的信息学院常任副教授傅旻帆，"
                "本科毕业于上海交通大学密西根学院。"
                "研究生招生咨询请联系高老师，办公室：信息学院1A-207。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "证据不足：当前检索到的官方来源不足以回答这个问题。")
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "接受过《“大学教授”VS“百万UP主”》专访的信息学院常任副教授傅旻帆，他本科毕业于上海交通大学密西根学院。"
        "请问他目前任职的上海科技大学信息学院中，负责研究生招生咨询的高老师官方办公室在几号楼的哪个房间？",
        mode="dense",
        top_k=2,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 5
    assert "高老师" in result.answer
    assert "信息学院1A-207" in result.answer
    assert "信息学院3-534" not in result.answer
    assert result.answer.endswith("[5].")


def test_source_derived_fallback_uses_exact_dated_notice_for_training_dates_and_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="信息学院计算机与通信学科平台机械加工间技能培训",
            url="https://sist.shanghaitech.edu.cn/2025/0403/c5304a1108817/page.htm",
            text=(
                "信息学院计算机与通信学科平台机械加工间技能培训通知。"
                "UG三维模型设计培训安排在4月10日、4月12日。"
                "如有疑问请联系吕老师，邮箱：lvky@shanghaitech.edu.cn。"
            ),
        ),
        _context(
            rank=5,
            title="信息学院计算机与通信学科平台机械加工间技能培训",
            url="https://sist.shanghaitech.edu.cn/2025/0915/c5304a1115123/page.htm",
            text=(
                "2025年9月15日发布机械加工间技能培训通知。"
                "UG三维模型设计培训安排在9月22日、9月24日。"
                "如有疑问请联系张老师，邮箱：zhangls@shanghaitech.edu.cn。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "证据不足：当前检索到的官方来源不足以回答这个问题。")
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "根据2025年9月15日发布的机械加工间技能培训通知，UG三维模型设计培训安排在哪些日期？"
        "如有疑问应联系谁、邮箱是什么？",
        mode="dense",
        top_k=2,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 5
    assert "9月22日" in result.answer
    assert "9月24日" in result.answer
    assert "张老师" in result.answer
    assert "zhangls@shanghaitech.edu.cn" in result.answer
    assert "4月10日" not in result.answer
    assert result.answer.endswith("[5].")


def test_incomplete_contact_generation_falls_back_to_source_phone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="磷烷气体供气设备采购询价公告",
            url="https://example.edu/2019/0911/procurement",
            text=(
                "报价供应商要求：具有独立承担民事责任的能力，具有企业法人营业执照、税务登记证、"
                "组织机构代码证复印件，本项目不接受联合体报价。"
                "报名资料请发送给刘老师，电话：021-20685370，邮箱：liutt1@shanghaitech.edu.cn。"
                "报价截止时间：2019年9月19日9:30。递交地点：华夏中路393号信息学院1号楼B区206。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            (
                "供应商需具备独立承担民事责任能力及相关证照，不接受联合体报价；"
                "报名资料发送给刘老师，邮箱liutt1@shanghaitech.edu.cn；"
                "报价截止时间为2019年9月19日9:30，递交地点为华夏中路393号信息学院1号楼B区206。 [1]"
            ),
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "如果供应商想参与上海科技大学磷烷气体供气设备询价，需要满足哪些报价供应商要求？"
        "报名资料应发送给哪位联系人，报价截止时间和递交地点是什么？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_phone_fact"
    assert result.fallback_source_rank == 1
    assert "021-20685370" in result.answer
    assert "liutt1@shanghaitech.edu.cn" in result.answer
    assert "2019年9月19日9:30" in result.answer
    assert "1号楼B区206" in result.answer
    assert result.answer.endswith("[1].")


def test_source_derived_fallback_composes_procurement_notice_answer_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="上海科技大学磷烷气体供气设备询价公告",
            url="https://sist.shanghaitech.edu.cn/2019/0911/c5124a44707/page.htm",
            text=(
                "项目名称 上海科技大学磷烷气体供气设备。\n"
                "报价供应商要求\n"
                "1，供应商须能独立承担民事责任，具有能从事该项目范围内的企业法人营业执照、"
                "税务登记证、组织机构代码证复印件；\n"
                "2，本项目不允许联合体报价。\n"
                "报名方式 请将以上所需报名资料复印件加盖公章扫描后发至联系人邮箱领取本次询价的需求文件。\n"
                "联系人：刘老师\n"
                "电话：021-20685370\n"
                "邮箱：liutt1@shanghaitech.edu.cn\n"
                "报价截止时间 2019年9月19日9:30时（北京时间）\n"
                "报价文件递交地点 上海市浦东新区华夏中路393号信息学院1号楼B区206"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "报名资料发送给刘老师，邮箱liutt1@shanghaitech.edu.cn。[1]",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "如果供应商想参与上海科技大学磷烷气体供气设备询价，需要满足哪些报价供应商要求？"
        "报名资料应发送给哪位联系人，报价截止时间和递交地点是什么？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert "独立承担民事责任" in result.answer
    assert "营业执照、税务登记证、组织机构代码证复印件" in result.answer
    assert "不允许联合体报价" in result.answer
    assert "刘老师" in result.answer
    assert "021-20685370" in result.answer
    assert "liutt1@shanghaitech.edu.cn" in result.answer
    assert "2019年9月19日9:30" in result.answer
    assert "信息学院1号楼B区206" in result.answer
    assert result.answer.endswith("[1].")


def test_source_derived_fallback_composes_procurement_objection_delivery_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="上海科技大学信息学院磷烷气体供气设备采购询价结果公告",
            url="https://sist.shanghaitech.edu.cn/2019/0923/c5124a44920/page.htm",
            text=(
                "上海科技大学信息学院磷烷气体供气设备采购询价结果公告。"
                "投标人如对询价结果有异议，请于本公告发布之日起三日内以书面形式提出质疑材料。"
                "受理人为刘老师。"
                "材料递交地址为上海市浦东新区华夏中路393号信息学院1号楼1B-206室。"
            ),
        )
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "假设某个外部投标供应商对信息学院公开发布的“磷烷气体供气设备采购”询价结果存有异议，"
        "如果其需要按照规定在发布之日起三日内递交书面质疑材料，其应该把材料具体递交到"
        "华夏中路校区哪栋建筑的哪一个办公室？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert "刘老师" in result.answer
    assert "信息学院1号楼1B-206室" in result.answer
    assert result.answer.endswith("[1].")


def test_incomplete_training_cap_generation_falls_back_to_source_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="移动机器人TurtleBot4技能培训通知",
            url="https://example.edu/2025/0606/turtlebot4",
            text=(
                "移动机器人TurtleBot4技能培训时间为2025年7月6日至7月11日，地点为信息学院3号楼201室。"
                "培训人数不超过36人。"
                "理论部分由Sören Schwertfeger教授主讲。"
                "咨询请联系王老师，电话：021-20685706，邮箱：wangyl3@shanghaitech.edu.cn。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            (
                "培训时间为2025年7月6日至7月11日，地点在信息学院3号楼201室，"
                "已有24位同学参加，理论部分由Sören Schwertfeger教授主讲；"
                "咨询请联系王老师，电话021-20685706，邮箱wangyl3@shanghaitech.edu.cn。 [1]"
            ),
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "根据2025年6月6日发布的移动机器人TurtleBot4技能培训通知，培训时间、地点、人数上限、"
        "理论主讲人和咨询联系方式分别是什么？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_capacity_limit"
    assert result.fallback_source_rank == 1
    assert "2025年7月6日至7月11日" in result.answer
    assert "信息学院3号楼201室" in result.answer
    assert "不超过36人" in result.answer
    assert "Sören Schwertfeger" in result.answer
    assert "021-20685706" in result.answer
    assert "wangyl3@shanghaitech.edu.cn" in result.answer
    assert "24位同学" not in result.answer
    assert result.answer.endswith("[1].")


def test_source_derived_fallback_answers_compact_person_profile_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="信息学院博士生李泽晖",
            url="https://sist.shanghaitech.edu.cn/profile/lizehui",
            text="李泽晖 身份：博士生 教育背景：上海大学本科 研究方向：算力电源",
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("李泽晖的身份、教育背景和研究方向是什么？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert result.answer == "李泽晖的身份是博士生，教育背景是上海大学本科，研究方向是算力电源。 [1]"


def test_source_derived_fallback_answers_lab_member_row_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=3,
            title="智能电源实验室成员",
            url="https://sist.shanghaitech.edu.cn/lab/power/members",
            text=(
                "姓名 身份 教育背景 研究方向 专利经历\n"
                "李泽晖 博士生 上海大学本科 算力电源 参与柔性电源控制相关专利\n"
                "王明 硕士生 浙江大学本科 智能芯片 参与芯片测试平台专利"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("李泽晖的身份、教育背景和研究方向是什么？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 3
    assert result.answer == "李泽晖的身份是博士生，教育背景是上海大学本科，研究方向是算力电源。 [3]"


def test_source_derived_fallback_resolves_singular_student_before_undergraduate_school(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=2,
            title="信息学院研究生风采",
            url="https://sist.shanghaitech.edu.cn/2026/0418/c2863a1120870/page.htm",
            text=(
                "信息学院研究生风采。"
                "博士研究生李泽辉，本科毕业于上海大学，研究方向为算力电源。"
                "博士研究生王明，本科毕业于浙江大学，研究方向为智能芯片。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "研究方向涉及算力电源的那位信息学院博士研究生，他的本科毕业院校是哪所？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 2
    assert "李泽辉" in result.answer
    assert "上海大学" in result.answer
    assert "王明" not in result.answer
    assert "浙江大学" not in result.answer
    assert result.answer.endswith("[2].")


def test_source_derived_fallback_uses_current_doctoral_lab_member_row_for_undergraduate_school(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="王浩宇课题组成员",
            url="https://sist.shanghaitech.edu.cn/lab/wanghy/members",
            text=(
                "姓名 身份 教育背景 研究方向\n"
                "张强 校友 浙江大学本科 算力电源\n"
                "李泽晖 博士生 上海大学本科 算力电源\n"
                "王明 博士生 北京理工大学本科 电源芯片"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "王浩宇课题组目前在读的博士生中，研究方向是算力电源的学生，本科毕业院校是哪所？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert result.answer == "李泽晖的本科毕业院校是上海大学。 [1]."
    assert "张强" not in result.answer
    assert "浙江大学" not in result.answer


def test_source_derived_fallback_answers_faculty_profile_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=2,
            title="王浩宇个人主页",
            url="https://sist.shanghaitech.edu.cn/faculty/wanghy",
            text=(
                "王浩宇，信息学院教授，博士毕业于浙江大学。"
                "办公室：信息学院3-530 邮箱：wanghy@shanghaitech.edu.cn "
                "研究方向：电力电子与智能电网。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("王浩宇教授的办公室、邮箱、博士毕业学校和研究方向是什么？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 2
    assert result.answer == (
        "王浩宇的办公室是信息学院3-530，邮箱是wanghy@shanghaitech.edu.cn，"
        "博士毕业学校是浙江大学，研究方向是电力电子与智能电网。 [2]"
    )


def test_source_derived_fallback_answers_requested_faculty_contact_slots_without_profile_slot_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="王浩宇个人主页",
            url="https://sist.shanghaitech.edu.cn/faculty/wanghy",
            text=(
                "王浩宇，信息学院教授。博士毕业院校：美国马里兰大学。"
                "办公室：信息学院3-530 邮箱：wanghy@shanghaitech.edu.cn "
                "研究方向：电力电子，算力电源，电源芯片，电动汽车，光伏储能。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("王浩宇教授的办公室具体在哪里？他的工作邮箱是什么？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert result.answer == "王浩宇的办公室是信息学院3-530，邮箱是wanghy@shanghaitech.edu.cn。 [1]."


def test_source_derived_fallback_uses_profile_where_identifying_anchors_cooccur(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="廉黎祥个人主页",
            url="https://sist.shanghaitech.edu.cn/faculty/lianlx",
            text=(
                "廉黎祥，信息学院教授。博士毕业院校：香港科技大学。"
                "办公室：信息学院3-421 邮箱：lianlx@shanghaitech.edu.cn "
                "研究方向：集成电路设计、芯片测试。"
            ),
        ),
        _context(
            rank=2,
            title="张芯韵个人主页",
            url="https://sist.shanghaitech.edu.cn/faculty/zhangxy",
            text=(
                "张芯韵，信息学院助理教授。博士毕业院校：香港中文大学。"
                "办公室：3-210 邮箱：zhangxy12@shanghaitech.edu.cn "
                "研究方向：AI驱动的芯片设计自动化 (AI4EDA)、IC可制造性设计、IC物理设计。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "博士毕业于香港中文大学且研究方向包括AI驱动的芯片设计自动化(AI4EDA)的教授，办公室和邮箱是什么？",
        mode="dense",
        top_k=2,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 2
    assert result.answer == "该教师的办公室是3-210，邮箱是zhangxy12@shanghaitech.edu.cn。 [2]."
    assert "lianlx@shanghaitech.edu.cn" not in result.answer


def test_source_derived_fallback_treats_profile_descriptors_as_identifying_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query, contexts = _artifact_query_and_contexts("q018")
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(query, mode="dense", top_k=5)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert result.answer == "该教师的办公室是3-210，邮箱是zhangxy12@shanghaitech.edu.cn。 [1]."
    assert "lianlx@shanghaitech.edu.cn" not in result.answer


def test_q002_artifact_model_refusal_recovers_office_email_from_top_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query, contexts = _artifact_query_and_contexts("q002")
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(query, mode="dense", top_k=5)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.fallback_source_rank == 1
    assert result.answer == "王浩宇的办公室是信息学院3-530，邮箱是wanghy@shanghaitech.edu.cn。 [1]."


def test_source_derived_fallback_answers_requested_professor_direction_and_phd_school(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="中文信息",
            url="https://sist.shanghaitech.edu.cn/wanghy/list.htm",
            text=(
                "王浩宇 副院长，正教授，博导。"
                "博士毕业院校： 美国马里兰大学。"
                "办公室： 信息学院3-530。"
                "邮箱： wanghy@shanghaitech.edu.cn。"
                "研究方向： 电力电子，算力电源，电源芯片，电动汽车，光伏储能。"
            ),
        )
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "证据不足：当前检索到的官方来源不足以回答这个问题。",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "信息科学与技术学院副院长王浩宇教授主要负责哪些研究方向？他的博士学位毕业于哪一所海外著名的高校？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.answer == (
        "王浩宇的博士毕业学校是美国马里兰大学，研究方向是电力电子，算力电源，电源芯片，电动汽车，光伏储能。 [1]"
    )


def test_source_derived_fallback_compares_two_professor_office_rooms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="英文信息",
            url="https://sist.shanghaitech.edu.cn/hexm_en/list.htm",
            text=(
                "Xuming He. Vice Dean, Associate Professor. "
                "Office: 1A-221, SIST Building. E-mail: hexm@shanghaitech.edu.cn."
            ),
        ),
        _context(
            rank=2,
            title="中文信息",
            url="https://sist.shanghaitech.edu.cn/tukw/main.htm",
            text=(
                "屠可伟 副院长、正教授、博导。"
                "电话：021-20685089。"
                "办公室： 1A-304B。"
                "邮箱： tukw@shanghaitech.edu.cn。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "何旭明的办公室是1A-221；屠可伟的办公室未包含在来源中。 [1]",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "对比信息学院两位副院长何旭明（Xuming He）与屠可伟（Kewei Tu）的长聘教职工概况主页，"
        "他们各自在信息学院大楼（SIST Building）的官方办公室房间号分别是什么？",
        mode="dense",
        top_k=2,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.answer == "何旭明的办公室房间号是1A-221；屠可伟的办公室房间号是1A-304B。 [1][2]"


def test_source_derived_fallback_uses_retest_formula_from_same_document_sibling_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="上海科技大学信息科学与技术学院2026年招收硕士研究生复试工作规程",
            url="https://sist.shanghaitech.edu.cn/_t335/2026/0316/c7339a1119896/page.htm",
            text="上海科技大学信息科学与技术学院2026年招收硕士研究生复试工作规程。首页 导航 复试分数线。",
        )
    ]
    sibling_chunks = [
        {
            "chunk_id": 2,
            "document_id": 1,
            "url": contexts[0].url,
            "text": (
                "五、复试工作细则 复试包括综合素质考核和专业面试两部分。"
                "复试成绩满分 100 分， 60 分及以上为合格。"
                "六、录取原则 总成绩满分 100 分，计算方法："
                "总成绩 =50* 初试成绩 / 初试满分 +50* 复试成绩 / 复试满分。"
            ),
        }
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "复试包括综合素质考核和专业面试，但来源中未明确列出复试满分和总成绩公式。 [1]",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(
        StaticContextRetrieverWithSiblingChunks(contexts, sibling_chunks),
        model_path=model_path,
        device="cpu",
    )  # type: ignore[arg-type]

    result = answerer.answer(
        "根据《上海科技大学信息科学与技术学院2026年招收硕士研究生复试工作规程》，"
        "复试包括哪两个部分？复试成绩满分和合格线分别是多少？总成绩如何计算？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.answer == (
        "2026年复试包括综合素质考核和专业面试；复试成绩满分为100分，60分为合格；"
        "考生总成绩=50*初试成绩/初试满分+50*复试成绩/复试满分。 [1]"
    )


def test_source_derived_fallback_binds_starred_uc_berkeley_course_from_sibling_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2025级本科生培养方案EE专业",
            url="https://faculty.sist.shanghaitech.edu.cn/office/Academics/Undergraduate/Degree%20Programmes/2025%20Bachelor%20Degree%20Programs%20in%20EE.htm",
            text="2025级本科生培养方案EE专业。电子信息工程专业。一、培养目标。二、学制、学位类型与要求。",
        )
    ]
    sibling_chunks = [
        {
            "chunk_id": 2,
            "document_id": 1,
            "url": contexts[0].url,
            "text": (
                "专业必修课程板块 课程代码 课程名称 学时 学分 开课学期 "
                "EE111 电路基础 * 64 4 一（2） "
                "EE111L 电路基础实验 * 48 1 一（2）。"
            ),
        },
        {
            "chunk_id": 3,
            "document_id": 1,
            "url": contexts[0].url,
            "text": "注：本课程设置仅作为推荐。加 “*” 号的课程为 UC Berkeley 课程。",
        },
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "来源中未明确列出任何一门具体课程标注为 UC Berkeley 合作课程。 [1]",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(
        StaticContextRetrieverWithSiblingChunks(contexts, sibling_chunks),
        model_path=model_path,
        device="cpu",
    )  # type: ignore[arg-type]

    result = answerer.answer(
        "针对2025级电子信息工程专业（EE专业）本科生，修读由美国加州大学伯克利分校（UC Berkeley）"
        "合作开设的专业必修课程，安排在哪个学年哪个学期推荐修读？这门课程的理论课和实验课课程代码分别是什么？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.answer == (
        "标有“*”号的UC Berkeley合作必修课程为《电路基础》（含电路基础实验），"
        "推荐修读时间为一（2）学期；理论课课程代码为EE111，实验课课程代码为EE111L。 [1]"
    )


def test_partial_contact_answer_falls_back_to_all_requested_teacher_profile_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="赵登吉个人主页",
            url="https://sist.shanghaitech.edu.cn/faculty/zhaodj",
            text=(
                "赵登吉，信息学院教授。博士毕业院校：澳大利亚西悉尼大学和法国图卢兹大学。"
                "办公室：信息学院1A-304E室 邮箱：zhaodj@shanghaitech.edu.cn "
                "研究方向：人工智能、多智能体系统、算法博弈论、机制设计、在线合作博弈、AI Agents。"
            ),
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "email: zhaodj@shanghaitech.edu.cn [1].",
            '{"status": "insufficient_evidence", "answer": ""}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("赵登吉老师的博士毕业院校、办公室、邮箱和研究方向是什么？", mode="dense", top_k=1)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_profile_fact"
    assert result.fallback_source_rank == 1
    assert result.answer == (
        "赵登吉的办公室是信息学院1A-304E室，邮箱是zhaodj@shanghaitech.edu.cn，"
        "博士毕业学校是澳大利亚西悉尼大学和法国图卢兹大学，"
        "研究方向是人工智能、多智能体系统、算法博弈论、机制设计、在线合作博弈、AI Agents。 [1]"
    )


def test_answer_context_selection_prefers_contact_fact_chunk_over_same_page_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="TurtleBot4机器人培训通知",
            url="https://sist.shanghaitech.edu.cn/2026/0417/c5304a1120842/page.htm",
            text="TurtleBot4机器人培训通知。首页 导航 新闻通知 培训对象 报名方式 活动背景。",
        ),
        _context(
            rank=5,
            title="TurtleBot4机器人培训通知",
            url="https://sist.shanghaitech.edu.cn/2026/0417/c5304a1120842/page.htm",
            text="联系人：李老师；办公室：信息学院1A-200；联系电话：021-20680000。",
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer("联系人是李老师，办公室是信息学院1A-200，联系电话是021-20680000 [5].")

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("TurtleBot4机器人培训通知的联系人和联系电话是什么？", mode="dense", top_k=2)

    assert result.status == "answered"
    assert result.answer_context_order[0]["source_id"] == 5
    assert tokenizer.chat_template_kwargs is not None
    user_message = tokenizer.chat_template_kwargs["messages"][1]["content"]  # type: ignore[index]
    assert user_message.index("[5] TurtleBot4") < user_message.index("[1] TurtleBot4")


def test_generation_prompt_uses_later_relevant_evidence_from_long_retrieved_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_front_matter = " ".join(f"首页 导航 学院概况 新闻列表 占位内容{i}" for i in range(80))
    contexts = [
        _context(
            rank=1,
            title="2026年“通计划”联合培养博士生专项计划通知",
            url="https://sist.shanghaitech.edu.cn/2026/0527/c2826a1123008/page.htm",
            text=(
                f"{long_front_matter}\n"
                "联系咨询方式：北京通用人工智能研究院座机010-85413687，"
                "邮箱tongprogram@bigai.ai。\n"
                "上海科技大学信息学院联系人为高老师，电话021-20684866，"
                "邮箱admission.sist@shanghaitech.edu.cn。"
            ),
        )
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer(
        "北京通用人工智能研究院座机010-85413687，邮箱tongprogram@bigai.ai；"
        "上海科技大学信息学院联系人为高老师，电话021-20684866，"
        "邮箱admission.sist@shanghaitech.edu.cn [1]."
    )

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "在2026年“通计划”联合培养博士生专项计划通知中，北京通用人工智能研究院和"
        "上海科技大学信息学院的联系咨询方式分别是什么？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert tokenizer.chat_template_kwargs is not None
    user_message = tokenizer.chat_template_kwargs["messages"][1]["content"]  # type: ignore[index]
    assert "010-85413687" in user_message
    assert "admission.sist@shanghaitech.edu.cn" in user_message
    assert "占位内容0" not in user_message
    assert result.answer.endswith("[1].")


def test_generation_prompt_uses_same_document_contact_sibling_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        ContextItem(
            rank=1,
            chunk_id=101,
            document_id=77,
            title="2026年“通计划”联合培养博士生专项计划通知",
            url="https://sist.shanghaitech.edu.cn/2026/0415/c2863a1120785/page.htm",
            category=None,
            language="zh",
            snippet="上海科技大学和北京通用人工智能研究院联合培养博士生专项计划。地址：上海市徐汇区岳阳路319号。",
            text="上海科技大学和北京通用人工智能研究院联合培养博士生专项计划。地址：上海市徐汇区岳阳路319号。邮编：200031。",
            trace_ref="test:chunk:101",
        )
    ]
    sibling_chunks = [
        {
            "chunk_id": 101,
            "document_id": 77,
            "title": contexts[0].title,
            "url": contexts[0].url,
            "category": None,
            "language": "zh",
            "text": contexts[0].text,
        },
        {
            "chunk_id": 103,
            "document_id": 77,
            "title": contexts[0].title,
            "url": contexts[0].url,
            "category": None,
            "language": "zh",
            "text": (
                "联系方式：北京通用人工智能研究院座机010-85413687（周一至周五9:00-18:00），"
                "邮箱tongprogram@bigai.ai。上海科技大学信息科学与技术学院联系人为高老师，"
                "电话021-20684866，邮箱admission.sist@shanghaitech.edu.cn。"
            ),
        },
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer(
        "北京通用人工智能研究院座机010-85413687，邮箱tongprogram@bigai.ai；"
        "上海科技大学信息学院联系人为高老师，电话021-20684866，"
        "邮箱admission.sist@shanghaitech.edu.cn [1]."
    )

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(
        StaticContextRetrieverWithSiblingChunks(contexts, sibling_chunks),
        model_path=model_path,
        device="cpu",
    )  # type: ignore[arg-type]

    result = answerer.answer(
        "在2026年“通计划”联合培养博士生专项计划通知中，北京通用人工智能研究院和"
        "上海科技大学信息学院的联系咨询方式分别是什么？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert tokenizer.chat_template_kwargs is not None
    user_message = tokenizer.chat_template_kwargs["messages"][1]["content"]  # type: ignore[index]
    assert "010-85413687" in user_message
    assert "tongprogram@bigai.ai" in user_message
    assert "021-20684866" in user_message
    assert "admission.sist@shanghaitech.edu.cn" in user_message
    assert result.answer.endswith("[1].")


def test_generation_prompt_uses_same_document_doctoral_practice_credit_sibling_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        ContextItem(
            rank=1,
            chunk_id=301,
            document_id=88,
            title="2025级电子信息工程企业联合培养博士项目培养方案",
            url="https://sist.shanghaitech.edu.cn/ee/doctoral-enterprise-2025.pdf",
            category=None,
            language="zh",
            snippet="企业联合培养博士项目要求总学分不低于42，课程学分不低于40。",
            text="企业联合培养博士项目培养方案。总学分不低于42，课程学分不低于40。",
            trace_ref="test:chunk:301",
        )
    ]
    sibling_chunks = [
        {
            "chunk_id": 301,
            "document_id": 88,
            "title": contexts[0].title,
            "url": contexts[0].url,
            "category": None,
            "language": "zh",
            "text": contexts[0].text,
        },
        {
            "chunk_id": 302,
            "document_id": 88,
            "title": contexts[0].title,
            "url": contexts[0].url,
            "category": None,
            "language": "zh",
            "text": "实践教学环节要求：课程实践部分不低于8学分，企业实践部分按培养方案执行。",
        },
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer("总学分不低于42，课程学分不低于40，课程实践部分不低于8学分 [1].")

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(
        StaticContextRetrieverWithSiblingChunks(contexts, sibling_chunks),
        model_path=model_path,
        device="cpu",
    )  # type: ignore[arg-type]

    result = answerer.answer(
        "2025级EE企业联合培养博士项目的总学分、课程学分和课程实践部分分别要求多少学分？",
        mode="dense",
        top_k=1,
    )

    assert result.status == "answered"
    assert tokenizer.chat_template_kwargs is not None
    user_message = tokenizer.chat_template_kwargs["messages"][1]["content"]  # type: ignore[index]
    assert "总学分不低于42" in user_message
    assert "课程学分不低于40" in user_message
    assert "课程实践部分不低于8学分" in user_message
    assert result.answer.endswith("[1].")


@pytest.mark.parametrize(
    ("query", "sibling_text", "answer", "expected_text"),
    [
        (
            "某位教师担任TIE、TTE、TPEA等IEEE Trans期刊副主编的情况是什么？",
            (
                "学术服务：担任IEEE Transactions on Industrial Electronics (TIE)、"
                "IEEE Transactions on Transportation Electrification (TTE)、"
                "IEEE Transactions on Power Electronics and Applications (TPEA)副主编。"
            ),
            "该教师担任TIE、TTE、TPEA等IEEE Trans期刊副主编 [1].",
            "TPEA)副主编",
        ),
        (
            "在校生作为第一发明人的专利申请号是什么？",
            "专利成果：在校生张三为第一发明人，专利名称为智能传感器系统，申请号CN202510123456.7。",
            "在校生张三作为第一发明人的专利申请号是CN202510123456.7 [1].",
            "申请号CN202510123456.7",
        ),
        (
            "直博招生的选拔方式或招生方式是否采用申请-考核制？",
            "招生方式：直博研究生选拔方式采用申请-考核制，申请人须按通知提交材料并参加学院考核。",
            "直博研究生选拔方式采用申请-考核制 [1].",
            "直博研究生选拔方式采用申请-考核制",
        ),
        (
            "2025级EE培养方案中，三选二课程计入本学科选修的规则是什么？",
            "2025级EE培养方案说明：三选二课程中未计入专业必修的课程，可按规则计入本学科选修模块。",
            "2025级EE三选二课程中未计入专业必修的课程可计入本学科选修模块 [1].",
            "三选二课程中未计入专业必修",
        ),
        (
            "电力电子课程是否会录制成视频供学生提前学习？",
            "教学安排：电力电子课程的核心知识点会录制成视频，供学生提前学习，再开展课堂讨论。",
            "电力电子课程核心知识点会录制成视频供学生提前学习 [1].",
            "录制成视频，供学生提前学习",
        ),
    ],
)
def test_generation_prompt_uses_query_anchor_local_sibling_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    sibling_text: str,
    answer: str,
    expected_text: str,
) -> None:
    context = ContextItem(
        rank=1,
        chunk_id=201,
        document_id=88,
        title="信息学院长页面",
        url="https://sist.shanghaitech.edu.cn/example/page.htm",
        category=None,
        language="zh",
        snippet="信息学院长页面包含多个栏目的概览信息。",
        text=" ".join(f"首页 导航 学院新闻 占位内容{i}" for i in range(90)),
        trace_ref="test:chunk:201",
    )
    sibling_chunks = [
        {
            "chunk_id": 201,
            "document_id": 88,
            "title": context.title,
            "url": context.url,
            "category": None,
            "language": "zh",
            "text": context.text,
        },
        {
            "chunk_id": 202,
            "document_id": 88,
            "title": context.title,
            "url": context.url,
            "category": None,
            "language": "zh",
            "text": sibling_text,
        },
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer(answer)

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(
        StaticContextRetrieverWithSiblingChunks([context], sibling_chunks),
        model_path=model_path,
        device="cpu",
    )  # type: ignore[arg-type]

    result = answerer.answer(query, mode="dense", top_k=1)

    assert result.status == "answered"
    assert [source.source_id for source in result.sources] == [1]
    assert tokenizer.chat_template_kwargs is not None
    user_message = tokenizer.chat_template_kwargs["messages"][1]["content"]  # type: ignore[index]
    assert expected_text in user_message
    assert "占位内容0" not in user_message
    assert result.answer.endswith("[1].")


def test_initial_generation_prompt_uses_answer_context_order_with_original_source_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2022级计算机科学与技术本科培养方案",
            url="https://example.edu/cs/2022-degree",
            text="2022级计算机科学与技术本科培养方案要求学生修满120学分。",
        ),
        _context(
            rank=3,
            title="2025级计算机科学与技术本科培养方案",
            url="https://example.edu/cs/2025-degree",
            text="2025级计算机科学与技术本科培养方案要求学生修满140学分。",
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer("2025级培养方案要求修满140学分 [3].")

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("2025级计算机科学与技术本科培养方案需要修满多少学分？", mode="dense", top_k=2)

    assert result.status == "answered"
    assert tokenizer.chat_template_kwargs is not None
    user_message = tokenizer.chat_template_kwargs["messages"][1]["content"]  # type: ignore[index]
    assert user_message.index("[3] 2025级") < user_message.index("[1] 2022级")
    assert "[2]" not in user_message


def test_weak_citation_support_triggers_repair_with_better_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts = [
        _context(
            rank=1,
            title="2024 admissions notice",
            url="https://example.edu/admissions-2024",
            text="The 2024 admissions notice says applications closed in 2024.",
        ),
        _context(
            rank=3,
            title="2025 admissions notice",
            url="https://example.edu/admissions-2025",
            text="The 2025 admissions notice says applications are open for the 2025 cohort.",
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation_sequence(
        monkeypatch,
        [
            "The 2025 admissions notice says applications are open [1].",
            '{"status": "answered", "answer": "The 2025 admissions notice says applications are open [3]."}',
        ],
    )
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("What does the 2025 admissions notice say?", mode="dense", top_k=2)

    assert result.status == "answered"
    assert result.generation_path == "repair"
    assert result.generation_rejection_reason == "weak_citation_support"
    assert result.answer.endswith("[3].")


def test_supported_citation_is_not_repaired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contexts = [
        _context(
            rank=1,
            title="2025 admissions notice",
            url="https://example.edu/admissions-2025",
            text="The 2025 admissions notice says applications are open for the 2025 cohort.",
        ),
        _context(
            rank=2,
            title="2024 admissions notice",
            url="https://example.edu/admissions-2024",
            text="The 2024 admissions notice says applications closed in 2024.",
        ),
    ]
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "The 2025 admissions notice says applications are open [1].")
    answerer = RagAnswerer(StaticContextRetriever(contexts), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer("What does the 2025 admissions notice say?", mode="dense", top_k=2)

    assert result.status == "answered"
    assert result.generation_path == "initial"
    assert result.answer.endswith("[1].")


def test_missing_answer_reranker_model_path_reports_local_path_error(
    tmp_path: Path, fake_hybrid_sentence_transformer_module
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()

    with pytest.raises(FileNotFoundError, match="Answer reranker model path"):
        RagAnswerer(
            _retriever_from_paths(paths),
            model_path=model_path,
            device="cpu",
            answer_reranker_model=tmp_path / "missing-answer-reranker",
        )


def test_answer_cli_json_outputs_structured_result(
    tmp_path: Path,
    fake_hybrid_sentence_transformer_module,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    _patch_generation(monkeypatch, "CLI answer cites the source [1].")

    exit_code = answer_main(
        [
            "--query",
            "exact bridge query",
            "--mode",
            "hybrid",
            "--model-path",
            str(model_path),
            "--device",
            "cpu",
            "--db",
            str(paths["db"]),
            "--bm25",
            str(paths["bm25"]),
            "--faiss",
            str(paths["faiss"]),
            "--chunk-index",
            str(paths["chunk_index"]),
            "--report",
            str(paths["report"]),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "answered"
    assert payload["answer"] == "CLI answer cites the source [1]."
    assert payload["sources"][0]["source_id"] == 1
    assert payload["retrieval"]["mode"] == "hybrid"
    assert payload["config"]["model_path"] == str(model_path)


def test_build_prompt_numbers_context_sources(tmp_path: Path, fake_hybrid_sentence_transformer_module) -> None:
    paths = _build_generation_artifacts(tmp_path)
    retriever = _retriever_from_paths(paths)
    retrieval_result = retriever.retrieve("exact bridge query", mode="hybrid", top_k=2)

    prompt = build_prompt("exact bridge query", retrieval_result.contexts)

    assert "[1]" in prompt
    assert "URL: https://example.edu/" in prompt
    assert "Every factual paragraph must include" in prompt


def test_generation_uses_chat_template_without_qwen_thinking(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_generation_artifacts(tmp_path)
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer("Chat template answer [1].")

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("exact bridge query", mode="hybrid", top_k=2)

    assert result.status == "answered"
    assert tokenizer.chat_template_kwargs is not None
    assert tokenizer.chat_template_kwargs["add_generation_prompt"] is True
    assert tokenizer.chat_template_kwargs["enable_thinking"] is False
    assert tokenizer.chat_template_kwargs["messages"][0]["role"] == "system"
    assert tokenizer.chat_template_kwargs["messages"][1]["role"] == "user"


def _patch_generation(monkeypatch: pytest.MonkeyPatch, generated_text: str) -> None:
    def fake_load_model(self: RagAnswerer) -> tuple[FakeTokenizer, FakeModel]:
        return FakeTokenizer(generated_text), FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)


def _patch_generation_sequence(monkeypatch: pytest.MonkeyPatch, generated_texts: list[str]) -> None:
    tokenizer = SequenceFakeTokenizer(generated_texts.copy())

    def fake_load_model(self: RagAnswerer) -> tuple[SequenceFakeTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)


def _retriever_from_paths(paths: dict[str, Path]) -> Retriever:
    return Retriever.from_paths(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )


def _build_generation_artifacts(
    tmp_path: Path,
    *,
    chunk_url: str | None = "https://example.edu/b",
    dense_text: str = "dense winner semantic source with enough full text for generation",
) -> dict[str, Path]:
    import faiss

    input_dir = tmp_path / "generation-merged"
    input_dir.mkdir()
    write_jsonl(
        input_dir / "documents.jsonl",
        [
            {"id": 10, "url": "https://example.edu/a", "canonical_url": "https://example.edu/a", "title": "A"},
            {"id": 11, "url": "https://example.edu/b", "canonical_url": "https://example.edu/b", "title": "B"},
        ],
    )
    write_jsonl(
        input_dir / "chunks.jsonl",
        [
            {
                "id": 100,
                "document_id": 10,
                "chunk_index": 0,
                "title": "Sparse Source",
                "url": "https://example.edu/a",
                "text": "sparse bridge official source",
                "char_count": 29,
            },
            {
                "id": 101,
                "document_id": 11,
                "chunk_index": 0,
                "title": "Dense Winner",
                "url": chunk_url,
                "text": dense_text,
                "char_count": len(dense_text),
            },
        ],
    )
    write_jsonl(input_dir / "courses.jsonl", [])
    write_jsonl(input_dir / "faculty_members.jsonl", [])
    write_jsonl(input_dir / "program_requirements.jsonl", [])
    write_jsonl(input_dir / "events.jsonl", [])

    db_path = tmp_path / "rag.sqlite"
    bm25_path = tmp_path / "bm25.pkl"
    faiss_path = tmp_path / "faiss.index"
    chunk_index_path = tmp_path / "chunk_index.jsonl"
    report_path = tmp_path / "report.json"
    build_database(input_dir, db_path, report_path)
    build_indexes(db_path, bm25_path, faiss_path, chunk_index_path, report_path, skip_faiss=True)

    index = faiss.IndexFlatIP(3)
    index.add(np.asarray([[0.95, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(faiss_path))
    atomic_json_dump(
        report_path,
        {"index": {"faiss": {"model_path": "/models/hub/snapshots/bge-m3-local", "model_id": DEFAULT_MODEL}}},
    )
    return {
        "db": db_path,
        "bm25": bm25_path,
        "faiss": faiss_path,
        "chunk_index": chunk_index_path,
        "report": report_path,
    }


def _context(*, rank: int, title: str, url: str, text: str) -> ContextItem:
    return ContextItem(
        rank=rank,
        chunk_id=rank,
        document_id=rank,
        title=title,
        url=url,
        category=None,
        language="zh",
        snippet=text,
        text=text,
        trace_ref=f"test:{rank}",
    )


def _artifact_query_and_contexts(question_id: str) -> tuple[str, list[ContextItem]]:
    artifact_path = Path(
        "data/eval/generation_hybrid_qwen35_20260615T100250Z/"
        "run_generation_hybrid_qwen35_20260615T100250Z.jsonl"
    )
    for line in artifact_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["id"] != question_id:
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
    raise AssertionError(f"{question_id} not found in {artifact_path}")
