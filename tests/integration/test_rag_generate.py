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
            '{"status": "answered", "answer": "Repair should not be needed [1]."}',
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
            '{"status": "answered", "answer": "Repair should not be needed [1]."}',
        ],
    )
    answerer = RagAnswerer(_retriever_from_paths(paths), model_path=model_path, device="cpu")

    result = answerer.answer("王浩宇教授的办公室具体在哪里？他的工作邮箱是什么？", mode="hybrid", top_k=2)

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
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
            text="2025级计算机科学与技术本科培养方案要求学生修满140学分。",
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
    assert "140学分" in result.answer
    assert result.answer.endswith("[3].")
    assert [source.source_id for source in result.sources] == [1, 2, 3]
    assert result.answer_context_order[0]["source_id"] == 3


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
