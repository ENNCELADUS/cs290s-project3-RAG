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


class FakeTensor:
    shape = (1, 3)

    def to(self, device: str) -> FakeTensor:
        return self


class FakeChatTokenizer:
    def __init__(self, generated_text: str) -> None:
        self.generated_text = generated_text
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

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        return self.generated_text


class FakeModel:
    def generate(self, **kwargs: object) -> list[list[int]]:
        return [[1, 2, 3, 4]]


class StaticHybridRetriever:
    def __init__(self, contexts: list[ContextItem]) -> None:
        self.contexts = contexts

    def retrieve(self, query: str, *, mode: str, top_k: int, **kwargs: Any) -> HybridRetrievalResult:
        assert mode == "hybrid"
        selected = self.contexts[:top_k]
        return HybridRetrievalResult(
            query=query,
            mode="hybrid",
            hits=[_hit_for_context(context) for context in selected],
            contexts=selected,
            config=HybridRetrievalConfig(final_top_k=top_k),
        )


def test_hybrid_prompt_preserves_flat_degree_table_bindings_for_ee_general_education(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(
        rank=1,
        title="2025级本科生培养方案EE专业",
        url="https://example.edu/ee/2025-degree",
        text=(
            "2025 级本科生培养方案 电子信息工程专业。"
            "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
            "类别 必修 选修 学分 人文社科通识 30 15 45 "
            "自然科学通识 16 16 32 专业课程 32 27 59 任选课程 9 145。"
        ),
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer("人文社科通识45学分，自然科学通识32学分。 [1]")

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(StaticHybridRetriever([context]), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "2025级EE本科培养方案中，人文社科通识和自然科学通识分别要求多少学分？",
        mode="hybrid",
        top_k=1,
    )

    assert result.status == "answered"
    assert tokenizer.chat_template_kwargs is not None
    user_message = tokenizer.chat_template_kwargs["messages"][1]["content"]  # type: ignore[index]
    assert "人文社科通识 - 必修 - 30" in user_message
    assert "人文社科通识 - 选修 - 15" in user_message
    assert "人文社科通识 - 学分 - 45" in user_message
    assert "自然科学通识 - 必修 - 16" in user_message
    assert "自然科学通识 - 选修 - 16" in user_message
    assert "自然科学通识 - 学分 - 32" in user_message


def test_hybrid_rejects_cs_credit_answer_missing_requested_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(
        rank=1,
        title="2025级本科生培养方案CS专业",
        url="https://example.edu/cs/2025-degree",
        text=(
            "2025 级本科生培养方案 计算机科学与技术。"
            "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
            "类别 必修 选修 学分 人文社科通识 30 15 45 "
            "自然科学通识 16 16 32 专业课程 20 39 59 任选课程 9 145。"
        ),
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer("专业课程板块必修20学分、选修39学分，合计59学分。 [1]")

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(StaticHybridRetriever([context]), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "2025级CS本科培养方案要求总学分、专业必修、专业选修和专业课程合计分别是多少学分？",
        mode="hybrid",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_credit_fields"
    assert (
        result.answer
        == "2025级计算机科学与技术专业毕业至少需要修满145学分；专业课程板块必修20学分、选修39学分，合计59学分。 [1]"
    )


def test_hybrid_recovers_ee_degree_total_and_free_choice_credits_from_flat_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(
        rank=1,
        title="2025级本科生培养方案EE专业",
        url="https://example.edu/ee/2025-degree",
        text=(
            "2025 级本科生培养方案 电子信息工程专业。"
            "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
            "类别 必修 选修 学分 人文社科通识 30 15 45 "
            "自然科学通识 16 16 32 专业课程 32 27 59 任选课程 9 145。"
        ),
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer("材料中具体数字缺失，无法回答。 [1]")

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(StaticHybridRetriever([context]), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "2025级EE本科培养方案总学分和任选课程分别是多少学分？",
        mode="hybrid",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_credit_fields"
    assert result.answer == "2025级电子信息工程专业毕业至少需要修满145学分，任选课程占9学分。 [1]"


def test_hybrid_recovers_ee_general_education_totals_from_flat_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(
        rank=1,
        title="2025级本科生培养方案EE专业",
        url="https://example.edu/ee/2025-degree",
        text=(
            "2025 级本科生培养方案 电子信息工程专业。"
            "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
            "类别 必修 选修 学分 人文社科通识 30 15 45 "
            "自然科学通识 16 16 32 专业课程 32 27 59 任选课程 9 145。"
        ),
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer("材料中没有提及具体数字，无法回答。 [1]")

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(StaticHybridRetriever([context]), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "2025级EE本科培养方案中，人文社科通识和自然科学通识总学分分别是多少？",
        mode="hybrid",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.answer == "2025级电子信息工程专业中，人文社科通识板块要求45学分，自然科学通识板块要求32学分。 [1]"


def test_hybrid_recovers_ee_professional_elective_minimum_from_flat_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(
        rank=1,
        title="2025级本科生培养方案EE专业",
        url="https://example.edu/ee/2025-degree",
        text=(
            "2025 级本科生培养方案 电子信息工程专业。"
            "学分： 修满至少 145 学分 的总学分数，具体要求如下。 "
            "类别 必修 选修 学分 人文社科通识 30 15 45 "
            "自然科学通识 16 16 32 专业课程 32 27 59 任选课程 9 145。"
        ),
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer("材料中具体数字缺失，无法回答。 [1]")

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(StaticHybridRetriever([context]), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "2025级EE本科培养方案中，专业课程板块最低选修多少学分？",
        mode="hybrid",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.answer == "2025级电子信息工程专业中，专业课程板块至少需要选修27学分。 [1]"


def test_hybrid_rejects_doctoral_credit_answer_missing_course_practice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(
        rank=1,
        title="2025级电子信息工程企业联合培养博士项目培养方案",
        url="https://example.edu/ee/doctoral-enterprise-2025.pdf",
        text=(
            "企业联合培养博士项目培养方案：基本修业年限5年，最长修业年限7年；"
            "总学分不低于42学分，课程学分不低于40学分，其中课程实践部分不低于8学分。"
        ),
    )
    model_path = tmp_path / "qwen-local"
    model_path.mkdir()
    tokenizer = FakeChatTokenizer("基本修业年限5年，最长7年；总学分不低于42学分，课程学分不低于40学分。 [1]")

    def fake_load_model(self: RagAnswerer) -> tuple[FakeChatTokenizer, FakeModel]:
        return tokenizer, FakeModel()

    monkeypatch.setattr(RagAnswerer, "_load_model", fake_load_model)
    answerer = RagAnswerer(StaticHybridRetriever([context]), model_path=model_path, device="cpu")  # type: ignore[arg-type]

    result = answerer.answer(
        "2025级EE企业联合培养博士项目的修业年限、总学分、课程学分和课程实践学分分别是多少？",
        mode="hybrid",
        top_k=1,
    )

    assert result.status == "answered"
    assert result.generation_path == "extractive_fallback"
    assert result.generation_rejection_reason == "missing_requested_credit_fields"
    assert "5年" in result.answer
    assert "7年" in result.answer
    assert "42学分" in result.answer
    assert "40学分" in result.answer
    assert "课程实践" in result.answer
    assert "8学分" in result.answer
    assert result.answer.endswith("[1]")


def _context(*, rank: int, title: str, url: str, text: str) -> ContextItem:
    return ContextItem(
        rank=rank,
        chunk_id=100 + rank,
        document_id=200 + rank,
        title=title,
        url=url,
        category=None,
        language="zh",
        snippet=text[:120],
        text=text,
        trace_ref=f"test:chunk:{100 + rank}",
    )


def _hit_for_context(context: ContextItem) -> OptimizedRetrievalHit:
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
            trace_id=f"test-trace-{context.rank}",
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
