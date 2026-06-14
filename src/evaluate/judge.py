from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .schema import QuestionSpec

JudgeStatus = Literal["correct", "incorrect", "manual_review", "evidence_insufficient"]


@dataclass(frozen=True)
class JudgeResult:
    status: JudgeStatus
    is_correct: int | None
    reason: str


def judge_answer(question: QuestionSpec, answer: str | None) -> JudgeResult:
    if answer is None or not answer.strip():
        return JudgeResult(status="manual_review", is_correct=None, reason="no generated answer")
    if _contains_any(answer, question.forbidden_facts):
        return JudgeResult(status="incorrect", is_correct=0, reason="contains forbidden fact")
    if question.judge_type == "exact_or_alias_match":
        if _contains_any(answer, question.acceptable_answers or [question.gt_answer]):
            return JudgeResult(status="correct", is_correct=1, reason="matched acceptable answer")
        return JudgeResult(status="incorrect", is_correct=0, reason="no acceptable answer matched")
    if question.judge_type == "required_facts_match":
        missing = [fact for fact in question.required_facts if not _contains(answer, fact)]
        if missing:
            return JudgeResult(status="incorrect", is_correct=0, reason=f"missing required facts: {missing}")
        return JudgeResult(status="correct", is_correct=1, reason="all required facts matched")
    if question.judge_type in {"required_facts_with_manual_review", "local_llm_judge_with_human_review"}:
        return JudgeResult(status="manual_review", is_correct=None, reason=f"{question.judge_type} requires review")
    return JudgeResult(status="manual_review", is_correct=None, reason=f"unknown judge_type: {question.judge_type}")


def _contains_any(answer: str, candidates: list[str]) -> bool:
    return any(_contains(answer, candidate) for candidate in candidates if candidate)


def _contains(answer: str, candidate: str) -> bool:
    return _normalize_text(candidate) in _normalize_text(answer)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())
