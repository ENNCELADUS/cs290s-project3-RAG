from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .schema import QuestionSpec

JudgeStatus = Literal["correct", "incorrect", "manual_review", "evidence_insufficient"]


@dataclass(frozen=True)
class JudgeResult:
    status: JudgeStatus
    is_correct: int | None
    reason: str


def judge_answer(
    question: QuestionSpec,
    answer: str | None,
    *,
    cited_expected_source_hit: bool = False,
) -> JudgeResult:
    if answer is None or not answer.strip():
        return JudgeResult(status="manual_review", is_correct=None, reason="no generated answer")
    if _contains_any(answer, question.forbidden_facts):
        return JudgeResult(status="incorrect", is_correct=0, reason="contains forbidden fact")
    if question.judge_type == "exact_or_alias_match":
        if _contains_any(answer, question.acceptable_answers or [question.gt_answer]):
            return JudgeResult(status="correct", is_correct=1, reason="matched acceptable answer")
        if cited_expected_source_hit:
            return _loose_cited_source_judge(question, answer)
        return JudgeResult(status="incorrect", is_correct=0, reason="no acceptable answer matched")
    if question.judge_type == "required_facts_match":
        missing = [fact for fact in question.required_facts if not _contains(answer, fact)]
        if missing:
            if cited_expected_source_hit:
                return _loose_cited_source_judge(question, answer)
            return JudgeResult(status="incorrect", is_correct=0, reason=f"missing required facts: {missing}")
        return JudgeResult(status="correct", is_correct=1, reason="all required facts matched")
    if question.judge_type in {"required_facts_with_manual_review", "local_llm_judge_with_human_review"}:
        if cited_expected_source_hit:
            return _loose_cited_source_judge(question, answer)
        return JudgeResult(status="manual_review", is_correct=None, reason=f"{question.judge_type} requires review")
    return JudgeResult(status="manual_review", is_correct=None, reason=f"unknown judge_type: {question.judge_type}")


def _contains_any(answer: str, candidates: list[str]) -> bool:
    return any(_contains(answer, candidate) for candidate in candidates if candidate)


def _contains(answer: str, candidate: str) -> bool:
    return _normalize_text(candidate) in _normalize_text(answer)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _loose_cited_source_judge(question: QuestionSpec, answer: str) -> JudgeResult:
    facts = question.required_facts or question.acceptable_answers or [question.gt_answer]
    facts = [fact for fact in facts if fact.strip()]
    if not facts:
        return JudgeResult(status="incorrect", is_correct=0, reason="loose cited-source judge has no facts")

    matched = [fact for fact in facts if _loose_fact_matches(answer, fact)]
    required = _minimum_loose_matches(len(facts))
    if len(matched) >= required:
        return JudgeResult(
            status="correct",
            is_correct=1,
            reason=f"loose cited-source judge matched {len(matched)}/{len(facts)} facts",
        )
    return JudgeResult(
        status="incorrect",
        is_correct=0,
        reason=f"loose cited-source judge matched {len(matched)}/{len(facts)} facts",
    )


def _minimum_loose_matches(fact_count: int) -> int:
    if fact_count <= 2:
        return fact_count
    if fact_count <= 4:
        return max(2, fact_count - 1)
    return max(3, int(fact_count * 0.6 + 0.999))


def _loose_fact_matches(answer: str, fact: str) -> bool:
    if _contains(answer, fact) or _compact_contains(answer, fact):
        return True
    atoms = _important_atoms(fact)
    if not atoms:
        return False
    return all(_compact_contains(answer, atom) for atom in atoms)


def _compact_contains(answer: str, candidate: str) -> bool:
    return _compact_text(candidate) in _compact_text(answer)


def _compact_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)


def _important_atoms(text: str) -> list[str]:
    atoms: list[str] = []
    atoms.extend(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text))
    atoms.extend(re.findall(r"\b\d{3,4}-?\d{6,8}\b", text))
    atoms.extend(re.findall(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|\d{4}年\d{1,2}月\d{1,2}日", text))
    atoms.extend(re.findall(r"\d{1,2}月\d{1,2}日|\d{1,2}:\d{2}", text))
    atoms.extend(re.findall(r"\b[A-Z]{2,}\d+[A-Z]?\b", text))
    atoms.extend(re.findall(r"\b\d+[A-Za-z]?-?\d{2,4}[A-Za-z]?\b", text))
    atoms.extend(re.findall(r"\d+(?:\.\d+)?\s*(?:学分|人|个|项|门|天)", text))
    for chunk in re.split(r"[，,；;。:：、（）()和与及]|包括|分别|以及|其中|是|为", text):
        chunk = chunk.strip()
        if 2 <= len(chunk) <= 24 and _has_content_word(chunk):
            atoms.append(chunk)
    return _dedupe(atoms)


def _has_content_word(text: str) -> bool:
    generic = {
        "办公室",
        "邮箱",
        "电话",
        "研究方向",
        "博士毕业院校",
        "学分",
        "课程",
        "页面",
        "该页面",
        "分别",
    }
    compact = _compact_text(text)
    return compact not in {_compact_text(item) for item in generic}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = _compact_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value.strip())
    return deduped
