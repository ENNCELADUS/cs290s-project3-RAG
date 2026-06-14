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
    has_citation: bool = False,
) -> JudgeResult:
    if answer is None or not answer.strip():
        return JudgeResult(status="manual_review", is_correct=None, reason="no generated answer")
    grounded_answer = cited_expected_source_hit or has_citation or _has_inline_citation(answer)
    if _contains_any(answer, question.forbidden_facts):
        return JudgeResult(status="incorrect", is_correct=0, reason="contains forbidden fact")
    if question.judge_type == "exact_or_alias_match":
        if _contains_any(answer, question.acceptable_answers or [question.gt_answer]):
            return JudgeResult(status="correct", is_correct=1, reason="matched acceptable answer")
        if grounded_answer:
            return _loose_cited_source_judge(question, answer)
        return JudgeResult(status="incorrect", is_correct=0, reason="no acceptable answer matched")
    if question.judge_type == "required_facts_match":
        missing = [fact for fact in question.required_facts if not _contains(answer, fact)]
        if missing:
            if grounded_answer:
                return _loose_cited_source_judge(question, answer)
            return JudgeResult(status="incorrect", is_correct=0, reason=f"missing required facts: {missing}")
        return JudgeResult(status="correct", is_correct=1, reason="all required facts matched")
    if question.judge_type in {"required_facts_with_manual_review", "local_llm_judge_with_human_review"}:
        if question.judge_type == "required_facts_with_manual_review" and grounded_answer:
            return _loose_cited_source_judge(question, answer, allow_manual_review=True)
        if grounded_answer:
            return _loose_cited_source_judge(question, answer, allow_manual_review=True)
        return JudgeResult(status="manual_review", is_correct=None, reason=f"{question.judge_type} requires review")
    return JudgeResult(status="manual_review", is_correct=None, reason=f"unknown judge_type: {question.judge_type}")


def _contains_any(answer: str, candidates: list[str]) -> bool:
    return any(_contains(answer, candidate) for candidate in candidates if candidate)


def _has_inline_citation(answer: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer))


def _contains(answer: str, candidate: str) -> bool:
    return _normalize_text(candidate) in _normalize_text(answer)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _loose_cited_source_judge(
    question: QuestionSpec,
    answer: str,
    *,
    allow_manual_review: bool = False,
) -> JudgeResult:
    facts = question.required_facts or question.acceptable_answers or [question.gt_answer]
    facts = [fact for fact in facts if fact.strip()]
    if not facts:
        if allow_manual_review:
            return JudgeResult(status="manual_review", is_correct=None, reason="loose atom judge has no facts")
        return JudgeResult(status="incorrect", is_correct=0, reason="loose atom judge has no facts")

    atoms = _dedupe([atom for fact in facts for atom in _important_atoms(fact)])
    if not atoms:
        if allow_manual_review:
            return JudgeResult(status="manual_review", is_correct=None, reason="loose atom judge has no atoms")
        return JudgeResult(status="incorrect", is_correct=0, reason="loose atom judge has no atoms")

    matched = [atom for atom in atoms if _compact_contains(answer, atom)]
    required = _minimum_loose_matches(len(atoms))
    if len(matched) >= required:
        return JudgeResult(
            status="correct",
            is_correct=1,
            reason=f"loose atom judge matched {len(matched)}/{len(atoms)} atoms",
        )
    return JudgeResult(
        status="incorrect",
        is_correct=0,
        reason=f"loose atom judge matched {len(matched)}/{len(atoms)} atoms",
    )


def _minimum_loose_matches(fact_count: int) -> int:
    if fact_count <= 3:
        return fact_count
    if fact_count <= 5:
        return max(3, int(fact_count * 0.8 + 0.999))
    return max(4, int(fact_count * 0.65 + 0.999))


def _compact_contains(answer: str, candidate: str) -> bool:
    candidate_date_keys = _date_keys(candidate)
    if candidate_date_keys:
        answer_date_keys = _date_keys(answer)
        return any(date_key in answer_date_keys for date_key in candidate_date_keys)
    return _compact_text(candidate) in _compact_text(answer)


def _date_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for year, month, day in re.findall(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", text):
        keys.add(f"{int(year):04d}{int(month):02d}{int(day):02d}")
        keys.add(f"{int(month):02d}{int(day):02d}")
    for year, month, day in re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text):
        keys.add(f"{int(year):04d}{int(month):02d}{int(day):02d}")
        keys.add(f"{int(month):02d}{int(day):02d}")
    for month, day in re.findall(r"(\d{1,2})月(\d{1,2})日", text):
        keys.add(f"{int(month):02d}{int(day):02d}")
    return keys


def _compact_text(text: str) -> str:
    compact = re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)
    for label in (
        "office",
        "email",
        "phone",
        "他的",
        "她的",
        "办公室",
        "工作邮箱",
        "邮箱",
        "电话",
        "在",
    ):
        compact = compact.replace(label, "")
    return compact


def _important_atoms(text: str) -> list[str]:
    text = _normalize_atom_text(text)
    atoms: list[str] = []
    atoms.extend(re.findall(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
    atoms.extend(re.findall(r"\b\d{3,4}-?\d{6,8}\b", text))
    atoms.extend(_date_keys(text))
    atoms.extend(re.findall(r"\d{1,2}月\d{1,2}日|\d{1,2}:\d{2}", text))
    atoms.extend(re.findall(r"\b[A-Z]{2,}\d+[A-Z]?\b", text))
    atoms.extend(
        re.findall(
            r"[\u4e00-\u9fffA-Za-z]+(?:学院|校区|楼|室)\d+[A-Za-z]?-?\d{2,4}[A-Za-z]?室?|"
            r"[\u4e00-\u9fffA-Za-z]+\d+[A-Za-z]?-\d{2,4}[A-Za-z]?室?",
            text,
        )
    )
    atoms.extend(re.findall(r"\b\d+[A-Za-z]?-?\d{2,4}[A-Za-z]?室?\b", text))
    atoms.extend(re.findall(r"\d+(?:\.\d+)?\s*(?:学分|人|个|项|门|天|年)", text))
    atoms.extend(re.findall(r"[\u4e00-\u9fffA-Za-z]+(?:一等奖|二等奖|三等奖|冠军|亚军|奖)", text))
    structured_atom_keys = {_compact_text(atom) for atom in atoms}
    for chunk in re.split(
        r"[，,；;。:：、（）()和与及]|包括|分别|以及|其中|要求|修满|官方|确切|发布|上线|日期|时间|地点|单位|邀请人|演讲者|是|为",
        text,
    ):
        chunk = chunk.strip()
        compact_chunk = _compact_text(chunk)
        if any(atom_key in compact_chunk and atom_key != compact_chunk for atom_key in structured_atom_keys):
            continue
        if 2 <= len(chunk) <= 24 and _has_content_word(chunk):
            atoms.append(chunk)
    return _dedupe(atoms)


def _normalize_atom_text(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("（", "(").replace("）", ")")


def _has_content_word(text: str) -> bool:
    generic = {
        "办公室",
        "工作邮箱",
        "邮箱",
        "电话",
        "研究方向",
        "博士毕业院校",
        "学分",
        "课程",
        "页面",
        "该页面",
        "分别",
        "要求修满",
        "发布日期",
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
