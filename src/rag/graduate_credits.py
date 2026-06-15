from __future__ import annotations

import re
from dataclasses import dataclass

from .answer_types import ExtractiveAnswer
from .retrieve import ContextItem


@dataclass(frozen=True)
class GraduateCreditSlots:
    label: str
    source_rank: int
    basic_years: str
    max_years: str
    total_credits: str
    course_credits: str | None
    practice_credits: str | None
    full_time: bool


def extract_graduate_credit_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if not _query_wants_graduate_credit_slots(query):
        return None
    if _query_wants_comparison(query):
        return _extract_graduate_credit_comparison(query, contexts)
    return _extract_single_graduate_credit_answer(query, contexts)


def graduate_credit_rejection_reason(query: str, answer: str, contexts: list[ContextItem]) -> str | None:
    expected_tokens = _expected_graduate_credit_tokens(query, contexts)
    if not expected_tokens:
        return None
    normalized_answer = _normalize_for_slot_check(answer)
    if all(token in normalized_answer for token in expected_tokens):
        return None
    return "missing_requested_credit_fields"


def _expected_graduate_credit_tokens(query: str, contexts: list[ContextItem]) -> list[str]:
    if not _query_wants_graduate_credit_slots(query):
        return []
    if _query_wants_comparison(query):
        comparison = _comparison_slots(query, contexts)
        if comparison is None:
            return []
        master, direct = comparison
        return _duration_total_tokens(master) + _duration_total_tokens(direct)

    tokens: list[str] = []
    for context in contexts:
        slots = _graduate_credit_slots(query, context)
        if slots is None:
            continue
        tokens.extend(_duration_total_tokens(slots))
        if slots.course_credits is not None and ("课程学分" in query or "course credit" in query.lower()):
            tokens.append(f"{slots.course_credits}学分")
        if slots.practice_credits is not None and ("实践" in query or "practice" in query.lower()):
            tokens.append(f"{slots.practice_credits}学分")
        return tokens
    return []


def _duration_total_tokens(slots: GraduateCreditSlots) -> list[str]:
    return [f"{slots.basic_years}年", f"{slots.max_years}年", f"{slots.total_credits}学分"]


def _extract_single_graduate_credit_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    for context in contexts:
        slots = _graduate_credit_slots(query, context)
        if slots is None:
            continue
        parts = []
        duration = f"{slots.label}基本学制{slots.basic_years}年、最长学制{slots.max_years}年"
        if slots.full_time and ("全日制" in query or "full-time" in query.lower()):
            duration += "，且为全日制"
        parts.append(duration)
        parts.append(f"总学分不低于{slots.total_credits}学分")
        if slots.course_credits is not None and ("课程学分" in query or "course credit" in query.lower()):
            parts.append(f"课程学分不低于{slots.course_credits}学分")
        if slots.practice_credits is not None and ("实践" in query or "practice" in query.lower()):
            parts.append(f"课程实践部分不低于{slots.practice_credits}学分")
        if len(parts) < 2:
            continue
        return ExtractiveAnswer(f"{'；'.join(parts)}。 [{slots.source_rank}]", slots.source_rank)
    return None


def _extract_graduate_credit_comparison(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "总学分" not in query:
        return None
    comparison = _comparison_slots(query, contexts)
    if comparison is None:
        return None
    master, direct = comparison
    return ExtractiveAnswer(
        (
            f"{master.label}基本学制{master.basic_years}年、最长学制{master.max_years}年、"
            f"总学分不低于{master.total_credits}学分；"
            f"{direct.label}基本学制{direct.basic_years}年、最长学制{direct.max_years}年、"
            f"总学分不低于{direct.total_credits}学分。 "
            f"[{master.source_rank}][{direct.source_rank}]"
        ),
        master.source_rank,
    )


def _comparison_slots(
    query: str, contexts: list[ContextItem]
) -> tuple[GraduateCreditSlots, GraduateCreditSlots] | None:
    master: GraduateCreditSlots | None = None
    direct: GraduateCreditSlots | None = None
    for context in contexts:
        slots = _graduate_credit_slots(query, context)
        if slots is None:
            continue
        text = _normalized_text(context)
        if master is None and _looks_like_master_plan(text):
            master = slots
        if direct is None and _looks_like_direct_phd_plan(text):
            direct = slots
    if master is None or direct is None:
        return None
    return master, direct


def _query_wants_graduate_credit_slots(query: str) -> bool:
    lowered = query.lower()
    wants_credit = "学分" in query or "credit" in lowered
    wants_duration = any(term in query for term in ("学制", "修业年限", "年限")) or "duration" in lowered
    graduate = any(term in query for term in ("硕士", "博士", "硕博", "直博", "研究生"))
    return graduate and wants_credit and wants_duration


def _query_wants_comparison(query: str) -> bool:
    return any(term in query for term in ("对比", "比较")) and any(term in query for term in ("和", "/", "与"))


def _graduate_credit_slots(query: str, context: ContextItem) -> GraduateCreditSlots | None:
    text = _normalized_text(context)
    basic_years = _first_number(
        (
            r"(?:基本学制|基本修业年限)(?:为)?\s*(\d+(?:\.\d+)?)\s*年",
            r"基本学制期限为\s*(\d+(?:\.\d+)?)\s*年",
        ),
        text,
    )
    max_years = _first_number((r"(?:最长学制|最长修业年限)(?:为)?\s*(\d+(?:\.\d+)?)\s*年",), text)
    total_credits = _first_number(
        (
            r"总学分不低于\s*(\d+(?:\.\d+)?)\s*(?:个)?\s*学分",
            r"总学分要求\s*(\d+(?:\.\d+)?)",
        ),
        text,
    )
    if basic_years is None or max_years is None or total_credits is None:
        return None
    return GraduateCreditSlots(
        label=_graduate_credit_label(query, text),
        source_rank=context.rank,
        basic_years=basic_years,
        max_years=max_years,
        total_credits=total_credits,
        course_credits=_first_number((r"课程学分不低于\s*(\d+(?:\.\d+)?)\s*学分",), text),
        practice_credits=_first_number(
            (r"(?:课程实践部分|实践教学课程实践部分)[^。；;，,\n]{0,12}?不低于\s*(\d+(?:\.\d+)?)\s*学分",),
            text,
        ),
        full_time="全日制" in text,
    )


def _graduate_credit_label(query: str, text: str) -> str:
    year_match = re.search(r"(20\d{2})\s*级", f"{query} {text}")
    prefix = f"{year_match.group(1)}级" if year_match is not None else ""
    if _looks_like_direct_phd_plan(text):
        return f"{prefix}硕博连读生（含硕士阶段）和直博生"
    if _looks_like_master_plan(text):
        return f"{prefix}硕士研究生"
    if "企业" in f"{query} {text}" and ("直博" in f"{query} {text}" or "博士" in f"{query} {text}"):
        return f"{prefix}电子信息企业联合培养直博项目"
    if "博士" in f"{query} {text}" or "直博" in f"{query} {text}":
        return f"{prefix}博士项目"
    return f"{prefix}研究生"


def _looks_like_master_plan(text: str) -> bool:
    return "硕士研究生" in text and "硕博连读" not in text and "直博" not in text


def _looks_like_direct_phd_plan(text: str) -> bool:
    return "硕博连读" in text or "直博" in text


def _first_number(patterns: tuple[str, ...], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            return _clean_number(match.group(1))
    return None


def _clean_number(number: str) -> str:
    return number[:-2] if number.endswith(".0") else number


def _normalized_text(context: ContextItem) -> str:
    return re.sub(r"\s+", " ", f"{context.title or ''} {context.text}").strip()


def _normalize_for_slot_check(text: str) -> str:
    return re.sub(r"\s+", "", text)
