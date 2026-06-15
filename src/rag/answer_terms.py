from __future__ import annotations

import re

from .retrieve import ContextItem


def _context_score_text(context: ContextItem) -> str:
    return "\n".join(str(part or "") for part in (context.title, context.url, context.snippet, context.text))


def _years(text: str) -> set[int]:
    return {int(year) for year in re.findall(r"(?<!\d)(20\d{2})(?!\d)", text)}


def _date_markers(text: str) -> set[str]:
    markers: set[str] = set()
    for year, month, day in re.findall(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text):
        markers.add(f"{year}-{int(month):02d}-{int(day):02d}")
    for year, month, day in re.findall(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", text):
        markers.add(f"{year}-{int(month):02d}-{int(day):02d}")
    for year, month, day in re.findall(r"(?<!\d)(20\d{2})[/_-](\d{2})(\d{2})(?!\d)", text):
        markers.add(f"{year}-{int(month):02d}-{int(day):02d}")
    return markers


def _exact_date_overlap_count(query: str, text: str) -> int:
    query_dates = _date_markers(query)
    if not query_dates:
        return 0
    return len(query_dates & _date_markers(text))


def _matched_terms(query: str, context_text: str, terms: tuple[str, ...]) -> list[str]:
    query_lower = query.lower()
    context_lower = context_text.lower()
    return [term for term in terms if term.lower() in query_lower and term.lower() in context_lower]


def _query_wants_contact(query: str) -> bool:
    lowered = query.lower()
    chinese_contact_terms = (
        "办公室",
        "邮箱",
        "教师",
        "教授",
        "老师",
        "联系方式",
        "联系人",
        "联系电话",
        "电话",
        "咨询",
        "联系",
    )
    return any(term in lowered for term in ("office", "email", "contact", "faculty", "professor", "phone")) or any(
        term in query for term in chinese_contact_terms
    )


def _has_contact_evidence(text: str) -> bool:
    lowered = text.lower()
    return (
        re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text) is not None
        or "office" in lowered
        or "phone" in lowered
        or "tel" in lowered
        or "办公室" in text
        or "邮箱" in text
        or "联系人" in text
        or "联系电话" in text
        or "电话" in text
        or "咨询" in text
        or "professor" in lowered
        or re.search(r"\b(?:Room|Rm\.?)\s+[A-Za-z0-9]", text, flags=re.IGNORECASE) is not None
        or re.search(r"(?:\d+\s*号楼\s*)?(?:\d?[A-Za-z]|[A-Za-z]区)[-－]?\s*\d{2,4}\s*室", text) is not None
    )


def _query_wants_degree_page(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("program", "degree", "credit", "credits", "curriculum")) or any(
        term in query for term in ("培养方案", "学分", "课程", "专业")
    )


def _query_wants_discipline_directions(query: str) -> bool:
    lowered = query.lower()
    return (
        "discipline direction" in lowered
        or "research direction" in lowered
        or "学科方向" in query
        or ("方向" in query and any(term in query for term in ("六个", "6个", "专业", "本科招生", "招生")))
    )


def _has_discipline_direction_anchor(text: str) -> bool:
    lowered = text.lower()
    return "学科方向" in text or "discipline direction" in lowered


def _looks_like_degree_page(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ("program", "degree", "credit", "credits", "curriculum")) or any(
        term in text for term in ("培养方案", "学分", "课程", "专业")
    )
