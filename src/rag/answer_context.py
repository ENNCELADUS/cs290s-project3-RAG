from __future__ import annotations

import re
from dataclasses import asdict, replace
from typing import Any

from .answer_recovery import (
    _anchor_overlap_count,
    _anchor_terms,
    _candidate_windows,
    _compact_sentence,
    _looks_like_navigation_span,
    _minimum_anchor_overlap,
    _query_requires_capacity_limit,
    _query_requires_phone_fact,
    _query_wants_schedule_and_contact,
)
from .answer_tables import append_structured_table_bindings
from .answer_terms import (
    _context_score_text,
    _date_markers,
    _exact_date_overlap_count,
    _has_contact_evidence,
    _has_discipline_direction_anchor,
    _looks_like_degree_page,
    _matched_terms,
    _query_wants_contact,
    _query_wants_degree_page,
    _query_wants_discipline_directions,
    _years,
)
from .answer_types import AnswerSource
from .retrieve import ContextItem, HybridRetrievalResult, Retriever

ANSWER_EVIDENCE_CONTEXT_CHARS = 1400
ANSWER_EVIDENCE_WINDOW_CHARS = 900


def _contexts_from_retrieval(retriever: Retriever, retrieval_result: object) -> list[ContextItem]:
    if isinstance(retrieval_result, HybridRetrievalResult):
        return retrieval_result.contexts
    return retriever.contexts_for_hits(retrieval_result)  # type: ignore[arg-type]


def _sources_from_contexts(contexts: list[ContextItem]) -> list[AnswerSource]:
    sources: list[AnswerSource] = []
    for context in contexts:
        if context.url is None:
            continue
        sources.append(
            AnswerSource(
                source_id=context.rank,
                title=context.title,
                url=context.url,
                chunk_id=context.chunk_id,
                document_id=context.document_id,
                trace_ref=context.trace_ref,
                snippet=context.snippet,
            )
        )
    return sources


def _retrieval_payload(retrieval_result: object, contexts: list[ContextItem]) -> dict[str, Any]:
    if isinstance(retrieval_result, HybridRetrievalResult):
        return {
            "mode": retrieval_result.mode,
            "hits": [asdict(hit) for hit in retrieval_result.hits],
            "contexts": [asdict(context) for context in contexts],
            "config": asdict(retrieval_result.config),
        }
    return {
        "mode": "dense",
        "hits": [_dataclass_or_value(hit) for hit in retrieval_result],  # type: ignore[union-attr]
        "contexts": [asdict(context) for context in contexts],
    }


def _dataclass_or_value(value: object) -> object:
    try:
        return asdict(value)
    except TypeError:
        return value


def _ordered_contexts(contexts: list[ContextItem], answer_context_order: list[dict[str, Any]]) -> list[ContextItem]:
    by_source_id = {context.rank: context for context in contexts}
    ordered = [
        by_source_id[int(item["source_id"])] for item in answer_context_order if int(item["source_id"]) in by_source_id
    ]
    if len(ordered) == len(contexts):
        return ordered
    selected_ids = {context.rank for context in ordered}
    return [*ordered, *[context for context in contexts if context.rank not in selected_ids]]


def _select_local_evidence_contexts(
    query: str,
    contexts: list[ContextItem],
    *,
    retriever: Retriever | None = None,
) -> list[ContextItem]:
    return [
        _select_local_evidence_context(query, context, sibling_texts=_sibling_chunk_texts(retriever, context))
        for context in contexts
    ]


def _select_local_evidence_context(
    query: str,
    context: ContextItem,
    *,
    sibling_texts: list[str],
) -> ContextItem:
    combined_text = "\n".join([context.text, *sibling_texts])
    if not sibling_texts and len(combined_text) <= ANSWER_EVIDENCE_CONTEXT_CHARS:
        structured_text = append_structured_table_bindings(context.text)
        if structured_text == context.text:
            return context
        return replace(context, text=structured_text, snippet=structured_text[:240])
    evidence_text = _local_evidence_text(query, context, combined_text)
    if evidence_text is None:
        structured_text = append_structured_table_bindings(context.text)
        if structured_text == context.text:
            return context
        return replace(context, text=structured_text, snippet=structured_text[:240])
    evidence_text = append_structured_table_bindings(evidence_text)
    return replace(context, text=evidence_text, snippet=evidence_text[:240])


def _sibling_chunk_texts(retriever: Retriever | None, context: ContextItem) -> list[str]:
    if retriever is None:
        return []
    chunks = getattr(retriever, "_chunks", None)
    if not isinstance(chunks, list):
        return []
    siblings: list[tuple[int, str]] = []
    for row in chunks:
        if not isinstance(row, dict):
            continue
        try:
            row_chunk_id = int(row.get("chunk_id", -1))
            row_document_id = int(row.get("document_id", -1))
        except (TypeError, ValueError):
            continue
        if row_chunk_id == context.chunk_id:
            continue
        same_document = row_document_id == context.document_id
        same_url = context.url is not None and row.get("url") == context.url
        if not same_document and not same_url:
            continue
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            siblings.append((row_chunk_id, text))
    return [text for _chunk_id, text in sorted(siblings)]


def _local_evidence_text(query: str, context: ContextItem, text: str) -> str | None:
    query_terms = _anchor_terms(query)
    evidence_pattern = _local_evidence_pattern(query)
    if evidence_pattern is None:
        return None
    scored: list[tuple[float, int, str]] = []
    context_header = " ".join(part for part in (context.title, context.url, context.snippet) if part)
    for index, window in enumerate(_candidate_windows(text)):
        candidate_match_text = f"{context_header} {window}"
        anchor_overlap = _anchor_overlap_count(query_terms, candidate_match_text)
        evidence_count = len(evidence_pattern.findall(window))
        if anchor_overlap < _minimum_anchor_overlap(query_terms) and evidence_count == 0:
            continue
        compact = _compact_sentence(window, max_chars=ANSWER_EVIDENCE_WINDOW_CHARS)
        if not compact:
            continue
        if _looks_like_navigation_span(compact):
            continue
        score = anchor_overlap * 1.5 + evidence_count * 6.0
        score += len(_years(query) & _years(candidate_match_text)) * 8.0
        score += _exact_date_overlap_count(query, candidate_match_text) * 12.0
        scored.append((score, -index, compact))
    if not scored:
        return None

    selected: list[str] = []
    total_chars = 0
    for _score, _negative_index, text in sorted(scored, reverse=True):
        normalized = re.sub(r"\s+", " ", text).strip()
        if any(normalized in existing or existing in normalized for existing in selected):
            continue
        if total_chars + len(normalized) > ANSWER_EVIDENCE_CONTEXT_CHARS:
            continue
        selected.append(normalized)
        total_chars += len(normalized)
        if total_chars >= ANSWER_EVIDENCE_WINDOW_CHARS:
            break
    if not selected:
        return None
    return "\n".join(selected)


def _local_evidence_pattern(query: str) -> re.Pattern[str] | None:
    patterns: list[str] = []
    lowered = query.lower()
    if _query_wants_contact(query) or _query_requires_phone_fact(query):
        patterns.extend(
            [
                r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
                r"(?<!\d)(?:0\d{2,3}[-\s]?)?\d{7,8}(?!\d)",
                r"联系咨询|咨询方式|联系人|联系电话|电话|座机|邮箱|办公室",
            ]
        )
    if _query_wants_schedule_and_contact(query) or _query_requires_capacity_limit(query):
        patterns.append(r"\d{1,2}\s*月\s*\d{1,2}\s*日|时间|地点|人数|上限|不超过|主讲")
    if "供应商" in query or "采购" in query or "procurement" in lowered:
        patterns.append(r"报价供应商要求|营业执照|税务登记证|组织机构代码证|联合体|报名资料|报价截止|递交地点")
    if "复试" in query or "总成绩" in query or "formula" in lowered:
        patterns.append(r"综合素质考核|专业面试|复试成绩|满分|合格|总成绩|初试成绩")
    if any(term in query for term in ("副主编", "IEEE Trans")) or any(
        term in lowered for term in ("tie", "tte", "tpea")
    ):
        patterns.append(r"副主编|IEEE\s*Trans(?:actions)?|TIE|TTE|TPEA")
    if any(term in query for term in ("专利", "第一发明人", "申请号", "在校生")):
        patterns.append(r"专利|第一发明人|申请号|在校生|CN\d{6,}")
    if any(term in query for term in ("选拔方式", "招生方式", "直博", "申请-考核制")):
        patterns.append(r"选拔方式|招生方式|直博|申请[-－—]考核制")
    if any(term in query for term in ("三选二", "本学科选修")) or ("2025级" in query and "ee" in lowered):
        patterns.append(r"三选二|本学科选修|2025\s*级|电子信息工程|EE")
    if any(term in query for term in ("录制成视频", "提前学习", "电力电子")):
        patterns.append(r"录制成视频|提前学习|电力电子")
    if not patterns:
        return None
    return re.compile("|".join(patterns), re.IGNORECASE)


def _metadata_answer_context_score(query: str, context: ContextItem) -> tuple[float, list[str]]:
    haystack = _context_score_text(context)
    normalized = haystack.lower()
    score = 0.0
    reasons: list[str] = []

    query_years = _years(query)
    context_years = _years(haystack)
    for year in sorted(query_years & context_years):
        score += 8.0
        reasons.append(f"query_year_match:{year}")
    exact_date_matches = _exact_date_overlap_count(query, haystack)
    if exact_date_matches:
        score += exact_date_matches * 8.0
        reasons.append("exact_date_match")
    elif _date_markers(query) and _date_markers(haystack):
        score -= 4.0
        reasons.append("date_mismatch_penalty")
    if query_years and context_years and _looks_like_degree_page(haystack):
        target_year = max(query_years)
        old_years = sorted(year for year in context_years if year < target_year)
        if old_years and target_year not in context_years:
            score -= 8.0
            reasons.append(f"old_year_penalty:{old_years[-1]}<{target_year}")

    anchor_count = sum(1 for term in _anchor_terms(query) if term in normalized)
    if anchor_count:
        score += min(anchor_count, 8) * 0.4
        reasons.append(f"anchor_overlap:{anchor_count}")

    if _query_wants_discipline_directions(query) and _has_discipline_direction_anchor(haystack):
        score += 6.0
        reasons.append("task_anchor:学科方向")

    program_matches = _matched_terms(
        query,
        haystack,
        ("cs", "computer science", "ee", "electrical", "electronic", "计算机", "电子", "电气", "信息"),
    )
    if program_matches:
        score += len(program_matches) * 1.5
        reasons.append(f"program_terms:{','.join(program_matches)}")

    course_matches = _matched_terms(
        query,
        haystack,
        ("培养方案", "学分", "课程", "program", "degree", "credit", "credits", "course", "curriculum"),
    )
    if course_matches:
        score += len(course_matches) * 1.5
        reasons.append(f"course_terms:{','.join(course_matches)}")

    if _query_wants_contact(query) and _has_contact_evidence(haystack):
        score += 3.0
        reasons.append("faculty_or_contact_evidence")
    if _query_wants_degree_page(query) and _looks_like_degree_page(haystack):
        score += 3.0
        reasons.append("page_type:degree_or_program")
    if not reasons:
        reasons.append("retrieval_rank_tiebreak")
    return score, reasons
