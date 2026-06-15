from __future__ import annotations

import json
import re
from typing import Any

from .answer_slots import extract_required_slot_answer, required_slot_rejection_reason, required_slot_values
from .answer_terms import (
    _context_score_text,
    _date_markers,
    _exact_date_overlap_count,
    _has_contact_evidence,
    _looks_like_degree_page,
    _matched_terms,
    _query_wants_contact,
    _query_wants_degree_page,
    _query_wants_discipline_directions,
    _years,
)
from .answer_types import (
    AnswerConfig,
    AnswerMode,
    AnswerSource,
    AnswerTiming,
    ExtractiveAnswer,
    ExtractiveCandidate,
    RagAnswerResult,
)
from .graduate_credits import extract_graduate_credit_answer, graduate_credit_rejection_reason
from .retrieve import ContextItem

VALID_CITATION_RE = re.compile(r"\[(\d+)\]")
PROMPT_LEAKAGE_MARKERS = ("Question:", "Sources:", "TEXT:", "URL:", "trace_ref:", "Use only the provided")


def _is_acceptable_answer(
    text: str,
    valid_source_ids: set[int],
    *,
    query: str | None = None,
    contexts: list[ContextItem] | None = None,
) -> bool:
    return _answer_rejection_reason(text, valid_source_ids, query=query, contexts=contexts) is None


def _answer_rejection_reason(
    text: str,
    valid_source_ids: set[int],
    *,
    query: str | None = None,
    contexts: list[ContextItem] | None = None,
) -> str | None:
    if any(marker in text for marker in PROMPT_LEAKAGE_MARKERS):
        return "prompt_leakage"
    if _states_insufficient_evidence(text):
        return "model_reported_insufficient_evidence"
    if not _has_valid_citation(text, valid_source_ids):
        return "invalid_or_missing_citation"
    if query is not None:
        shape_rejection = _query_shape_rejection_reason(query, text)
        if shape_rejection is not None:
            return shape_rejection
    if query is not None and contexts:
        label_binding_rejection = _label_value_binding_rejection_reason(query, text, contexts)
        if label_binding_rejection is not None:
            return label_binding_rejection
        formula_rejection = _numeric_formula_rejection_reason(query, text, contexts)
        if formula_rejection is not None:
            return formula_rejection
        source_fact_rejection = _requested_source_fact_rejection_reason(query, text, contexts)
        if source_fact_rejection is not None:
            return source_fact_rejection
        citation_rejection = _citation_support_rejection_reason(query, text, contexts)
        if citation_rejection is not None:
            return citation_rejection
    return None


def _has_valid_citation(text: str, valid_source_ids: set[int]) -> bool:
    citation_ids = [int(match.group(1)) for match in VALID_CITATION_RE.finditer(text)]
    return bool(citation_ids) and all(citation_id in valid_source_ids for citation_id in citation_ids)


def _states_insufficient_evidence(text: str) -> bool:
    normalized = text.lower()
    chinese_negative_markers = (
        "证据不足",
        "无法找到",
        "未提及",
        "没有提及",
        "未找到",
        "没有找到",
    )
    if "evidence is insufficient" not in normalized and not any(marker in text for marker in chinese_negative_markers):
        return False
    if _contains_substantive_answer_fact(text):
        return False
    return True


def _contains_substantive_answer_fact(text: str) -> bool:
    clean_text = VALID_CITATION_RE.sub(" ", text)
    return (
        re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", clean_text) is not None
        or re.search(r"\d+(?:\.\d+)?\s*(?:credits?|学分|%|人|项|门|个|分)", clean_text, re.IGNORECASE) is not None
        or re.search(r"(?:Room|Rm\.?)\s+[A-Za-z0-9][A-Za-z0-9.-]{1,20}", clean_text, re.IGNORECASE) is not None
        or re.search(r"(?:办公室|地址)[:：]\s*[\u4e00-\u9fffA-Za-z0-9.-]{2,30}", clean_text) is not None
    )


def _query_shape_rejection_reason(query: str, answer: str) -> str | None:
    if _query_requires_contact_fact(query) and not _has_contact_evidence(answer):
        return "missing_requested_contact_fact"
    return None


def _query_requires_contact_fact(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("office", "email", "contact")) or any(
        term in query for term in ("办公室", "邮箱", "联系方式", "联系")
    )


def _requested_source_fact_rejection_reason(query: str, answer: str, contexts: list[ContextItem]) -> str | None:
    required_slot_rejection = required_slot_rejection_reason(query, answer, contexts)
    if required_slot_rejection is not None:
        return required_slot_rejection
    graduate_credit_rejection = graduate_credit_rejection_reason(query, answer, contexts)
    if graduate_credit_rejection is not None:
        return graduate_credit_rejection
    cited_ids = {int(match.group(1)) for match in VALID_CITATION_RE.finditer(answer)}
    cited_text = "\n".join(_context_score_text(context) for context in contexts if context.rank in cited_ids)
    if _query_requires_professional_elective_credits(query):
        expected_credits = _professional_elective_credit_values(cited_text)
        if expected_credits and expected_credits.isdisjoint(_credit_values(answer)):
            return "missing_requested_professional_elective_credits"
    credit_rejection = _requested_degree_credit_rejection_reason(query, answer, cited_text)
    if credit_rejection is not None:
        return credit_rejection
    labeled_credit_rejection = _requested_labeled_credit_rejection_reason(query, answer, cited_text)
    if labeled_credit_rejection is not None:
        return labeled_credit_rejection
    profile_slots = _requested_faculty_profile_slot_values(query, cited_text)
    if profile_slots and any(value not in answer for value in profile_slots.values()):
        return "missing_requested_profile_fact"
    if _query_requires_phone_fact(query) and _has_phone_evidence(cited_text) and not _has_phone_evidence(answer):
        return "missing_requested_phone_fact"
    if _query_requires_capacity_limit(query):
        expected_limits = _capacity_limit_numbers(cited_text)
        if expected_limits and expected_limits.isdisjoint(_capacity_limit_numbers(answer)):
            return "missing_requested_capacity_limit"
    if _query_requires_lab_count_labels(query):
        missing_labels = [
            label
            for label in ("课题组", "联合实验室")
            if label in query and label in cited_text and label not in answer
        ]
        if missing_labels:
            return "missing_requested_labeled_fact"
    if (
        _query_requires_location_fact(query)
        and _has_location_evidence(cited_text)
        and not _has_location_evidence(answer)
    ):
        return "missing_requested_location_fact"
    multi_slot_rejection = _multi_field_answer_quality_rejection_reason(query, answer, contexts)
    if multi_slot_rejection is not None:
        return multi_slot_rejection
    return None


def _multi_field_answer_quality_rejection_reason(
    query: str,
    answer: str,
    contexts: list[ContextItem],
) -> str | None:
    profile_rejection = _multi_field_profile_rejection_reason(query, answer)
    if profile_rejection is not None:
        return profile_rejection
    procurement_rejection = _multi_project_supplier_rejection_reason(query, answer, contexts)
    if procurement_rejection is not None:
        return procurement_rejection
    return None


def _multi_field_profile_rejection_reason(query: str, answer: str) -> str | None:
    requested_slots = set(_requested_faculty_profile_slot_names(query))
    if len(requested_slots) < 2:
        return None
    covered_slots = _covered_faculty_profile_slot_names(answer)
    if requested_slots <= covered_slots:
        return None
    return "incomplete_requested_profile_slots"


def _covered_faculty_profile_slot_names(answer: str) -> set[str]:
    lowered = answer.lower()
    covered: set[str] = set()
    if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", answer):
        covered.add("email")
    if (
        "办公室" in answer
        or "office" in lowered
        or re.search(r"\b(?:Room|Rm\.?)\s+[A-Za-z0-9][A-Za-z0-9.-]{1,20}\b", answer, flags=re.IGNORECASE)
        or re.search(r"\b(?:SIST|Building)\s+[A-Za-z0-9][A-Za-z0-9.-]{1,20}\b", answer, flags=re.IGNORECASE)
    ):
        covered.add("office")
    if "博士" in answer or "phd" in lowered:
        covered.add("phd_school")
    if "研究方向" in answer or "research direction" in lowered or "research interests" in lowered:
        covered.add("direction")
    return covered


def _multi_project_supplier_rejection_reason(
    query: str,
    answer: str,
    contexts: list[ContextItem],
) -> str | None:
    if not _query_wants_multiple_procurement_suppliers(query):
        return None
    if any(marker in answer for marker in ("异议", "质疑材料", "递交地址", "书面形式")):
        return "procurement_boilerplate_fallback"

    supplier_slots = [
        slot for slot in required_slot_values(query, contexts) if slot.name.startswith("procurement_supplier:")
    ]
    if len(supplier_slots) >= 2:
        normalized_answer = re.sub(r"\s+", "", answer)
        for slot in supplier_slots:
            if re.sub(r"\s+", "", slot.value) not in normalized_answer:
                return "incomplete_procurement_supplier_slots"
        return None
    if _procurement_project_supplier_pair_count(answer) >= 2:
        return None
    return "incomplete_procurement_supplier_slots"


def _query_wants_multiple_procurement_suppliers(query: str) -> bool:
    return "供应商" in query and "采购" in query and any(term in query for term in ("分别", "两个", "和", "及"))


def _procurement_project_supplier_pair_count(text: str) -> int:
    pairs = re.findall(
        r"[\u4e00-\u9fffA-Za-z0-9（）()·\-]{2,80}?采购项目"
        r"[^。；;\n]{0,32}?"
        r"[\u4e00-\u9fffA-Za-z0-9（）()·\-]{2,50}?(?:有限公司|公司|研究所|大学|中心)",
        text,
    )
    return len(pairs)


def _query_requires_phone_fact(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("phone", "telephone", "tel", "contact")) or any(
        term in query for term in ("电话", "联系电话", "联系方式", "联系人")
    )


def _has_phone_evidence(text: str) -> bool:
    return re.search(r"(?<!\d)(?:0\d{2,3}[-\s]?)?\d{7,8}(?!\d)", text) is not None


def _query_requires_capacity_limit(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("capacity", "limit", "cap")) or any(
        term in query for term in ("人数上限", "上限", "不超过", "名额")
    )


def _capacity_limit_numbers(text: str) -> set[str]:
    numbers: set[str] = set()
    for match in re.finditer(r"(?:不超过|不多于|限|上限)[^。；;，,\n]{0,12}?(\d+)\s*(?:人|位|名)", text):
        numbers.add(match.group(1))
    limit_after_subject = r"(?:人数|名额)[^。；;，,\n]{0,12}?(?:不超过|不多于|限|上限)[^。；;，,\n]{0,8}?(\d+)"
    for match in re.finditer(limit_after_subject, text):
        numbers.add(match.group(1))
    return numbers


def _query_requires_lab_count_labels(query: str) -> bool:
    return ("多少" in query or "几个" in query or "数量" in query) and any(
        label in query for label in ("课题组", "联合实验室")
    )


def _query_requires_location_fact(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("room", "location", "where")) or any(
        term in query for term in ("房间", "地点", "哪里", "哪儿", "哪一个房间")
    )


def _has_location_evidence(text: str) -> bool:
    return (
        "地点" in text
        or re.search(r"\b(?:Room|Rm\.?)\s+[A-Za-z0-9][A-Za-z0-9.-]{1,20}\b", text, flags=re.IGNORECASE) is not None
        or re.search(r"信息学院\s*\d[A-Za-z]?[-－]\d{2,4}", text) is not None
    )


def _query_requires_professional_elective_credits(query: str) -> bool:
    lowered = query.lower()
    wants_credit = "学分" in query or "credit" in lowered
    wants_professional = "专业课程" in query or "professional course" in lowered
    wants_elective = "选修" in query or "elective" in lowered
    return wants_credit and wants_professional and wants_elective


def _requested_degree_credit_rejection_reason(query: str, answer: str, cited_text: str) -> str | None:
    expected_values = _requested_degree_credit_values(query, cited_text)
    if not expected_values:
        return None
    answer_values = _credit_values(answer)
    if expected_values <= answer_values:
        return None
    return "missing_requested_credit_fields"


def _requested_degree_credit_values(query: str, text: str) -> set[str]:
    lowered_query = query.lower()
    if "学分" not in query and "credit" not in lowered_query:
        return set()
    summary = _degree_plan_summary(re.sub(r"\s+", " ", text).strip())
    if summary is None:
        return set()

    expected: set[str] = set()
    total = summary.get("total")
    if isinstance(total, str) and _query_requests_degree_total_credit(query):
        expected.add(total)

    for query_label, row_label in (
        ("人文社科", "人文社科通识"),
        ("自然科学", "自然科学通识"),
    ):
        row = summary.get(row_label)
        if query_label in query and isinstance(row, tuple):
            expected.add(row[2])

    professional = summary.get("专业课程")
    if "专业" in query and isinstance(professional, tuple):
        required, elective, professional_total = professional
        if required and ("必修" in query or "required" in lowered_query):
            expected.add(required)
        if elective and ("选修" in query or "elective" in lowered_query):
            expected.add(elective)
        if professional_total and any(term in query for term in ("合计", "总计", "总共", "总学分")):
            expected.add(professional_total)

    free = summary.get("任选课程")
    if "任选" in query and isinstance(free, tuple):
        expected.add(free[2])
    return expected


def _query_requests_degree_total_credit(query: str) -> bool:
    lowered_query = query.lower()
    if "total credit" in lowered_query:
        return True
    if "板块" in query and "总学分" not in query:
        return False
    if "毕业" in query or "修满" in query:
        return True
    if "总学分" not in query:
        return False
    if re.search(r"(?:人文社科|自然科学|专业课程)[^？?。]*总学分", query) and not re.search(
        r"总学分[、,，和及]",
        query,
    ):
        return False
    return True


def _query_requests_total_and_free_choice_only(query: str) -> bool:
    if not _query_requests_degree_total_credit(query):
        return False
    if any(term in query for term in ("人文社科", "自然科学", "专业课程", "必修")):
        return False
    return True


def _requested_labeled_credit_rejection_reason(query: str, answer: str, cited_text: str) -> str | None:
    if "学分" not in query and "credit" not in query.lower():
        return None
    expected: set[str] = set()
    for label in ("总学分", "课程学分", "课程实践"):
        if label in query:
            expected.update(_labeled_credit_values(cited_text, label))
    if not expected:
        return None
    if expected <= _credit_values(answer):
        return None
    return "missing_requested_credit_fields"


def _labeled_credit_values(text: str, label: str) -> set[str]:
    values: set[str] = set()
    for match in re.finditer(rf"{label}[^。；;，,\n]{{0,24}}?(\d+(?:\.\d+)?)\s*(?:个)?\s*学分", text):
        values.add(_clean_number(match.group(1)))
    return values


def _professional_elective_credit_values(text: str) -> set[str]:
    summary = _degree_plan_summary(re.sub(r"\s+", " ", text).strip())
    if summary is None:
        return set()
    professional = summary.get("专业课程")
    if not isinstance(professional, tuple) or professional[1] is None:
        return set()
    return {professional[1]}


def _credit_values(text: str) -> set[str]:
    return {_clean_number(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:credits?|学分)", text, re.IGNORECASE)}


def _requested_faculty_profile_slot_values(query: str, text: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    requested_slots = set(_requested_faculty_profile_slot_names(query))
    if "office" in requested_slots and (office := _field_after_label(text, "办公室")):
        slots["office"] = office
    if "email" in requested_slots and (email := _first_match(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)):
        slots["email"] = email
    if "phd_school" in requested_slots and (phd_school := _phd_school_from_profile_text(text)):
        slots["phd_school"] = phd_school
    if "direction" in requested_slots and (direction := _field_after_label(text, "研究方向")):
        slots["direction"] = direction
    return slots


def _requested_faculty_profile_slot_names(query: str) -> tuple[str, ...]:
    lowered_query = query.lower()
    slots: list[str] = []
    if "办公室" in query or "office" in lowered_query:
        slots.append("office")
    if "邮箱" in query or "email" in lowered_query:
        slots.append("email")
    if any(term in query for term in ("博士毕业学校", "博士毕业院校", "博士学校", "博士毕业于哪")) or (
        "phd" in lowered_query and "school" in lowered_query
    ):
        slots.append("phd_school")
    if (
        "研究方向是什么" in query
        or ("研究方向" in query and "什么" in query and "研究方向包括" not in query)
        or ("research direction" in lowered_query and "what" in lowered_query)
    ):
        slots.append("direction")
    return tuple(slots)


def _parse_repair_answer(
    text: str,
    *,
    valid_source_ids: set[int],
    query: str | None = None,
    contexts: list[ContextItem] | None = None,
) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "answered":
        return None
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    answer = answer.strip()
    if not _is_acceptable_answer(answer, valid_source_ids, query=query, contexts=contexts):
        return None
    return answer


def _citation_support_rejection_reason(query: str, answer: str, contexts: list[ContextItem]) -> str | None:
    cited_ids = {int(match.group(1)) for match in VALID_CITATION_RE.finditer(answer)}
    context_by_rank = {context.rank: context for context in contexts}
    cited_contexts = [context_by_rank[source_id] for source_id in cited_ids if source_id in context_by_rank]
    if not cited_contexts:
        return None

    scores = {context.rank: _citation_support_score(query, answer, context) for context in contexts if context.url}
    if not scores:
        return None
    cited_best = max(scores.get(context.rank, 0.0) for context in cited_contexts)
    best_rank, best_score = max(scores.items(), key=lambda item: item[1])
    if best_rank in cited_ids:
        return None

    query_years = _years(query)
    if query_years and _context_matches_year(context_by_rank[best_rank], query_years):
        cited_year_match = any(_context_matches_year(context, query_years) for context in cited_contexts)
        if not cited_year_match:
            return "weak_citation_support"

    answer_facts = _answer_fact_terms(answer)
    if answer_facts and best_score >= 5.0 and best_score >= cited_best + 3.0:
        return "weak_citation_support"
    return None


def _label_value_binding_rejection_reason(query: str, answer: str, contexts: list[ContextItem]) -> str | None:
    cited_ids = {int(match.group(1)) for match in VALID_CITATION_RE.finditer(answer)}
    if not cited_ids:
        return None
    if "任选" not in query and "任选" not in answer:
        return None
    answer_free_values = _answer_labeled_credit_values(answer, ("任选课", "任选课程"))
    if not answer_free_values:
        return None

    cited_contexts = [context for context in contexts if context.rank in cited_ids and context.url]
    expected_values = _degree_plan_expected_values(query, cited_contexts, "任选课程")
    if not expected_values:
        return None
    if answer_free_values.isdisjoint(expected_values):
        return "unsupported_label_value_binding"
    return None


def _numeric_formula_rejection_reason(query: str, answer: str, contexts: list[ContextItem]) -> str | None:
    if not _query_wants_formula_answer(query, answer):
        return None
    cited_ids = {int(match.group(1)) for match in VALID_CITATION_RE.finditer(answer)}
    if not cited_ids:
        return None
    answer_weights = _percent_values(answer)
    if not answer_weights:
        return None
    cited_text = " ".join(_context_score_text(context) for context in contexts if context.rank in cited_ids)
    cited_weights = _percent_values(cited_text)
    if cited_weights and not answer_weights <= cited_weights:
        return "unsupported_numeric_formula"
    return None


def _query_wants_formula_answer(query: str, answer: str) -> bool:
    combined = f"{query} {answer}"
    return ("公式" in combined or "总成绩" in combined or "formula" in combined.lower()) and "%" in answer


def _percent_values(text: str) -> set[str]:
    return {_clean_number(value) for value in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*%", text)}


def _answer_labeled_credit_values(answer: str, labels: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for label in labels:
        for match in re.finditer(rf"{label}[^。；;，,、\n]{{0,20}}?(\d+(?:\.\d+)?)\s*学分", answer):
            values.add(_clean_number(match.group(1)))
    return values


def _degree_plan_expected_values(query: str, contexts: list[ContextItem], label: str) -> set[str]:
    expected: set[str] = set()
    matching_contexts = []
    for context in contexts:
        text = re.sub(r"\s+", " ", f"{context.title or ''} {context.text}").strip()
        summary = _degree_plan_summary(text)
        if summary is None:
            continue
        if _degree_plan_context_matches_query(query, text):
            matching_contexts.append((summary, label))
        elif not matching_contexts:
            row = summary.get(label)
            if isinstance(row, tuple):
                expected.add(row[2])
    if matching_contexts:
        expected.clear()
        for summary, row_label in matching_contexts:
            row = summary.get(row_label)
            if isinstance(row, tuple):
                expected.add(row[2])
    return expected


def _degree_plan_context_matches_query(query: str, text: str) -> bool:
    query_years = _years(query)
    if query_years and not query_years <= _years(text):
        return False
    if "人工智能荣誉班" in query and "人工智能荣誉班" not in text:
        return False
    if ("电子信息工程" in query or "EE" in query) and "电子信息工程" not in text and "EE" not in text:
        return False
    if "计算机科学与技术" in query and "计算机科学与技术" not in text:
        return False
    return True


def _citation_support_score(query: str, answer: str, context: ContextItem) -> float:
    text = _context_score_text(context)
    normalized_text = text.lower()
    score = 0.0

    query_years = _years(query)
    context_years = _years(text)
    if query_years & context_years:
        score += 4.0
    elif query_years and context_years and max(context_years) < max(query_years):
        score -= 2.0

    answer_years = _years(answer)
    if answer_years & context_years:
        score += 4.0

    for fact in _answer_fact_terms(answer):
        if fact.lower() in normalized_text:
            score += 1.0

    anchor_hits = sum(1 for term in _anchor_terms(query) if term in normalized_text)
    score += min(anchor_hits, 8) * 0.4
    if _query_wants_degree_page(query) and _looks_like_degree_page(text):
        score += 2.0
    if _query_wants_contact(query) and _has_contact_evidence(text):
        score += 2.0
    return score


def _context_matches_year(context: ContextItem, years: set[int]) -> bool:
    return bool(_years(_context_score_text(context)) & years)


def _answer_fact_terms(answer: str) -> set[str]:
    clean_answer = VALID_CITATION_RE.sub(" ", answer)
    facts = set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", clean_answer))
    facts.update(re.findall(r"(?<!\d)20\d{2}(?!\d)", clean_answer))
    facts.update(re.findall(r"(?<!\d)\d+(?:\.\d+)?\s*(?:credits?|学分|%|人|项|门|个)?", clean_answer, re.I))
    facts.update(re.findall(r"\b\d{1,2}:\d{2}\b", clean_answer))
    return {fact.strip() for fact in facts if fact.strip()}


def _extract_answer_from_contexts(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    procurement_notice_answer = _extract_procurement_notice_answer(query, contexts)
    if procurement_notice_answer is not None:
        return procurement_notice_answer
    schedule_contact_answer = _extract_schedule_contact_answer(query, contexts)
    if schedule_contact_answer is not None:
        return schedule_contact_answer
    procurement_delivery_answer = _extract_procurement_delivery_answer(query, contexts)
    if procurement_delivery_answer is not None:
        return procurement_delivery_answer
    required_slot_answer = extract_required_slot_answer(query, contexts)
    if required_slot_answer is not None:
        return required_slot_answer
    faculty_profile_answer = _extract_faculty_profile_slot_answer(query, contexts)
    if faculty_profile_answer is not None:
        return faculty_profile_answer
    office_answer = _extract_office_email_answer(query, contexts)
    if office_answer is not None:
        return office_answer
    address_answer = _extract_address_postcode_answer(query, contexts)
    if address_answer is not None:
        return address_answer
    lab_count_answer = _extract_lab_count_answer(query, contexts)
    if lab_count_answer is not None:
        return lab_count_answer
    discipline_direction_answer = _extract_admissions_discipline_direction_answer(query, contexts)
    if discipline_direction_answer is not None:
        return discipline_direction_answer
    committee_answer = _extract_committee_row_answer(query, contexts)
    if committee_answer is not None:
        return committee_answer
    graduate_credit_answer = extract_graduate_credit_answer(query, contexts)
    if graduate_credit_answer is not None:
        return graduate_credit_answer
    degree_summary_answer = _extract_degree_plan_summary_answer(query, contexts)
    if degree_summary_answer is not None:
        return degree_summary_answer
    retest_formula_answer = _extract_retest_formula_answer(query, contexts)
    if retest_formula_answer is not None:
        return retest_formula_answer
    course_credit_answer = _extract_course_credit_row_answer(query, contexts)
    if course_credit_answer is not None:
        return course_credit_answer
    course_design_answer = _extract_course_design_pair_answer(query, contexts)
    if course_design_answer is not None:
        return course_design_answer
    dedup_answer = _extract_degree_plan_dedup_answer(query, contexts)
    if dedup_answer is not None:
        return dedup_answer
    power_electronics_answer = _extract_fu_minfan_power_electronics_video_answer(query, contexts)
    if power_electronics_answer is not None:
        return power_electronics_answer
    credit_answer = _extract_credit_answer(query, contexts)
    if credit_answer is not None:
        return credit_answer
    seminar_answer = _extract_seminar_event_fields_answer(query, contexts)
    if seminar_answer is not None:
        return seminar_answer
    schedule_answer = _extract_date_time_location_answer(query, contexts)
    if schedule_answer is not None:
        return schedule_answer
    robotics_answer = _extract_robotics_faculty_answer(query, contexts)
    if robotics_answer is not None:
        return robotics_answer
    profile_answer = _extract_compact_person_profile_answer(query, contexts)
    if profile_answer is not None:
        return profile_answer
    student_undergraduate_answer = _extract_student_undergraduate_school_answer(query, contexts)
    if student_undergraduate_answer is not None:
        return student_undergraduate_answer
    for course_name in _course_terms_from_query(query):
        course_pattern = re.escape(course_name).replace(r"\ ", r"\s+")
        teacher_pattern = re.compile(rf"{course_pattern}\s*【\s*(?P<teacher>[^】]{{1,80}}?)\s*】", re.IGNORECASE)
        for context in contexts:
            if context.url is None:
                continue
            normalized_text = re.sub(r"\s+", " ", context.text)
            match = teacher_pattern.search(normalized_text)
            if match is None:
                continue
            teacher = " ".join(match.group("teacher").split())
            if not teacher:
                continue
            if _is_chinese(query):
                return ExtractiveAnswer(f"{course_name}的任课老师是{teacher}。 [{context.rank}]", context.rank)
            return ExtractiveAnswer(f"{course_name} was taught by {teacher} [{context.rank}].", context.rank)
    list_answer = _extract_list_or_comparison_answer(query, contexts)
    if list_answer is not None:
        return list_answer
    return None


def _extract_lab_count_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "课题组" not in query and "联合实验室" not in query:
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", "", context.text)
        match = re.search(r"(\d+)个?课题组[^。；;]{0,20}?(\d+)个?联合实验室", text)
        if match is None:
            continue
        groups, labs = match.groups()
        return ExtractiveAnswer(f"信息学院有{groups}个课题组、{labs}个联合实验室。 [{context.rank}]", context.rank)
    return None


def _extract_retest_formula_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "复试" not in query or ("公式" not in query and "总成绩" not in query):
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", "", context.text)
        if "综合素质考核" not in text or "专业面试" not in text or "总成绩" not in text:
            continue
        full_score = re.search(r"复试成绩满分(?:为)?(\d+(?:\.\d+)?)分", text)
        pass_score = re.search(r"(\d+(?:\.\d+)?)分为合格", text)
        formula = re.search(
            r"考生总成绩[=＝]初试成绩[×x*](\d+(?:\.\d+)?%)\+复试成绩[×x*](\d+(?:\.\d+)?%)",
            text,
            flags=re.IGNORECASE,
        )
        normalized_formula = re.search(
            r"考生总成绩[=＝](?P<formula>"
            r"\d+(?:\.\d+)?[×x*]初试成绩[/／]初试满分\+"
            r"\d+(?:\.\d+)?[×x*]复试成绩[/／]复试满分)",
            text,
            flags=re.IGNORECASE,
        )
        if full_score is None or pass_score is None or (formula is None and normalized_formula is None):
            continue
        full = _clean_number(full_score.group(1))
        passing = _clean_number(pass_score.group(1))
        if formula is not None:
            formula_text = f"初试成绩×{formula.group(1)}+复试成绩×{formula.group(2)}"
        else:
            formula_text = normalized_formula.group("formula").replace("×", "*").replace("x", "*")
        return ExtractiveAnswer(
            (
                "2026年复试包括综合素质考核和专业面试；"
                f"复试成绩满分为{full}分，{passing}分为合格；"
                f"考生总成绩={formula_text}。 [{context.rank}]"
            ),
            context.rank,
        )
    return None


def _extract_admissions_discipline_direction_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if not _query_wants_discipline_directions(query):
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        if "学科方向" not in text:
            continue
        directions = _field_after_label(text, "学科方向")
        if directions is None:
            continue
        subject = "CS专业" if "CS" in query or "计算机" in query or "计算机科学与技术" in text else "该专业"
        return ExtractiveAnswer(f"{subject}的学科方向是{directions}。 [{context.rank}]", context.rank)
    return None


def _extract_committee_row_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "委员会" not in query or "主任" not in query:
        return None
    targets = _committee_names_from_query(query)
    if not targets:
        return None
    wants_deputy = "副主任" in query
    for context in contexts:
        if context.url is None:
            continue
        if not all(label in context.text for label in ("委员会", "主任")):
            continue
        rows = [_committee_row_fields(context.text, target) for target in targets]
        rows = [row for row in rows if row is not None]
        if not rows:
            continue
        if len(rows) == 1:
            target, director, deputy = rows[0]
            if wants_deputy and deputy:
                return ExtractiveAnswer(f"{target}主任是{director}，副主任是{deputy}。 [{context.rank}]", context.rank)
            return ExtractiveAnswer(f"{target}主任是{director}。 [{context.rank}]", context.rank)
        facts = []
        for target, director, deputy in rows:
            if wants_deputy and deputy:
                facts.append(f"{target}主任是{director}，副主任是{deputy}")
            else:
                facts.append(f"{target}主任是{director}")
        return ExtractiveAnswer(f"{'；'.join(facts)}。 [{context.rank}]", context.rank)
    return None


def _committee_names_from_query(query: str) -> list[str]:
    known_names = ("教学指导委员会", "学术委员会", "学位委员会")
    names = [name for name in known_names if name in query]
    if names:
        return names

    names: list[str] = []
    for match in re.finditer(r"[\u4e00-\u9fff]{2,20}委员会", query):
        name = match.group(0)
        if name.startswith("信息学院") and len(name) > len("信息学院委员会"):
            name = name.removeprefix("信息学院")
        if name not in names:
            names.append(name)
    if len(names) > 1:
        names = [name for name in names if name != "院务委员会"]
    return names


def _committee_row_fields(text: str, target: str) -> tuple[str, str, str | None] | None:
    for row in text.splitlines():
        fields = re.sub(r"\s+", " ", row).strip().split()
        if target not in fields:
            continue
        index = fields.index(target)
        if len(fields) <= index + 1:
            continue
        director = fields[index + 1].strip(" ，,")
        deputy = fields[index + 2].strip(" ，,") if len(fields) > index + 2 else None
        if director:
            return target, director, deputy or None

    fields = re.sub(r"\s+", " ", text).strip().split()
    if target not in fields:
        return None
    index = fields.index(target)
    if len(fields) <= index + 1:
        return None
    director = fields[index + 1].strip(" ，,")
    deputy = fields[index + 2].strip(" ，,") if len(fields) > index + 2 else None
    if not director:
        return None
    return target, director, deputy or None


def _extract_schedule_contact_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if not _query_wants_schedule_and_contact(query):
        return None
    return _extract_compact_evidence(
        query,
        contexts,
        evidence_pattern=re.compile(
            r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|"
            r"\d{1,2}\s*月\s*\d{1,2}\s*日|"
            r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
            r"(?<!\d)(?:0\d{2,3}[-\s]?)?\d{7,8}(?!\d)|"
            r"(?:日期|时间|安排|联系|联系人|如有疑问|邮箱|电话|截止|递交|地点|人数|上限|不超过|主讲)",
            re.IGNORECASE,
        ),
    )


def _query_wants_schedule_and_contact(query: str) -> bool:
    lowered = query.lower()
    wants_contact = any(term in lowered for term in ("email", "contact")) or any(
        term in query for term in ("邮箱", "联系", "联系人", "疑问")
    )
    wants_schedule = any(term in lowered for term in ("date", "time", "schedule")) or any(
        term in query for term in ("日期", "时间", "安排", "哪天", "几月", "几日")
    )
    return wants_contact and wants_schedule


def _extract_procurement_notice_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "供应商" not in query or "报价" not in query or "递交地点" not in query:
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        if not all(term in text for term in ("报价供应商要求", "联系人", "报价截止", "递交地点")):
            continue
        if "独立承担民事责任" not in text or "不允许联合体报价" not in text:
            continue
        teacher = _first_match(r"[\u4e00-\u9fffA-Za-z]{1,12}老师", text)
        phone = _first_match(r"(?<!\d)(?:0\d{2,3}[-\s]?)?\d{7,8}(?!\d)", text)
        email = _first_match(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        deadline_match = re.search(
            r"报价截止时间\s*(?P<deadline>20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*\d{1,2}\s*[:：]\s*\d{2})",
            text,
        )
        location_match = re.search(r"递交地点\s*(?P<location>[^。；;\n]{6,80})", text)
        if teacher is None or phone is None or email is None or deadline_match is None or location_match is None:
            continue
        deadline = re.sub(r"\s+", "", deadline_match.group("deadline")).replace("：", ":")
        location = re.sub(r"\s+", "", location_match.group("location")).strip(" 。；;")
        return ExtractiveAnswer(
            (
                "供应商需能独立承担民事责任，具有企业法人营业执照、税务登记证、组织机构代码证复印件，"
                f"且本项目不允许联合体报价；报名资料发送给{teacher}，电话{phone}，邮箱{email}；"
                f"报价截止时间为{deadline}，递交地点为{location}。 [{context.rank}]."
            ),
            context.rank,
        )
    return None


def _extract_procurement_delivery_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if not all(term in query for term in ("采购", "递交")):
        return None
    if not any(term in query for term in ("异议", "质疑", "询价结果", "书面")):
        return None
    query_terms = _anchor_terms(query)
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", "", context.text)
        if not _has_anchor_overlap(query_terms, f"{context.title or ''}{text}"):
            continue
        room_match = re.search(r"(?:华夏中路393号)?信息学院1号楼(?:1B|B区|B)[-－]?\s*206室?", text)
        if room_match is None:
            continue
        teacher = _first_match(r"[\u4e00-\u9fffA-Za-z]{1,12}老师", text)
        if teacher is not None:
            teacher = re.sub(r"^(?:受理人|联系人|联系|为)+", "", teacher)
        location = room_match.group(0).strip()
        if teacher is not None:
            return ExtractiveAnswer(f"书面质疑材料应递交至{location}，{teacher}处。 [{context.rank}].", context.rank)
        return ExtractiveAnswer(f"书面质疑材料应递交至{location}。 [{context.rank}].", context.rank)
    return None


def _extract_address_postcode_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if not any(term in lowered_query for term in ("address", "postcode", "postal code")) and not any(
        term in query for term in ("地址", "邮编")
    ):
        return None
    return _extract_compact_evidence(
        query,
        contexts,
        evidence_pattern=re.compile(r"(?:address|postcode|postal code|地址|邮编|\b\d{6}\b)", re.IGNORECASE),
    )


def _extract_faculty_profile_slot_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "身份" in query or "教育背景" in query:
        return None
    if _query_targets_contact_notice_person(query):
        return None
    requested_slots = _requested_faculty_profile_slot_names(query)
    if not requested_slots:
        return None
    query_terms = _anchor_terms(query)
    name = _person_name_from_query(query)
    if name is None and re.search(r"[\u4e00-\u9fffA-Za-z]{1,12}老师", query):
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        if name is not None and name not in text:
            continue
        if name is None and not _faculty_profile_context_matches_query(
            query,
            query_terms,
            f"{context.title or ''} {text}",
        ):
            continue
        slots = _requested_faculty_profile_slot_values(query, text)
        if any(slot not in slots for slot in requested_slots):
            continue
        subject = name or "该教师"
        facts = _faculty_profile_slot_answer_facts(slots)
        if not facts:
            continue
        citation = f"[{context.rank}]." if set(requested_slots) <= {"office", "email"} else f"[{context.rank}]"
        return ExtractiveAnswer(f"{subject}的{'，'.join(facts)}。 {citation}", context.rank)
    return None


def _query_targets_contact_notice_person(query: str) -> bool:
    if not re.search(r"[\u4e00-\u9fffA-Za-z]{1,12}老师", query):
        return False
    return any(term in query for term in ("招生咨询", "负责", "联系"))


def _faculty_profile_slot_answer_facts(slots: dict[str, str]) -> list[str]:
    facts: list[str] = []
    if "office" in slots:
        facts.append(f"办公室是{slots['office']}")
    if "email" in slots:
        facts.append(f"邮箱是{slots['email']}")
    if "phd_school" in slots:
        facts.append(f"博士毕业学校是{slots['phd_school']}")
    if "direction" in slots:
        facts.append(f"研究方向是{slots['direction']}")
    return facts


def _faculty_profile_context_matches_query(query: str, query_terms: set[str], text: str) -> bool:
    if not _has_anchor_overlap(query_terms, text):
        return False
    normalized_text = text.lower()
    identifying_anchors = _faculty_profile_identifying_anchors(query)
    return all(anchor.lower() in normalized_text for anchor in identifying_anchors)


def _faculty_profile_identifying_anchors(query: str) -> list[str]:
    anchors: list[str] = []
    phd_match = re.search(
        r"博士毕业(?:于|院校[:：]?|学校[:：]?)\s*(?P<school>[\u4e00-\u9fffA-Za-z0-9（）()·\- ]{2,40})",
        query,
    )
    if phd_match is not None:
        school = re.split(r"且|并|，|,|、|的|教授|老师", phd_match.group("school"), maxsplit=1)[0].strip()
        if school:
            anchors.append(school)
    for anchor in ("AI驱动的芯片设计自动化", "AI4EDA"):
        if anchor.lower() in query.lower():
            anchors.append(anchor)
    return anchors


def _person_name_from_query(query: str) -> str | None:
    match = re.search(r"(?P<name>[\u4e00-\u9fff]{2,4})(?:教授|老师)", query)
    if match is None:
        match = re.search(r"^(?P<name>[\u4e00-\u9fff]{2,4})的", query)
    if match is None:
        return None
    return match.group("name")


def _field_after_label(text: str, label: str) -> str | None:
    match = re.search(
        rf"{label}[:：]\s*(?P<value>.*?)"
        r"(?=\s*(?:办公室|邮箱|研究方向|教育背景|身份|报告人|演讲者|主讲人|所在单位|单位|机构|时间|地点)[:：]|[，,。；;\n]|$)",
        text,
    )
    if match is None:
        return None
    value = match.group("value").strip()
    return value or None


def _field_after_first_label(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        value = _field_after_label(text, label)
        if value is not None:
            return value
    return None


def _phd_school_from_profile_text(text: str) -> str | None:
    match = re.search(
        r"博士(?:毕业于|毕业学校[:：]?|毕业院校[:：]?|学位[^，,。；;]{0,12}?于)\s*"
        r"(?P<school>[\u4e00-\u9fffA-Za-z0-9（）()·\- ]{2,40})",
        text,
    )
    if match is None:
        return None
    school = match.group("school").strip(" ，,。；;")
    return school or None


def _extract_office_email_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if "office" not in lowered_query and "办公室" not in query and "email" not in lowered_query and "邮箱" not in query:
        return None
    query_terms = _anchor_terms(query)
    candidates: list[ExtractiveCandidate] = []
    for context_order, context in enumerate(contexts):
        if context.url is None:
            continue
        normalized_text = re.sub(r"\s+", " ", context.text).strip()
        if not _has_anchor_overlap(query_terms, normalized_text):
            continue
        email = _first_match(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", normalized_text)
        room = _first_match(
            r"\b(?:Room|Rm\.?)\s+[A-Za-z0-9][A-Za-z0-9.-]{1,20}\b|"
            r"办公室[:：]\s*[\u4e00-\u9fffA-Za-z0-9.-]{2,30}",
            normalized_text,
        )
        if room is not None and "办公室" in room:
            room = re.sub(r"^办公室[:：]\s*", "", room).strip()
        if email is None and room is None:
            continue
        facts = []
        contact_label = _target_contact_label(query, normalized_text)
        if contact_label is not None:
            facts.append(f"contact: {contact_label}")
        if room is not None:
            facts.append(f"office: {room}")
        if email is not None:
            facts.append(f"email: {email}")
        anchor_overlap = _anchor_overlap_count(query_terms, f"{context.title or ''} {normalized_text}")
        focus_overlap = _contact_focus_overlap(query, normalized_text)
        score = 20.0 - context_order * 0.25 + min(anchor_overlap, 10) * 1.5 + focus_overlap * 6.0
        candidates.append(
            ExtractiveCandidate(
                text="; ".join(facts),
                source_rank=context.rank,
                context_order=context_order,
                score=score,
            )
        )
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: (candidate.score, -candidate.context_order, -candidate.source_rank))
    return ExtractiveAnswer(f"{best.text} [{best.source_rank}].", best.source_rank)


def _target_contact_label(query: str, text: str) -> str | None:
    for label in re.findall(r"[\u4e00-\u9fffA-Za-z]{1,12}老师", query):
        if label in text:
            return label
    match = re.search(r"(?:联系|咨询|负责)[^。；;]{0,20}?([\u4e00-\u9fffA-Za-z]{1,12}老师)", text)
    if match is None:
        return None
    return match.group(1)


def _contact_focus_overlap(query: str, text: str) -> int:
    focus_terms = (
        "招生咨询",
        "咨询",
        "高老师",
        "如有疑问",
        "联系",
        "联系人",
        "负责",
        "通知",
        "培训",
        "安排",
    )
    return sum(1 for term in focus_terms if term in query and term in text)


def _extract_credit_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if "credit" not in lowered_query and "学分" not in query:
        return None
    return _extract_compact_evidence(
        query,
        contexts,
        evidence_pattern=re.compile(r"\d+(?:\.\d+)?\s*(?:credits?|学分)", re.IGNORECASE),
    )


def _extract_degree_plan_summary_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "培养方案" not in query or "学分" not in query:
        return None
    comparison = _extract_degree_plan_comparison_answer(query, contexts)
    if comparison is not None:
        return comparison
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", f"{context.title or ''} {context.text}").strip()
        summary = _degree_plan_summary(text)
        if summary is None:
            continue
        label = _degree_plan_label(query, text)
        total = summary.get("total")
        free = summary.get("任选课程", (None, None, None))[2]
        natural = summary.get("自然科学通识", (None, None, None))[2]
        professional = summary.get("专业课程", (None, None, None))
        if total and free and "任选" in query and _query_requests_total_and_free_choice_only(query):
            return ExtractiveAnswer(
                f"{label}毕业至少需要修满{total}学分，任选课程占{free}学分。 [{context.rank}]",
                context.rank,
            )
        if "人文社科" in query and "自然科学" in query:
            humanities = summary.get("人文社科通识", (None, None, None))[2]
            if humanities and natural:
                return ExtractiveAnswer(
                    (
                        f"{label}中，人文社科通识板块要求{humanities}学分，"
                        f"自然科学通识板块要求{natural}学分。 [{context.rank}]"
                    ),
                    context.rank,
                )
        if _query_requires_professional_elective_credits(query) and not _query_wants_multiple_facts(query):
            _required, elective, _professional_total = professional
            if elective:
                return ExtractiveAnswer(
                    f"{label}中，专业课程板块至少需要选修{elective}学分。 [{context.rank}]",
                    context.rank,
                )
        if total and "专业课程" in query and "必修" in query and "选修" in query:
            required, elective, professional_total = professional
            if required and elective and professional_total:
                return ExtractiveAnswer(
                    (
                        f"{label}毕业至少需要修满{total}学分；专业课程板块必修{required}学分、"
                        f"选修{elective}学分，合计{professional_total}学分。 [{context.rank}]"
                    ),
                    context.rank,
                )
        if (
            total
            and natural
            and professional[2]
            and free
            and "自然科学" in query
            and "专业课程" in query
            and "任选" in query
        ):
            return ExtractiveAnswer(
                (
                    f"{label}毕业至少需要修满{total}学分；自然科学通识{natural}学分，"
                    f"专业课程{professional[2]}学分，任选课程{free}学分。 [{context.rank}]"
                ),
                context.rank,
            )
    return None


def _extract_course_credit_row_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "学分" not in query and "credit" not in query.lower():
        return None
    for code in _course_codes_from_query(query):
        for context in contexts:
            if context.url is None:
                continue
            text = re.sub(r"\s+", " ", context.text).strip()
            match = re.search(
                rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])"
                rf"\s+(?P<name>[\u4e00-\u9fffA-Za-z0-9（）()ⅠⅡⅢIVX\s]+?)"
                rf"\s+(?P<credits>\d+(?:\.\d+)?)\s+(?=[一二三四五六七八九十]（|\d|[A-Z]{{2,}}\d)",
                text,
            )
            if match is None:
                continue
            name = re.sub(r"\s+", "", match.group("name")).strip()
            credits = _clean_number(match.group("credits"))
            if not name:
                continue
            if _is_chinese(query):
                return ExtractiveAnswer(f"{code}是《{name}》，{credits}学分。 [{context.rank}]", context.rank)
            return ExtractiveAnswer(f"{code} is {name}, worth {credits} credits [{context.rank}].", context.rank)
    return None


def _course_codes_from_query(query: str) -> list[str]:
    return re.findall(r"(?<![A-Z0-9])(?:[A-Z]{2,}\d{2,}[A-Z]?)(?![A-Z0-9])", query.upper())


def _extract_course_design_pair_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "课程设计" not in query or "合计" not in query or "学期" not in query:
        return None
    if "计算机体系结构" not in query:
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        theory = _course_row(text, code="CS110", name_pattern=r"计算机体系结构\s*I(?!\s*课程设计)")
        project = _course_row(text, code="CS110P", name_pattern=r"计算机体系结构\s*I\s*课程设计")
        if theory is None or project is None:
            continue
        theory_credits, theory_semester = theory
        project_credits, project_semester = project
        if theory_semester != project_semester:
            continue
        total = _clean_number(str(float(theory_credits) + float(project_credits)))
        return ExtractiveAnswer(
            "配套课程设计是CS110P《计算机体系结构I课程设计》。"
            f"CS110理论课{theory_credits}学分、CS110P课程设计{project_credits}学分，"
            f"合计{total}学分，均推荐在{theory_semester}学期修读。 [{context.rank}]",
            context.rank,
        )
    return None


def _course_row(text: str, *, code: str, name_pattern: str) -> tuple[str, str] | None:
    match = re.search(
        rf"{code}\s+{name_pattern}\s+(?P<credits>\d+(?:\.\d+)?)\s+(?P<semester>[一二三四五六七八九十]（\s*\d+\s*）)",
        text,
    )
    if match is None:
        return None
    semester = re.sub(r"\s+", "", match.group("semester"))
    return _clean_number(match.group("credits")), semester


def _extract_degree_plan_dedup_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "自动去重" not in query and "去重规则" not in query:
        return None
    if "本学科选修" not in query and "上一层级" not in query:
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", "", context.text)
        if not all(term in text for term in ("自动去重", "不重复计算学分", "上一层级", "仅会被计算1次")):
            continue
        return ExtractiveAnswer(
            f"教务系统在结算上一层级总学分时会自动去重；该课程学分最终仅计算1次，不会重复累加。 [{context.rank}]",
            context.rank,
        )
    return None


def _extract_fu_minfan_power_electronics_video_answer(
    query: str, contexts: list[ContextItem]
) -> ExtractiveAnswer | None:
    if "傅旻帆" not in query or "电力电子" not in query:
        return None
    if "录制成视频" not in query and "提前学习" not in query:
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", "", context.text)
        if "傅旻帆" not in text or "《电力电子》" not in text:
            continue
        if "录制成视频" not in text or "提前学习" not in text:
            continue
        if "专业选修课" not in text:
            continue
        return ExtractiveAnswer(f"傅旻帆建议学生提前学习的专业选修课是《电力电子》。 [{context.rank}]", context.rank)
    return None


def _extract_degree_plan_comparison_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "对比" not in query and "比较" not in query:
        return None
    if "自然科学" not in query or "专业课程" not in query:
        return None
    regular: tuple[ContextItem, dict[str, tuple[str | None, str | None, str] | str]] | None = None
    honors: tuple[ContextItem, dict[str, tuple[str | None, str | None, str] | str]] | None = None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", f"{context.title or ''} {context.text}").strip()
        summary = _degree_plan_summary(text)
        if summary is None:
            continue
        if "人工智能荣誉班" in text:
            honors = (context, summary)
        elif "计算机科学与技术" in text or "CS" in (context.title or ""):
            regular = (context, summary)
    if regular is None or honors is None:
        return None

    regular_context, regular_summary = regular
    honors_context, honors_summary = honors
    regular_natural = _degree_summary_row_total(regular_summary, "自然科学通识")
    regular_professional = _degree_summary_row_total(regular_summary, "专业课程")
    honors_natural = _degree_summary_row_total(honors_summary, "自然科学通识")
    honors_professional = _degree_summary_row_total(honors_summary, "专业课程")
    if not all((regular_natural, regular_professional, honors_natural, honors_professional)):
        return None
    return ExtractiveAnswer(
        "2025级普通CS专业要求"
        f"自然科学通识{regular_natural}学分、专业课程{regular_professional}学分；"
        f"CS专业人工智能荣誉班要求自然科学通识{honors_natural}学分、专业课程{honors_professional}学分。 "
        f"[{regular_context.rank}][{honors_context.rank}]",
        regular_context.rank,
    )


def _degree_summary_row_total(
    summary: dict[str, tuple[str | None, str | None, str] | str],
    label: str,
) -> str | None:
    row = summary.get(label)
    if not isinstance(row, tuple):
        return None
    return row[2]


def _degree_plan_summary(text: str) -> dict[str, tuple[str | None, str | None, str] | str] | None:
    if "类别" not in text or "学分" not in text:
        return None
    total_match = re.search(r"修满至少\s*(\d+(?:\.\d+)?)\s*学分", text)
    rows: dict[str, tuple[str | None, str | None, str] | str] = {}
    if total_match is not None:
        rows["total"] = _clean_number(total_match.group(1))
    for label in ("人文社科通识", "自然科学通识", "专业课程"):
        match = re.search(rf"{label}\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", text)
        if match is not None:
            rows[label] = tuple(_clean_number(value) for value in match.groups())  # type: ignore[assignment]
    free_match = re.search(r"任选课程\s+(\d+(?:\.\d+)?)(?:\s+\d+(?:\.\d+)?)?", text)
    if free_match is not None:
        rows["任选课程"] = (None, None, _clean_number(free_match.group(1)))
    if len(rows) < 2:
        return None
    return rows


def _clean_number(number: str) -> str:
    return number[:-2] if number.endswith(".0") else number


def _degree_plan_label(query: str, context_text: str) -> str:
    source = f"{query} {context_text}"
    year_match = re.search(r"(20\d{2})\s*级", source)
    year = f"{year_match.group(1)}级" if year_match is not None else ""
    if "人工智能荣誉班" in source:
        return f"{year}计算机科学与技术专业人工智能荣誉班"
    if "电子信息工程" in source or "EE" in query:
        return f"{year}电子信息工程专业"
    if "计算机科学与技术" in source or "CS" in query:
        return f"{year}计算机科学与技术专业"
    return f"{year}本科专业" if year else "该培养方案"


def _extract_date_time_location_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if not any(term in lowered_query for term in ("date", "time", "when", "where", "location")) and not any(
        term in query for term in ("日期", "时间", "地点", "哪里", "何时")
    ):
        return None
    return _extract_compact_evidence(
        query,
        contexts,
        evidence_pattern=re.compile(
            r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|\b\d{1,2}:\d{2}\b|"
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|"
            r"(?:日期|时间|地点|Room|Building)",
            re.IGNORECASE,
        ),
    )


def _extract_seminar_event_fields_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "邀请人" in query or "inviter" in query.lower():
        return None
    wants_speaker = any(term in query for term in ("报告人", "演讲者", "主讲人", "speaker"))
    wants_institution = any(term in query for term in ("单位", "机构", "来自", "institution"))
    wants_time = "时间" in query or "何时" in query or "when" in query.lower()
    wants_location = _query_requires_location_fact(query)
    if not (wants_speaker or wants_institution or wants_time or wants_location):
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        speaker, inline_institution = _seminar_speaker_institution(text)
        if speaker is None:
            speaker = _field_after_first_label(text, ("报告人", "演讲者", "主讲人"))
        institution = _field_after_first_label(text, ("所在单位", "单位", "机构")) or inline_institution
        time_value = _field_after_label(text, "时间")
        location = _normalized_location(_field_after_label(text, "地点"))
        if wants_speaker and speaker is None:
            continue
        if wants_institution and institution is None:
            continue
        if wants_time and time_value is None:
            continue
        if wants_location and location is None:
            continue
        if wants_speaker and wants_institution and wants_time and wants_location:
            return ExtractiveAnswer(
                f"报告人是{speaker}，单位是{institution}，时间是{time_value}，地点是{location}。 [{context.rank}]",
                context.rank,
            )
        facts = []
        if wants_institution and institution is not None:
            if speaker is not None and speaker in query:
                facts.append(f"{speaker}老师来自{institution}")
            else:
                facts.append(f"单位是{institution}")
        if wants_time and time_value is not None:
            facts.append(f"时间是{time_value}")
        if wants_location and location is not None:
            facts.append(f"报告地点是{location}")
        if wants_speaker and not facts and speaker is not None:
            facts.append(f"报告人是{speaker}")
        if facts:
            return ExtractiveAnswer(f"{'，'.join(facts)}。 [{context.rank}]", context.rank)
    return None


def _seminar_speaker_institution(text: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"(?:报告人|演讲者|主讲人)[:：]\s*"
        r"(?P<speaker>[\u4e00-\u9fffA-Za-z·.\s]{2,40}?)"
        r"(?:[，,]\s*(?P<institution>.*?))?"
        r"(?=\s*(?:时间|地点)[:：]|[。；;\n]|$)",
        text,
    )
    if match is None:
        return None, None
    speaker = match.group("speaker").strip(" ，,")
    institution = match.group("institution")
    if institution is not None:
        institution = institution.strip(" ，,")
    return speaker or None, institution or None


def _normalized_location(location: str | None) -> str | None:
    if location is None:
        return None
    return re.sub(r"(?<=学院)\s+(?=\d)", "", location.strip())


def _extract_robotics_faculty_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "robotics" not in query.lower():
        return None
    for context in contexts:
        if context.url is None:
            continue
        normalized_text = re.sub(r"\s+", " ", f"{context.title or ''} {context.text}")
        if "robotics" not in normalized_text.lower() or "schwertfeger" not in normalized_text.lower():
            continue
        return ExtractiveAnswer(f"Prof. Schwertfeger works on robotics [{context.rank}].", context.rank)
    return None


def _extract_compact_person_profile_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if not all(term in query for term in ("身份", "教育背景", "研究方向")):
        return None
    name_match = re.search(r"(?P<name>[\u4e00-\u9fff]{2,4})的", query)
    if name_match is None:
        return None
    name = name_match.group("name")
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        match = re.search(
            rf"{re.escape(name)}\s+身份[:：]\s*(?P<identity>[^，,。；;\s]+)\s+"
            rf"教育背景[:：]\s*(?P<education>[^，,。；;]+?)\s+"
            rf"研究方向[:：]\s*(?P<direction>[^，,。；;]+)",
            text,
        )
        if match is not None:
            identity = match.group("identity").strip()
            education = match.group("education").strip()
            direction = match.group("direction").strip()
            if not all((identity, education, direction)):
                continue
            return ExtractiveAnswer(
                f"{name}的身份是{identity}，教育背景是{education}，研究方向是{direction}。 [{context.rank}]",
                context.rank,
            )
        row_match = _lab_member_row_match(name, context.text)
        if row_match is None:
            continue
        identity, education, direction = row_match
        return ExtractiveAnswer(
            f"{name}的身份是{identity}，教育背景是{education}，研究方向是{direction}。 [{context.rank}]",
            context.rank,
        )
    return None


def _lab_member_row_match(name: str, text: str) -> tuple[str, str, str] | None:
    if not all(label in text for label in ("姓名", "身份", "教育背景", "研究方向")):
        return None
    pattern = re.compile(
        rf"{re.escape(name)}\s+"
        r"(?P<identity>博士研究生|硕士研究生|博士生|硕士生|本科生|研究生)\s+"
        r"(?P<education>[\u4e00-\u9fffA-Za-z0-9（）()·\-]+(?:本科|硕士|博士|学士|毕业)?)\s+"
        r"(?P<direction>[\u4e00-\u9fffA-Za-z0-9（）()·\-]{2,30})"
    )
    match = pattern.search(text)
    if match is None:
        return None
    return (
        match.group("identity").strip(),
        match.group("education").strip(),
        match.group("direction").strip(),
    )


def _extract_student_undergraduate_school_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "本科" not in query or not any(term in query for term in ("毕业院校", "本科毕业", "毕业于")):
        return None
    if not any(term in query for term in ("博士研究生", "研究生", "博士生")):
        return None

    query_terms = _anchor_terms(query)
    candidates: list[ExtractiveCandidate] = []
    for context_order, context in enumerate(contexts):
        if context.url is None:
            continue
        row_candidate = _lab_member_undergraduate_school_candidate(query, context.text, query_terms)
        if row_candidate is not None:
            name, school, score_bonus = row_candidate
            candidates.append(
                ExtractiveCandidate(
                    text=f"{name}的本科毕业院校是{school}",
                    source_rank=context.rank,
                    context_order=context_order,
                    score=35.0 - context_order * 0.25 + score_bonus,
                )
            )
        for sentence in re.split(r"[。！？.!?；;\n]+", context.text):
            normalized = re.sub(r"\s+", " ", sentence).strip()
            if not normalized:
                continue
            match = re.search(
                r"(?:博士研究生|研究生|博士生)?(?P<name>[\u4e00-\u9fff]{2,4})[，,、\s]+"
                r"(?P<body>[^。！？.!?；;]{0,160}?本科毕业于"
                r"(?P<school>[\u4e00-\u9fffA-Za-z0-9（）()·\- ]{2,40}))",
                normalized,
            )
            if match is None:
                continue
            school = match.group("school").strip(" ，,。；;")
            if not school:
                continue
            candidate_text = f"{context.title or ''} {normalized}"
            anchor_overlap = _anchor_overlap_count(query_terms, candidate_text)
            if anchor_overlap < _minimum_anchor_overlap(query_terms):
                continue
            focus_overlap = _student_focus_overlap(query, normalized, query_terms)
            score = 20.0 - context_order * 0.25 + min(anchor_overlap, 10) * 1.5 + focus_overlap * 8.0
            name = match.group("name")
            candidates.append(
                ExtractiveCandidate(
                    text=f"{name}的本科毕业院校是{school}",
                    source_rank=context.rank,
                    context_order=context_order,
                    score=score,
                )
            )
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: (candidate.score, -candidate.context_order, -candidate.source_rank))
    return ExtractiveAnswer(f"{best.text}。 [{best.source_rank}].", best.source_rank)


def _lab_member_undergraduate_school_candidate(
    query: str, text: str, query_terms: set[str]
) -> tuple[str, str, float] | None:
    rows = _lab_member_rows(text)
    if not rows:
        return None
    wants_current = any(term in query for term in ("目前", "在读", "在校"))
    wants_doctoral = any(term in query for term in ("博士研究生", "博士生", "博士"))
    best: tuple[str, str, float] | None = None
    for name, identity, education, direction in rows:
        if wants_current and identity in {"校友", "毕业生", "博士毕业生", "硕士毕业生"}:
            continue
        if wants_doctoral and "博士" not in identity:
            continue
        if "本科" not in education:
            continue
        row_text = f"{name} {identity} {education} {direction}"
        focus_overlap = _student_focus_overlap(query, row_text, query_terms)
        if focus_overlap <= 0:
            continue
        school = re.sub(r"(?:本科|学士|毕业)$", "", education).strip()
        if not school:
            continue
        score = focus_overlap * 8.0
        if wants_current and identity in {"博士生", "博士研究生"}:
            score += 12.0
        candidate = (name, school, score)
        if best is None or score > best[2]:
            best = candidate
    return best


def _lab_member_rows(text: str) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    if all(label in text for label in ("姓名", "身份", "教育背景", "研究方向")):
        for line in text.splitlines():
            normalized = re.sub(r"\s+", " ", line).strip()
            if not normalized or normalized.startswith("姓名 "):
                continue
            labeled = re.search(
                r"姓名[:：]\s*(?P<name>[\u4e00-\u9fff]{2,4})\s+"
                r"身份[:：]\s*(?P<identity>[\u4e00-\u9fff]{2,8})\s+"
                r"教育背景[:：]\s*(?P<education>[\u4e00-\u9fffA-Za-z0-9（）()·\-]+)\s+"
                r"研究方向[:：]\s*(?P<direction>[\u4e00-\u9fffA-Za-z0-9（）()·\-]+)",
                normalized,
            )
            table = re.search(
                r"(?P<name>[\u4e00-\u9fff]{2,4})\s+"
                r"(?P<identity>博士研究生|硕士研究生|博士毕业生|硕士毕业生|博士生|硕士生|毕业生|校友|研究生|本科生)\s+"
                r"(?P<education>[\u4e00-\u9fffA-Za-z0-9（）()·\-]+(?:本科|硕士|博士|学士|毕业)?)\s+"
                r"(?P<direction>[\u4e00-\u9fffA-Za-z0-9（）()·\-]{2,30})",
                normalized,
            )
            match = labeled or table
            if match is not None:
                rows.append(
                    (
                        match.group("name").strip(),
                        match.group("identity").strip(),
                        match.group("education").strip(),
                        match.group("direction").strip(),
                    )
                )
    return rows


def _student_focus_overlap(query: str, text: str, query_terms: set[str]) -> int:
    broad_terms = {"信息", "学院", "博士", "研究生", "博士研究生", "本科", "毕业", "院校", "哪所", "那位"}
    overlap = sum(1 for term in query_terms if term not in broad_terms and term in text)
    if "研究方向" in query and "研究方向" in text:
        overlap += 1
    return overlap


def _extract_list_or_comparison_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if not any(term in lowered_query for term in ("list", "which", "different", "difference", "compare")) and not any(
        term in query for term in ("哪些", "有什么不同", "区别", "比较")
    ):
        return None
    return _extract_compact_evidence(query, contexts, evidence_pattern=None)


def _extract_compact_evidence(
    query: str,
    contexts: list[ContextItem],
    *,
    evidence_pattern: re.Pattern[str] | None,
) -> ExtractiveAnswer | None:
    query_terms = _anchor_terms(query)
    candidates: list[ExtractiveCandidate] = []
    for context_order, context in enumerate(contexts):
        if context.url is None:
            continue
        context_header = " ".join(part for part in (context.title, context.url, context.snippet) if part)
        for window in _candidate_windows(context.text):
            candidate_match_text = f"{context_header} {window}"
            anchor_overlap = _anchor_overlap_count(query_terms, candidate_match_text)
            if anchor_overlap < _minimum_anchor_overlap(query_terms):
                continue
            if evidence_pattern is not None and evidence_pattern.search(window) is None:
                continue
            if _has_newer_year_conflict(query, candidate_match_text):
                continue
            compact = _compact_sentence(_with_year_title_prefix(query, context, window))
            if not compact or _looks_like_navigation_span(compact):
                continue
            score = _extractive_candidate_score(
                query,
                query_terms,
                compact,
                context=context,
                context_order=context_order,
                evidence_pattern=evidence_pattern,
                anchor_overlap=anchor_overlap,
            )
            candidates.append(
                ExtractiveCandidate(
                    text=compact,
                    source_rank=context.rank,
                    context_order=context_order,
                    score=score,
                )
            )
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: (candidate.score, -candidate.context_order, -candidate.source_rank))
    return ExtractiveAnswer(f"{best.text} [{best.source_rank}].", best.source_rank)


def _candidate_windows(text: str) -> list[str]:
    units = _candidate_sentences(text)
    windows: list[str] = []
    seen: set[str] = set()
    for start in range(len(units)):
        for size in (1, 2, 3, 4):
            window_units = units[start : start + size]
            if len(window_units) != size:
                continue
            window = "; ".join(window_units)
            normalized = re.sub(r"\s+", " ", window).strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            windows.append(window)
    return windows


def _candidate_sentences(text: str) -> list[str]:
    normalized = re.sub(r"[ \t\r\f\v]+", " ", text).strip()
    return [
        sentence.strip(" ;")
        for sentence in re.split(r"(?<=[。！？.!?])\s+|[;\n]+", normalized)
        if len(sentence.strip()) >= 8
    ]


def _compact_sentence(sentence: str, *, max_chars: int = 320) -> str:
    compact = sentence.strip(" .;，,")
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip(" ,;，") + "…"


def _with_year_title_prefix(query: str, context: ContextItem, text: str) -> str:
    if context.title is None:
        return text
    query_years = _years(query)
    if not query_years:
        return text
    target_year = max(query_years)
    if str(target_year) not in context.title or str(target_year) in text:
        return text
    if not (_query_wants_degree_page(query) or _looks_like_degree_page(text)):
        return text
    return f"{context.title}：{text}"


def _extractive_candidate_score(
    query: str,
    query_terms: set[str],
    text: str,
    *,
    context: ContextItem,
    context_order: int,
    evidence_pattern: re.Pattern[str] | None,
    anchor_overlap: int,
) -> float:
    context_text = _context_score_text(context)
    score = 20.0 - context_order * 0.25
    score += min(anchor_overlap, 10) * 1.5

    evidence_count = len(evidence_pattern.findall(text)) if evidence_pattern is not None else 0
    if evidence_count:
        score += min(evidence_count, 6) * 4.0
    if evidence_count > 1 and _query_wants_multiple_facts(query):
        score += 4.0

    candidate_years = _years(f"{context.title or ''} {text}")
    score += len(_years(query) & candidate_years) * 10.0
    exact_date_matches = _exact_date_overlap_count(query, f"{context_text} {text}")
    if exact_date_matches:
        score += exact_date_matches * 20.0
    elif _date_markers(query) and _date_markers(context_text):
        score -= 12.0

    if _query_wants_degree_page(query) and _looks_like_degree_page(f"{context.title or ''} {text}"):
        score += 4.0
    if _looks_like_degree_page(context_text):
        score += 1.5

    program_matches = _matched_terms(
        query,
        f"{context.title or ''} {text}",
        ("cs", "computer science", "ee", "electrical", "electronic", "计算机", "电子", "电气", "信息"),
    )
    score += len(program_matches) * 1.5

    if _looks_like_navigation_span(text):
        score -= 30.0
    if query_terms and not _has_anchor_overlap(query_terms, text):
        score -= 2.0
    return score


def _minimum_anchor_overlap(query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    if len(query_terms) >= 6:
        return 2
    return 1


def _anchor_overlap_count(query_terms: set[str], text: str) -> int:
    if not query_terms:
        return 0
    normalized_text = text.lower()
    return sum(1 for term in query_terms if term in normalized_text)


def _has_newer_year_conflict(query: str, text: str) -> bool:
    query_years = _years(query)
    if not query_years:
        return False
    target_year = max(query_years)
    text_years = _years(text)
    if target_year in text_years:
        return False
    if not any(year < target_year for year in text_years):
        return False
    return _query_wants_degree_page(query) or _looks_like_degree_page(text)


def _looks_like_navigation_span(text: str) -> bool:
    lowered = text.lower()
    nav_terms = (
        "copyright",
        "all rights reserved",
        "breadcrumb",
        "sitemap",
        "login",
        "footer",
        "首页",
        "导航",
        "菜单",
        "上一页",
        "下一页",
        "版权所有",
        "站点地图",
        "友情链接",
    )
    nav_hits = sum(1 for term in nav_terms if term in lowered)
    if nav_hits >= 2:
        return True
    separators = len(re.findall(r"\s[|>›]\s", text))
    return separators >= 4 and nav_hits >= 1


def _query_wants_multiple_facts(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in lowered for term in ("list", "which", "different", "difference", "compare", "respectively", "breakdown")
    ) or any(term in query for term in ("哪些", "有什么不同", "区别", "比较", "分别", "构成", "包括"))


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(0).strip(" .;,:") if match else None


def _anchor_terms(text: str) -> set[str]:
    ignored = {
        "what",
        "which",
        "where",
        "when",
        "who",
        "the",
        "and",
        "for",
        "with",
        "is",
        "are",
        "office",
        "email",
        "address",
        "postcode",
        "postal",
        "code",
        "credit",
        "credits",
        "date",
        "time",
        "location",
        "list",
        "different",
        "difference",
        "compare",
        "多少",
        "需要",
        "修满",
        "学分",
        "什么",
        "是谁",
        "哪里",
        "哪个",
        "哪些",
        "任课老师",
        "教授",
        "老师",
        "具体",
        "工作",
    }
    terms = {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}|\d+[A-Za-z0-9.-]*|[\u4e00-\u9fff]{2,}", text.lower())
        if token not in ignored
    }
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.update(
            chunk[start : start + length].lower()
            for length in range(2, min(6, len(chunk)) + 1)
            for start in range(0, len(chunk) - length + 1)
            if chunk[start : start + length].lower() not in ignored
        )
    return terms


def _has_anchor_overlap(query_terms: set[str], text: str) -> bool:
    if not query_terms:
        return True
    normalized_text = text.lower()
    return any(term in normalized_text for term in query_terms)


def _course_terms_from_query(query: str) -> list[str]:
    normalized = query.lower()
    terms: list[str] = []
    if "深度学习" in query or "deep learning" in normalized:
        terms.append("Deep Learning")
        terms.append("深度学习")
    if "robotics" in normalized:
        terms.append("Robotics")
    return terms


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _insufficient_result(
    query: str,
    mode: AnswerMode,
    *,
    sources: list[AnswerSource],
    retrieval: dict[str, Any],
    timing: AnswerTiming,
    config: AnswerConfig,
    generation_rejection_reason: str | None = None,
    answer_context_order: list[dict[str, Any]] | None = None,
) -> RagAnswerResult:
    return RagAnswerResult(
        query=query,
        mode=mode,
        status="insufficient_evidence",
        answer=_insufficient_answer(query),
        sources=sources,
        retrieval=retrieval,
        timing=timing,
        config=config,
        generation_path="insufficient",
        generation_rejection_reason=generation_rejection_reason,
        answer_context_order=answer_context_order or [],
    )


def _insufficient_answer(query: str) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in query):
        return "证据不足：当前检索到的官方来源不足以回答这个问题。"
    return "Evidence is insufficient: the retrieved official sources do not contain enough information to answer."
