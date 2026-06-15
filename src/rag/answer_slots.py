from __future__ import annotations

import re
from dataclasses import dataclass

from .answer_types import ExtractiveAnswer
from .retrieve import ContextItem


@dataclass(frozen=True)
class RequiredSlotValue:
    name: str
    label: str
    value: str
    source_rank: int


def required_slot_rejection_reason(query: str, answer: str, contexts: list[ContextItem]) -> str | None:
    slot_values = required_slot_values(query, contexts)
    if not slot_values:
        return None
    normalized_answer = _normalize_answer_text(answer)
    for slot in slot_values:
        if _normalize_answer_text(slot.value) not in normalized_answer:
            return f"missing_required_slot:{slot.name}"
    return None


def extract_required_slot_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    slot_values = required_slot_values(query, contexts)
    if not slot_values:
        return None
    facts = [f"{slot.label}是{slot.value}" for slot in slot_values]
    citations = " ".join(f"[{rank}]" for rank in sorted({slot.source_rank for slot in slot_values}))
    return ExtractiveAnswer(f"{'；'.join(facts)}。 {citations}", slot_values[0].source_rank)


def required_slot_values(query: str, contexts: list[ContextItem]) -> list[RequiredSlotValue]:
    requested_slots = _requested_slot_names(query)
    if not requested_slots:
        return []

    found: dict[str, RequiredSlotValue] = {}
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        for slot_name in requested_slots:
            if slot_name == "procurement_suppliers":
                for slot in _procurement_supplier_values(context, text):
                    found.setdefault(slot.name, slot)
                continue
            if slot_name in found:
                continue
            value = _slot_value(slot_name, text)
            if value is None:
                continue
            found[slot_name] = RequiredSlotValue(
                name=slot_name,
                label=_slot_label(slot_name),
                value=value,
                source_rank=context.rank,
            )

    ordered: list[RequiredSlotValue] = []
    for name in requested_slots:
        if name == "procurement_suppliers":
            ordered.extend(slot for slot in found.values() if slot.name.startswith("procurement_supplier:"))
        elif name in found:
            ordered.append(found[name])
    return ordered


def _requested_slot_names(query: str) -> list[str]:
    lowered = query.lower()
    slots: list[str] = []
    if _query_wants_procurement_suppliers(query):
        slots.append("procurement_suppliers")
        return slots

    lab_slots: list[str] = []
    if "研究方向" in query or "research direction" in lowered:
        lab_slots.append("research_directions")
    if any(term in query for term in ("招生名额", "名额", "招收")) or "quota" in lowered:
        lab_slots.append("quota")
    if any(term in query for term in ("邮箱", "联系邮箱")) or "email" in lowered:
        lab_slots.append("email")
    if any(term in query for term in ("组长", "负责人", "课题组负责人", "PI")) or "principal investigator" in lowered:
        lab_slots.append("group_leader")
    if "quota" in lab_slots or "group_leader" in lab_slots:
        slots.extend(lab_slots)
    return slots


def _slot_value(slot_name: str, text: str) -> str | None:
    if slot_name == "research_directions":
        return _field_after_label(text, "研究方向")
    if slot_name == "quota":
        return _quota_value(text)
    if slot_name == "email":
        return _first_match(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    if slot_name == "group_leader":
        return _field_after_first_label(text, ("组长", "负责人", "课题组负责人", "PI")) or _first_match(
            r"由\s*([\u4e00-\u9fff]{2,4})课题组负责", text
        )
    return None


def _slot_label(slot_name: str) -> str:
    if slot_name.startswith("procurement_supplier:"):
        return f"{slot_name.split(':', maxsplit=1)[1]}供应商"
    return {
        "research_directions": "研究方向",
        "quota": "招生名额",
        "email": "联系邮箱",
        "group_leader": "组长",
    }[slot_name]


def _field_after_first_label(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        value = _field_after_label(text, label)
        if value is not None:
            return value
    return None


def _field_after_label(text: str, label: str) -> str | None:
    match = re.search(
        rf"{label}[:：]\s*(?P<value>.*?)"
        r"(?=\s*(?:研究方向|招生名额|名额|联系邮箱|邮箱|组长|负责人|课题组负责人|PI|时间|地点)[:：]|[。；;\n]|$)",
        text,
    )
    if match is None:
        return None
    value = match.group("value").strip(" ，,。；;")
    return value or None


def _quota_value(text: str) -> str | None:
    match = re.search(
        r"(?:拟招收|招收|招生名额[:：]?)\s*"
        r"(?P<value>\d+\s*[-－—]\s*\d+\s*个?[^。；;，,\n]{0,20}?名额|\d+\s*个?[^。；;，,\n]{0,20}?名额)",
        text,
    )
    if match is None:
        return _field_after_first_label(text, ("招生名额", "名额"))
    return re.sub(r"\s+", "", match.group("value")).strip(" ，,。；;")


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    value = match.group(1) if match.lastindex else match.group(0)
    value = value.strip(" ，,。；;")
    return value or None


def _query_wants_procurement_suppliers(query: str) -> bool:
    return "供应商" in query and "采购" in query and any(term in query for term in ("分别", "两个", "和", "及"))


def _procurement_supplier_values(context: ContextItem, text: str) -> list[RequiredSlotValue]:
    project = _field_after_label(text, "项目名称") or _project_name_from_title(context.title)
    supplier = _field_after_first_label(text, ("成交供应商", "中标供应商", "成交单位", "中选供应商"))
    if project is None or supplier is None:
        return []
    return [
        RequiredSlotValue(
            name=f"procurement_supplier:{project}",
            label=f"{project}供应商",
            value=supplier,
            source_rank=context.rank,
        )
    ]


def _project_name_from_title(title: str | None) -> str | None:
    if title is None:
        return None
    match = re.search(r"(?P<project>[\u4e00-\u9fffA-Za-z0-9（）()·\- ]{2,80}?采购项目)", title)
    if match is None:
        return None
    return match.group("project").strip()


def _normalize_answer_text(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("－", "-").replace("—", "-")
