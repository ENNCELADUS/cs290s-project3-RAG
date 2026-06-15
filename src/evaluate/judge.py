from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .schema import QuestionSpec

JudgeStatus = Literal["correct", "incorrect", "manual_review", "evidence_insufficient"]

ALIAS_GROUPS = (
    ("acl", "计算语言学协会年会", "associationforcomputationallinguistics"),
    (
        "sandiegocaliforniausa",
        "sandiegocalifornia",
        "sandiego",
        "美国加利福尼亚州圣迭戈",
        "美国加州圣迭戈",
        "美国加利福尼亚州圣地亚哥",
        "美国加州圣地亚哥",
        "圣迭戈",
        "圣地亚哥",
    ),
    ("california", "加利福尼亚州", "加州"),
    ("usa", "unitedstates", "美国"),
    ("semiconductoranalyzer", "半导体分析仪"),
    ("microelectronicmeasurementsystem", "微电子测量系统"),
    ("lowtemperaturemeasurementsystem", "低温测量系统"),
    ("powerdevicemeasurementsystem", "功率器件测量系统"),
    ("loadpullsystem", "负载牵引系统", "负载拉移系统"),
    ("impedanceanalyzer", "阻抗分析仪"),
    ("otf1200x开启式管式炉", "otf1200xopeningtubefurnace", "otf1200xopentubefurnace", "开启式管式炉", "开放式管式炉"),
    ("1b206", "1b-206", "b区206", "b206", "1号楼1b206", "1号楼b区206"),
)


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
            return _loose_cited_source_judge(question, answer)
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
    if any(_special_fact_match(answer, fact) for fact in facts):
        return JudgeResult(status="correct", is_correct=1, reason="matched special fact")

    atoms = _dedupe([atom for fact in facts for atom in _important_atoms(fact)])
    if not atoms:
        if allow_manual_review:
            return JudgeResult(status="manual_review", is_correct=None, reason="loose atom judge has no atoms")
        return JudgeResult(status="incorrect", is_correct=0, reason="loose atom judge has no atoms")

    matched = [atom for atom in atoms if _compact_contains(answer, atom)]
    missing_quantity_atoms = [
        atom for atom in atoms if _quantity_keys(atom) and not _is_year_only_quantity_atom(atom) and atom not in matched
    ]
    if missing_quantity_atoms:
        return JudgeResult(
            status="incorrect",
            is_correct=0,
            reason=f"loose atom judge missing required quantity atoms: {len(missing_quantity_atoms)}",
        )
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
    if _is_grade_atom(candidate):
        return _grade_key(candidate) in {_grade_key(grade) for grade in _grade_atoms(answer)}
    candidate_date_keys = _date_keys(candidate)
    if candidate_date_keys:
        answer_date_keys = _date_keys(answer)
        return any(date_key in answer_date_keys for date_key in candidate_date_keys)
    candidate_quantity_keys = _quantity_keys(candidate)
    if candidate_quantity_keys:
        answer_quantity_keys = _quantity_keys(answer)
        return any(quantity_key in answer_quantity_keys for quantity_key in candidate_quantity_keys)
    candidate_keys = _match_keys(candidate)
    answer_keys = _match_keys(answer)
    if candidate_keys & answer_keys:
        return True
    return _compact_text(candidate) in _compact_text(answer)


def _special_fact_match(answer: str, fact: str) -> bool:
    if _is_course_dedup_fact(fact):
        return _matches_course_dedup_fact(answer, fact)
    if _is_room_1b206_fact(fact):
        return _matches_room_1b206(answer)
    return False


def _is_course_dedup_fact(text: str) -> bool:
    compact = _compact_text(text)
    has_dedup_concept = any(term in compact for term in ("自动去重", "不重复计算学分", "不重复累加", "不能重复累加"))
    return has_dedup_concept and _has_once_credit_concept(text)


def _matches_course_dedup_fact(answer: str, fact: str) -> bool:
    compact_answer = _compact_text(answer)
    if not _has_once_credit_concept(answer):
        return False
    if not any(term in compact_answer for term in ("自动去重", "去重", "不重复累加", "不会重复累加", "不能重复累加")):
        return False
    if not any(term in compact_answer for term in ("上一层级", "上层", "本学科选修", "总学分")):
        return False
    if "同时计入" in _compact_text(fact):
        has_two_modules = "两个模块" in compact_answer or (
            "专业方向必修" in compact_answer and "专业任选" in compact_answer
        )
        has_module_counts_and_credits = "门数" in compact_answer and "学分" in compact_answer
        if not (has_two_modules and has_module_counts_and_credits):
            return False
    return True


def _has_once_credit_concept(text: str) -> bool:
    compact = _compact_text(text)
    return any(term in compact for term in ("1次", "一次", "只算1次", "仅算1次", "只计算一次", "仅计算一次"))


def _is_room_1b206_fact(text: str) -> bool:
    compact = _compact_text(text)
    return any(alias in compact for alias in ("1b206", "1b-206", "b区206", "1号楼1b206", "1号楼b区206"))


def _matches_room_1b206(answer: str) -> bool:
    compact = _compact_text(answer)
    return any(alias in compact for alias in ("1b206", "1b-206", "b区206", "1号楼1b206", "1号楼b区206"))


def _date_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for year, month, day in re.findall(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", text):
        _add_date_key(keys, int(month), int(day), int(year))
    for month, day in re.findall(r"(?<!\d)(\d{2})(\d{2})(?!\d)", text):
        _add_date_key(keys, int(month), int(day))
    for year, month, day in re.findall(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", text):
        _add_date_key(keys, int(month), int(day), int(year))
    for year, month in re.findall(r"(?<!\d)(\d{4})[-/.](\d{1,2})(?![-/.\d])", text):
        _add_month_key(keys, int(month), int(year))
    for year, month, day in re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text):
        _add_date_key(keys, int(month), int(day), int(year))
    for year, month, start_day, end_day in re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日?[至到-](\d{1,2})日", text):
        _add_date_key(keys, int(month), int(start_day), int(year))
        _add_date_key(keys, int(month), int(end_day), int(year))
    for month, day in re.findall(r"(\d{1,2})月(\d{1,2})日", text):
        _add_date_key(keys, int(month), int(day))
    return keys


def _quantity_keys(text: str) -> set[str]:
    keys: set[str] = set()
    normalized = _normalize_readable_text(text)
    unit_pattern = r"学分|分|年|门|人|项|天"
    for start, end, unit in re.findall(rf"(\d+(?:\.\d+)?)\s*[-至到]\s*(\d+(?:\.\d+)?)\s*({unit_pattern})", normalized):
        keys.add(f"{_quantity_number_key(start)}{unit}")
        keys.add(f"{_quantity_number_key(end)}{unit}")
    for number, unit in re.findall(
        rf"(\d+(?:\.\d+)?)\s*(?:个)?\s*(?:总|课程|选修|专业|实践|课程实践)?\s*({unit_pattern})",
        normalized,
    ):
        keys.add(f"{_quantity_number_key(number)}{unit}")
    for unit, number in re.findall(
        rf"(?:总|课程|选修|必修|合计|专业|实践|课程实践)?\s*({unit_pattern})\s*(?:为|是|:|：)?\s*(\d+(?:\.\d+)?)",
        normalized,
    ):
        keys.add(f"{_quantity_number_key(number)}{unit}")
    return keys


def _quantity_number_key(number: str) -> str:
    return number[:-2] if number.endswith(".0") else number


def _is_year_only_quantity_atom(text: str) -> bool:
    quantity_keys = _quantity_keys(text)
    return bool(quantity_keys) and all(re.fullmatch(r"\d{4}年", key) for key in quantity_keys)


def _add_date_key(keys: set[str], month: int, day: int, year: int | None = None) -> None:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return
    if year is not None:
        _add_month_key(keys, month, year)
        keys.add(f"{year:04d}{month:02d}{day:02d}")
    keys.add(f"{month:02d}{day:02d}")


def _add_month_key(keys: set[str], month: int, year: int) -> None:
    if 1 <= month <= 12:
        keys.add(f"{year:04d}{month:02d}")


def _compact_text(text: str) -> str:
    compact = re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)
    compact = compact.replace("委员会的委员会主任", "委员会主任")
    for label in (
        "office",
        "sist",
        "schoolofinformationscienceandtechnology",
        "email",
        "phone",
        "他的",
        "她的",
        "办公室",
        "办公地点",
        "信息学院",
        "工作邮箱",
        "邮箱",
        "电话",
        "教授",
        "担任",
        "由",
        "是",
        "在",
    ):
        compact = compact.replace(label, "")
    return compact


def _important_atoms(text: str) -> list[str]:
    readable_text = _normalize_readable_text(text)
    text = _normalize_atom_text(text)
    atoms: list[str] = []
    atoms.extend(re.findall(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
    atoms.extend(re.findall(r"\b\d{3,4}-?\d{6,8}\b", text))
    atoms.extend(_grade_atoms(readable_text))
    atoms.extend(re.findall(r"\b[A-Z][A-Za-z]+ ?\d{4}\b|\b[A-Z]{2,} ?\d{4}\b", readable_text))
    atoms.extend(_date_keys(text))
    atoms.extend(re.findall(r"\d{1,2}月\d{1,2}日|\d{1,2}:\d{2}", text))
    atoms.extend(re.findall(r"\b[A-Z]{2,}\d+[A-Z]?\b", text))
    atoms.extend(_known_program_atoms(text))
    atoms.extend(
        re.findall(
            r"[\u4e00-\u9fffA-Za-z]+(?:学院|校区|楼|室)\d+[A-Za-z]?-?\d{2,4}[A-Za-z]?室?|"
            r"[\u4e00-\u9fffA-Za-z]+\d+[A-Za-z]?-\d{2,4}[A-Za-z]?室?",
            text,
        )
    )
    atoms.extend(re.findall(r"\b\d+[A-Za-z]?-?\d{2,4}[A-Za-z]?室?\b", text))
    atoms.extend(re.findall(r"\d+(?:\.\d+)?\s*(?:学分|人|个|项|门|天|年|分)", text))
    atoms.extend(re.findall(r"[\u4e00-\u9fffA-Za-z]+(?:一等奖|二等奖|三等奖|冠军|亚军|奖)", text))
    atoms.extend(_english_title_atoms(readable_text))
    structured_atom_keys = {_compact_text(atom) for atom in atoms}
    for chunk in re.split(
        r"[，,；;。:：、（）()和与及而]|包括|分别|以及|其中|要求|修满|官方|确切|发布|上线|日期|时间|地点|单位|邀请人|演讲者|是|为",
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
    return _normalize_readable_text(text).replace(" ", "")


def _normalize_readable_text(text: str) -> str:
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("–", "-").replace("—", "-").replace("－", "-")
    text = text.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", text).strip()


def _english_title_atoms(text: str) -> list[str]:
    atoms: list[str] = []
    for match in re.finditer(
        r"\b(?:[A-Z][A-Za-z0-9]+|[A-Z]{2,})(?:[- ][A-Z][A-Za-z0-9]+|[- ][A-Z]{2,}){1,6}\b",
        text,
    ):
        atom = match.group(0).strip(" ,.;:()")
        if 4 <= len(_compact_text(atom)) <= 80:
            atoms.append(atom)
    return atoms


def _grade_atoms(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])[A-D][+-](?![A-Za-z])", text, flags=re.IGNORECASE)


def _is_grade_atom(text: str) -> bool:
    return _grade_key(text) in {"A+", "A-", "B+", "B-", "C+", "C-", "D+", "D-"}


def _grade_key(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


def _known_program_atoms(text: str) -> list[str]:
    compact = _compact_text(text)
    programs = (
        "计算机科学与技术",
        "电子科学与技术",
        "信息与通信工程",
        "电子信息",
    )
    return [program for program in programs if _compact_text(program) in compact]


def _match_keys(text: str) -> set[str]:
    compact = _compact_text(text)
    keys = {compact} if compact else set()
    for group in ALIAS_GROUPS:
        compact_group = {_compact_text(alias) for alias in group}
        if compact_group & keys or any(alias in compact for alias in compact_group):
            keys.update(compact_group)
    return keys


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
