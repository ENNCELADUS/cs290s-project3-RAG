from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime

DATE_RE = re.compile(r"(?P<date>20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)")
EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
COURSE_RE = re.compile(r"\b(?P<code>(?:CS|EE|SI|AI|MATH|BIO)\d{2,4}[A-Z]?)\b")
CREDIT_RE = re.compile(r"(?P<credits>\d+(?:\.\d+)?)\s*(?:credits?|学分)")
TITLE_RE = re.compile(r"\b(Professor|Associate Professor|Assistant Professor|Research Professor|Lecturer)\b", re.I)

REQUIREMENT_KEYWORDS = (
    "培养方案",
    "毕业",
    "学位",
    "学分",
    "必修",
    "选修",
    "课程体系",
    "graduation",
    "degree",
    "credit",
)
NOISY_REQUIREMENT_KEYWORDS = ("毕业生故事", "青春榜样", "新闻", "活动", "招聘")
FACULTY_NAME_STOPWORDS = {
    "faculty",
    "people",
    "home",
    "homepage",
    "introduction",
    "profile",
    "teacher",
    "teachers",
    "师资队伍",
    "常任教授",
    "特聘教授",
    "研究人员",
    "支撑人员",
    "行政人员",
}


def extract_structured_records(document: dict[str, object], text: str) -> dict[str, list[dict[str, object]]]:
    observed_at = str(document.get("fetched_at") or datetime.now(UTC).isoformat(timespec="seconds"))
    base = {
        "source_document_id": document.get("id"),
        "source_url": document.get("url"),
        "observed_at": observed_at,
    }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "courses": _extract_courses(lines, base) if _allows_structured_kind(document, "courses") else [],
        "faculty_members": _extract_faculty(lines, base) if _allows_structured_kind(document, "faculty") else [],
        "program_requirements": (
            _extract_program_requirements(lines, base)
            if _allows_structured_kind(document, "program_requirements")
            else []
        ),
        "events": _extract_events(lines, base, str(document.get("category") or "")),
    }


def _extract_courses(lines: list[str], base: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in lines:
        match = COURSE_RE.search(line)
        if not match:
            continue
        code = match.group("code")
        credit_match = CREDIT_RE.search(line)
        course_name = _trim_evidence(line.replace(code, "").strip(" -:：|"))
        if not course_name and not credit_match:
            continue
        records.append(
            {
                **base,
                "school": "School of Information Science and Technology",
                "course_code": code,
                "course_name": course_name,
                "credits": float(credit_match.group("credits")) if credit_match else None,
                "evidence": _trim_evidence(line),
                "confidence": 0.68 if credit_match else 0.55,
            }
        )
    return _dedupe(records, ("course_code", "evidence", "source_url"))


def _extract_faculty(lines: list[str], base: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        emails = EMAIL_RE.findall(line)
        title_match = TITLE_RE.search(line)
        if not title_match:
            continue
        if not emails and not any(keyword in line.lower() for keyword in ("professor", "lecturer", "faculty")):
            continue
        context = _context(lines, index, radius=2)
        records.append(
            {
                **base,
                "school": "School of Information Science and Technology",
                "name": _guess_name(context),
                "title": title_match.group(1) if title_match else None,
                "email": emails[0] if emails else None,
                "evidence": _trim_evidence(context),
                "confidence": 0.76 if emails and title_match else 0.62,
            }
        )
    return _dedupe(records, ("email", "title", "source_url", "evidence"))


def _extract_program_requirements(lines: list[str], base: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(keyword in line for keyword in NOISY_REQUIREMENT_KEYWORDS):
            continue
        if not any(keyword in lowered or keyword in line for keyword in REQUIREMENT_KEYWORDS):
            continue
        credit_match = CREDIT_RE.search(line)
        if not credit_match and not any(keyword in line for keyword in ("必修", "选修", "毕业要求", "学位要求")):
            continue
        context = _context(lines, index, radius=1)
        records.append(
            {
                **base,
                "school": "School of Information Science and Technology",
                "program_name": _guess_program_name(context),
                "requirement_type": "graduation" if "毕业" in context or "graduation" in context.lower() else "program",
                "requirement_text": _trim_evidence(line),
                "min_credits": float(credit_match.group("credits")) if credit_match else None,
                "evidence": _trim_evidence(context),
                "confidence": 0.8 if credit_match else 0.66,
            }
        )
    return _dedupe(records, ("requirement_text", "source_url"))


def _extract_events(lines: list[str], base: dict[str, object], category: str) -> list[dict[str, object]]:
    if category not in {"events", "news", "admission", "career", "general"}:
        return []
    records: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        date_match = DATE_RE.search(line)
        if not date_match:
            continue
        previous = lines[index - 1] if index > 0 else ""
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        title = previous if 5 <= len(previous) <= 120 else next_line
        if not title or title == line:
            continue
        org = (
            "ShanghaiTech University"
            if "sist" not in str(base.get("source_url", "")).lower()
            else "School of Information Science and Technology"
        )
        records.append(
            {
                **base,
                "org": org,
                "event_type": category,
                "title": title,
                "published_at": _normalize_date(date_match.group("date")),
                "language": "zh" if any("\u4e00" <= char <= "\u9fff" for char in title) else "en",
                "evidence": _trim_evidence(_context(lines, index, radius=1)),
                "confidence": 0.7,
            }
        )
    return _dedupe(records, ("title", "published_at", "source_url"))


def _allows_structured_kind(document: dict[str, object], kind: str) -> bool:
    category = str(document.get("category") or "")
    url = str(document.get("url") or "").lower()
    if kind == "courses":
        return category == "courses" or any(token in url for token in ("course", "courses", "bkjx", "yjsjx"))
    if kind == "faculty":
        return category == "faculty" or "faculty.sist" in url or any(token in url for token in ("szdw", "teacher"))
    if kind == "program_requirements":
        return category == "program_requirements" or any(
            token in url for token in ("pyfa", "training", "培养方案", "degree")
        )
    return False


def _context(lines: list[str], index: int, radius: int) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(lines[start:end])


def _guess_name(context: str) -> str | None:
    lines = context.splitlines()
    for line in lines:
        inline_name = _name_before_title(line)
        if inline_name:
            return inline_name
    for line in reversed(lines):
        clean = line.strip()
        if _is_plausible_faculty_name(clean):
            return clean
    return None


def _name_before_title(line: str) -> str | None:
    match = TITLE_RE.search(line)
    if not match:
        return None
    prefix = line[: match.start()].strip(" -:：|")
    words = [word for word in prefix.split() if word.lower() not in FACULTY_NAME_STOPWORDS]
    if len(words) < 2:
        return None
    candidate = " ".join(words[-2:])
    return candidate if _is_plausible_faculty_name(candidate) else None


def _is_plausible_faculty_name(value: str) -> bool:
    if not 2 <= len(value) <= 80:
        return False
    if value.lower() in FACULTY_NAME_STOPWORDS:
        return False
    if EMAIL_RE.search(value) or TITLE_RE.search(value):
        return False
    if any(token in value.lower() for token in ("room ", "building", "地址", "学院", "university")):
        return False
    return True


def _guess_program_name(context: str) -> str | None:
    for candidate in ("计算机科学与技术", "电子信息", "信息科学与技术", "Computer Science", "Electronic Information"):
        if candidate in context:
            return candidate
    return None


def _normalize_date(value: str) -> str:
    return value.replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").replace(".", "-")


def _trim_evidence(value: str, limit: int = 500) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def _dedupe(records: Iterable[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    seen: set[tuple[object, ...]] = set()
    deduped: list[dict[str, object]] = []
    for record in records:
        fingerprint = tuple(record.get(key) for key in keys)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(record)
    return deduped
