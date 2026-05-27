from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

CORE_CATEGORIES = {
    "school_info",
    "program",
    "program_requirements",
    "courses",
    "faculty",
    "research",
    "admission",
    "career",
    "news",
    "events",
}


def quality_flags(text: str, status_code: int | None = None, parser: str | None = None) -> list[str]:
    flags: list[str] = []
    if status_code is not None and status_code >= 400:
        flags.append("http_error")
    if not text.strip():
        flags.append("empty_text")
    elif len(text.strip()) < 120:
        flags.append("short_text")
    if "\ufffd" in text:
        flags.append("replacement_chars")
    if parser == "ocr":
        flags.append("ocr_text")
    if _looks_garbled(text):
        flags.append("possibly_garbled")
    return flags


def _looks_garbled(text: str) -> bool:
    if not text:
        return False
    suspicious = sum(1 for char in text if ord(char) < 32 and char not in "\n\t")
    return suspicious / max(1, len(text)) > 0.01


def write_quality_report(
    run_dir: Path,
    documents: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    structured_counts: dict[str, int],
    failures: list[str],
    expected_categories: set[str] | None = None,
) -> None:
    coverage_categories = expected_categories or CORE_CATEGORIES
    category_counts = Counter(str(doc.get("category") or "unknown") for doc in documents)
    host_counts = Counter(str(row.get("host") or "unknown") for row in manifest_rows)
    language_counts = Counter(str(doc.get("language") or "unknown") for doc in documents)
    low_quality = [
        row
        for row in manifest_rows
        if any(
            flag in str(row.get("quality_flags", ""))
            for flag in ("empty_text", "possibly_garbled", "replacement_chars")
        )
    ]
    duplicate_sha = _duplicate_count(manifest_rows, "sha256")
    missing_categories = sorted(coverage_categories - set(category_counts))
    ocr_count = sum(1 for row in manifest_rows if str(row.get("ocr_used", "")).lower() == "true")

    lines = [
        "# Data Collection Quality Report",
        "",
        f"- generated_at_utc: `{datetime.now(UTC).isoformat(timespec='seconds')}`",
        f"- run_dir: `{run_dir}`",
        f"- documents: **{len(documents)}**",
        f"- manifest rows: **{len(manifest_rows)}**",
        f"- failed URLs: **{len(failures)}**",
        f"- duplicate sha256 values: **{duplicate_sha}**",
        f"- OCR documents: **{ocr_count}**",
        f"- low-quality rows: **{len(low_quality)}**",
        f"- low-quality ratio: **{_ratio(len(low_quality), len(manifest_rows)):.2%}**",
        "",
        "## Coverage Matrix",
        "",
        "| category | documents | status |",
        "| --- | ---: | --- |",
    ]
    for category in sorted(coverage_categories):
        count = category_counts.get(category, 0)
        status = "covered" if count else "missing"
        lines.append(f"| {category} | {count} | {status} |")

    lines.extend(["", "## Category Counts", "", "| category | documents |", "| --- | ---: |"])
    for category, count in category_counts.most_common():
        lines.append(f"| {category} | {count} |")

    lines.extend(["", "## Host Counts", "", "| host | rows |", "| --- | ---: |"])
    for host, count in host_counts.most_common(20):
        lines.append(f"| {host} | {count} |")

    lines.extend(["", "## Language Counts", "", "| language | documents |", "| --- | ---: |"])
    for language, count in language_counts.most_common():
        lines.append(f"| {language} | {count} |")

    lines.extend(["", "## Structured Outputs", "", "| file | rows |", "| --- | ---: |"])
    for name, count in sorted(structured_counts.items()):
        lines.append(f"| {name} | {count} |")

    lines.extend(["", "## Quality Gate Notes", ""])
    if missing_categories:
        lines.append(f"- Missing core categories: {', '.join(missing_categories)}.")
    else:
        lines.append("- All core categories are represented.")
    lines.append("- Manual audit required: sample 5-10 documents per major category before final indexing.")
    lines.append(
        "- Program requirement rows should be checked for explicit degree, cohort, credit, or requirement language."
    )

    if failures:
        lines.extend(["", "## Failed URLs", ""])
        lines.extend(f"- {failure}" for failure in failures[:100])

    run_dir.joinpath("quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _duplicate_count(rows: list[dict[str, object]], key: str) -> int:
    values = [str(row.get(key) or "") for row in rows if row.get(key)]
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _ratio(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return part / total
