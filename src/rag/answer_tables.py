from __future__ import annotations

import re

CREDIT_TABLE_COLUMNS = ("必修", "选修", "学分")
CREDIT_TABLE_ROWS = ("人文社科通识", "自然科学通识", "专业课程")


def append_structured_table_bindings(text: str) -> str:
    bindings = degree_credit_table_bindings(text)
    if not bindings:
        return text
    return f"{text.rstrip()}\n\nStructured table bindings:\n" + "\n".join(bindings)


def degree_credit_table_bindings(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if "类别" not in normalized or "学分" not in normalized:
        return []

    bindings: list[str] = []
    total_match = re.search(r"修满至少\s*(\d+(?:\.\d+)?)\s*学分", normalized)
    if total_match is not None:
        bindings.append(f"毕业最低总学分 - 学分 - {_clean_number(total_match.group(1))}")

    for row_label in CREDIT_TABLE_ROWS:
        match = re.search(rf"{row_label}\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", normalized)
        if match is None:
            continue
        for column, value in zip(CREDIT_TABLE_COLUMNS, match.groups(), strict=True):
            bindings.append(f"{row_label} - {column} - {_clean_number(value)}")

    free_match = re.search(r"任选课程\s+(\d+(?:\.\d+)?)(?:\s+\d+(?:\.\d+)?)?", normalized)
    if free_match is not None:
        bindings.append(f"任选课程 - 学分 - {_clean_number(free_match.group(1))}")
    return bindings


def _clean_number(number: str) -> str:
    return number[:-2] if number.endswith(".0") else number
