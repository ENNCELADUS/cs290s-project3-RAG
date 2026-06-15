from __future__ import annotations

import re

CREDIT_TABLE_COLUMNS = ("必修", "选修", "学分")
CREDIT_TABLE_ROWS = ("人文社科通识", "自然科学通识", "专业课程")


def append_structured_table_bindings(text: str) -> str:
    bindings = structured_table_bindings(text)
    if not bindings:
        return text
    return f"{text.rstrip()}\n\nStructured table bindings:\n" + "\n".join(bindings)


def structured_table_bindings(text: str) -> list[str]:
    bindings = degree_credit_table_bindings(text)
    seen = set(bindings)
    for binding in markdown_table_bindings(text):
        if binding not in seen:
            bindings.append(binding)
            seen.add(binding)
    return bindings


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


def markdown_table_bindings(text: str) -> list[str]:
    rows = [_pipe_cells(line) for line in text.splitlines()]
    rows = [row for row in rows if row]
    bindings: list[str] = []
    for index, header in enumerate(rows[:-1]):
        if len(header) < 2:
            continue
        if index + 1 < len(rows) and _is_markdown_separator(rows[index + 1]):
            data_rows = rows[index + 2 :]
        else:
            data_rows = rows[index + 1 :]
        if not data_rows:
            continue
        row_label_column, *value_columns = header
        if not row_label_column or not value_columns:
            continue
        for row in data_rows:
            if len(row) != len(header) or _is_markdown_separator(row):
                continue
            row_label = row[0]
            if not row_label:
                continue
            for column, value in zip(value_columns, row[1:], strict=True):
                if column and value:
                    bindings.append(f"{row_label} - {column} - {value}")
        if bindings:
            return bindings
    return bindings


def _pipe_cells(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        return []
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return [cell for cell in cells if cell]


def _is_markdown_separator(row: list[str]) -> bool:
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in row)


def _clean_number(number: str) -> str:
    return number[:-2] if number.endswith(".0") else number
