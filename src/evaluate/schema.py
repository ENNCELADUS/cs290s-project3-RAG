from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

DEFAULT_QUESTIONS = Path("data/test/question_final_structured_100.csv")

REQUIRED_COLUMNS = {
    "id",
    "category",
    "language",
    "query",
    "gt_answer",
    "primary_source_url",
    "acceptable_source_urls",
    "required_facts",
    "acceptable_answers",
    "forbidden_facts",
    "judge_type",
}


@dataclass(frozen=True)
class QuestionSpec:
    id: str
    category: str
    language: str
    query: str
    gt_answer: str
    primary_source_url: str
    acceptable_source_urls: list[str]
    evidence_snippet: str
    required_facts: list[str]
    acceptable_answers: list[str]
    forbidden_facts: list[str]
    grading_notes: str
    judge_type: str
    complexity: str

    @property
    def question_id(self) -> str:
        return self.id


def load_questions(path: Path, *, offset: int = 0, limit: int | None = None) -> list[QuestionSpec]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    questions = [_question_from_row(row, path=path) for row in rows]
    selected = questions[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _question_from_row(row: dict[str, str], *, path: Path) -> QuestionSpec:
    return QuestionSpec(
        id=row["id"],
        category=row["category"],
        language=row["language"],
        query=row["query"],
        gt_answer=row["gt_answer"],
        primary_source_url=row["primary_source_url"],
        acceptable_source_urls=_json_list(row["acceptable_source_urls"], path=path, row_id=row["id"]),
        evidence_snippet=row.get("evidence_snippet", ""),
        required_facts=_json_list(row.get("required_facts", "[]"), path=path, row_id=row["id"]),
        acceptable_answers=_json_list(row.get("acceptable_answers", "[]"), path=path, row_id=row["id"]),
        forbidden_facts=_json_list(row.get("forbidden_facts", "[]"), path=path, row_id=row["id"]),
        grading_notes=row.get("grading_notes", ""),
        judge_type=row["judge_type"],
        complexity=row.get("complexity", ""),
    )


def _json_list(value: str, *, path: Path, row_id: str) -> list[str]:
    text = value.strip() if value else "[]"
    try:
        parsed = json.loads(text)
    except JSONDecodeError as error:
        raise ValueError(f"{path} row {row_id} expected JSON list") from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{path} row {row_id} expected JSON list of strings")
    return [item.strip() for item in parsed if item.strip()]


EvaluationQuestion = QuestionSpec
load_questions_csv = load_questions
