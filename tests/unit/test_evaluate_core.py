from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evaluate.judge import judge_answer
from evaluate.metrics import normalize_url, source_matches, source_metrics
from evaluate.schema import load_questions


def test_load_questions_parses_json_columns(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)

    questions = load_questions(questions_path)

    assert len(questions) == 1
    assert questions[0].id == "q1"
    assert questions[0].acceptable_source_urls == ["https://example.edu/source"]
    assert questions[0].required_facts == ["office 3-530", "wanghy@example.edu"]


def test_load_questions_rejects_missing_required_columns(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text("id,query\nq1,hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_questions(questions_path)


def test_source_metrics_are_rank_aware_and_url_prefix_based() -> None:
    observed = [
        "http://example.edu/noise/",
        "https://example.edu/source/detail",
        "https://example.edu/other",
    ]
    expected = ["https://example.edu/source", "https://example.edu/other"]

    metrics = source_metrics(observed, expected)

    assert normalize_url("http://EXAMPLE.edu/source/") == "https://example.edu/source"
    assert source_matches("https://example.edu/source/detail", "https://example.edu/source")
    assert source_matches("https://example.edu/source/detail", "https://example.edu/")
    assert metrics["source_hit@1"] == 0.0
    assert metrics["source_hit@5"] == 1.0
    assert metrics["source_recall@5"] == 1.0
    assert metrics["mrr@5"] == 0.5
    assert metrics["precision@5"] == 0.4
    assert 0.0 < metrics["ndcg@5"] < 1.0


def test_source_metrics_match_sist_template_and_query_variants() -> None:
    expected = [
        "https://sist.shanghaitech.edu.cn/2024/0115/c7339a1097189/page.htm",
        "https://sist.shanghaitech.edu.cn/list.htm",
    ]
    observed = [
        "https://sist.shanghaitech.edu.cn/_t335/2024/0115/c7339a1097189/page.htm?from=nav#section",
        "https://sist.shanghaitech.edu.cn/list.htm?lang=en",
    ]

    metrics = source_metrics(observed, expected)

    assert source_matches(observed[0], expected[0])
    assert source_matches(observed[1], expected[1])
    assert metrics["source_hit@1"] == 1.0
    assert metrics["source_recall@5"] == 1.0


def test_judge_exact_required_manual_and_forbidden_cases(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]

    assert judge_answer(question, "The office is office 3-530 and email is wanghy@example.edu.").is_correct == 1

    required_question = question.__class__(
        **{
            **question.__dict__,
            "judge_type": "required_facts_match",
            "acceptable_answers": [],
        }
    )
    assert judge_answer(required_question, "office 3-530 only").status == "incorrect"
    assert judge_answer(required_question, "office 3-530, wanghy@example.edu").status == "correct"

    manual_question = question.__class__(**{**question.__dict__, "judge_type": "required_facts_with_manual_review"})
    assert judge_answer(manual_question, "office 3-530, wanghy@example.edu").status == "manual_review"
    assert judge_answer(question, "wrong forbidden fact").status == "incorrect"


def _write_question_csv(path: Path) -> None:
    row = {
        "id": "q1",
        "category": "Factual",
        "language": "en",
        "query": "Where is the office?",
        "gt_answer": "office 3-530, wanghy@example.edu",
        "primary_source_url": "https://example.edu/source",
        "acceptable_source_urls": json.dumps(["https://example.edu/source"]),
        "evidence_snippet": "office 3-530",
        "required_facts": json.dumps(["office 3-530", "wanghy@example.edu"]),
        "acceptable_answers": json.dumps(["office 3-530 and email is wanghy@example.edu"]),
        "forbidden_facts": json.dumps(["forbidden fact"]),
        "grading_notes": "test row",
        "judge_type": "exact_or_alias_match",
        "complexity": "Low",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
