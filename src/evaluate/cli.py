from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path

from .export import write_outputs
from .runner import ArtifactPaths, EvalMode, EvaluationConfig, run_evaluation
from .schema import DEFAULT_QUESTIONS, load_questions

DEFAULT_OUTPUT_DIR = Path("data/eval")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5 ShanghaiTech/SIST RAG evaluation.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runner", choices=["retrieve", "answer", "both"], default="both")
    parser.add_argument("--modes", nargs="+", choices=["dense", "hybrid"], default=["dense", "hybrid"])
    parser.add_argument("--include-diagnostic-bm25", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--bm25", type=Path, default=None)
    parser.add_argument("--faiss", type=Path, default=None)
    parser.add_argument("--chunk-index", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--review-decisions", type=Path, default=None)
    parser.add_argument("--require-final-labels", action="store_true")
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args(argv)

    modes: list[EvalMode] = list(args.modes)
    if args.include_diagnostic_bm25:
        modes.insert(0, "bm25")
    artifacts = ArtifactPaths()
    artifacts = ArtifactPaths(
        db=args.db or artifacts.db,
        bm25=args.bm25 or artifacts.bm25,
        faiss=args.faiss or artifacts.faiss,
        chunk_index=args.chunk_index or artifacts.chunk_index,
        report=args.report or artifacts.report,
        dense_model=args.dense_model,
    )
    questions = load_questions(args.questions, offset=args.offset, limit=args.limit)
    config = EvaluationConfig(
        runner=args.runner,
        modes=tuple(modes),
        top_k=args.top_k,
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        artifacts=artifacts,
    )
    timestamp = args.timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    records = run_evaluation(questions, config)
    review_decisions = _load_review_decisions(args.review_decisions) if args.review_decisions is not None else {}
    paths = write_outputs(
        records,
        args.output_dir,
        timestamp=timestamp,
        review_decisions=review_decisions,
        require_final_labels=args.require_final_labels,
    )
    for label, path in paths.items():
        print(f"wrote {label}: {path}")
    return 1 if any(record["status"] != "ok" for record in records) else 0


def _load_review_decisions(path: Path) -> dict[tuple[str, str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "mode", "is_correct"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    decisions: dict[tuple[str, str], int] = {}
    for row in rows:
        value = row["is_correct"].strip()
        if value not in {"0", "1"}:
            raise ValueError(f"{path} row {row['id']} {row['mode']} has invalid is_correct={value!r}")
        decisions[(row["id"], row["mode"])] = int(value)
    return decisions


if __name__ == "__main__":
    raise SystemExit(main())
