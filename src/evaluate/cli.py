from __future__ import annotations

import argparse
import csv
import json
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
    parser.add_argument("--diagnostic-depth", type=int, default=None)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--dense-top-k", type=int, default=None)
    parser.add_argument("--fused-top-k", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--rerank-preserve-top-k", type=int, default=None)
    parser.add_argument("--rrf-k", type=int, default=None)
    parser.add_argument("--sparse-weight", type=float, default=None)
    parser.add_argument("--dense-weight", type=float, default=None)
    parser.add_argument("--url-cap", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--answer-reranker-model", type=Path, default=None)
    parser.add_argument("--answer-reranker-device", default="cpu")
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--expanded-query", action="append", default=None)
    parser.add_argument("--expanded-queries-jsonl", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
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
    if args.diagnostic_depth is not None and args.diagnostic_depth <= 0:
        parser.error("--diagnostic-depth must be a positive integer")

    modes: list[EvalMode] = list(args.modes)
    if (args.expanded_query or args.expanded_queries_jsonl) and "hybrid" not in modes:
        parser.error("query expansion requires --modes hybrid")
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
        diagnostic_depth=args.diagnostic_depth,
        sparse_top_k=args.sparse_top_k,
        dense_top_k=args.dense_top_k,
        fused_top_k=args.fused_top_k,
        rerank_top_k=args.rerank_top_k,
        rerank_preserve_top_k=args.rerank_preserve_top_k,
        rrf_k=args.rrf_k,
        sparse_weight=args.sparse_weight,
        dense_weight=args.dense_weight,
        url_cap=args.url_cap,
        model_path=args.model_path,
        answer_reranker_model=args.answer_reranker_model,
        answer_reranker_device=args.answer_reranker_device,
        reranker_model=args.reranker_model,
        reranker_device=args.reranker_device,
        expanded_queries=tuple(args.expanded_query or ()),
        expanded_queries_by_id=(
            _load_expanded_queries_jsonl(args.expanded_queries_jsonl)
            if args.expanded_queries_jsonl is not None
            else None
        ),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
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


def _load_expanded_queries_jsonl(path: Path) -> dict[str, tuple[str, ...]]:
    expanded_queries: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} expected JSON object")
            question_id = row.get("id")
            queries = row.get("expanded_queries")
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(f"{path}:{line_number} expected non-empty string id")
            if not isinstance(queries, list) or not all(isinstance(query, str) for query in queries):
                raise ValueError(f"{path}:{line_number} expected expanded_queries as a list of strings")
            cleaned = tuple(query.strip() for query in queries if query.strip())
            if cleaned:
                expanded_queries[question_id.strip()] = cleaned
    return expanded_queries


if __name__ == "__main__":
    raise SystemExit(main())
