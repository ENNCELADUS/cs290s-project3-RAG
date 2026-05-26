from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .crawler import CollectorConfig, OfficialCollector
from .io import prepare_run_dir
from .merge import merge_existing_with_run
from .pdf import ocr_environment_status
from .seeds import load_seed_urls

DEFAULT_SEEDS = Path("config/official_seed_urls.csv")
DEFAULT_COLLECTION_RUNS = Path("data/collection_runs")
DEFAULT_EXISTING_JSONL = Path("data/jsonl")
DEFAULT_MERGED = Path("data/merged")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ShanghaiTech/SIST official-source data collection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check local collection environment")
    doctor_parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)

    collect_parser = subparsers.add_parser("collect", help="Run an append-only collection pass")
    collect_parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    collect_parser.add_argument("--collection-runs", type=Path, default=DEFAULT_COLLECTION_RUNS)
    collect_parser.add_argument("--run-name", default=None)
    collect_parser.add_argument("--max-pages", type=int, default=50)
    collect_parser.add_argument("--delay", type=float, default=1.0)
    collect_parser.add_argument("--timeout", type=float, default=20.0)
    collect_parser.add_argument("--retries", type=int, default=1)
    collect_parser.add_argument("--dry-run", action="store_true")
    collect_parser.add_argument("--ignore-robots", action="store_true")
    collect_parser.add_argument(
        "--skip-known",
        action="store_true",
        help="Do not write URLs already present in data/jsonl or previous collection runs.",
    )
    collect_parser.add_argument("--existing-jsonl", type=Path, default=DEFAULT_EXISTING_JSONL)

    merge_parser = subparsers.add_parser("merge", help="Merge existing data/jsonl with one collection run")
    merge_parser.add_argument("--existing-jsonl", type=Path, default=DEFAULT_EXISTING_JSONL)
    merge_parser.add_argument("--run-jsonl", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, default=DEFAULT_MERGED)

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.seeds)
    if args.command == "collect":
        return _collect(args)
    if args.command == "merge":
        return _merge(args)
    return 2


def _doctor(seeds_path: Path) -> int:
    seeds = load_seed_urls(seeds_path)
    ocr_status = ocr_environment_status()
    print(f"seed_count={len(seeds)}")
    for key, value in ocr_status.items():
        print(f"{key}={value}")
    if not ocr_status["has_chi_sim"]:
        print("warning=tesseract chi_sim language data is missing; Chinese OCR fallback will be disabled")
    return 0


def _collect(args: argparse.Namespace) -> int:
    run_dir = prepare_run_dir(args.collection_runs, args.run_name)
    known_urls = (
        _load_known_urls(args.existing_jsonl, args.collection_runs, current_run_dir=run_dir)
        if args.skip_known
        else set()
    )
    config = CollectorConfig(
        seeds_path=args.seeds,
        run_dir=run_dir,
        max_pages=args.max_pages,
        request_delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        retries=args.retries,
        dry_run=args.dry_run,
        respect_robots=not args.ignore_robots,
        known_urls=frozenset(known_urls),
        progress_factory=None if args.dry_run else _tqdm_progress,
    )
    stats = OfficialCollector(config).run()
    print(f"run_dir={run_dir}")
    for key, value in stats.items():
        print(f"{key}={value}")
    if args.skip_known:
        print(f"known_urls_skipped={len(known_urls)}")
    return 0


def _merge(args: argparse.Namespace) -> int:
    stats = merge_existing_with_run(args.existing_jsonl, args.run_jsonl, args.output)
    print(f"output={args.output}")
    for key, value in stats.items():
        print(f"{key}={value}")
    return 0


def _tqdm_progress(total: int) -> tqdm:
    return tqdm(total=total, desc="collecting pages", unit="page")


def _load_known_urls(existing_jsonl: Path, collection_runs: Path, current_run_dir: Path) -> set[str]:
    known_urls: set[str] = set()
    for documents_path in [existing_jsonl / "documents.jsonl", *collection_runs.glob("*/jsonl/documents.jsonl")]:
        if current_run_dir in documents_path.parents or not documents_path.exists():
            continue
        with documents_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                url = row.get("canonical_url") or row.get("url")
                if isinstance(url, str) and url:
                    known_urls.add(url)
    return known_urls
