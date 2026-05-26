from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

MANIFEST_FIELDS = [
    "url",
    "canonical_url",
    "title",
    "host",
    "category",
    "language",
    "content_type",
    "status_code",
    "fetched_at",
    "raw_path",
    "text_path",
    "sha256",
    "depth",
    "parent_url",
    "parser",
    "ocr_used",
    "text_chars",
    "quality_flags",
]


def prepare_run_dir(base_dir: Path, run_name: str | None = None) -> Path:
    run_name = run_name or datetime.now().date().isoformat()
    candidate = base_dir / run_name
    if not candidate.exists():
        _create_run_dirs(candidate)
        return candidate

    suffix = 1
    while True:
        candidate = base_dir / f"{run_name}_{suffix:03d}"
        if not candidate.exists():
            _create_run_dirs(candidate)
            return candidate
        suffix += 1


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            normalized = {field: row.get(field, "") for field in MANIFEST_FIELDS}
            if isinstance(normalized["quality_flags"], list):
                normalized["quality_flags"] = ";".join(normalized["quality_flags"])
            writer.writerow(normalized)


def _create_run_dirs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    run_dir.joinpath("raw").mkdir()
    run_dir.joinpath("texts").mkdir()
    run_dir.joinpath("jsonl").mkdir()
