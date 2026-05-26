from __future__ import annotations

import csv
import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .chunking import iter_chunk_records
from .crawler import CollectorConfig, OfficialCollector, build_eval_seed_candidates
from .io import read_jsonl, write_jsonl, write_manifest
from .quality import quality_flags, write_quality_report
from .structured import extract_structured_records
from .urls import canonicalize_url

STRUCTURED_FILES = ("courses", "faculty_members", "program_requirements", "events")


def reparse_run(
    source_run_dir: Path,
    output_run_dir: Path,
    *,
    seeds_path: Path,
    only_flags: set[str] | None = None,
    url_filter: set[str] | None = None,
    limit: int | None = None,
    chunk_chars: int = 1200,
    chunk_overlap: int = 120,
) -> dict[str, int]:
    """Reparse saved raw files from a prior collection run into a new append-only run."""
    source_documents = read_jsonl(source_run_dir / "jsonl" / "documents.jsonl")
    manifest_by_url = _load_manifest_by_url(source_run_dir / "source_manifest.csv")
    selected = _select_documents(source_documents, manifest_by_url, only_flags, url_filter)
    if limit is not None:
        selected = selected[:limit]

    collector = OfficialCollector(
        CollectorConfig(
            seeds_path=seeds_path,
            run_dir=output_run_dir,
            max_pages=max(1, len(selected)),
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
        )
    )

    documents: list[dict[str, object]] = []
    chunks: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    structured: dict[str, list[dict[str, object]]] = {name: [] for name in STRUCTURED_FILES}
    failures: list[str] = []
    reparsed_at = datetime.now(UTC).isoformat(timespec="seconds")

    for source_document in selected:
        source_url = str(source_document.get("canonical_url") or source_document.get("url") or "")
        raw_path_value = source_document.get("raw_path")
        if not source_url or not isinstance(raw_path_value, str) or not raw_path_value:
            failures.append(f"missing source url or raw_path for document {source_document.get('id')}")
            continue

        source_raw_path = source_run_dir / raw_path_value
        if not source_raw_path.exists():
            failures.append(f"{source_url}: missing raw file {source_raw_path}")
            continue

        body = source_raw_path.read_bytes()
        sha256 = hashlib.sha256(body).hexdigest()
        fetched = {
            "url": source_url,
            "body": body,
            "status_code": int(source_document.get("status_code") or 200),
            "content_type": str(source_document.get("content_type") or ""),
            "fetched_at": str(source_document.get("fetched_at") or reparsed_at),
            "sha256": sha256,
        }

        try:
            parsed = collector._parse_response(fetched, str(source_document.get("category") or "general"))
        except Exception as error:  # pragma: no cover - defensive around external parsers.
            failures.append(f"{source_url}: parse failed: {error}")
            continue

        document_id = len(documents) + 1
        raw_path = output_run_dir / "raw" / f"{sha256}{parsed['extension']}"
        if source_raw_path.resolve() != raw_path.resolve():
            shutil.copy2(source_raw_path, raw_path)
        text_path = output_run_dir / "texts" / f"{sha256}.txt"
        text_path.write_text(str(parsed["text"]), encoding="utf-8")

        document = {
            **source_document,
            "id": document_id,
            "run_id": output_run_dir.name,
            "url": source_url,
            "canonical_url": source_url,
            "title": parsed["title"] or source_document.get("title"),
            "category": parsed["category"],
            "language": parsed["language"],
            "raw_path": raw_path.relative_to(output_run_dir).as_posix(),
            "text_path": text_path.relative_to(output_run_dir).as_posix(),
            "sha256": sha256,
            "text_chars": len(str(parsed["text"])),
            "parser": parsed["parser"],
            "ocr_used": parsed["ocr_used"],
            "reparsed_at": reparsed_at,
            "reparsed_from_run": source_run_dir.name,
            "reparsed_from_document_id": source_document.get("id"),
        }
        flags = quality_flags(str(parsed["text"]), int(document.get("status_code") or 200), str(parsed["parser"]))
        flags.extend(str(flag) for flag in parsed["flags"])

        documents.append(document)
        manifest_rows.append(
            {
                "url": source_url,
                "canonical_url": source_url,
                "title": document.get("title") or "",
                "host": document.get("host") or "",
                "category": document.get("category") or "",
                "language": document.get("language") or "",
                "content_type": document.get("content_type") or "",
                "status_code": document.get("status_code") or "",
                "fetched_at": document.get("fetched_at") or "",
                "raw_path": document["raw_path"],
                "text_path": document["text_path"],
                "sha256": sha256,
                "depth": document.get("depth") or 0,
                "parent_url": document.get("parent_url") or "",
                "parser": document.get("parser") or "",
                "ocr_used": document.get("ocr_used") or False,
                "text_chars": document.get("text_chars") or 0,
                "quality_flags": flags,
            }
        )

        chunks.extend(
            {
                "id": len(chunks) + index + 1,
                **chunk,
            }
            for index, chunk in enumerate(
                iter_chunk_records(
                    document_id=document_id,
                    title=document.get("title") if isinstance(document.get("title"), str) else None,
                    url=source_url,
                    category=str(document.get("category") or "general"),
                    language=str(document.get("language") or "unknown"),
                    text=str(parsed["text"]),
                    max_chars=chunk_chars,
                    overlap=chunk_overlap,
                )
            )
        )

        extracted = extract_structured_records(document, str(parsed["text"]))
        for name, rows in extracted.items():
            structured[name].extend(rows)

    _write_reparse_outputs(output_run_dir, documents, chunks, manifest_rows, structured, failures)
    return {
        "source_documents": len(source_documents),
        "selected": len(selected),
        "documents": len(documents),
        "chunks": len(chunks),
        "manifest_rows": len(manifest_rows),
        "failed": len(failures),
    }


def _select_documents(
    documents: list[dict[str, object]],
    manifest_by_url: dict[str, dict[str, str]],
    only_flags: set[str] | None,
    url_filter: set[str] | None,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for document in documents:
        url = str(document.get("canonical_url") or document.get("url") or "")
        canonical_url = canonicalize_url(url) if url else ""
        if url_filter and canonical_url not in url_filter:
            continue
        if only_flags:
            row = manifest_by_url.get(canonical_url, {})
            flags = {flag for flag in row.get("quality_flags", "").split(";") if flag}
            if not flags.intersection(only_flags):
                continue
        selected.append(document)
    return selected


def _load_manifest_by_url(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            canonicalize_url(row.get("canonical_url") or row.get("url") or ""): row
            for row in csv.DictReader(handle)
            if row.get("canonical_url") or row.get("url")
        }


def _write_reparse_outputs(
    run_dir: Path,
    documents: list[dict[str, object]],
    chunks: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    structured: dict[str, list[dict[str, object]]],
    failures: list[str],
) -> None:
    jsonl_dir = run_dir / "jsonl"
    write_jsonl(jsonl_dir / "documents.jsonl", documents)
    write_jsonl(jsonl_dir / "chunks.jsonl", chunks)
    structured_counts = {
        f"{name}.jsonl": write_jsonl(jsonl_dir / f"{name}.jsonl", rows) for name, rows in structured.items()
    }
    write_jsonl(run_dir / "eval_seed_candidates.jsonl", build_eval_seed_candidates(documents))
    write_manifest(run_dir / "source_manifest.csv", manifest_rows)
    write_quality_report(run_dir, documents, manifest_rows, structured_counts, failures)
