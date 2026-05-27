from __future__ import annotations

from pathlib import Path

from .io import read_jsonl, write_jsonl

STRUCTURED_FILES = ["courses.jsonl", "faculty_members.jsonl", "program_requirements.jsonl", "events.jsonl"]
INVALID_FACULTY_NAMES = {
    "faculty",
    "people",
    "home",
    "homepage",
    "introduction",
    "profile",
    "teacher",
    "teachers",
    "师资队伍",
    "常任教授",
    "特聘教授",
    "研究人员",
    "支撑人员",
    "行政人员",
}
NON_INDEXABLE_PARSERS = {
    "unsupported_binary",
    "doc_unsupported",
    "xls_unsupported",
    "ppt_unsupported",
    "office_unsupported",
}


def merge_existing_with_run(existing_jsonl_dir: Path, run_jsonl_dir: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    existing_documents = read_jsonl(existing_jsonl_dir / "documents.jsonl")
    run_documents = _dedupe_run_documents(
        [document for document in read_jsonl(run_jsonl_dir / "documents.jsonl") if _is_indexable(document)]
    )
    run_url_keys = {_document_url_key(document) for document in run_documents}
    kept_existing_documents = [
        document for document in existing_documents if _document_url_key(document) not in run_url_keys
    ]

    existing_id_map: dict[int, int] = {}
    document_id_map: dict[int, int] = {}
    merged_documents: list[dict[str, object]] = []

    for document in kept_existing_documents:
        old_id = int(document.get("id", len(merged_documents) + 1))
        new_id = len(merged_documents) + 1
        existing_id_map[old_id] = new_id
        merged_documents.append({**document, "id": new_id})

    normalized_run_documents: list[dict[str, object]] = []
    for document in run_documents:
        old_id = int(document.get("id", len(normalized_run_documents) + 1))
        new_id = len(merged_documents) + len(normalized_run_documents) + 1
        document_id_map[old_id] = new_id
        normalized = {**document, "id": new_id}
        normalized_run_documents.append(normalized)

    merged_documents.extend(normalized_run_documents)
    stats["documents"] = write_jsonl(output_dir / "documents.jsonl", merged_documents)

    existing_chunks = read_jsonl(existing_jsonl_dir / "chunks.jsonl")
    existing_chunks = [
        _remap_document_id(row, existing_id_map)
        for row in existing_chunks
        if int(row.get("document_id", 0)) in existing_id_map
    ]
    run_chunks = [
        _remap_document_id(row, document_id_map)
        for row in read_jsonl(run_jsonl_dir / "chunks.jsonl")
        if int(row.get("document_id", 0)) in document_id_map
    ]
    merged_chunks = [*existing_chunks, *run_chunks]
    for index, chunk in enumerate(merged_chunks, start=1):
        chunk["id"] = index
    stats["chunks"] = write_jsonl(output_dir / "chunks.jsonl", merged_chunks)

    for filename in STRUCTURED_FILES:
        existing_rows = [
            _remap_source_document_id(row, existing_id_map)
            for row in read_jsonl(existing_jsonl_dir / filename)
            if _keeps_structured_row(filename, row, existing_id_map)
        ]
        run_rows = [
            _remap_source_document_id(row, document_id_map)
            for row in read_jsonl(run_jsonl_dir / filename)
            if _keeps_structured_row(filename, row, document_id_map)
        ]
        stats[filename] = write_jsonl(output_dir / filename, [*existing_rows, *run_rows])

    return stats


def _remap_document_id(row: dict[str, object], document_id_map: dict[int, int]) -> dict[str, object]:
    document_id = int(row.get("document_id", 0))
    return {**row, "document_id": document_id_map.get(document_id, document_id)}


def _remap_source_document_id(row: dict[str, object], document_id_map: dict[int, int]) -> dict[str, object]:
    source_document_id = row.get("source_document_id")
    if source_document_id is None:
        return row
    old_id = int(source_document_id)
    return {**row, "source_document_id": document_id_map.get(old_id, old_id)}


def _keeps_structured_row(filename: str, row: dict[str, object], document_id_map: dict[int, int]) -> bool:
    source_document_id = row.get("source_document_id")
    if source_document_id is not None and int(source_document_id) not in document_id_map:
        return False
    return _has_required_structured_fields(filename, row)


def _has_required_structured_fields(filename: str, row: dict[str, object]) -> bool:
    if filename == "courses.jsonl":
        return bool(str(row.get("course_code") or "").strip()) and (
            bool(str(row.get("course_name") or "").strip()) or row.get("credits") is not None
        )
    if filename == "faculty_members.jsonl":
        name = str(row.get("name") or "").strip()
        return bool(name) and name.lower() not in INVALID_FACULTY_NAMES
    if filename == "program_requirements.jsonl":
        return bool(str(row.get("requirement_text") or "").strip()) and bool(str(row.get("evidence") or "").strip())
    if filename == "events.jsonl":
        return bool(str(row.get("title") or "").strip()) and bool(str(row.get("published_at") or "").strip())
    return True


def _document_url_key(document: dict[str, object]) -> str:
    return str(document.get("canonical_url") or document.get("url") or "")


def _is_indexable(document: dict[str, object]) -> bool:
    parser = str(document.get("parser") or "")
    text_chars = int(document.get("text_chars") or 0)
    return parser not in NON_INDEXABLE_PARSERS and (not parser or text_chars > 0)


def _dedupe_run_documents(documents: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_url: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for document in documents:
        key = _document_url_key(document)
        if key not in best_by_url:
            best_by_url[key] = document
            order.append(key)
            continue
        if int(document.get("text_chars") or 0) > int(best_by_url[key].get("text_chars") or 0):
            best_by_url[key] = document
    return [best_by_url[key] for key in order]
