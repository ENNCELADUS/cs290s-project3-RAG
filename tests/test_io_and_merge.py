from pathlib import Path

from rag_collection.cli import _load_known_urls
from rag_collection.io import prepare_run_dir, read_jsonl, write_jsonl
from rag_collection.merge import merge_existing_with_run


def test_prepare_run_dir_is_append_only(tmp_path: Path) -> None:
    first = prepare_run_dir(tmp_path, "2026-05-26")
    second = prepare_run_dir(tmp_path, "2026-05-26")

    assert first.name == "2026-05-26"
    assert second.name == "2026-05-26_001"
    assert first.joinpath("raw").is_dir()
    assert second.joinpath("jsonl").is_dir()


def test_merge_offsets_run_document_ids(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    run = tmp_path / "run"
    output = tmp_path / "merged"
    existing.mkdir()
    run.mkdir()
    write_jsonl(existing / "documents.jsonl", [{"id": 10, "url": "old"}])
    write_jsonl(existing / "chunks.jsonl", [{"id": 1, "document_id": 10, "text": "old"}])
    write_jsonl(run / "documents.jsonl", [{"id": 1, "url": "new"}])
    write_jsonl(run / "chunks.jsonl", [{"id": 1, "document_id": 1, "text": "new"}])
    for filename in ["courses.jsonl", "faculty_members.jsonl", "program_requirements.jsonl", "events.jsonl"]:
        write_jsonl(existing / filename, [])
        write_jsonl(run / filename, [])

    stats = merge_existing_with_run(existing, run, output)

    assert stats["documents"] == 2
    merged_documents = read_jsonl(output / "documents.jsonl")
    merged_chunks = read_jsonl(output / "chunks.jsonl")
    assert merged_documents[1]["id"] == 2
    assert merged_chunks[1]["document_id"] == 2


def test_merge_prefers_run_document_for_same_url(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    run = tmp_path / "run"
    output = tmp_path / "merged"
    existing.mkdir()
    run.mkdir()
    write_jsonl(existing / "documents.jsonl", [{"id": 10, "url": "same", "canonical_url": "same"}])
    write_jsonl(existing / "chunks.jsonl", [{"id": 1, "document_id": 10, "text": "old"}])
    write_jsonl(run / "documents.jsonl", [{"id": 1, "url": "same", "canonical_url": "same"}])
    write_jsonl(run / "chunks.jsonl", [{"id": 1, "document_id": 1, "text": "new"}])
    for filename in ["courses.jsonl", "faculty_members.jsonl", "program_requirements.jsonl", "events.jsonl"]:
        write_jsonl(existing / filename, [])
        write_jsonl(run / filename, [])

    merge_existing_with_run(existing, run, output)

    merged_documents = read_jsonl(output / "documents.jsonl")
    merged_chunks = read_jsonl(output / "chunks.jsonl")
    assert len(merged_documents) == 1
    assert merged_chunks[0]["text"] == "new"


def test_load_known_urls_ignores_current_run(tmp_path: Path) -> None:
    existing = tmp_path / "data" / "jsonl"
    previous_run = tmp_path / "data" / "collection_runs" / "previous" / "jsonl"
    current_run = tmp_path / "data" / "collection_runs" / "current"
    current_jsonl = current_run / "jsonl"
    existing.mkdir(parents=True)
    previous_run.mkdir(parents=True)
    current_jsonl.mkdir(parents=True)
    write_jsonl(existing / "documents.jsonl", [{"canonical_url": "https://old.example/"}])
    write_jsonl(previous_run / "documents.jsonl", [{"url": "https://previous.example/"}])
    write_jsonl(current_jsonl / "documents.jsonl", [{"url": "https://current.example/"}])

    known_urls = _load_known_urls(existing, tmp_path / "data" / "collection_runs", current_run)

    assert known_urls == {"https://old.example/", "https://previous.example/"}
