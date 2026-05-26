from pathlib import Path

from rag_collection.cli import _load_known_urls
from rag_collection.io import prepare_run_dir, read_jsonl, write_jsonl, write_manifest
from rag_collection.merge import merge_existing_with_run
from rag_collection.reparse import reparse_run


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


def test_merge_excludes_documents_that_are_not_indexable(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    run = tmp_path / "run"
    output = tmp_path / "merged"
    existing.mkdir()
    run.mkdir()
    write_jsonl(existing / "documents.jsonl", [])
    write_jsonl(existing / "chunks.jsonl", [])
    write_jsonl(
        run / "documents.jsonl",
        [
            {
                "id": 1,
                "url": "https://sist.shanghaitech.edu.cn/image.jpg",
                "parser": "unsupported_binary",
                "text_chars": 0,
            },
            {"id": 2, "url": "https://sist.shanghaitech.edu.cn/course.htm", "parser": "html", "text_chars": 180},
        ],
    )
    write_jsonl(
        run / "chunks.jsonl",
        [{"id": 1, "document_id": 1, "text": ""}, {"id": 2, "document_id": 2, "text": "ok"}],
    )
    for filename in ["courses.jsonl", "faculty_members.jsonl", "program_requirements.jsonl", "events.jsonl"]:
        write_jsonl(existing / filename, [])
        write_jsonl(run / filename, [])

    stats = merge_existing_with_run(existing, run, output)

    merged_documents = read_jsonl(output / "documents.jsonl")
    merged_chunks = read_jsonl(output / "chunks.jsonl")
    assert stats["documents"] == 1
    assert merged_documents[0]["url"] == "https://sist.shanghaitech.edu.cn/course.htm"
    assert merged_chunks == [{"document_id": 1, "id": 1, "text": "ok"}]


def test_merge_keeps_best_run_document_for_duplicate_canonical_url(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    run = tmp_path / "run"
    output = tmp_path / "merged"
    existing.mkdir()
    run.mkdir()
    write_jsonl(existing / "documents.jsonl", [])
    write_jsonl(existing / "chunks.jsonl", [])
    write_jsonl(
        run / "documents.jsonl",
        [
            {"id": 1, "url": "https://sist.shanghaitech.edu.cn/", "canonical_url": "same", "text_chars": 20},
            {"id": 2, "url": "https://sist.shanghaitech.edu.cn/main.htm", "canonical_url": "same", "text_chars": 200},
        ],
    )
    write_jsonl(
        run / "chunks.jsonl",
        [{"id": 1, "document_id": 1, "text": "short"}, {"id": 2, "document_id": 2, "text": "long"}],
    )
    for filename in ["courses.jsonl", "faculty_members.jsonl", "program_requirements.jsonl", "events.jsonl"]:
        write_jsonl(existing / filename, [])
        write_jsonl(run / filename, [])

    merge_existing_with_run(existing, run, output)

    merged_documents = read_jsonl(output / "documents.jsonl")
    merged_chunks = read_jsonl(output / "chunks.jsonl")
    assert len(merged_documents) == 1
    assert merged_documents[0]["text_chars"] == 200
    assert merged_chunks == [{"document_id": 1, "id": 1, "text": "long"}]


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


def test_reparse_run_writes_append_only_outputs_for_flagged_docs(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.csv"
    seeds.write_text(
        "url,category,depth_limit,priority,notes\n"
        "https://www.shanghaitech.edu.cn/,school_info,1,1,test\n",
        encoding="utf-8",
    )
    source_run = tmp_path / "collection_runs" / "source"
    output_run = tmp_path / "collection_runs" / "reparse"
    for run_dir in (source_run, output_run):
        run_dir.joinpath("raw").mkdir(parents=True)
        run_dir.joinpath("texts").mkdir()
        run_dir.joinpath("jsonl").mkdir()

    raw_name = "sample.html"
    source_run.joinpath("raw", raw_name).write_bytes(
        b"<html><head><title>Fixed</title></head><body><main>Fixed parsed body with enough text.</main></body></html>"
    )
    write_jsonl(
        source_run / "jsonl" / "documents.jsonl",
        [
            {
                "id": 7,
                "run_id": "source",
                "url": "http://www.shanghaitech.edu.cn/test/main.htm",
                "canonical_url": "http://www.shanghaitech.edu.cn/test/main.htm",
                "title": "Old",
                "host": "www.shanghaitech.edu.cn",
                "category": "school_info",
                "language": "unknown",
                "content_type": "text/html",
                "status_code": 200,
                "fetched_at": "2026-05-26T00:00:00+00:00",
                "raw_path": f"raw/{raw_name}",
                "text_path": "texts/old.txt",
                "sha256": "old",
                "depth": 0,
                "parent_url": None,
                "text_chars": 0,
                "parser": "html",
                "ocr_used": False,
            }
        ],
    )
    write_manifest(
        source_run / "source_manifest.csv",
        [
            {
                "url": "https://www.shanghaitech.edu.cn/test/",
                "canonical_url": "https://www.shanghaitech.edu.cn/test/",
                "quality_flags": ["empty_text"],
            }
        ],
    )

    stats = reparse_run(source_run, output_run, seeds_path=seeds, only_flags={"empty_text"})

    documents = read_jsonl(output_run / "jsonl" / "documents.jsonl")
    chunks = read_jsonl(output_run / "jsonl" / "chunks.jsonl")
    assert stats["documents"] == 1
    assert documents[0]["title"] == "Fixed"
    assert documents[0]["canonical_url"] == "https://www.shanghaitech.edu.cn/test/"
    assert documents[0]["reparsed_from_run"] == "source"
    assert "Fixed parsed body with enough text." in chunks[0]["text"]
