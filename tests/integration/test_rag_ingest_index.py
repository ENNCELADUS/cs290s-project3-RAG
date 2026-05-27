import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rag.index import DEFAULT_MODEL, build_indexes
from rag.ingest import build_database
from rag.io import atomic_json_dump, read_jsonl, write_jsonl
from rag.retrieve import Retriever
from rag.retrieve import main as retrieve_main


def test_build_database_preserves_ids_and_filters_invalid_chunks(tmp_path: Path, merged_input_dir: Path) -> None:
    db_path = tmp_path / "rag.sqlite"
    report_path = tmp_path / "report.json"

    report = build_database(merged_input_dir, db_path, report_path)

    assert report["output_rows"]["documents"] == 2
    assert report["output_rows"]["chunks"] == 1
    assert report["output_rows"]["faculty_members"] == 1
    assert report["filtered_rows"]["chunks"]["missing_document"] == 1
    assert report["filtered_rows"]["chunks"]["empty_text"] == 2
    assert report["foreign_key_errors"] == []


def test_failed_database_rebuild_preserves_existing_sqlite(tmp_path: Path, merged_input_dir: Path) -> None:
    db_path = tmp_path / "rag.sqlite"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)

    bad_input_dir = tmp_path / "bad-merged"
    bad_input_dir.mkdir()
    for input_file in merged_input_dir.glob("*.jsonl"):
        write_jsonl(bad_input_dir / input_file.name, read_jsonl(input_file))
    write_jsonl(
        bad_input_dir / "chunks.jsonl",
        [
            {
                "id": 100,
                "document_id": 10,
                "chunk_index": 0,
                "text": "valid text",
                "char_count": "not-an-int",
            }
        ],
    )

    with pytest.raises(ValueError):
        build_database(bad_input_dir, db_path, report_path)

    rebuilt_report = build_indexes(
        db_path,
        tmp_path / "bm25.pkl",
        tmp_path / "faiss.index",
        tmp_path / "chunk_index.jsonl",
        report_path,
        skip_faiss=True,
    )
    assert rebuilt_report["chunk_count"] == 1


def test_build_bm25_index_returns_stable_chunk_ids(tmp_path: Path, merged_input_dir: Path) -> None:
    db_path = tmp_path / "rag.sqlite"
    bm25_path = tmp_path / "bm25.pkl"
    chunk_index_path = tmp_path / "chunk_index.jsonl"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)

    report = build_indexes(
        db_path,
        bm25_path,
        tmp_path / "faiss.index",
        chunk_index_path,
        report_path,
        skip_faiss=True,
    )

    assert report["bm25"]["chunk_count"] == 1
    assert report["faiss"]["status"] == "skipped"
    assert read_jsonl(chunk_index_path)[0]["chunk_id"] == 100
    deep_learning_hits = report["smoke_queries"][0]["hits"]
    assert deep_learning_hits[0]["chunk_id"] == 100


def test_bm25_retrieval_returns_cited_chunks(tmp_path: Path, merged_input_dir: Path) -> None:
    db_path = tmp_path / "rag.sqlite"
    bm25_path = tmp_path / "bm25.pkl"
    chunk_index_path = tmp_path / "chunk_index.jsonl"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)
    build_indexes(
        db_path,
        bm25_path,
        tmp_path / "faiss.index",
        chunk_index_path,
        report_path,
        skip_faiss=True,
    )

    retriever = Retriever.from_paths(db_path=db_path, bm25_path=bm25_path)
    hits = retriever.retrieve("深度学习 任课老师", mode="bm25", top_k=1)

    assert hits[0].rank == 1
    assert hits[0].chunk_id == 100
    assert hits[0].title == "Deep Learning"
    assert hits[0].url == "https://example.edu/a"
    assert hits[0].mode == "bm25"
    assert "Alice" in hits[0].snippet


def test_faiss_mapping_length_matches_vector_count(
    tmp_path: Path, merged_input_dir: Path, fake_sentence_transformer_module
) -> None:
    db_path = tmp_path / "rag.sqlite"
    faiss_path = tmp_path / "faiss.index"
    chunk_index_path = tmp_path / "chunk_index.jsonl"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)

    report = build_indexes(
        db_path,
        tmp_path / "bm25.pkl",
        faiss_path,
        chunk_index_path,
        report_path,
        model_name="/models/hub/snapshots/bge-m3-local",
        model_id=DEFAULT_MODEL,
        batch_size=2,
    )

    import faiss

    index = faiss.read_index(str(faiss_path))
    mapping = read_jsonl(chunk_index_path)
    assert index.ntotal == len(mapping) == 1
    assert report["faiss"]["model_id"] == DEFAULT_MODEL
    assert report["faiss"]["model_path"] == "/models/hub/snapshots/bge-m3-local"


def test_dense_retrieval_returns_cited_chunks(
    tmp_path: Path, merged_input_dir: Path, fake_sentence_transformer_module
) -> None:
    import faiss

    db_path = tmp_path / "rag.sqlite"
    faiss_path = tmp_path / "faiss.index"
    chunk_index_path = tmp_path / "chunk_index.jsonl"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)
    index = faiss.IndexFlatIP(3)
    index.add(np.ones((1, 3), dtype="float32"))
    faiss.write_index(index, str(faiss_path))
    write_jsonl(
        chunk_index_path,
        [
            {
                "row_index": 0,
                "chunk_id": 100,
                "document_id": 10,
                "title": "Deep Learning",
                "url": "https://example.edu/a",
                "category": None,
                "language": None,
                "char_count": 18,
            }
        ],
    )
    atomic_json_dump(
        report_path,
        {"index": {"faiss": {"model_path": "/models/hub/snapshots/bge-m3-local", "model_id": DEFAULT_MODEL}}},
    )

    retriever = Retriever.from_paths(
        db_path=db_path,
        faiss_path=faiss_path,
        chunk_index_path=chunk_index_path,
        report_path=report_path,
    )
    hits = retriever.retrieve("SIST faculty robotics", mode="dense", top_k=1)

    assert hits[0].rank == 1
    assert hits[0].chunk_id == 100
    assert hits[0].url == "https://example.edu/a"
    assert hits[0].mode == "dense"


def test_dense_retrieval_reports_missing_dense_index(tmp_path: Path, merged_input_dir: Path) -> None:
    db_path = tmp_path / "rag.sqlite"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)

    retriever = Retriever.from_paths(db_path=db_path)

    with pytest.raises(FileNotFoundError, match="faiss_path"):
        retriever.retrieve("SIST faculty robotics", mode="dense")


def test_retrieval_cli_prints_cited_bm25_hits(
    tmp_path: Path, merged_input_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "rag.sqlite"
    bm25_path = tmp_path / "bm25.pkl"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)
    build_indexes(
        db_path,
        bm25_path,
        tmp_path / "faiss.index",
        tmp_path / "chunk_index.jsonl",
        report_path,
        skip_faiss=True,
    )

    exit_code = retrieve_main(
        ["--query", "深度学习 任课老师", "--mode", "bm25", "--db", str(db_path), "--bm25", str(bm25_path)]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "1. Deep Learning" in output
    assert "https://example.edu/a" in output
    assert "score=" in output


def test_retrieval_cli_outputs_json_hits(
    tmp_path: Path, merged_input_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "rag.sqlite"
    bm25_path = tmp_path / "bm25.pkl"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)
    build_indexes(
        db_path,
        bm25_path,
        tmp_path / "faiss.index",
        tmp_path / "chunk_index.jsonl",
        report_path,
        skip_faiss=True,
    )

    exit_code = retrieve_main(
        [
            "--query",
            "深度学习 任课老师",
            "--mode",
            "bm25",
            "--db",
            str(db_path),
            "--bm25",
            str(bm25_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["query"] == "深度学习 任课老师"
    assert payload["mode"] == "bm25"
    assert payload["hits"][0]["chunk_id"] == 100
    assert payload["hits"][0]["url"] == "https://example.edu/a"


def test_require_cuda_fails_before_dense_index_build_when_cuda_is_unavailable(
    tmp_path: Path, monkeypatch, merged_input_dir: Path
) -> None:
    db_path = tmp_path / "rag.sqlite"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class UnexpectedSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            raise AssertionError("SentenceTransformer should not load when CUDA is required but unavailable")

    fake_sentence_transformers = types.SimpleNamespace(SentenceTransformer=UnexpectedSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        build_indexes(
            db_path,
            tmp_path / "bm25.pkl",
            tmp_path / "faiss.index",
            tmp_path / "chunk_index.jsonl",
            report_path,
            require_cuda=True,
        )
