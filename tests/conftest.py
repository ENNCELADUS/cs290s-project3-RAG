from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from rag.io import write_jsonl


def pytest_configure(config: pytest.Config) -> None:
    if _is_scoped_test_run(config):
        _disable_subset_coverage_gate(config)


def pytest_sessionstart(session: pytest.Session) -> None:
    if _is_scoped_test_run(session.config):
        _disable_subset_coverage_gate(session.config)


def _is_scoped_test_run(config: pytest.Config) -> bool:
    explicit_paths = [arg for arg in config.invocation_params.args if str(arg).startswith("tests/")]
    return bool(explicit_paths or config.option.markexpr)


def _disable_subset_coverage_gate(config: pytest.Config) -> None:
    # Full-suite pytest enforces coverage. Scoped runs keep the report but should stay usable during development.
    config.option.cov_fail_under = 0
    cov_plugin = config.pluginmanager.getplugin("_cov")
    if cov_plugin is not None and hasattr(cov_plugin, "options"):
        cov_plugin.options.cov_fail_under = 0


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    run_real_data = os.getenv("RAG_TEST_REAL_DATA") == "1"
    skip_real_data = pytest.mark.skip(reason="set RAG_TEST_REAL_DATA=1 to run real artifact checks")
    for item in items:
        path_parts = set(Path(str(item.fspath)).parts)
        if "unit" in path_parts:
            item.add_marker(pytest.mark.unit)
        if "integration" in path_parts:
            item.add_marker(pytest.mark.integration)
        if "real_data" in item.keywords and not run_real_data:
            item.add_marker(skip_real_data)


@pytest.fixture
def merged_input_dir(tmp_path: Path) -> Path:
    input_dir = tmp_path / "merged"
    input_dir.mkdir()
    write_jsonl(
        input_dir / "documents.jsonl",
        [
            {"id": 10, "url": "https://example.edu/a", "title": "A", "host": "example.edu"},
            {"id": 11, "url": "https://example.edu/b", "title": "B", "host": "example.edu"},
        ],
    )
    write_jsonl(
        input_dir / "chunks.jsonl",
        [
            {
                "id": 100,
                "document_id": 10,
                "chunk_index": 0,
                "title": "Deep Learning",
                "url": "https://example.edu/a",
                "text": "深度学习 任课老师 Alice",
                "char_count": 18,
            },
            {"id": 101, "document_id": 99, "chunk_index": 0, "text": "missing document", "char_count": 16},
            {"id": 102, "document_id": 11, "chunk_index": 0, "text": "", "char_count": 0},
            {"id": 103, "document_id": 11, "chunk_index": 1, "text": "\x00\x00\x00", "char_count": 3},
        ],
    )
    write_jsonl(input_dir / "courses.jsonl", [{"source_document_id": 10, "course_code": "CS181", "course_name": "AI"}])
    write_jsonl(
        input_dir / "faculty_members.jsonl",
        [{"source_document_id": 10, "name": "All"}, {"source_document_id": 10, "name": "Alice"}],
    )
    write_jsonl(input_dir / "program_requirements.jsonl", [])
    write_jsonl(input_dir / "events.jsonl", [])
    return input_dir


@pytest.fixture
def real_rag_artifacts() -> dict[str, Path]:
    paths = {
        "db": Path("data/rag/sist_merged_2026-05-27.sqlite"),
        "bm25": Path("data/rag/bm25_2026-05-27.pkl"),
        "faiss": Path("data/rag/faiss_bge_m3_2026-05-27.index"),
        "chunk_index": Path("data/rag/chunk_index_2026-05-27.jsonl"),
        "report": Path("data/rag/build_report_2026-05-27.json"),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        pytest.skip(f"missing real RAG artifacts: {', '.join(missing)}")
    return paths


@pytest.fixture
def fake_sentence_transformer_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    import sys
    import types

    import numpy as np

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            self.model_name = model_name
            self.device = device

        def encode(
            self,
            texts: list[str],
            batch_size: int,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
            show_progress_bar: bool,
        ) -> np.ndarray:
            return np.ones((len(texts), 3), dtype="float32")

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return fake_module


@pytest.fixture
def fake_hybrid_sentence_transformer_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    import sys
    import types

    import numpy as np

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            self.model_name = model_name
            self.device = device

        def encode(
            self,
            texts: list[str],
            batch_size: int,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
            show_progress_bar: bool,
        ) -> np.ndarray:
            vectors = []
            for text in texts:
                if text == "exact bridge query":
                    vectors.append([1.0, 0.0, 0.0])
                elif "dense winner" in text:
                    vectors.append([1.0, 0.0, 0.0])
                elif "bridge" in text:
                    vectors.append([0.95, 0.0, 0.0])
                else:
                    vectors.append([0.0, 0.1, 0.0])
            return np.asarray(vectors, dtype="float32")

    class FakeCrossEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            self.model_name = model_name
            self.device = device

        def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
            return np.asarray([10.0 if "dense winner" in passage else 1.0 for _, passage in pairs], dtype="float32")

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer, CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return fake_module
