from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag.app import (
    AppState,
    _build_sources_from_retrieval,
    _build_sources_html,
    _build_status_html,
    _format_full_result,
    _format_retrieval_result,
    handle_query,
)
from rag.generate import AnswerSource, RagAnswerer, RagAnswerResult, AnswerConfig, AnswerTiming
from rag.index import build_indexes
from rag.ingest import build_database
from rag.io import atomic_json_dump, write_jsonl
from rag.retrieve import HybridRetrievalResult, Retriever


class _FakeTensor:
    shape = (1, 3)

    def to(self, device: str) -> _FakeTensor:
        return self


class _FakeTokenizer:
    def __init__(self, generated_text: str) -> None:
        self.generated_text = generated_text

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, _FakeTensor]:
        return {"input_ids": _FakeTensor()}

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        return self.generated_text


class _FakeModel:
    def generate(self, **kwargs: object) -> list[list[int]]:
        return [[1, 2, 3, 4]]


def _build_test_artifacts(tmp_path: Path) -> dict[str, Path]:
    import faiss

    input_dir = tmp_path / "app-merged"
    input_dir.mkdir()
    write_jsonl(
        input_dir / "documents.jsonl",
        [
            {"id": 1, "url": "https://shanghaitech.edu.cn/about", "canonical_url": "https://shanghaitech.edu.cn/about", "title": "About ShanghaiTech"},
            {"id": 2, "url": "https://sist.shanghaitech.edu.cn/faculty", "canonical_url": "https://sist.shanghaitech.edu.cn/faculty", "title": "SIST Faculty"},
        ],
    )
    write_jsonl(
        input_dir / "chunks.jsonl",
        [
            {
                "id": 1,
                "document_id": 1,
                "chunk_index": 0,
                "title": "ShanghaiTech Overview",
                "url": "https://shanghaitech.edu.cn/about",
                "text": "ShanghaiTech University has six schools including SIST, SPST, SLST, SEM, SCA, and BME.",
                "char_count": 90,
            },
            {
                "id": 2,
                "document_id": 2,
                "chunk_index": 0,
                "title": "SIST Robotics Faculty",
                "url": "https://sist.shanghaitech.edu.cn/faculty",
                "text": "SIST has several faculty members working on robotics including Prof. Sören Schwertfeger.",
                "char_count": 90,
            },
        ],
    )
    write_jsonl(input_dir / "courses.jsonl", [])
    write_jsonl(input_dir / "faculty_members.jsonl", [])
    write_jsonl(input_dir / "program_requirements.jsonl", [])
    write_jsonl(input_dir / "events.jsonl", [])

    db_path = tmp_path / "rag.sqlite"
    bm25_path = tmp_path / "bm25.pkl"
    faiss_path = tmp_path / "faiss.index"
    chunk_index_path = tmp_path / "chunk_index.jsonl"
    report_path = tmp_path / "report.json"
    build_database(input_dir, db_path, report_path)
    build_indexes(db_path, bm25_path, faiss_path, chunk_index_path, report_path, skip_faiss=True)

    index = faiss.IndexFlatIP(3)
    index.add(np.asarray([[0.9, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(faiss_path))
    atomic_json_dump(
        report_path,
        {"index": {"faiss": {"model_path": "/models/bge-m3-local", "model_id": "BAAI/bge-m3"}}},
    )
    return {"db": db_path, "bm25": bm25_path, "faiss": faiss_path, "chunk_index": chunk_index_path, "report": report_path}


def _test_app_state(tmp_path: Path, *, model_path: Path | None = None) -> AppState:
    paths = _build_test_artifacts(tmp_path)
    return AppState(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
        model_path=model_path,
        device="cpu",
    )


class TestAppState:
    def test_init_with_artifacts_only(self, tmp_path: Path, fake_hybrid_sentence_transformer_module) -> None:
        state = _test_app_state(tmp_path)
        assert state.has_artifacts()
        assert not state.has_generator()
        assert state.mode_label == "retrieval-only"
        assert state.retriever is not None

    def test_init_with_missing_model_path_falls_back(self, tmp_path: Path, fake_hybrid_sentence_transformer_module) -> None:
        state = _test_app_state(tmp_path, model_path=tmp_path / "nonexistent-model")
        assert state.has_artifacts()
        assert not state.has_generator()
        assert "retrieval-only" in state.mode_label

    def test_init_with_existing_model_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_hybrid_sentence_transformer_module) -> None:
        from rag import app as app_module

        model_dir = tmp_path / "qwen-local"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        def _fake_load_model(self: RagAnswerer) -> tuple[object, object]:
            return object(), object()

        monkeypatch.setattr(app_module.RagAnswerer, "_load_model", _fake_load_model)

        state = _test_app_state(tmp_path, model_path=model_dir)
        assert state.has_artifacts()
        assert state.has_generator()
        assert state.mode_label == "full RAG"

    def test_init_with_missing_artifacts(self, tmp_path: Path) -> None:
        state = AppState(
            db_path=tmp_path / "nonexistent.sqlite",
            bm25_path=tmp_path / "nonexistent.pkl",
        )
        assert not state.has_artifacts()
        assert state.init_error is not None
        assert "RAG artifacts not found" in state.init_error

    def test_build_status_html_retrieval_only(self, tmp_path: Path, fake_hybrid_sentence_transformer_module) -> None:
        state = _test_app_state(tmp_path)
        html = _build_status_html(state)
        assert "retrieval-only" in html

    def test_build_status_html_full_rag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_hybrid_sentence_transformer_module) -> None:
        from rag import app as app_module

        model_dir = tmp_path / "qwen-local"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        def _fake_load_model(self: RagAnswerer) -> tuple[object, object]:
            return object(), object()

        monkeypatch.setattr(app_module.RagAnswerer, "_load_model", _fake_load_model)

        state = _test_app_state(tmp_path, model_path=model_dir)
        html = _build_status_html(state)
        assert "full RAG" in html


class TestHandleQuery:
    def test_retrieval_only_mode_returns_contexts(self, tmp_path: Path, fake_hybrid_sentence_transformer_module) -> None:
        state = _test_app_state(tmp_path)
        answer, sources, status = handle_query(
            "ShanghaiTech schools", "bm25", 3, True, state
        )
        assert "Retrieved" in answer
        assert "ShanghaiTech" in answer
        assert "source-table" in sources
        assert "retrieval-only" in status

    def test_retrieval_only_hybrid_mode(self, tmp_path: Path, fake_hybrid_sentence_transformer_module) -> None:
        state = _test_app_state(tmp_path)
        answer, sources, status = handle_query(
            "SIST robotics faculty", "hybrid", 5, True, state
        )
        assert "Retrieved" in answer
        assert "source-table" in sources

    def test_empty_query_returns_prompt(self, tmp_path: Path, fake_hybrid_sentence_transformer_module) -> None:
        state = _test_app_state(tmp_path)
        answer, sources, status = handle_query(
            "  ", "hybrid", 5, False, state
        )
        assert "Enter a question above" in answer

    def test_missing_artifacts_shows_error(self, tmp_path: Path) -> None:
        state = AppState(db_path=tmp_path / "nope.sqlite")
        answer, sources, status = handle_query(
            "test", "hybrid", 5, True, state
        )
        assert "Setup required" in answer

    def test_full_rag_mode_uses_generator(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_hybrid_sentence_transformer_module) -> None:
        from rag import app as app_module

        model_dir = tmp_path / "qwen-local"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        def _fake_load_model(self: RagAnswerer) -> tuple[object, object]:
            return _FakeTokenizer("ShanghaiTech has six schools [1]."), _FakeModel()

        monkeypatch.setattr(app_module.RagAnswerer, "_load_model", _fake_load_model)

        state = _test_app_state(tmp_path, model_path=model_dir)
        answer, sources, status = handle_query(
            "How many schools does ShanghaiTech have?", "hybrid", 3, False, state
        )
        assert "[1]" in answer
        assert "source-table" in sources
        assert "full RAG" in status

    def test_retrieval_only_checkbox_bypasses_generator(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_hybrid_sentence_transformer_module) -> None:
        from rag import app as app_module

        model_dir = tmp_path / "qwen-local"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        def _fake_load_model(self: RagAnswerer) -> tuple[object, object]:
            return _FakeTokenizer("Generated [1]."), _FakeModel()

        monkeypatch.setattr(app_module.RagAnswerer, "_load_model", _fake_load_model)

        state = _test_app_state(tmp_path, model_path=model_dir)
        answer, sources, status = handle_query(
            "SIST faculty", "hybrid", 3, True, state  # retrieval_only=True
        )
        assert "Retrieved" in answer
        assert "Generated" not in answer

    def test_handle_query_returns_error_on_exception(self, tmp_path: Path, fake_hybrid_sentence_transformer_module) -> None:
        state = _test_app_state(tmp_path)
        state.retriever = None
        state.init_error = "test error"
        answer, sources, status = handle_query(
            "test", "hybrid", 5, True, state
        )
        assert "Setup required" in answer


class TestFormatFunctions:
    def test_format_full_result_answered(self) -> None:
        result = RagAnswerResult(
            query="test",
            mode="hybrid",
            status="answered",
            answer="ShanghaiTech has six schools [1].",
            sources=[
                AnswerSource(source_id=1, title="About", url="https://example.com", chunk_id=1, document_id=1, trace_ref="x", snippet="ShanghaiTech..."),
            ],
            retrieval={"mode": "hybrid", "hits": [], "contexts": []},
            timing=AnswerTiming(retrieval_s=0.5, generation_s=1.0, total_s=1.5),
            config=AnswerConfig(model_path="/m", device="cpu", max_new_tokens=512, temperature=0.2, top_k=5),
        )
        output = _format_full_result(result)
        assert "[1]" in output
        assert "1.50s" in output

    def test_format_full_result_insufficient_evidence(self) -> None:
        result = RagAnswerResult(
            query="test",
            mode="hybrid",
            status="insufficient_evidence",
            answer="Evidence is insufficient.",
            sources=[],
            retrieval={"mode": "hybrid", "hits": [], "contexts": []},
            timing=AnswerTiming(retrieval_s=0.5, generation_s=0.0, total_s=0.5),
            config=AnswerConfig(model_path="/m", device="cpu", max_new_tokens=512, temperature=0.2, top_k=5),
        )
        output = _format_full_result(result)
        assert "Insufficient Evidence" in output

    def test_format_full_result_none(self) -> None:
        output = _format_full_result(None)
        assert "No answer generated" in output

    def test_format_retrieval_result_empty(self) -> None:
        output = _format_retrieval_result("no match", [], "bm25")
        assert "No results" in output

    def test_format_retrieval_result_none(self) -> None:
        output = _format_retrieval_result("test", None, "dense")
        assert "No results" in output

    def test_build_sources_html_empty(self) -> None:
        html = _build_sources_html([])
        assert "No sources" in html

    def test_build_sources_html(self) -> None:
        sources = [
            AnswerSource(source_id=1, title="About", url="https://example.com", chunk_id=1, document_id=1, trace_ref="x", snippet="Test snippet."),
        ]
        html = _build_sources_html(sources)
        assert "source-table" in html
        assert "About" in html
        assert "https://example.com" in html

    def test_build_sources_from_retrieval(self, tmp_path: Path, fake_hybrid_sentence_transformer_module) -> None:
        state = _test_app_state(tmp_path)
        result = state.retriever.retrieve("ShanghaiTech schools", mode="bm25", top_k=2)
        html = _build_sources_from_retrieval(result)
        assert "source-table" in html

    def test_build_sources_from_retrieval_none(self) -> None:
        html = _build_sources_from_retrieval(None)
        assert "No sources" in html
