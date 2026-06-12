import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rag.index import DEFAULT_MODEL, build_indexes
from rag.ingest import build_database
from rag.io import atomic_json_dump, read_jsonl, write_jsonl
from rag.retrieve import HybridRetrievalResult, Retriever, _dedupe_candidates
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


def test_bm25_index_uses_metadata_for_matching_but_returns_raw_chunk_text(tmp_path: Path) -> None:
    input_dir = _metadata_enrichment_input(tmp_path)
    db_path = tmp_path / "rag.sqlite"
    bm25_path = tmp_path / "bm25.pkl"
    chunk_index_path = tmp_path / "chunk_index.jsonl"
    report_path = tmp_path / "report.json"
    build_database(input_dir, db_path, report_path)
    build_indexes(
        db_path,
        bm25_path,
        tmp_path / "faiss.index",
        chunk_index_path,
        report_path,
        skip_faiss=True,
    )

    retriever = Retriever.from_paths(db_path=db_path, bm25_path=bm25_path)
    hits = retriever.retrieve("robotics", mode="bm25", top_k=1)
    contexts = retriever.contexts_for_hits(hits)

    assert hits[0].chunk_id == 100
    assert hits[0].snippet == "General lab introduction with contacts and office hours."
    assert contexts[0].text == "General lab introduction with contacts and office hours."


def test_faiss_index_uses_metadata_for_embeddings_but_returns_raw_chunk_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = _metadata_enrichment_input(tmp_path)
    db_path = tmp_path / "rag.sqlite"
    bm25_path = tmp_path / "bm25.pkl"
    faiss_path = tmp_path / "faiss.index"
    chunk_index_path = tmp_path / "chunk_index.jsonl"
    report_path = tmp_path / "report.json"
    build_database(input_dir, db_path, report_path)

    class MetadataAwareSentenceTransformer:
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
            vectors = [[1.0, 0.0, 0.0] if "catalog" in text.lower() else [0.0, 1.0, 0.0] for text in texts]
            return np.asarray(vectors, dtype="float32")

    fake_sentence_transformers = types.SimpleNamespace(SentenceTransformer=MetadataAwareSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)

    build_indexes(
        db_path,
        bm25_path,
        faiss_path,
        chunk_index_path,
        report_path,
        model_name="/models/hub/snapshots/bge-m3-local",
        model_id=DEFAULT_MODEL,
    )

    retriever = Retriever.from_paths(
        db_path=db_path,
        faiss_path=faiss_path,
        chunk_index_path=chunk_index_path,
        report_path=report_path,
    )
    hits = retriever.retrieve("catalog", mode="dense", top_k=1)

    assert hits[0].chunk_id == 101
    assert hits[0].snippet == "Degree planning notes with credits and prerequisites."


def test_retrieval_reports_missing_sqlite_without_creating_file(tmp_path: Path) -> None:
    db_path = tmp_path / "missing.sqlite"

    with pytest.raises(FileNotFoundError, match="SQLite database"):
        Retriever.from_paths(db_path=db_path, bm25_path=tmp_path / "bm25.pkl")

    assert not db_path.exists()


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


def test_retriever_reuses_dense_model_for_dense_and_hybrid_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_hybrid_artifacts(tmp_path)
    stats = {"construct_count": 0}

    class CountingSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            stats["construct_count"] += 1
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
            return _hybrid_test_vectors(texts)

    fake_module = types.SimpleNamespace(SentenceTransformer=CountingSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    retriever = Retriever.from_paths(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )

    dense_hits = retriever.retrieve("exact bridge query", mode="dense", top_k=2)
    hybrid_result = retriever.retrieve(
        "exact bridge query",
        mode="hybrid",
        top_k=2,
        sparse_top_k=2,
        dense_top_k=2,
        fused_top_k=3,
        rerank_top_k=0,
    )

    assert isinstance(hybrid_result, HybridRetrievalResult)
    assert stats == {"construct_count": 1}
    assert [hit.chunk_id for hit in dense_hits] == [101, 102]
    assert [hit.chunk_id for hit in hybrid_result.hits] == [102, 101]


def test_retriever_uses_separate_dense_model_for_different_report_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_hybrid_artifacts(tmp_path)
    first_model = tmp_path / "local-dense-a"
    second_model = tmp_path / "local-dense-b"
    first_model.mkdir()
    second_model.mkdir()
    constructed: list[tuple[str, str]] = []

    class CountingSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            constructed.append((model_name, device))
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
            return _hybrid_test_vectors(texts)

    fake_module = types.SimpleNamespace(SentenceTransformer=CountingSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    atomic_json_dump(
        paths["report"],
        {"index": {"faiss": {"model_path": str(first_model), "model_id": DEFAULT_MODEL}}},
    )
    retriever = Retriever.from_paths(
        db_path=paths["db"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )

    first_hits = retriever.retrieve("exact bridge query", mode="dense", top_k=2)
    atomic_json_dump(
        paths["report"],
        {"index": {"faiss": {"model_path": str(second_model), "model_id": DEFAULT_MODEL}}},
    )
    second_hits = retriever.retrieve("exact bridge query", mode="dense", top_k=2)

    assert constructed == [(str(first_model), "cpu"), (str(second_model), "cpu")]
    assert [hit.chunk_id for hit in first_hits] == [101, 102]
    assert [hit.chunk_id for hit in second_hits] == [101, 102]


def test_dense_retrieval_reports_missing_dense_index(tmp_path: Path, merged_input_dir: Path) -> None:
    db_path = tmp_path / "rag.sqlite"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)

    retriever = Retriever.from_paths(db_path=db_path)

    with pytest.raises(FileNotFoundError, match="faiss_path"):
        retriever.retrieve("SIST faculty robotics", mode="dense")


def test_dense_retrieval_reports_missing_chunk_index(
    tmp_path: Path, merged_input_dir: Path, fake_sentence_transformer_module
) -> None:
    import faiss

    db_path = tmp_path / "rag.sqlite"
    faiss_path = tmp_path / "faiss.index"
    missing_chunk_index_path = tmp_path / "missing_chunk_index.jsonl"
    report_path = tmp_path / "report.json"
    build_database(merged_input_dir, db_path, report_path)
    index = faiss.IndexFlatIP(3)
    index.add(np.ones((1, 3), dtype="float32"))
    faiss.write_index(index, str(faiss_path))
    atomic_json_dump(
        report_path,
        {"index": {"faiss": {"model_path": "/models/hub/snapshots/bge-m3-local", "model_id": DEFAULT_MODEL}}},
    )

    retriever = Retriever.from_paths(
        db_path=db_path,
        faiss_path=faiss_path,
        chunk_index_path=missing_chunk_index_path,
        report_path=report_path,
    )

    with pytest.raises(FileNotFoundError, match="chunk_index_path"):
        retriever.retrieve("SIST faculty robotics", mode="dense")


def test_hybrid_retrieval_rrf_fuses_sparse_and_dense(
    tmp_path: Path, fake_hybrid_sentence_transformer_module
) -> None:
    paths = _build_hybrid_artifacts(tmp_path)

    retriever = Retriever.from_paths(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )

    result = retriever.retrieve(
        "exact bridge query",
        mode="hybrid",
        top_k=3,
        sparse_top_k=2,
        dense_top_k=2,
        fused_top_k=3,
        rerank_top_k=2,
    )

    assert isinstance(result, HybridRetrievalResult)
    assert result.hits[0].chunk_id == 102
    assert result.hits[0].trace.sparse_rank is not None
    assert result.hits[0].trace.dense_rank == 2
    assert result.hits[0].trace.rerank_score is None
    assert result.contexts[0].trace_ref == result.hits[0].trace.trace_id


def test_hybrid_retrieval_preserves_sparse_matches_with_non_positive_bm25_scores(
    tmp_path: Path, fake_hybrid_sentence_transformer_module
) -> None:
    paths = _build_hybrid_artifacts(tmp_path)
    retriever = Retriever.from_paths(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )

    result = retriever.retrieve(
        "semantic",
        mode="hybrid",
        top_k=2,
        sparse_top_k=3,
        dense_top_k=2,
        fused_top_k=3,
        rerank_top_k=2,
    )

    assert isinstance(result, HybridRetrievalResult)
    traces_by_chunk = {hit.chunk_id: hit.trace for hit in result.hits}
    assert traces_by_chunk[101].sparse_rank is not None
    assert traces_by_chunk[101].sparse_score is not None
    assert traces_by_chunk[101].sparse_score <= 0


def test_hybrid_retrieval_defaults_weight_dense_rrf_above_sparse_rrf_at_same_rank(
    tmp_path: Path, fake_hybrid_sentence_transformer_module
) -> None:
    paths = _build_weighted_rrf_artifacts(tmp_path)
    retriever = Retriever.from_paths(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )

    result = retriever.retrieve(
        "exact bridge query",
        mode="hybrid",
        top_k=2,
        sparse_top_k=1,
        dense_top_k=1,
        fused_top_k=2,
        rerank_top_k=0,
    )

    assert isinstance(result, HybridRetrievalResult)
    assert result.config.sparse_weight == 1.0
    assert result.config.dense_weight == 1.5
    chunk_ids = [hit.chunk_id for hit in result.hits]
    assert chunk_ids.index(101) < chunk_ids.index(100)
    assert result.hits[chunk_ids.index(101)].trace.dense_rank == 1
    assert result.hits[chunk_ids.index(101)].trace.sparse_rank is None
    assert result.hits[chunk_ids.index(100)].trace.sparse_rank == 1
    assert result.hits[chunk_ids.index(100)].trace.dense_rank is None


def test_hybrid_cli_json_includes_hits_contexts_and_config(
    tmp_path: Path, fake_hybrid_sentence_transformer_module, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _build_hybrid_artifacts(tmp_path)

    exit_code = retrieve_main(
        [
            "--query",
            "exact bridge query",
            "--mode",
            "hybrid",
            "--db",
            str(paths["db"]),
            "--bm25",
            str(paths["bm25"]),
            "--faiss",
            str(paths["faiss"]),
            "--chunk-index",
            str(paths["chunk_index"]),
            "--report",
            str(paths["report"]),
            "--sparse-top-k",
            "2",
            "--dense-top-k",
            "2",
            "--rerank-preserve-top-k",
            "1",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert set(payload) == {"config", "contexts", "hits", "mode", "query"}
    assert payload["mode"] == "hybrid"
    assert payload["hits"][0]["trace"]["rrf_score"] > 0
    assert payload["contexts"][0]["text"]
    assert payload["config"]["sparse_top_k"] == 2
    assert payload["config"]["rerank_preserve_top_k"] == 1
    assert payload["config"]["sparse_weight"] == 1.0
    assert payload["config"]["dense_weight"] == 1.5


def test_hybrid_deduplicates_text_and_caps_canonical_url() -> None:
    chunks_by_id = {
        1: {"chunk_id": 1, "canonical_url": "https://example.edu/a", "url": "https://example.edu/a", "text": "same"},
        2: {"chunk_id": 2, "canonical_url": "https://example.edu/a", "url": "https://example.edu/a", "text": "same"},
        3: {"chunk_id": 3, "canonical_url": "https://example.edu/a", "url": "https://example.edu/a", "text": "second"},
        4: {"chunk_id": 4, "canonical_url": "https://example.edu/a", "url": "https://example.edu/a", "text": "third"},
        5: {"chunk_id": 5, "canonical_url": "https://example.edu/b", "url": "https://example.edu/b", "text": "other"},
    }
    candidates = [
        {"chunk_id": chunk_id, "rrf_score": 1.0 / chunk_id}
        for chunk_id in [1, 2, 3, 4, 5]
    ]

    selected = _dedupe_candidates(candidates, chunks_by_id, final_top_k=5, url_cap=2)

    assert [candidate["chunk_id"] for candidate in selected] == [1, 3, 5]


def test_hybrid_deduplicates_source_keys_by_default() -> None:
    chunks_by_id = {
        1: {
            "chunk_id": 1,
            "document_id": 10,
            "canonical_url": "https://sist.shanghaitech.edu.cn/_t335/2026/0327/c2863a1120270/page.htm",
            "url": "https://sist.shanghaitech.edu.cn/_t335/2026/0327/c2863a1120270/page.htm",
            "text": "first article chunk",
        },
        2: {
            "chunk_id": 2,
            "document_id": 11,
            "canonical_url": "https://sist.shanghaitech.edu.cn/2026/0327/c7339a1120270/page.htm",
            "url": "https://sist.shanghaitech.edu.cn/2026/0327/c7339a1120270/page.htm",
            "text": "same article different column",
        },
        3: {
            "chunk_id": 3,
            "document_id": 12,
            "canonical_url": "https://sist.shanghaitech.edu.cn/2026/0327/c7339a1120271/page.htm",
            "url": "https://sist.shanghaitech.edu.cn/2026/0327/c7339a1120271/page.htm",
            "text": "different article",
        },
        4: {
            "chunk_id": 4,
            "document_id": 12,
            "canonical_url": "https://sist.shanghaitech.edu.cn/2026/0327/c7339a1120271/page.htm",
            "url": "https://sist.shanghaitech.edu.cn/2026/0327/c7339a1120271/page.htm",
            "text": "same document second chunk",
        },
        5: {
            "chunk_id": 5,
            "document_id": 13,
            "canonical_url": "https://example.edu/other",
            "url": "https://example.edu/other",
            "text": "external source",
        },
    }
    candidates = [
        {"chunk_id": chunk_id, "rrf_score": 1.0 / chunk_id}
        for chunk_id in [1, 2, 3, 4, 5]
    ]

    selected = _dedupe_candidates(candidates, chunks_by_id, final_top_k=5, url_cap=1)

    assert [candidate["chunk_id"] for candidate in selected] == [1, 3, 5]


def test_hybrid_retrieval_reports_missing_dense_artifacts(tmp_path: Path, merged_input_dir: Path) -> None:
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

    retriever = Retriever.from_paths(db_path=db_path, bm25_path=bm25_path)

    with pytest.raises(FileNotFoundError, match="faiss_path"):
        retriever.retrieve("exact bridge query", mode="hybrid")


def test_hybrid_reranker_reorders_with_local_model(
    tmp_path: Path, fake_hybrid_sentence_transformer_module
) -> None:
    paths = _build_hybrid_artifacts(tmp_path)
    reranker_model = tmp_path / "local-reranker"
    reranker_model.mkdir()
    retriever = Retriever.from_paths(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )

    result = retriever.retrieve(
        "exact bridge query",
        mode="hybrid",
        top_k=2,
        sparse_top_k=2,
        dense_top_k=2,
        fused_top_k=3,
        rerank_top_k=2,
        reranker_model=str(reranker_model),
    )

    assert isinstance(result, HybridRetrievalResult)
    assert result.hits[0].chunk_id == 101
    assert result.hits[0].trace.rerank_score == 10.0


def test_hybrid_reranker_can_preserve_fused_prefix(
    tmp_path: Path, fake_hybrid_sentence_transformer_module
) -> None:
    paths = _build_hybrid_artifacts(tmp_path)
    reranker_model = tmp_path / "local-reranker"
    reranker_model.mkdir()
    retriever = Retriever.from_paths(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )

    result = retriever.retrieve(
        "exact bridge query",
        mode="hybrid",
        top_k=3,
        sparse_top_k=2,
        dense_top_k=2,
        fused_top_k=3,
        rerank_top_k=3,
        rerank_preserve_top_k=1,
        reranker_model=str(reranker_model),
    )

    assert isinstance(result, HybridRetrievalResult)
    assert [hit.chunk_id for hit in result.hits] == [102, 101, 100]
    assert result.hits[0].trace.rerank_score is None
    assert result.hits[1].trace.rerank_score == 10.0


def test_hybrid_reranker_skips_prediction_when_preserved_prefix_covers_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_hybrid_artifacts(tmp_path)
    reranker_model = tmp_path / "local-reranker"
    reranker_model.mkdir()

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

    class UnexpectedCrossEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            raise AssertionError("CrossEncoder should not load when the preserved prefix covers the rerank window")

    fake_module = types.SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer,
        CrossEncoder=UnexpectedCrossEncoder,
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    retriever = Retriever.from_paths(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )

    result = retriever.retrieve(
        "exact bridge query",
        mode="hybrid",
        top_k=3,
        sparse_top_k=2,
        dense_top_k=2,
        fused_top_k=3,
        rerank_top_k=3,
        rerank_preserve_top_k=3,
        reranker_model=str(reranker_model),
    )

    assert isinstance(result, HybridRetrievalResult)
    assert [hit.chunk_id for hit in result.hits] == [102, 101, 100]
    assert all(hit.trace.rerank_score is None for hit in result.hits)


def test_hybrid_reranker_reuses_local_model_for_same_retriever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_hybrid_artifacts(tmp_path)
    reranker_model = tmp_path / "local-reranker"
    reranker_model.mkdir()
    other_reranker_model = tmp_path / "other-local-reranker"
    other_reranker_model.mkdir()
    stats = {"construct_count": 0, "predict_count": 0}

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

    class CountingCrossEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            stats["construct_count"] += 1
            self.model_name = model_name
            self.device = device

        def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
            stats["predict_count"] += 1
            return np.asarray([10.0 if "dense winner" in passage else 1.0 for _, passage in pairs], dtype="float32")

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer, CrossEncoder=CountingCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    retriever = Retriever.from_paths(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )

    first_result = retriever.retrieve(
        "exact bridge query",
        mode="hybrid",
        top_k=2,
        sparse_top_k=2,
        dense_top_k=2,
        fused_top_k=3,
        rerank_top_k=2,
        reranker_model=str(reranker_model),
    )
    second_result = retriever.retrieve(
        "exact bridge query",
        mode="hybrid",
        top_k=2,
        sparse_top_k=2,
        dense_top_k=2,
        fused_top_k=3,
        rerank_top_k=2,
        reranker_model=str(reranker_model),
    )

    assert isinstance(first_result, HybridRetrievalResult)
    assert isinstance(second_result, HybridRetrievalResult)
    assert stats == {"construct_count": 1, "predict_count": 2}
    assert [hit.chunk_id for hit in first_result.hits] == [101, 102]
    assert [hit.chunk_id for hit in second_result.hits] == [101, 102]

    other_result = retriever.retrieve(
        "exact bridge query",
        mode="hybrid",
        top_k=2,
        sparse_top_k=2,
        dense_top_k=2,
        fused_top_k=3,
        rerank_top_k=2,
        reranker_model=str(other_reranker_model),
    )

    assert isinstance(other_result, HybridRetrievalResult)
    assert stats == {"construct_count": 2, "predict_count": 3}
    assert [hit.chunk_id for hit in other_result.hits] == [101, 102]


def test_hybrid_reranker_reports_missing_local_model(
    tmp_path: Path, fake_hybrid_sentence_transformer_module
) -> None:
    paths = _build_hybrid_artifacts(tmp_path)
    retriever = Retriever.from_paths(
        db_path=paths["db"],
        bm25_path=paths["bm25"],
        faiss_path=paths["faiss"],
        chunk_index_path=paths["chunk_index"],
        report_path=paths["report"],
    )

    with pytest.raises(FileNotFoundError, match="Reranker model path"):
        retriever.retrieve(
            "exact bridge query",
            mode="hybrid",
            sparse_top_k=2,
            dense_top_k=2,
            reranker_model=str(tmp_path / "missing-reranker"),
        )


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
    assert set(payload) == {"hits", "mode", "query"}
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


def _metadata_enrichment_input(tmp_path: Path) -> Path:
    input_dir = tmp_path / "metadata-enrichment-merged"
    input_dir.mkdir()
    write_jsonl(
        input_dir / "documents.jsonl",
        [
            {
                "id": 10,
                "url": "https://example.edu/research/robotics-lab",
                "canonical_url": "https://example.edu/research/robotics-lab",
                "title": "Robotics Laboratory",
                "host": "example.edu",
            },
            {
                "id": 11,
                "url": "https://example.edu/academics/course-catalog",
                "canonical_url": "https://example.edu/academics/course-catalog",
                "title": "Course Catalog",
                "host": "example.edu",
            },
            {
                "id": 12,
                "url": "https://example.edu/admissions/calendar",
                "canonical_url": "https://example.edu/admissions/calendar",
                "title": "Admissions Calendar",
                "host": "example.edu",
            },
        ],
    )
    write_jsonl(
        input_dir / "chunks.jsonl",
        [
            {
                "id": 100,
                "document_id": 10,
                "chunk_index": 0,
                "title": "Robotics Laboratory",
                "url": "https://example.edu/research/robotics-lab",
                "category": "Research",
                "text": "General lab introduction with contacts and office hours.",
                "char_count": 54,
            },
            {
                "id": 101,
                "document_id": 11,
                "chunk_index": 0,
                "title": "Course Catalog",
                "url": "https://example.edu/academics/course-catalog",
                "category": "Academics",
                "text": "Degree planning notes with credits and prerequisites.",
                "char_count": 51,
            },
            {
                "id": 102,
                "document_id": 12,
                "chunk_index": 0,
                "title": "Admissions Calendar",
                "url": "https://example.edu/admissions/calendar",
                "category": "Admissions",
                "text": "Application schedule details with registration reminders.",
                "char_count": 57,
            },
        ],
    )
    write_jsonl(input_dir / "courses.jsonl", [])
    write_jsonl(input_dir / "faculty_members.jsonl", [])
    write_jsonl(input_dir / "program_requirements.jsonl", [])
    write_jsonl(input_dir / "events.jsonl", [])
    return input_dir


def _build_hybrid_artifacts(tmp_path: Path) -> dict[str, Path]:
    import faiss

    input_dir = tmp_path / "hybrid-merged"
    input_dir.mkdir()
    write_jsonl(
        input_dir / "documents.jsonl",
        [
            {"id": 10, "url": "https://example.edu/a", "canonical_url": "https://example.edu/a", "title": "A"},
            {"id": 11, "url": "https://example.edu/b", "canonical_url": "https://example.edu/b", "title": "B"},
            {"id": 12, "url": "https://example.edu/c", "canonical_url": "https://example.edu/c", "title": "C"},
        ],
    )
    write_jsonl(
        input_dir / "chunks.jsonl",
        [
            {
                "id": 100,
                "document_id": 10,
                "chunk_index": 0,
                "title": "Sparse Source",
                "url": "https://example.edu/a",
                "text": "sparse sparse exact source",
                "char_count": 26,
            },
            {
                "id": 101,
                "document_id": 11,
                "chunk_index": 0,
                "title": "Dense Winner",
                "url": "https://example.edu/b",
                "text": "dense winner semantic source",
                "char_count": 28,
            },
            {
                "id": 102,
                "document_id": 12,
                "chunk_index": 0,
                "title": "Bridge Source",
                "url": "https://example.edu/c",
                "text": "sparse bridge dense semantic",
                "char_count": 28,
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
    index.add(
        np.asarray(
            [
                [0.0, 0.1, 0.0],
                [1.0, 0.0, 0.0],
                [0.95, 0.0, 0.0],
            ],
            dtype="float32",
        )
    )
    faiss.write_index(index, str(faiss_path))
    atomic_json_dump(
        report_path,
        {"index": {"faiss": {"model_path": "/models/hub/snapshots/bge-m3-local", "model_id": DEFAULT_MODEL}}},
    )
    return {
        "db": db_path,
        "bm25": bm25_path,
        "faiss": faiss_path,
        "chunk_index": chunk_index_path,
        "report": report_path,
    }


def _build_weighted_rrf_artifacts(tmp_path: Path) -> dict[str, Path]:
    import faiss

    input_dir = tmp_path / "weighted-rrf-merged"
    input_dir.mkdir()
    write_jsonl(
        input_dir / "documents.jsonl",
        [
            {"id": 10, "url": "https://example.edu/a", "canonical_url": "https://example.edu/a", "title": "A"},
            {"id": 11, "url": "https://example.edu/b", "canonical_url": "https://example.edu/b", "title": "B"},
        ],
    )
    write_jsonl(
        input_dir / "chunks.jsonl",
        [
            {
                "id": 100,
                "document_id": 10,
                "chunk_index": 0,
                "title": "Sparse Source",
                "url": "https://example.edu/a",
                "text": "exact bridge query sparse source",
                "char_count": 31,
            },
            {
                "id": 101,
                "document_id": 11,
                "chunk_index": 0,
                "title": "Dense Winner",
                "url": "https://example.edu/b",
                "text": "dense winner semantic source",
                "char_count": 28,
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
    index.add(np.asarray([[0.0, 0.1, 0.0], [1.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(faiss_path))
    atomic_json_dump(
        report_path,
        {"index": {"faiss": {"model_path": "/models/hub/snapshots/bge-m3-local", "model_id": DEFAULT_MODEL}}},
    )
    return {
        "db": db_path,
        "bm25": bm25_path,
        "faiss": faiss_path,
        "chunk_index": chunk_index_path,
        "report": report_path,
    }


def _hybrid_test_vectors(texts: list[str]) -> np.ndarray:
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
