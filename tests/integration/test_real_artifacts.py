from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pytest

from rag.index import _tokenize
from rag.io import read_jsonl

pytestmark = pytest.mark.real_data


def test_real_sqlite_artifact_has_expected_counts_and_clean_foreign_keys(real_rag_artifacts: dict[str, Path]) -> None:
    expected_counts = {
        "documents": 7190,
        "chunks": 28481,
        "courses": 707,
        "faculty_members": 489,
        "program_requirements": 1777,
        "events": 5199,
    }
    with sqlite3.connect(real_rag_artifacts["db"]) as conn:
        actual_counts = {
            table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in expected_counts
        }
        foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()

    assert actual_counts == expected_counts
    assert foreign_key_errors == []


def test_real_faiss_artifact_matches_chunk_mapping(real_rag_artifacts: dict[str, Path]) -> None:
    index = faiss.read_index(str(real_rag_artifacts["faiss"]))
    chunk_mapping = read_jsonl(real_rag_artifacts["chunk_index"])

    assert index.ntotal == len(chunk_mapping) == 28481
    assert chunk_mapping[0]["row_index"] == 0


def test_real_bm25_artifact_returns_cited_smoke_hits(real_rag_artifacts: dict[str, Path]) -> None:
    with real_rag_artifacts["bm25"].open("rb") as handle:
        payload = pickle.load(handle)
    scores = payload["bm25"].get_scores(_tokenize("SIST faculty robotics"))
    top_indices = np.argsort(scores)[::-1][:5]
    chunk_ids = [payload["chunk_ids"][int(index)] for index in top_indices]

    placeholders = ",".join("?" for _ in chunk_ids)
    with sqlite3.connect(real_rag_artifacts["db"]) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT id, title, url, substr(text, 1, 120) AS snippet FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()

    assert len(rows) == 5
    assert all(row["url"] for row in rows)
    assert any(row["snippet"] for row in rows)
