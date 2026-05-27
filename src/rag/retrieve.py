from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .index import DEFAULT_BM25, DEFAULT_CHUNK_INDEX, DEFAULT_DB, DEFAULT_FAISS, DEFAULT_REPORT, _tokenize
from .io import read_jsonl

RetrievalMode = Literal["bm25", "dense"]


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    chunk_id: int
    document_id: int
    title: str | None
    url: str | None
    category: str | None
    language: str | None
    score: float
    snippet: str
    mode: RetrievalMode


class Retriever:
    def __init__(
        self,
        db_path: Path,
        bm25_path: Path | None = None,
        faiss_path: Path | None = None,
        chunk_index_path: Path | None = None,
        report_path: Path | None = None,
        dense_model: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.bm25_path = bm25_path
        self.faiss_path = faiss_path
        self.chunk_index_path = chunk_index_path
        self.report_path = report_path
        self.dense_model = dense_model
        self._chunks = _load_chunks(db_path)
        self._chunks_by_id = {int(row["chunk_id"]): row for row in self._chunks}

    @classmethod
    def from_paths(
        cls,
        *,
        db_path: Path,
        bm25_path: Path | None = None,
        faiss_path: Path | None = None,
        chunk_index_path: Path | None = None,
        report_path: Path | None = None,
        dense_model: str | None = None,
    ) -> Retriever:
        return cls(
            db_path=db_path,
            bm25_path=bm25_path,
            faiss_path=faiss_path,
            chunk_index_path=chunk_index_path,
            report_path=report_path,
            dense_model=dense_model,
        )

    def retrieve(self, query: str, *, mode: RetrievalMode, top_k: int = 5) -> list[RetrievalHit]:
        if mode == "bm25":
            return self._retrieve_bm25(query, top_k)
        if mode == "dense":
            return self._retrieve_dense(query, top_k)
        raise ValueError(f"unsupported retrieval mode: {mode}")

    def _retrieve_bm25(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self.bm25_path is None:
            raise FileNotFoundError("BM25 retrieval requires a bm25_path")
        with self.bm25_path.open("rb") as handle:
            payload = pickle.load(handle)
        scores = payload["bm25"].get_scores(_tokenize(query))
        chunk_ids = [int(chunk_id) for chunk_id in payload["chunk_ids"]]
        top_indices = np.argsort(scores)[::-1][:top_k]
        hits: list[RetrievalHit] = []
        for rank, index in enumerate(top_indices, start=1):
            chunk_id = chunk_ids[int(index)]
            row = self._chunks_by_id[chunk_id]
            hits.append(_hit_from_row(row, rank=rank, score=float(scores[int(index)]), mode="bm25"))
        return hits

    def _retrieve_dense(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self.faiss_path is None:
            raise FileNotFoundError("Dense retrieval requires a faiss_path")
        if self.chunk_index_path is None:
            raise FileNotFoundError("Dense retrieval requires a chunk_index_path")
        model_path = self.dense_model or _dense_model_from_report(self.report_path)

        _allow_duplicate_openmp_on_macos()
        import faiss
        from sentence_transformers import SentenceTransformer

        index = faiss.read_index(str(self.faiss_path))
        chunk_mapping = read_jsonl(self.chunk_index_path)
        model = SentenceTransformer(model_path, device="cpu")
        embedding = model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector = np.asarray(embedding, dtype="float32")
        scores, indices = index.search(vector, top_k)
        hits: list[RetrievalHit] = []
        for rank, index_position in enumerate(indices[0], start=1):
            if int(index_position) < 0:
                continue
            chunk_id = int(chunk_mapping[int(index_position)]["chunk_id"])
            row = self._chunks_by_id[chunk_id]
            hits.append(_hit_from_row(row, rank=rank, score=float(scores[0][rank - 1]), mode="dense"))
        return hits


def _load_chunks(db_path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                id AS chunk_id,
                document_id,
                title,
                url,
                category,
                language,
                text
            FROM chunks
            ORDER BY id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _hit_from_row(row: dict[str, object], *, rank: int, score: float, mode: RetrievalMode) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        chunk_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        title=_optional_str(row.get("title")),
        url=_optional_str(row.get("url")),
        category=_optional_str(row.get("category")),
        language=_optional_str(row.get("language")),
        score=score,
        snippet=str(row["text"])[:240],
        mode=mode,
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _dense_model_from_report(report_path: Path | None) -> str:
    if report_path is None:
        raise FileNotFoundError("Dense retrieval requires --dense-model or a build report with index.faiss.model_path")
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    model_path = dict(dict(report.get("index", {})).get("faiss", {})).get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise FileNotFoundError("Dense retrieval requires --dense-model or a build report with index.faiss.model_path")
    return model_path


def _allow_duplicate_openmp_on_macos() -> None:
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cited retrieval over existing RAG artifacts.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--faiss", type=Path, default=DEFAULT_FAISS)
    parser.add_argument("--chunk-index", type=Path, default=DEFAULT_CHUNK_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    args = parser.parse_args(argv)

    retriever = Retriever.from_paths(
        db_path=args.db,
        bm25_path=args.bm25,
        faiss_path=args.faiss,
        chunk_index_path=args.chunk_index,
        report_path=args.report,
        dense_model=args.dense_model,
    )
    hits = retriever.retrieve(args.query, mode=args.mode, top_k=args.top_k)
    if args.json:
        print(
            json.dumps(
                {"query": args.query, "mode": args.mode, "hits": [asdict(hit) for hit in hits]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    for hit in hits:
        title = hit.title or "(untitled)"
        print(f"{hit.rank}. {title}")
        print(f"   url={hit.url or ''}")
        print(f"   chunk_id={hit.chunk_id} score={hit.score:.6g}")
        print(f"   snippet={hit.snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
