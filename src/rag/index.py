from __future__ import annotations

import argparse
import json
import pickle
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .io import atomic_json_dump, write_jsonl

DEFAULT_DB = Path("data/rag/sist_merged_2026-05-27.sqlite")
DEFAULT_BM25 = Path("data/rag/bm25_2026-05-27.pkl")
DEFAULT_FAISS = Path("data/rag/faiss_bge_m3_2026-05-27.index")
DEFAULT_CHUNK_INDEX = Path("data/rag/chunk_index_2026-05-27.jsonl")
DEFAULT_REPORT = Path("data/rag/build_report_2026-05-27.json")
DEFAULT_MODEL = "BAAI/bge-m3"
SMOKE_QUERIES = ["深度学习 任课老师", "计算机科学与技术 毕业 学分", "SIST faculty robotics"]
TOKEN_RE = re.compile(r"[A-Za-z0-9_./+-]+")


def build_indexes(
    db_path: Path,
    bm25_path: Path,
    faiss_path: Path,
    chunk_index_path: Path,
    report_path: Path,
    model_name: str = DEFAULT_MODEL,
    model_id: str = DEFAULT_MODEL,
    batch_size: int = 32,
    max_seq_length: int | None = 512,
    require_cuda: bool = False,
    skip_bm25: bool = False,
    skip_faiss: bool = False,
) -> dict[str, object]:
    chunks = _load_chunks(db_path)
    chunk_index_rows = [
        {
            "row_index": index,
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "title": row["title"],
            "url": row["url"],
            "category": row["category"],
            "language": row["language"],
            "char_count": row["char_count"],
        }
        for index, row in enumerate(chunks)
    ]
    write_jsonl(chunk_index_path, chunk_index_rows)

    if skip_bm25:
        if not bm25_path.exists():
            raise FileNotFoundError(f"cannot skip BM25 because {bm25_path} does not exist")
        bm25_report = {"path": str(bm25_path), "chunk_count": len(chunks), "status": "reused"}
    else:
        bm25_report = _build_bm25(chunks, bm25_path)
    smoke_results = _smoke_bm25(chunks, bm25_path)
    faiss_report: dict[str, object]
    if skip_faiss:
        faiss_report = {"status": "skipped"}
    else:
        faiss_report = _build_faiss(chunks, faiss_path, model_name, model_id, batch_size, max_seq_length, require_cuda)

    report = {
        "built_at": datetime.now(UTC).isoformat(),
        "sqlite_path": str(db_path),
        "chunk_count": len(chunks),
        "chunk_index_path": str(chunk_index_path),
        "bm25": bm25_report,
        "faiss": faiss_report,
        "smoke_queries": smoke_results,
    }
    existing_report = _load_report(report_path)
    atomic_json_dump(report_path, {**existing_report, "index": report})
    return report


def _load_chunks(db_path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.title,
                c.url,
                c.category,
                c.language,
                c.text,
                c.char_count,
                d.host
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    tokens.extend(token.strip().lower() for token in jieba.lcut(text) if token.strip())
    return tokens


def _build_bm25(chunks: list[dict[str, object]], output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenized = [_tokenize(str(row["text"])) for row in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "chunk_ids": [int(row["chunk_id"]) for row in chunks],
        "config": {
            "tokenizer": "regex_ascii_plus_jieba",
            "chunk_order": "chunks.id ASC",
        },
    }
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle)
    tmp_path.replace(output_path)
    return {"path": str(output_path), "chunk_count": len(chunks)}


def _smoke_bm25(chunks: list[dict[str, object]], bm25_path: Path, top_k: int = 5) -> list[dict[str, object]]:
    with bm25_path.open("rb") as handle:
        payload = pickle.load(handle)
    bm25: BM25Okapi = payload["bm25"]
    results: list[dict[str, object]] = []
    for query in SMOKE_QUERIES:
        scores = bm25.get_scores(_tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        hits = []
        for index in top_indices:
            row = chunks[int(index)]
            hits.append(
                {
                    "chunk_id": row["chunk_id"],
                    "score": float(scores[int(index)]),
                    "title": row["title"],
                    "url": row["url"],
                    "snippet": str(row["text"])[:240],
                }
            )
        results.append({"query": query, "hits": hits})
    return results


def _build_faiss(
    chunks: list[dict[str, object]],
    output_path: Path,
    model_name: str,
    model_id: str,
    batch_size: int,
    max_seq_length: int | None,
    require_cuda: bool,
) -> dict[str, object]:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required for FAISS dense index build, but torch.cuda.is_available() is false")
    device = "cuda" if cuda_available else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    texts = [str(row["text"]) for row in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(embeddings, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise RuntimeError(f"embedding shape {vectors.shape} does not match chunk count {len(chunks)}")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    faiss.write_index(index, str(tmp_path))
    tmp_path.replace(output_path)
    return {
        "path": str(output_path),
        "model": model_id,
        "model_id": model_id,
        "model_path": model_name,
        "device": device,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "chunk_count": len(chunks),
        "dimension": int(vectors.shape[1]),
    }


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indexes from the RAG SQLite database.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--bm25-output", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--faiss-output", type=Path, default=DEFAULT_FAISS)
    parser.add_argument("--chunk-index-output", type=Path, default=DEFAULT_CHUNK_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--skip-faiss", action="store_true")
    args = parser.parse_args(argv)

    report = build_indexes(
        args.db,
        args.bm25_output,
        args.faiss_output,
        args.chunk_index_output,
        args.report,
        model_name=args.model,
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        require_cuda=args.require_cuda,
        skip_bm25=args.skip_bm25,
        skip_faiss=args.skip_faiss,
    )
    print(f"chunk_count={report['chunk_count']}")
    print(f"chunk_index_path={args.chunk_index_output}")
    print(f"bm25_path={args.bm25_output}")
    print(f"faiss_status={dict(report['faiss']).get('status', 'built')}")
    if dict(report["faiss"]).get("path"):
        print(f"faiss_path={dict(report['faiss'])['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
