from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .index import DEFAULT_BM25, DEFAULT_CHUNK_INDEX, DEFAULT_DB, DEFAULT_FAISS, DEFAULT_REPORT, _tokenize
from .io import read_jsonl
from .source_urls import normalize_url, sist_article_id

BaselineRetrievalMode = Literal["bm25", "dense"]
RetrievalMode = Literal["bm25", "dense", "hybrid"]
HYBRID_MODE: Literal["hybrid"] = "hybrid"
DEFAULT_SPARSE_TOP_K = 20
DEFAULT_DENSE_TOP_K = 20
DEFAULT_FUSED_TOP_K = 20
DEFAULT_RERANK_TOP_K = 10
DEFAULT_RRF_K = 60
DEFAULT_SPARSE_WEIGHT = 1.0
DEFAULT_DENSE_WEIGHT = 1.5
DEFAULT_URL_CAP = 1
SNIPPET_CHARS = 240
WHITESPACE_RE = re.compile(r"\s+")


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
    mode: BaselineRetrievalMode


@dataclass(frozen=True)
class HybridRetrievalConfig:
    sparse_top_k: int = DEFAULT_SPARSE_TOP_K
    dense_top_k: int = DEFAULT_DENSE_TOP_K
    fused_top_k: int = DEFAULT_FUSED_TOP_K
    rerank_top_k: int = DEFAULT_RERANK_TOP_K
    rerank_preserve_top_k: int = 0
    final_top_k: int = 5
    rrf_k: int = DEFAULT_RRF_K
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT
    dense_weight: float = DEFAULT_DENSE_WEIGHT
    reranker_model: str | None = None
    url_cap: int = DEFAULT_URL_CAP


@dataclass(frozen=True)
class RetrievalTrace:
    trace_id: str
    chunk_id: int
    sparse_rank: int | None
    sparse_score: float | None
    dense_rank: int | None
    dense_score: float | None
    rrf_score: float
    rerank_score: float | None
    final_rank: int | None


@dataclass(frozen=True)
class OptimizedRetrievalHit:
    rank: int
    chunk_id: int
    document_id: int
    title: str | None
    url: str | None
    canonical_url: str | None
    category: str | None
    language: str | None
    score: float
    rrf_score: float
    rerank_score: float | None
    snippet: str
    mode: Literal["hybrid"]
    trace: RetrievalTrace


@dataclass(frozen=True)
class ContextItem:
    rank: int
    chunk_id: int
    document_id: int
    title: str | None
    url: str | None
    category: str | None
    language: str | None
    snippet: str
    text: str
    trace_ref: str


@dataclass(frozen=True)
class HybridRetrievalResult:
    query: str
    mode: Literal["hybrid"]
    hits: list[OptimizedRetrievalHit]
    contexts: list[ContextItem]
    config: HybridRetrievalConfig


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
        self._reranker_models: dict[str, Any] = {}

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

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode,
        top_k: int = 5,
        sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        fused_top_k: int = DEFAULT_FUSED_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        rerank_preserve_top_k: int = 0,
        rrf_k: int = DEFAULT_RRF_K,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        reranker_model: str | None = None,
        url_cap: int = DEFAULT_URL_CAP,
    ) -> list[RetrievalHit] | HybridRetrievalResult:
        if mode == "bm25":
            return self._retrieve_bm25(query, top_k)
        if mode == "dense":
            return self._retrieve_dense(query, top_k)
        if mode == "hybrid":
            config = HybridRetrievalConfig(
                sparse_top_k=sparse_top_k,
                dense_top_k=dense_top_k,
                fused_top_k=fused_top_k,
                rerank_top_k=rerank_top_k,
                rerank_preserve_top_k=rerank_preserve_top_k,
                final_top_k=top_k,
                rrf_k=rrf_k,
                sparse_weight=sparse_weight,
                dense_weight=dense_weight,
                reranker_model=reranker_model,
                url_cap=url_cap,
            )
            return self._retrieve_hybrid(query, config)
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
        if not self.faiss_path.exists():
            raise FileNotFoundError(f"Dense retrieval FAISS index does not exist: {self.faiss_path}")
        if not self.chunk_index_path.exists():
            raise FileNotFoundError(f"Dense retrieval chunk_index_path does not exist: {self.chunk_index_path}")
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

    def _retrieve_hybrid(self, query: str, config: HybridRetrievalConfig) -> HybridRetrievalResult:
        sparse_hits = self._retrieve_bm25_matching(query, config.sparse_top_k)
        dense_hits = self._retrieve_dense(query, config.dense_top_k)
        fused = _reciprocal_rank_fuse(
            sparse_hits,
            dense_hits,
            config.rrf_k,
            sparse_weight=config.sparse_weight,
            dense_weight=config.dense_weight,
        )[: config.fused_top_k]
        preserve_top_k = max(0, min(config.rerank_preserve_top_k, config.rerank_top_k))
        reranked = _rerank_candidates(
            query,
            fused[preserve_top_k : config.rerank_top_k],
            self._chunks_by_id,
            config.reranker_model,
            self._reranker_models,
        )
        ordered = [*fused[:preserve_top_k], *reranked, *fused[config.rerank_top_k :]]
        selected = _dedupe_candidates(
            ordered,
            self._chunks_by_id,
            final_top_k=config.final_top_k,
            url_cap=config.url_cap,
        )

        hits: list[OptimizedRetrievalHit] = []
        contexts: list[ContextItem] = []
        for final_rank, candidate in enumerate(selected, start=1):
            chunk_id = candidate["chunk_id"]
            row = self._chunks_by_id[chunk_id]
            trace = _trace_from_candidate(candidate, final_rank=final_rank)
            score = trace.rerank_score if trace.rerank_score is not None else trace.rrf_score
            hits.append(_optimized_hit_from_row(row, rank=final_rank, score=score, trace=trace))
            contexts.append(_context_from_row(row, rank=final_rank, trace_ref=trace.trace_id))
        return HybridRetrievalResult(query=query, mode=HYBRID_MODE, hits=hits, contexts=contexts, config=config)

    def contexts_for_hits(self, hits: list[RetrievalHit]) -> list[ContextItem]:
        return [
            _context_from_row(
                self._chunks_by_id[hit.chunk_id],
                rank=hit.rank,
                trace_ref=f"{hit.mode}:chunk:{hit.chunk_id}",
            )
            for hit in hits
        ]

    def _retrieve_bm25_matching(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self.bm25_path is None:
            raise FileNotFoundError("BM25 retrieval requires a bm25_path")
        with self.bm25_path.open("rb") as handle:
            payload = pickle.load(handle)
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []
        bm25 = payload["bm25"]
        scores = bm25.get_scores(list(query_tokens))
        chunk_ids = [int(chunk_id) for chunk_id in payload["chunk_ids"]]
        matching_indices = [
            int(index)
            for index in np.argsort(scores)[::-1]
            if _bm25_document_matches_query(bm25, int(index), query_tokens)
        ]
        hits: list[RetrievalHit] = []
        for rank, index in enumerate(matching_indices[:top_k], start=1):
            chunk_id = chunk_ids[index]
            row = self._chunks_by_id[chunk_id]
            hits.append(_hit_from_row(row, rank=rank, score=float(scores[index]), mode="bm25"))
        return hits


def _load_chunks(db_path: Path) -> list[dict[str, object]]:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {db_path}")
    sqlite_uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(sqlite_uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.title,
                c.url,
                d.canonical_url AS canonical_url,
                c.category,
                c.language,
                c.text
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _hit_from_row(row: dict[str, object], *, rank: int, score: float, mode: BaselineRetrievalMode) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        chunk_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        title=_optional_str(row.get("title")),
        url=_optional_str(row.get("url")),
        category=_optional_str(row.get("category")),
        language=_optional_str(row.get("language")),
        score=score,
        snippet=str(row["text"])[:SNIPPET_CHARS],
        mode=mode,
    )


def _optimized_hit_from_row(
    row: dict[str, object], *, rank: int, score: float, trace: RetrievalTrace
) -> OptimizedRetrievalHit:
    return OptimizedRetrievalHit(
        rank=rank,
        chunk_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        title=_optional_str(row.get("title")),
        url=_optional_str(row.get("url")),
        canonical_url=_optional_str(row.get("canonical_url")),
        category=_optional_str(row.get("category")),
        language=_optional_str(row.get("language")),
        score=score,
        rrf_score=trace.rrf_score,
        rerank_score=trace.rerank_score,
        snippet=str(row["text"])[:SNIPPET_CHARS],
        mode=HYBRID_MODE,
        trace=trace,
    )


def _context_from_row(row: dict[str, object], *, rank: int, trace_ref: str) -> ContextItem:
    return ContextItem(
        rank=rank,
        chunk_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        title=_optional_str(row.get("title")),
        url=_optional_str(row.get("url")),
        category=_optional_str(row.get("category")),
        language=_optional_str(row.get("language")),
        snippet=str(row["text"])[:SNIPPET_CHARS],
        text=str(row["text"]),
        trace_ref=trace_ref,
    )


def _reciprocal_rank_fuse(
    sparse_hits: list[RetrievalHit],
    dense_hits: list[RetrievalHit],
    rrf_k: int,
    *,
    sparse_weight: float,
    dense_weight: float,
) -> list[dict[str, object]]:
    candidates: dict[int, dict[str, object]] = {}
    for hit in sparse_hits:
        candidate = candidates.setdefault(
            hit.chunk_id,
            {
                "chunk_id": hit.chunk_id,
                "sparse_rank": None,
                "sparse_score": None,
                "dense_rank": None,
                "dense_score": None,
                "rrf_score": 0.0,
                "rerank_score": None,
            },
        )
        candidate["sparse_rank"] = hit.rank
        candidate["sparse_score"] = hit.score
        candidate["rrf_score"] = float(candidate["rrf_score"]) + sparse_weight / (rrf_k + hit.rank)
    for hit in dense_hits:
        candidate = candidates.setdefault(
            hit.chunk_id,
            {
                "chunk_id": hit.chunk_id,
                "sparse_rank": None,
                "sparse_score": None,
                "dense_rank": None,
                "dense_score": None,
                "rrf_score": 0.0,
                "rerank_score": None,
            },
        )
        candidate["dense_rank"] = hit.rank
        candidate["dense_score"] = hit.score
        candidate["rrf_score"] = float(candidate["rrf_score"]) + dense_weight / (rrf_k + hit.rank)
    return sorted(candidates.values(), key=_candidate_sort_key)


def _candidate_sort_key(candidate: dict[str, object]) -> tuple[float, int, int]:
    ranks = [rank for rank in (candidate["sparse_rank"], candidate["dense_rank"]) if isinstance(rank, int)]
    best_rank = min(ranks) if ranks else sys.maxsize
    return (-float(candidate["rrf_score"]), best_rank, int(candidate["chunk_id"]))


def _bm25_document_matches_query(bm25: object, index: int, query_tokens: set[str]) -> bool:
    doc_freqs = getattr(bm25, "doc_freqs", None)
    if not isinstance(doc_freqs, list) or index >= len(doc_freqs):
        return True
    return any(token in doc_freqs[index] for token in query_tokens)


def _rerank_candidates(
    query: str,
    candidates: list[dict[str, object]],
    chunks_by_id: dict[int, dict[str, object]],
    reranker_model: str | None,
    reranker_models: dict[str, Any],
) -> list[dict[str, object]]:
    if reranker_model is None or not candidates:
        return candidates
    model_path = Path(reranker_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Reranker model path does not exist: {model_path}")

    _allow_duplicate_openmp_on_macos()
    from sentence_transformers import CrossEncoder

    model_key = str(model_path.resolve())
    model = reranker_models.get(model_key)
    if model is None:
        model = CrossEncoder(str(model_path), device="cpu")
        reranker_models[model_key] = model
    pairs = [(query, str(chunks_by_id[int(candidate["chunk_id"])]["text"])) for candidate in candidates]
    scores = model.predict(pairs)
    scored: list[dict[str, object]] = []
    for candidate, score in zip(candidates, scores, strict=True):
        scored.append({**candidate, "rerank_score": float(score)})
    return sorted(scored, key=lambda candidate: (-float(candidate["rerank_score"]), _candidate_sort_key(candidate)))


def _dedupe_candidates(
    candidates: list[dict[str, object]],
    chunks_by_id: dict[int, dict[str, object]],
    *,
    final_top_k: int,
    url_cap: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_text_hashes: set[str] = set()
    counts_by_source_key: dict[str, int] = {}
    for candidate in candidates:
        row = chunks_by_id[int(candidate["chunk_id"])]
        text_hash = _normalized_text_hash(str(row["text"]))
        if text_hash in seen_text_hashes:
            continue
        source_keys = _source_keys(row)
        if any(counts_by_source_key.get(key, 0) >= url_cap for key in source_keys):
            continue
        selected.append(candidate)
        seen_text_hashes.add(text_hash)
        for key in source_keys:
            counts_by_source_key[key] = counts_by_source_key.get(key, 0) + 1
        if len(selected) >= final_top_k:
            break
    return selected


def _normalized_text_hash(text: str) -> str:
    normalized = WHITESPACE_RE.sub(" ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _source_keys(row: dict[str, object]) -> set[str]:
    keys: set[str] = set()
    url = _optional_str(row.get("canonical_url")) or _optional_str(row.get("url"))
    if url is not None:
        normalized_url = normalize_url(url)
        keys.add(f"url:{normalized_url}")
        article_id = sist_article_id(normalized_url)
        if article_id is not None:
            keys.add(f"sist_article:{article_id}")
    document_id = _optional_int(row.get("document_id"))
    if document_id is not None:
        keys.add(f"document:{document_id}")
    return keys


def _trace_from_candidate(candidate: dict[str, object], *, final_rank: int) -> RetrievalTrace:
    chunk_id = int(candidate["chunk_id"])
    return RetrievalTrace(
        trace_id=f"chunk:{chunk_id}",
        chunk_id=chunk_id,
        sparse_rank=_optional_int(candidate.get("sparse_rank")),
        sparse_score=_optional_float(candidate.get("sparse_score")),
        dense_rank=_optional_int(candidate.get("dense_rank")),
        dense_score=_optional_float(candidate.get("dense_score")),
        rrf_score=float(candidate["rrf_score"]),
        rerank_score=_optional_float(candidate.get("rerank_score")),
        final_rank=final_rank,
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


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
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sparse-top-k", type=int, default=DEFAULT_SPARSE_TOP_K)
    parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--fused-top-k", type=int, default=DEFAULT_FUSED_TOP_K)
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--rerank-preserve-top-k", type=int, default=0)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--sparse-weight", type=float, default=DEFAULT_SPARSE_WEIGHT)
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_DENSE_WEIGHT)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--url-cap", type=int, default=DEFAULT_URL_CAP)
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
    result = retriever.retrieve(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        sparse_top_k=args.sparse_top_k,
        dense_top_k=args.dense_top_k,
        fused_top_k=args.fused_top_k,
        rerank_top_k=args.rerank_top_k,
        rerank_preserve_top_k=args.rerank_preserve_top_k,
        rrf_k=args.rrf_k,
        sparse_weight=args.sparse_weight,
        dense_weight=args.dense_weight,
        reranker_model=args.reranker_model,
        url_cap=args.url_cap,
    )
    if args.json:
        if isinstance(result, HybridRetrievalResult):
            print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
            return 0
        print(
            json.dumps(
                {"query": args.query, "mode": args.mode, "hits": [asdict(hit) for hit in result]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    hits = result.hits if isinstance(result, HybridRetrievalResult) else result
    for hit in hits:
        title = hit.title or "(untitled)"
        print(f"{hit.rank}. {title}")
        print(f"   url={hit.url or ''}")
        print(f"   chunk_id={hit.chunk_id} score={hit.score:.6g}")
        if isinstance(hit, OptimizedRetrievalHit):
            print(f"   rrf_score={hit.rrf_score:.6g} rerank_score={hit.rerank_score}")
        print(f"   snippet={hit.snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
