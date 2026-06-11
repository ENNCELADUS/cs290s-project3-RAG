# Phase 2 Retrieval Core

Last updated: 2026-06-01

This document describes the retrieval core that is currently implemented. It includes baseline BM25/dense retrieval and Phase 2 hybrid retrieval with RRF, optional local reranking, trace fields, de-duplication, and packed contexts. It does not describe generation or UI behavior.

## Implemented Surface

### Artifact Builders

The retrieval core consumes artifacts produced by the existing build commands:

```bash
uv run rag-build-db
uv run rag-build-index
```

Default artifact paths are:

| artifact | path |
| --- | --- |
| SQLite corpus DB | `data/rag/sist_merged_2026-05-27.sqlite` |
| BM25 payload | `data/rag/bm25_2026-05-27.pkl` |
| FAISS dense index | `data/rag/faiss_bge_m3_2026-05-27.index` |
| chunk mapping | `data/rag/chunk_index_2026-05-27.jsonl` |
| build report | `data/rag/build_report_2026-05-27.json` |

`rag-build-db` reads `data/merged/all-collection-runs-clean-2026-05-27` by default and writes SQLite tables for documents, chunks, courses, faculty, program requirements, and events.

`rag-build-index` writes the BM25 payload, chunk mapping, optional FAISS index, and build report. BM25 tokenization uses ASCII token matching plus `jieba` for Chinese text. Dense FAISS vectors use normalized `BAAI/bge-m3` embeddings and inner-product search.

Offline dense builds should pass a local model path explicitly:

```bash
uv run rag-build-index \
  --skip-bm25 \
  --model /path/to/local/bge-m3-snapshot \
  --model-id BAAI/bge-m3
```

### Python API

The runtime retrieval API is in `src/rag/retrieve.py`.

```python
from pathlib import Path

from rag.retrieve import Retriever

retriever = Retriever.from_paths(
    db_path=Path("data/rag/sist_merged_2026-05-27.sqlite"),
    bm25_path=Path("data/rag/bm25_2026-05-27.pkl"),
    faiss_path=Path("data/rag/faiss_bge_m3_2026-05-27.index"),
    chunk_index_path=Path("data/rag/chunk_index_2026-05-27.jsonl"),
    report_path=Path("data/rag/build_report_2026-05-27.json"),
)

baseline_hits = retriever.retrieve("SIST faculty robotics", mode="bm25", top_k=5)
hybrid_result = retriever.retrieve("SIST faculty robotics", mode="hybrid", top_k=5)
```

`retrieve()` supports:

| mode | behavior |
| --- | --- |
| `bm25` | Loads the BM25 pickle payload, scores query tokens, maps ranked payload positions to stable chunk IDs, and returns cited SQLite metadata. |
| `dense` | Loads FAISS, `chunk_index_2026-05-27.jsonl`, and a local `sentence-transformers` model, embeds the query, searches FAISS, and returns cited SQLite metadata. |
| `hybrid` | Runs query-matching BM25 candidates and dense candidates, merges them with reciprocal rank fusion, optionally reranks with a local CrossEncoder, de-duplicates final sources, and returns optimized hits plus packed contexts. |

Dense and hybrid modes resolve the query embedding model in this order:

1. `dense_model` passed to `Retriever.from_paths()`.
2. `index.faiss.model_path` in `build_report_2026-05-27.json`.

Hybrid reranking is optional. When a reranker is requested, `reranker_model` or `--reranker-model` must point to an existing local model path. The runtime does not intentionally use hosted inference for reranking. If no reranker is provided, hybrid retrieval keeps the RRF order and leaves rerank scores empty.

The SQLite DB is opened read-only. Missing SQLite, FAISS, chunk mapping, dense model metadata, or a requested local reranker path produces explicit `FileNotFoundError` messages.

### Result Shapes

Baseline `bm25` and `dense` modes return `RetrievalHit` values with the same fields:

| field | meaning |
| --- | --- |
| `rank` | 1-based rank within the returned list. |
| `chunk_id` | Stable chunk ID from SQLite and `chunk_index_2026-05-27.jsonl`. |
| `document_id` | Source document ID from SQLite. |
| `title` | Chunk/source title when available. |
| `url` | Source URL for citation. |
| `category` | Source category when available. |
| `language` | Source language when available. |
| `score` | BM25 score or FAISS inner-product score. |
| `snippet` | First 240 characters of chunk text. |
| `mode` | `bm25` or `dense`. |

Hybrid mode returns a `HybridRetrievalResult` with:

| field | meaning |
| --- | --- |
| `query` | Original query. |
| `mode` | Always `hybrid`. |
| `hits` | Optimized ranked hits with RRF score, optional rerank score, and nested trace. |
| `contexts` | Packed final contexts for future generation and UI display. |
| `config` | Effective sparse, dense, fused, rerank, final, RRF, reranker, and URL-cap settings. |

Each hybrid trace records sparse rank/score, dense rank/score, RRF score, optional rerank score, and final rank. Final contexts include rank, chunk ID, document ID, title, URL, category, language, snippet, full text, and `trace_ref`.

Hybrid de-duplication removes duplicate normalized text and keeps at most two final chunks per canonical URL by default. If a document canonical URL is unavailable, the chunk URL is used.

### CLI

The smoke retrieval CLI is `rag-retrieve`.

```bash
uv run rag-retrieve \
  --query "SIST faculty robotics" \
  --mode hybrid \
  --top-k 5
```

Supported flags:

| flag | default |
| --- | --- |
| `--query` | required |
| `--mode bm25\|dense\|hybrid` | `bm25` |
| `--top-k` | `5` final hits or contexts |
| `--sparse-top-k` | `20` |
| `--dense-top-k` | `20` |
| `--fused-top-k` | `20` |
| `--rerank-top-k` | `10` |
| `--rrf-k` | `60` |
| `--reranker-model` | unset; no reranking |
| `--url-cap` | `2` |
| `--db` | `data/rag/sist_merged_2026-05-27.sqlite` |
| `--bm25` | `data/rag/bm25_2026-05-27.pkl` |
| `--faiss` | `data/rag/faiss_bge_m3_2026-05-27.index` |
| `--chunk-index` | `data/rag/chunk_index_2026-05-27.jsonl` |
| `--report` | `data/rag/build_report_2026-05-27.json` |
| `--dense-model` | unset; falls back to build report |
| `--json` | off |

Baseline JSON output remains:

```json
{
  "query": "深度学习 任课老师",
  "mode": "bm25",
  "hits": []
}
```

Hybrid JSON output has this top-level shape:

```json
{
  "query": "深度学习 任课老师",
  "mode": "hybrid",
  "hits": [],
  "contexts": [],
  "config": {}
}
```

## Verified Behavior

Local verification on 2026-05-31:

```bash
uv run --locked --no-sync --offline python -m pytest tests/integration/test_rag_ingest_index.py -q
uv run --locked --no-sync --offline ruff check src tests
uv run --locked --no-sync --offline python -m pytest
```

The focused retrieval suite passed 19 tests, including baseline retrieval, dense retrieval failure messages, hybrid RRF ordering, hybrid JSON/context output, de-duplication, missing dense artifact failures, reranker fallback, non-positive BM25 sparse match preservation, and fake local reranker reordering. Ruff passed. The full local suite passed with 77 tests passing and 3 real-artifact tests skipped because generated `data/rag/` artifacts are not present locally.

Earlier remote WSL artifact verification on 2026-05-30:

- `rag-build-db` built `data/rag/sist_merged_2026-05-27.sqlite` with 7190 documents, 28481 chunks, and clean foreign keys.
- `rag-build-index --skip-bm25 --model <local bge-m3 snapshot> --model-id BAAI/bge-m3` rebuilt FAISS for 28481 chunks with dimension 1024.
- `rag-retrieve` returned cited source URLs for BM25 and dense retrieval on:
  - `深度学习 任课老师`
  - `计算机科学与技术 毕业 学分`
  - `SIST faculty robotics`
- `RAG_TEST_REAL_DATA=1 uv run --locked --offline python -m pytest tests/integration/test_real_artifacts.py -q` passed 3 real-artifact checks.

Remote retrieval pilot verification on 2026-05-31:

- The 12-question manifest in `data/eval/retrieval_pilot_manifest_2026-05-31.jsonl` ran against the full remote
  `data/rag/` artifacts.
- Official before-optimization retrieval was `dense`; after-optimization retrieval was `hybrid`; `bm25` was diagnostic.
- Expected-source hit@5 was `bm25` 6/12, `dense` 9/12, and `hybrid` 10/12.
- The Phase 3 gate passed because hybrid was at least as good as dense and showed no critical top-5 citation regression.
- Full per-hit logs were kept on the remote at
  `/home/richard/cs290s-project3-RAG/data/eval/20260531T155503Z_retrieval_pilot/retrieval_pilot_20260531T155503Z.jsonl`.

Full retrieval evaluation verification on 2026-06-11:

- The structured 100-question set in `data/test/question_final_structured_100.csv` ran against the full remote
  `data/rag/` artifacts through `src/evaluate`.
- Official before-optimization retrieval was `dense`; after-optimization retrieval was `hybrid`; `bm25` remained
  diagnostic and was not part of the official comparison.
- Final root-prefix-corrected retrieval results were `dense` source_hit@5 0.69 and `hybrid` source_hit@5 0.68.
- `hybrid` improved top-rank retrieval quality: source_hit@1 0.46 versus dense 0.43, MRR@5 0.561167 versus 0.509000,
  and nDCG@5 0.572978 versus 0.529621.
- Full generated artifacts are grouped by timestamp on the remote under
  `/home/richard/cs290s-project3-RAG/data/eval/20260611T222046Z_remote_retrieve_full_rootfix/`.

## Todo

The following retrieval-adjacent features are not implemented yet:

- Real local reranker smoke verification on full generated artifacts.
- Phase 5 answer-generation run with final manual review decisions for assignment correctness labels.
- Local generation with citation prompt and insufficient-evidence policy.
- Product-quality Gradio UI.
