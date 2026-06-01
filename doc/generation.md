# Local Generation

Last updated: 2026-06-01

This document describes the Phase 3 local answer-generation path. Retrieval still produces cited hits and packed
contexts; generation turns those contexts into a **Generated Answer** with numbered citations.

## Runtime Contract

The generation API is in `src/rag/generate.py`.

```python
from pathlib import Path

from rag.generate import RagAnswerer
from rag.retrieve import Retriever

retriever = Retriever.from_paths(
    db_path=Path("data/rag/sist_merged_2026-05-27.sqlite"),
    bm25_path=Path("data/rag/bm25_2026-05-27.pkl"),
    faiss_path=Path("data/rag/faiss_bge_m3_2026-05-27.index"),
    chunk_index_path=Path("data/rag/chunk_index_2026-05-27.jsonl"),
    report_path=Path("data/rag/build_report_2026-05-27.json"),
)

answerer = RagAnswerer(
    retriever,
    model_path=Path("/models/hub/snapshots/qwen3-4b-instruct-2507-local"),
    device="auto",
)
result = answerer.answer("SIST faculty robotics", mode="hybrid", top_k=5)
```

Supported answer modes:

| mode | role |
| --- | --- |
| `dense` | Official **Before Optimization** answer condition for Phase 5 evaluation. |
| `hybrid` | Official **After Optimization** answer condition and default `rag-answer` mode. |

`bm25` remains a **Diagnostic Baseline** for retrieval analysis and is not exposed as a Phase 3 answer mode.

## Local Model Policy

`rag-answer` requires an explicit local `--model-path`. The command does not default to a Hugging Face model ID and
uses `local_files_only=True` when loading `transformers` artifacts.

Recommended model family:

- `Qwen/Qwen3-4B-Instruct-2507` for the first working local demo.
- `Qwen/Qwen3-30B-A3B-Instruct-2507` only as a later AIStation experiment path.

The model ID is a download/acquisition hint only. Runtime commands should point to the already-present local snapshot
directory.

## CLI

```bash
uv run rag-answer \
  --query "SIST faculty robotics" \
  --mode hybrid \
  --model-path /models/hub/snapshots/qwen3-4b-instruct-2507-local \
  --device auto \
  --json
```

Useful flags:

| flag | default |
| --- | --- |
| `--query` | required |
| `--mode dense\|hybrid` | `hybrid` |
| `--top-k` | `5` |
| `--model-path` | required local path |
| `--device auto\|cpu\|cuda` | `auto` |
| `--max-new-tokens` | `512` |
| `--temperature` | `0.2` |
| `--db` | `data/rag/sist_merged_2026-05-27.sqlite` |
| `--bm25` | `data/rag/bm25_2026-05-27.pkl` |
| `--faiss` | `data/rag/faiss_bge_m3_2026-05-27.index` |
| `--chunk-index` | `data/rag/chunk_index_2026-05-27.jsonl` |
| `--report` | `data/rag/build_report_2026-05-27.json` |
| `--dense-model` | unset; falls back to build report |
| `--json` | off |

Device behavior:

- `auto` uses CUDA when `torch.cuda.is_available()` is true, otherwise CPU.
- `cpu` is allowed for local smoke and tests.
- `cuda` fails clearly if CUDA is unavailable.

## Answer Policy

The prompt gives the model the user question and numbered source contexts. It requires the answer to:

- use the user's language,
- use only the provided official-source context,
- cite factual paragraphs with numbered citations such as `[1]`,
- say that evidence is insufficient when the provided sources do not support an answer.

The code returns an **Evidence-Insufficient Answer** when retrieval returns no contexts, no usable source URL is present,
or the generated text does not contain a valid citation number.

## Output Shape

`rag-answer --json` returns:

```json
{
  "query": "SIST faculty robotics",
  "mode": "hybrid",
  "status": "answered",
  "answer": "Generated answer with citation [1].",
  "sources": [],
  "retrieval": {},
  "timing": {},
  "config": {}
}
```

`sources` maps citation numbers to title, URL, chunk ID, document ID, trace reference, and snippet. `retrieval` keeps
the underlying hits and packed contexts so the Gradio UI and Phase 5 evaluation runner can inspect evidence.

## Verification

Default tests use fake tokenizer/model objects and temporary retrieval artifacts:

```bash
uv run python -m pytest tests/integration/test_rag_generate.py
```

Opt-in real-model smoke checks require a local Qwen snapshot and generated `data/rag/` artifacts:

```bash
uv run rag-answer \
  --query "深度学习这门课的任课老师是谁？" \
  --mode hybrid \
  --model-path /models/hub/snapshots/qwen3-4b-instruct-2507-local

uv run rag-answer \
  --query "Which SIST faculty work on robotics?" \
  --mode hybrid \
  --model-path /models/hub/snapshots/qwen3-4b-instruct-2507-local

uv run rag-answer \
  --query "What is the SIST cafeteria menu tomorrow?" \
  --mode hybrid \
  --model-path /models/hub/snapshots/qwen3-4b-instruct-2507-local
```

The last query should produce an evidence-insufficient response unless the indexed official-source corpus contains
usable evidence.

For regression work on the real local generator, run the opt-in e2e tests. These tests load the real model, call the
public `RagAnswerer` API, and pin the expected answer shape for representative Chinese, English, and unanswerable
questions:

```bash
RAG_TEST_REAL_DATA=1 \
RAG_TEST_REAL_LLM=1 \
RAG_TEST_MODEL_PATH=/home/richard/models/Qwen3-0.6B \
RAG_TEST_DEVICE=cuda \
uv run python -m pytest tests/e2e/test_rag_answer_real_llm.py -q
```

They are expected to fail until the prompt/generation policy can produce concise cited answers and refuse unsupported
questions.
