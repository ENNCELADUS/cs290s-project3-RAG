# Repository Guidelines

## Project Structure & Module Organization

This repository builds an official-source ShanghaiTech/SIST RAG system. Core Python packages live under `src/`: `src/rag_collection/` contains the crawler, parsers, structured extraction, reparse, merge, and collection CLI; `src/rag/` contains ingestion, SQLite export, BM25/FAISS indexing, baseline retrieval, and hybrid retrieval. Tests in `tests/` mirror package behavior, for example `tests/integration/test_crawler.py` and `tests/integration/test_rag_ingest_index.py`.

Configuration lives in `config/`, including seed CSVs such as `config/official_seed_urls_sist_nav_deep.csv`. Treat `doc/tech_stack_plan.md` and `doc/data_collection.md` as current report references. Generated crawl runs belong under `data/collection_runs/`, merged datasets under `data/merged/`, and RAG artifacts under `data/rag/`.

## Build, Test, and Development Commands

- `uv sync --locked --dev`: install the Python environment.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/integration/test_rag_ingest_index.py`: run focused ingestion/index/retrieval tests.
- `uv run ruff check src tests`: lint imports, pyupgrade rules, and bugbear checks.
- `uv run ruff format src tests`: format Python files.
- `uv run collect-data doctor --seeds config/official_seed_urls_sist_nav_deep.csv`: verify crawl dependencies and seed loading.
- `uv run rag-build-db`: build the default SQLite database from the clean merged dataset.
- `uv run rag-build-index --skip-faiss`: build BM25 and metadata without dense embeddings.
- `uv run rag-retrieve --query "SIST faculty robotics" --mode hybrid --json`: run optimized retrieval against existing artifacts.

## Coding Style & Naming Conventions

Use Python with type hints and `from __future__ import annotations`, matching existing modules. Keep path defaults as `pathlib.Path` constants near the top of CLI modules. Use snake_case for functions, variables, CLI flags, and JSONL fields. Ruff targets 120-character lines and Python 3.11.

## Testing Guidelines

Add or update tests for behavior changes. Prefer small fixtures with `tmp_path` and local JSONL writers, as in `tests/integration/test_rag_ingest_index.py`, instead of large real datasets. For crawler/parser changes, cover canonical URLs, content parsing, quality flags, and merge behavior.

## Security & Configuration Tips

The final system must use self-hosted/local models only. Do not add OpenAI, Claude, Gemini, DashScope, DeepSeek hosted API, Hugging Face hosted inference, or similar hosted LLM calls. Collection should remain official-source and append-only: create new run directories rather than overwriting prior runs.

## Commit & Pull Request Guidelines

Use concise, imperative commit subjects like `Add RAG ingestion and indexing pipeline`. Keep diffs surgical and avoid formatting unrelated files. PRs should state the changed pipeline stage, commands run, generated artifacts, and skipped expensive steps such as dense FAISS embedding.

## Agent-Specific Instructions

Before editing, read the relevant module and current docs. State assumptions when scope is ambiguous. Touch only files needed, preserve user changes, and verify with targeted tests or explain skipped checks.
