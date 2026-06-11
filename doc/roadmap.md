# CS290S SIST RAG Product Roadmap

Last updated: 2026-06-11

## Vision

A polished local-model RAG web app for answering ShanghaiTech/SIST questions with citations, evaluation evidence, and reproducible official-source data.

Primary users are course reviewers and students. The product-quality target is a reliable Gradio QA demo deployed on AIStation, backed by audited official-source data, local retrieval/generation models, and report-ready evaluation artifacts. The roadmap is deadline-driven for the June 14, 2026 course submission.

## Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language and env | Python 3.11/3.12, `uv` | Matches the existing repo and model/data tooling. |
| Collection | `src/rag_collection` CLI | Existing official-source crawl, reparse, quality, and merge pipeline. |
| Corpus artifacts | JSONL + SQLite | JSONL is inspectable; SQLite is queryable for retrieval metadata. |
| Sparse retrieval | BM25 with regex tokens + `jieba` | Handles exact course, faculty, Chinese, and English matches. |
| Dense retrieval | FAISS + `BAAI/bge-m3` | Multilingual dense retrieval for semantic matching. |
| Optimization | RRF hybrid merge + reranker | Clear before/after improvement axis for report and demo. |
| Generator | Local Qwen3 family; Qwen3-0.6B smoke, Qwen3-4B/30B optional | Keeps local-only generation testable on 8GB GPUs, with stronger AIStation experiment paths. |
| UI | Gradio | Fastest route to a polished interactive web demo. |
| Deployment | AIStation GPU + local smoke mode | Meets assignment resources while preserving lightweight dev checks. |

## Current State

- Official-source collection and merge workflow is documented in `doc/data_collection.md`.
- Accepted clean merged dataset is `data/merged/all-collection-runs-clean-2026-05-27`.
- SQLite ingestion exists in `src/rag/ingest.py`.
- BM25 and FAISS index building exist in `src/rag/index.py`.
- Baseline and hybrid retrieval exist in `src/rag/retrieve.py`.
- The current 100-question Phase 5 retrieval evaluation is documented in `doc/retrieval_experiments.md`.
- Local answer generation exists in `src/rag/generate.py` and the `rag-answer` CLI.
- Phase 3 answer policy uses local `transformers` loading only, explicit `--model-path`, chat-template rendering when
  available, citation validation, prompt-leakage rejection, and evidence-insufficient responses.
- Opt-in real LLM e2e tests exist in `tests/e2e/test_rag_answer_real_llm.py` and passed on the remote WSL host with
  `/home/richard/models/Qwen3-0.6B` and generated `data/rag/` artifacts.
- Docker packaging and mounted-runtime smoke commands are documented in `README.md`.
- A structured 100-question evaluation CSV exists at `data/test/question_final_structured_100.csv`.
- A Phase 5 evaluation module exists at `src/evaluate/` with the `rag-evaluate` CLI for retrieval or answer runs over the
  structured question set, producing JSONL records, summary JSON, review queues, gap notes, and Excel output under
  `data/eval/`.
- A full retrieval-only run on 2026-06-11 completed with `dense` source_hit@5 0.69 and `hybrid` source_hit@5 0.68;
  `hybrid` improved source_hit@1, MRR@5, and nDCG@5.
- Tests cover collection, parsing, merge, ingestion, indexing, retrieval, generation, and opt-in real LLM e2e behavior.

## Build Order

| Phase | Goal | Primary artifacts | Sessions |
| --- | --- | --- | --- |
| 1 | Baseline retrieval app core | SQLite, BM25, FAISS, retrieval CLI | 1-2 |
| 2 | Hybrid retrieval and reranking | Optimized retriever, baseline comparison path | 1-2 |
| 3 | Local generator and answer policy | Qwen generation, citation prompt, refusal policy | 1-2 |
| 4 | Product-quality Gradio UI | Reviewer-facing chat UI and retrieval trace | 1-2 |
| 5 | Evaluation and targeted refresh | 50+ question set, JSONL/Excel eval runs | 2-3 |
| 6 | AIStation deployment | Server setup notes, launch command, smoke checks | 1 |
| 7 | Report, demo, submission package | Report tables, video script, zip checklist | 1-2 |

## Phase 1 - Baseline Retrieval App Core

*Goal: A developer can build corpus artifacts and run deterministic cited retrieval for representative questions.*

### What's New

- Baseline retrieval callable from tests and CLI without the generator.
- Reusable loader for SQLite chunk metadata, BM25 payload, FAISS index, and chunk mapping.
- Smoke query command that prints ranked chunks with title, URL, score, and snippet.

### Data and Artifact Changes

- Reuse `data/merged/all-collection-runs-clean-2026-05-27` as the default input.
- Generate default artifacts under `data/rag/`:
  - `sist_merged_2026-05-27.sqlite`
  - `bm25_2026-05-27.pkl`
  - `faiss_bge_m3_2026-05-27.index`
  - `chunk_index_2026-05-27.jsonl`
  - `build_report_2026-05-27.json`

### Task Checklist

#### Retrieval Core
- [x] Add a retrieval module that loads existing `rag-build-db` and `rag-build-index` outputs.
- [x] Implement baseline BM25 retrieval with stable chunk IDs and cited metadata.
- [x] Implement baseline FAISS retrieval when dense artifacts are present.
- [x] Add a CLI command for smoke retrieval with a `--mode bm25|dense` option.

#### Tests
- [x] Add unit tests using `tmp_path` artifacts, following `tests/integration/test_rag_ingest_index.py`.
- [x] Verify missing dense index behavior is explicit and useful.
- [x] Verify Chinese and English smoke queries return source URLs.

#### Definition of Done
- [x] `uv run pytest tests/integration/test_rag_ingest_index.py` passes.
- [x] `uv run pytest` passes.
- [x] `uv run ruff check src tests` passes.
- [x] Smoke queries return deterministic cited chunks for:
  - `深度学习 任课老师`
  - `计算机科学与技术 毕业 学分`
  - `SIST faculty robotics`

## Phase 2 - Hybrid Retrieval and Reranking

*Goal: The app can compare baseline retrieval against optimized hybrid retrieval on the same questions.*

### What's New

- Hybrid retrieval combines BM25 and dense rankings with reciprocal rank fusion.
- Reranker reorders fused candidates for final context selection.
- Context packer removes near-duplicate sources and keeps source metadata for citations.
- Baseline and optimized modes stay available for evaluation.

### Task Checklist

#### Retrieval Optimization
- [x] Add RRF merge for BM25 and FAISS candidate lists.
- [x] Add configurable top-k values for sparse, dense, fused, reranked, and final context counts.
- [x] Add optional local `sentence_transformers.CrossEncoder` reranking via `--reranker-model`.
- [x] Add source de-duplication by canonical URL and text/content hash when available.
- [x] Add context packing with title, URL, language, category, snippet, and rank fields.

#### Comparison Support
- [x] Preserve baseline retrieval output shape for before/after evaluation.
- [x] Add retrieval trace fields: sparse score, dense score, RRF score, rerank score, final rank.

#### Definition of Done
- [x] The same question can run in baseline and optimized modes.
- [x] Optimized output includes final citations and retrieval trace.
- [x] Unit tests cover RRF ordering, de-duplication, and missing reranker fallback.
- [x] A 12-question real-artifact retrieval pilot passed the Phase 3 gate: `hybrid` reached 10/12 expected-source hit@5 versus `dense` at 9/12.

## Phase 3 - Local Generator and Answer Policy

*Goal: End-to-end local RAG answers work with citations and no hosted LLM API calls.*

### What's New

- Local Qwen generation path with a conservative citation-first prompt.
- Qwen chat-template rendering when available, with Qwen3 thinking disabled for answer-mode generation.
- Structural answer validation for missing citations, unresolved citations, prompt/source leakage, and explicit
  evidence-insufficient model text.
- Evidence-grounded fallback for narrow explicit course-teacher and robotics-faculty patterns exposed by the real
  Qwen3-0.6B smoke tests.
- Optional Qwen3-30B-A3B AIStation experiment path.
- Answer policy: answer in the user's language, cite sources, and say when evidence is insufficient.

### Task Checklist

#### Generation
- [x] Add generator module using local `transformers` model loading or a local self-hosted endpoint.
- [x] Define prompt template with question, ranked context, answer rules, and citation requirements.
- [x] Add configurable model path, max tokens, temperature, and device settings.
- [x] Add insufficient-evidence response when retrieval context is absent, uncited, or lacks usable source URLs.

#### Safety and Constraints
- [x] Ensure no hosted OpenAI, Claude, Gemini, DashScope, hosted DeepSeek, or Hugging Face hosted inference calls are used.
- [x] Add clear error messages for missing local model files or unavailable CUDA.

#### Implementation Notes
- [x] Add `rag-answer` for local cited answer generation over `dense` and `hybrid` retrieval modes.
- [x] Require explicit local `--model-path`; recommended Qwen model IDs are documentation hints, not runtime download defaults.
- [x] Add default fake-model tests for prompt, citation, refusal, device, and JSON output behavior.
- [x] Add opt-in real LLM e2e tests guarded by `RAG_TEST_REAL_LLM`, `RAG_TEST_REAL_DATA`, and `RAG_TEST_MODEL_PATH`.
- [x] Run opt-in real Qwen3-0.6B smoke checks with a local model snapshot on the remote WSL host.
- [ ] Run opt-in Qwen3-4B or larger AIStation smoke checks if GPU memory allows.

#### Definition of Done
- [x] Local Qwen3 smoke model answers representative Chinese and English questions.
- [x] Answers include numbered citations mapped to source titles and URLs in the structured `sources` list.
- [x] An unanswerable question produces an evidence-insufficient response.

## Phase 4 - Product-Quality Gradio UI

*Goal: A reviewer can use a polished browser demo to ask questions, inspect sources, and compare retrieval modes.*

### What's New

- Gradio chat UI with answer, citations, retrieved snippets, scores, and trace/debug panel.
- Controls for baseline vs optimized mode, top-k, and sample questions.
- Clear loading, error, empty, and insufficient-evidence states.

### Task Checklist

#### UI
- [ ] Add Gradio app entrypoint under `src/rag/`.
- [ ] Implement chat-style question input and answer output.
- [ ] Show citations as clickable source URLs when available.
- [ ] Show retrieved snippets in ranked order with title, category, language, and scores.
- [ ] Add sample questions covering factual, course, faculty, time-sensitive, and English cases.

#### Product Polish
- [ ] Keep the first screen as the usable QA app, not a landing page.
- [ ] Add compact controls for mode and retrieval settings.
- [ ] Add graceful messages for missing indexes, missing model, and no evidence.

#### Definition of Done
- [ ] Local smoke launch starts the Gradio app.
- [ ] Reviewer can run a full question-answer-citation flow in the browser.
- [ ] Baseline and optimized modes can be demonstrated without code changes.

## Phase 5 - Evaluation and Targeted Data Refresh

*Goal: The project has report-ready before/after evaluation evidence and uses targeted data refresh only for verified gaps.*

### What's New

- 50+ bilingual evaluation questions with ground truth and source evidence.
- Evaluation runner that captures baseline answer, optimized answer, correctness labels, retrieval hit notes, latency, and source URLs.
- Excel output matching the assignment columns plus product-quality diagnostics.
- Targeted official-source refresh process for known coverage gaps.

### Artifact Layout

- `data/test/question_final.csv`
- `data/test/question_final_structured_100.csv`
- `data/eval/questions_YYYY-MM-DD.xlsx`
- `data/eval/retrieval_pilot_manifest_2026-05-31.jsonl`
- `data/eval/<timestamp>_<run_id>/run_<run_id>.jsonl`
- `data/eval/<timestamp>_<run_id>/summary_<run_id>.json`
- `data/eval/<timestamp>_<run_id>/review_queue_<run_id>.csv`
- `data/eval/<timestamp>_<run_id>/results_before_after_<run_id>.xlsx`
- `data/eval/<timestamp>_<run_id>/gap_notes_<run_id>.md`

### Task Checklist

#### Evaluation Set
- [x] Create at least 50 questions across factual, multi-hop, time-sensitive, comparative, conditional, course, faculty, and English categories.
- [x] Store ground-truth answers, source URLs, expected evidence snippets, category, and language.
- [ ] Include reviewer-friendly examples for the demo.

#### Evaluation Runner
- [x] Add a formal `src/evaluate` runner that can run each question through `dense` and `hybrid` modes for retrieval or answer paths.
- [x] Record retrieved or cited source URLs, source-hit flags, top titles, latency, answer status, and answer text to JSONL.
- [x] Write summary JSON with per-mode ok/error counts, retrieval metrics, correctness draft counts, and average latency.
- [x] Record before/after correctness drafts and support review-decision overrides for final labels.
- [x] Export the required Excel columns:
  - `query`
  - `gt_answer`
  - `sys_resp_before_opt`
  - `sys_resp_after_opt`
  - `is_correct_before_opt`
  - `is_correct_after_opt`

#### Targeted Refresh
- [ ] Use current merged data as the baseline; do not perform a major recrawl.
- [ ] For verified gaps, add targeted seed URLs and run bounded official-source collection.
- [ ] Merge accepted refresh runs into a new generated dataset path.
- [ ] Document any refresh in `doc/data_collection.md` or a linked run note.

#### Definition of Done
- [ ] Excel workbook supports assignment submission.
- [ ] Report tables can cite evaluation counts, accuracy delta, latency, and failure categories.
- [ ] At least five failed or weak cases have documented causes and next actions.

## Phase 6 - AIStation Deployment

*Goal: The Gradio app runs on AIStation with local models and reproducible artifact paths.*

### What's New

- AIStation setup instructions for environment, model paths, index paths, and launch command.
- Local smoke-mode instructions for developers without GPU access.
- Deployment smoke checklist for representative questions.

### Task Checklist

#### Server Setup
- [ ] Document `uv sync --locked --dev` or AIStation equivalent environment setup.
- [ ] Document local model placement for Qwen3-4B and optional Qwen3-30B-A3B.
- [ ] Document index artifact placement under `data/rag/`.
- [ ] Add launch command for Gradio with host, port, model path, and artifact paths.

#### Verification
- [ ] Run collection doctor check:
  - `uv run collect-data doctor --seeds config/official_seed_urls_sist_nav_deep.csv`
- [ ] Run retrieval smoke queries on deployed artifacts.
- [ ] Run at least five end-to-end UI questions on AIStation.
- [ ] Record any GPU memory, latency, or model-loading constraints.

#### Definition of Done
- [ ] Deployed Gradio app answers representative questions on GPU.
- [ ] Local smoke-mode remains documented and usable.
- [ ] No hosted LLM or hosted inference service is required.

## Phase 7 - Report, Demo, and Submission Package

*Goal: The code, report, evaluation workbook, and demo materials are ready for course submission.*

### What's New

- Report sections are populated from actual artifacts.
- Demo script covers the product value, optimization, citations, and failure handling.
- Submission packaging checklist prevents missing required files.

### Task Checklist

#### Report
- [ ] Write data collection and preprocessing section from `doc/data_collection.md`.
- [ ] Write architecture section from `doc/tech_stack_plan.md` and current implementation.
- [ ] Write implementation details for retrieval, reranking, generation, UI, and deployment.
- [ ] Write optimization section comparing baseline vs hybrid+rerank.
- [ ] Write results section using Excel evaluation tables and failure analysis.
- [ ] Write limitations and future work, including stale data and model/resource constraints.

#### Demo
- [ ] Prepare a 10-minute demo script with 3-5 representative questions.
- [ ] Include one success case, one multi-hop case, one course/faculty case, and one insufficient-evidence case.
- [ ] Capture UI flow showing answer, citations, snippets, and retrieval mode comparison.

#### Submission
- [ ] Confirm source code excludes raw datasets and large model checkpoints.
- [ ] Include README instructions, report PDF, Excel results, and demo video.
- [ ] Package required files as `studentID-name-project3.zip`.

#### Definition of Done
- [ ] Final report is English PDF.
- [ ] Evaluation workbook is complete and spot-checked against source citations.
- [ ] Demo video or live presentation materials are complete.
- [ ] Submission package is ready before June 14, 2026.

## Data and Artifact Policy

- Keep corpus records in SQLite tables: documents, chunks, courses, faculty members, program requirements, and events.
- Keep generated retrieval artifacts under `data/rag/`.
- Keep evaluation runs and query logs as versioned JSONL/Excel artifacts under `data/eval/`.
- Keep `data/collection_runs/` append-only.
- Treat `data/merged/` as generated downstream output; do not overwrite clean merged baselines silently.
- Do not commit raw datasets, large generated indexes, model checkpoints, or secrets unless the submission process explicitly requires a small artifact.

## API and Interface Map

| Interface | Phase | Purpose |
| --- | --- | --- |
| `uv run rag-build-db` | Existing/Phase 1 | Build SQLite corpus database. |
| `uv run rag-build-index` | Existing/Phase 1 | Build BM25, FAISS, chunk index, and report artifacts. |
| `uv run rag-retrieve --mode bm25\|dense` | Existing/Phase 1 | Run baseline smoke retrieval and print cited chunks. |
| `uv run rag-retrieve --mode hybrid` | Existing/Phase 2 | Run optimized RRF retrieval with optional local reranking, trace output, and packed contexts. |
| Retriever Python API | Existing/Phase 1-2 | Power CLI, eval runner, generator, and UI. |
| Retrieval pilot manifest | Historical/Phase 2 | Preserve the 12-question smoke spec for provenance; do not use it for current report metrics. |
| Generator Python API | Existing/Phase 3 | Convert retrieved context into cited local-model answers. |
| `uv run rag-answer --mode dense\|hybrid` | Existing/Phase 3 | Generate structured local answers for official before/after answer conditions. |
| Real LLM e2e tests | Existing/Phase 3 | Opt-in local-Qwen regression checks for answer templates and insufficient-evidence behavior. |
| Gradio app | Phase 4 | Reviewer-facing web demo. |
| `uv run rag-evaluate` | Existing/Phase 5 | Run structured questions through retrieval and/or answer paths, summarize retrieval metrics, emit review queues, and export assignment-ready before/after Excel outputs. |

## Test and Acceptance Plan

Run these checks at the relevant phase boundaries:

```bash
uv run pytest
uv run pytest tests/integration/test_rag_ingest_index.py
uv run pytest tests/integration/test_rag_generate.py
uv run ruff check src tests
uv run collect-data doctor --seeds config/official_seed_urls_sist_nav_deep.csv
```

Opt-in real LLM regression:

```bash
RAG_TEST_REAL_DATA=1 \
RAG_TEST_REAL_LLM=1 \
RAG_TEST_MODEL_PATH=/home/richard/models/Qwen3-0.6B \
RAG_TEST_DEVICE=cuda \
uv run python -m pytest tests/e2e/test_rag_answer_real_llm.py -q
```

Structured question benchmark smoke:

```bash
uv run rag-evaluate --runner retrieve --modes dense hybrid --limit 5
```

Locked 100-question retrieval evaluation:

```bash
uv run rag-evaluate --runner retrieve --modes dense hybrid --top-k 5
```

Manual acceptance scenarios:

- Chinese factual: `上海科技大学一共有几个学院？`
- Course: `《深度学习》这门课的任课老师是谁？`
- Program requirement: `计算机科学与技术专业需要修满多少学分才能毕业？`
- Faculty/advisor: `我想做机器人方向，有哪些导师可以推荐？`
- English: `Which SIST faculty work on robotics?`
- Insufficient evidence: ask for a fact that should not be in the corpus and verify the app refuses or qualifies the answer.

Evaluation acceptance:

- At least 50 bilingual questions.
- Required Excel columns are present.
- Baseline and optimized runs are comparable.
- Accuracy, latency, retrieval hit rate, and failure categories are summarized for the report.

## Deliberately Not Building for v1

- Hosted/commercial LLM API integration: forbidden by project constraints.
- Full public campus search product: broader audience and moderation needs are out of scope.
- Authentication and user accounts: unnecessary for the reviewer-facing demo unless deployment policy requires external access control.
- Major recrawl: use targeted refreshes only after evaluation identifies concrete gaps.
- Multi-tenant SaaS, billing, plugin system, or collaborative features: not relevant to the course submission.
- Native mobile app: Gradio web UI is sufficient for demo and deployment.

## Operating Rules for Future Sessions

- Start each implementation session by reading this roadmap, `doc/tech_stack_plan.md`, and `doc/data_collection.md`.
- Work phase-by-phase and keep the first incomplete checklist item as the current task.
- After completing a phase, update this file with checked tasks, verification commands, generated artifact paths, and any commit hash if committed.
- Preserve user changes in the working tree; do not rewrite unrelated docs or generated data.
- If a phase exposes a data coverage gap, create a targeted refresh task rather than launching a broad crawl.
- Keep uncommitted evaluation artifacts under `data/test/` and generated benchmark outputs under `data/eval/` visibly
  documented before using them for report claims.
