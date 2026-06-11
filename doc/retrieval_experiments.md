# Retrieval Evaluation Experiments

Last updated: 2026-06-11

This document records the current report-facing retrieval evaluation for the ShanghaiTech/SIST RAG system. The old
12-question Phase 2 pilot is historical only; the locked retrieval test paradigm is now the structured 100-question
Phase 5 set and the `src/evaluate` runner.

This is still a retrieval-only experiment. It measures whether the retriever surfaces expected official-source evidence,
not whether the local generator produces a correct final answer.

## Locked Test Paradigm

The official before/after retrieval convention for the report is:

| role | retrieval mode | purpose |
| --- | --- | --- |
| Before optimization | `dense` | Pre-optimization condition using FAISS over `BAAI/bge-m3` embeddings. |
| After optimization | `hybrid` | Optimized condition using BM25+dense RRF fusion, source de-duplication, and packed contexts. |
| Diagnostic baseline | `bm25` | Optional sparse diagnostic only; not part of the official before/after comparison. |

Locked inputs:

| input | value |
| --- | --- |
| Question file | `data/test/question_final_structured_100.csv` |
| Question count | 100 |
| Corpus snapshot | `data/merged/all-collection-runs-clean-2026-05-27` |
| SQLite DB | `data/rag/sist_merged_2026-05-27.sqlite` |
| BM25 payload | `data/rag/bm25_2026-05-27.pkl` |
| FAISS index | `data/rag/faiss_bge_m3_2026-05-27.index` |
| Chunk index | `data/rag/chunk_index_2026-05-27.jsonl` |
| Build report | `data/rag/build_report_2026-05-27.json` |
| Retrieval depth | `top_k=5` |
| Evaluation runner | `src/evaluate` via `python -m evaluate.cli` or `rag-evaluate` |

Question distribution:

| field | distribution |
| --- | --- |
| category | Factual 45; Time-sensitive 20; Comparative 13; Conditional 10; Multi-hop 12 |
| language | zh 100 |
| complexity | Low 45; Medium 45; High 10 |
| judge type | exact_or_alias_match 35; required_facts_match 27; required_facts_with_manual_review 29; local_llm_judge_with_human_review 9 |

Metric policy:

- Source metrics use normalized URL-prefix qrels from the structured question source fields.
- URL normalization removes query strings and fragments and treats SIST template path segments such as `_t335` as aliases
  of the same non-template official path after commit `44bfec4`.
- Root expected URLs, such as `https://sist.shanghaitech.edu.cn/`, match same-site subpages after commit `6230f75`.
- `source_hit@k` is 1 when any expected official source appears in the top `k`.
- `source_recall@k`, `mrr@k`, `ndcg@k`, and `precision@k` are retrieval diagnostics, not answer correctness.
- Retrieval-only runs intentionally produce `manual_review` correctness status because no generated answer is judged.

Canonical command used on the remote runner:

```bash
uv run --locked --no-sync --offline python -m evaluate.cli \
  --runner retrieve \
  --modes dense hybrid \
  --top-k 5 \
  --timestamp remote_retrieve_full_rootfix_20260611
```

## Remote Artifact Layout

Generated evaluation artifacts stay on the remote runner and are grouped by timestamp under `data/eval/`. The tracked
manifest `data/eval/retrieval_pilot_manifest_2026-05-31.jsonl` remains at the root as a small historical Phase 2 spec.

```text
/home/richard/cs290s-project3-RAG/data/eval/
  retrieval_pilot_manifest_2026-05-31.jsonl
  20260531T155503Z_retrieval_pilot/
    retrieval_pilot_20260531T155503Z.jsonl
    retrieval_pilot_20260531T155503Z.md
  20260611T220856Z_remote_retrieve_full/
    run_remote_retrieve_full_20260611.jsonl
    summary_remote_retrieve_full_20260611.json
    review_queue_remote_retrieve_full_20260611.csv
    gap_notes_remote_retrieve_full_20260611.md
    results_before_after_remote_retrieve_full_20260611.xlsx
  20260611T222046Z_remote_retrieve_full_rootfix/
    run_remote_retrieve_full_rootfix_20260611.jsonl
    summary_remote_retrieve_full_rootfix_20260611.json
    review_queue_remote_retrieve_full_rootfix_20260611.csv
    gap_notes_remote_retrieve_full_rootfix_20260611.md
    results_before_after_remote_retrieve_full_rootfix_20260611.xlsx
```

Use the `20260611T222046Z_remote_retrieve_full_rootfix/` directory for report numbers. The earlier
`20260611T220856Z_remote_retrieve_full/` run is retained only as a pre-root-prefix-fix diagnostic.

## Final Retrieval Results

Final run:

```text
run_id: remote_retrieve_full_rootfix_20260611
remote directory: /home/richard/cs290s-project3-RAG/data/eval/20260611T222046Z_remote_retrieve_full_rootfix
records: 200
status: dense 100 ok / 0 errors; hybrid 100 ok / 0 errors
```

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 | avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 0.43 | 0.69 | 0.663333 | 0.509000 | 0.529621 | 0.156 | 1.944186 |
| `hybrid` | 0.46 | 0.68 | 0.648333 | 0.561167 | 0.572978 | 0.154 | 2.814903 |

Per-question top-5 source-hit overlap:

| dense hit@5 | hybrid hit@5 | questions |
| ---: | ---: | ---: |
| 0 | 0 | 25 |
| 0 | 1 | 6 |
| 1 | 0 | 7 |
| 1 | 1 | 62 |

Per-question top-1 source-hit overlap:

| dense hit@1 | hybrid hit@1 | questions |
| ---: | ---: | ---: |
| 0 | 0 | 46 |
| 0 | 1 | 11 |
| 1 | 0 | 8 |
| 1 | 1 | 35 |

Interpretation:

- Hybrid improves top-rank quality: `source_hit@1`, `mrr@5`, and `ndcg@5` are higher than dense.
- Dense is slightly better on broad top-5 coverage: `source_hit@5`, `source_recall@5`, and `precision@5` are marginally higher.
- Hybrid-only top-5 wins (6 questions) and dense-only top-5 wins (7 questions) show that both channels recover evidence
  the other misses; the report should describe this as a ranking-quality improvement rather than a simple top-5 win.
- The 25 questions missed by both modes are the first targets for bounded official-source refresh or qrels inspection.

## Controlled Optimization Runs

The `remote_retrieve_full_rootfix_20260611` run above is the control anchor. Each follow-up run changes one factor and
uses the same 100-question file, corpus snapshot, RAG artifacts, retrieval modes, and `top_k=5`.

### Fix 1: URL Canonicalization and Qrels Alias Matching

Change:

- commit: `44bfec4` (`Canonicalize source URL variants`)
- branch: `codex/retrieval-control-url-canonicalization`
- behavior: source metrics now ignore query strings/fragments and treat SIST `_tNNN` template path segments as aliases of
  the same non-template official path.
- no retrieval ranking, candidate selection, indexing, or generation changes.

Remote run:

```text
run_id: remote_retrieve_urlcanon_20260611
remote worktree: /home/richard/cs290s-project3-RAG-retrieval-urlcanon
remote artifacts:
  data/eval/run_remote_retrieve_urlcanon_20260611.jsonl
  data/eval/summary_remote_retrieve_urlcanon_20260611.json
  data/eval/review_queue_remote_retrieve_urlcanon_20260611.csv
  data/eval/gap_notes_remote_retrieve_urlcanon_20260611.md
  data/eval/results_before_after_remote_retrieve_urlcanon_20260611.xlsx
records: 200
status: dense 100 ok / 0 errors; hybrid 100 ok / 0 errors
```

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 | avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 0.51 | 0.74 | 0.723333 | 0.588500 | 0.608518 | 0.172 | 2.035649 |
| `hybrid` | 0.56 | 0.71 | 0.693333 | 0.623167 | 0.625689 | 0.162 | 2.882323 |

Delta versus the control anchor:

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | +0.08 | +0.05 | +0.060000 | +0.079500 | +0.078897 | +0.016 |
| `hybrid` | +0.10 | +0.03 | +0.045000 | +0.062000 | +0.052711 | +0.008 |

Per-question top-5 source-hit overlap after fix 1:

| dense hit@5 | hybrid hit@5 | questions |
| ---: | ---: | ---: |
| 0 | 0 | 23 |
| 0 | 1 | 3 |
| 1 | 0 | 6 |
| 1 | 1 | 68 |

Per-question top-1 source-hit overlap after fix 1:

| dense hit@1 | hybrid hit@1 | questions |
| ---: | ---: | ---: |
| 0 | 0 | 38 |
| 0 | 1 | 11 |
| 1 | 0 | 6 |
| 1 | 1 | 45 |

Interpretation:

- This fix improves both retrieval conditions without changing retrieved candidates, so the gain is attributable to
  stricter source URL canonicalization rather than ranking.
- The remaining both-missed top-5 set is 23 questions, down from 25 in the control anchor.
- Dense still has broader top-5 coverage, while hybrid keeps stronger top-rank quality.

## Verification

The root URL matching fix, URL-canonicalization fix, and evaluation module were checked locally:

```bash
uv run --locked --no-sync --offline python -m pytest tests/unit/test_evaluate_core.py tests/integration/test_evaluate_phase5.py -q
uv run --locked --no-sync --offline ruff check src/evaluate tests/unit/test_evaluate_core.py tests/integration/test_evaluate_phase5.py
```

Remote validation:

```text
run_remote_retrieve_full_rootfix_20260611.jsonl: 200 lines
review_queue_remote_retrieve_full_rootfix_20260611.csv: 201 lines including header
gap_notes_remote_retrieve_full_rootfix_20260611.md: 202 lines
run_remote_retrieve_urlcanon_20260611.jsonl: 200 lines
review_queue_remote_retrieve_urlcanon_20260611.csv: 201 lines including header
gap_notes_remote_retrieve_urlcanon_20260611.md: 202 lines
workbook sheets: submission, diagnostics, retrieval_metrics, review_queue
workbook rows: submission 101; diagnostics 201; retrieval_metrics 3; review_queue 201
```

The remote focused unit check passed:

```bash
uv run --locked --no-sync --offline python -m pytest tests/unit/test_evaluate_core.py -q
```

## Locked Follow-up Rules

- Do not compare future report retrieval numbers against the old 12-question pilot.
- Do not use `bm25` as the official before-optimization condition.
- Do not treat retrieval-only `manual_review` labels as answer correctness.
- For final assignment Excel correctness, run answer generation over the same 100-question CSV and apply manual review
  decisions where the deterministic judge emits `manual_review`.
- Keep generated run logs remote under timestamp directories; commit only small specs and documentation unless a
  submission checklist explicitly asks for the generated workbook.
