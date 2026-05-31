# Retrieval Pilot Experiments

Last updated: 2026-05-31

This document records the small retrieval-only pilot used before Phase 3 generation work. It validates the current
Phase 1-2 retrievers on real `data/rag/` artifacts and checks whether hybrid retrieval is a defensible source of
generator context. It is not the Phase 5 Excel evaluation.

## Experiment Definition

The official before/after retrieval convention for the report is:

| role | retrieval mode | purpose |
| --- | --- | --- |
| Before optimization | `dense` | Pre-optimization condition for report and later Excel comparison. |
| After optimization | `hybrid` | Optimized condition using BM25+dense RRF fusion, de-duplication, and packed contexts. |
| Diagnostic baseline | `bm25` | Sparse baseline used to explain retrieval behavior, not the official before condition. |

No `sys_resp_before_opt` or `sys_resp_after_opt` values are produced in this pilot because generation is not
implemented yet. The pilot inspects retrieved sources and packed contexts only.

## Question Manifest

The 12-question manifest is tracked at `data/eval/retrieval_pilot_manifest_2026-05-31.jsonl`.

| category | count | examples |
| --- | ---: | --- |
| course | 2 | `《深度学习》这门课的任课老师是谁？`; `CS181 Artificial Intelligence 的任课老师是谁？` |
| program credits | 2 | CS master and doctor credit requirements. |
| program comparison | 1 | Professional/project master versus academic master training plans. |
| faculty/research | 2 | Xuming He and robotics direction questions. |
| institution fact | 1 | SIST founding-year question. |
| time-sensitive corpus-latest | 2 | Latest training-plan notice and latest lecture/activity in the indexed corpus. |
| English | 2 | Robotics faculty and SIST research-center questions. |

Time-sensitive questions are evaluated against the current artifact corpus, not live web truth as of the run date.

## Metrics

Primary metric:

- `expected_source_hit_at_5`: true if any top-5 retrieved URL matches one of the query's normalized expected official
  URL prefixes.

Diagnostics:

- `overlap_at_5_vs_dense`: count of shared top-5 chunk IDs with dense retrieval for the same query.
- `latency_s`: wall-clock runtime after one warmup per mode.
- `notes`: manual citation-quality notes or obvious irrelevant-source issues.

Latency is smoke latency for the current retrieval runtime. It is not final UI serving latency because the Phase 2
runtime may load dense models during retrieval calls.

## Remote Run

Run host:

```text
ssh -p 2222 richard@10.20.97.163
repo: /home/richard/cs290s-project3-RAG
artifacts: data/rag/
```

Execution policy:

- Use a temporary remote Python script, not a tracked runner.
- Run modes `bm25`, `dense`, and `hybrid` for all 12 questions.
- Run `hybrid_rerank` only if a local reranker snapshot already exists.
- Keep full per-hit JSONL logs on the remote under `data/eval/`.

The remote cache did not contain a local reranker snapshot, so this pilot did not run a `hybrid_rerank` condition.

## Results

A first diagnostic dry run exposed overly narrow expected URL prefixes for list pages, current-year degree-program
PDFs, and broad robotics/activity questions. The tracked manifest now uses normalized official source prefixes rather
than one exact page version.

Final run:

```text
run_id: retrieval_pilot_20260531T155503Z
jsonl: /home/richard/cs290s-project3-RAG/data/eval/retrieval_pilot_20260531T155503Z.jsonl
summary: /home/richard/cs290s-project3-RAG/data/eval/retrieval_pilot_20260531T155503Z.md
```

| mode | expected-source hit@5 | avg latency (s) | max latency (s) |
| --- | ---: | ---: | ---: |
| `bm25` | 6/12 | 0.605 | 0.643 |
| `dense` | 9/12 | 1.883 | 2.001 |
| `hybrid` | 10/12 | 2.457 | 2.625 |

| query_id | category | bm25 hit | dense hit | hybrid hit | dense latency (s) | hybrid latency (s) | hybrid overlap@5 vs dense |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilot_001` | course | 0 | 0 | 0 | 1.915 | 2.345 | 2 |
| `pilot_002` | course | 1 | 1 | 1 | 1.898 | 2.439 | 2 |
| `pilot_003` | program_credits | 0 | 1 | 1 | 1.980 | 2.486 | 3 |
| `pilot_004` | program_credits | 1 | 0 | 1 | 1.850 | 2.460 | 3 |
| `pilot_005` | program_comparison | 1 | 1 | 1 | 2.001 | 2.385 | 4 |
| `pilot_006` | faculty_research | 1 | 1 | 1 | 1.838 | 2.515 | 3 |
| `pilot_007` | faculty_research | 0 | 1 | 1 | 1.837 | 2.384 | 2 |
| `pilot_008` | institution_fact | 0 | 0 | 0 | 1.818 | 2.625 | 3 |
| `pilot_009` | time_sensitive_corpus_latest | 0 | 1 | 1 | 1.778 | 2.465 | 2 |
| `pilot_010` | time_sensitive_corpus_latest | 0 | 1 | 1 | 1.951 | 2.496 | 3 |
| `pilot_011` | english_faculty_research | 1 | 1 | 1 | 1.888 | 2.448 | 2 |
| `pilot_012` | english_research | 1 | 1 | 1 | 1.847 | 2.435 | 2 |

Open issues from the pilot:

- `pilot_001` (`深度学习` instructor) did not retrieve the expected course catalog source in any mode.
- `pilot_008` (SIST founding year) did not retrieve the expected college-introduction source in any mode.
- Hybrid improves expected-source hit@5 by one query over dense, but adds about 0.574 s average latency in this smoke
  setup.

## Phase 3 Gate

Gate: pass. `hybrid` has expected-source hit@5 of 10/12 versus `dense` at 9/12, with no critical citation-quality
regression in the top-5 inspection. The two shared misses should be handled as retrieval/data-refresh follow-ups, not as
blockers for Phase 3 generation.
