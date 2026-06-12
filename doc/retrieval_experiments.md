# Retrieval Evaluation Experiments

Last updated: 2026-06-13

This document records the current report-facing retrieval evaluation for the ShanghaiTech/SIST RAG system. The old
12-question Phase 2 pilot is historical only; the locked retrieval test paradigm is now the structured 100-question
Phase 5 set and the `src/evaluate` runner.

This is still a retrieval-only experiment. It measures whether the retriever surfaces expected official-source evidence,
not whether the local generator produces a correct final answer.

## Locked Test Paradigm

The official before/after retrieval convention for the report is:

| role | retrieval mode | purpose |
| --- | --- | --- |
| Before optimization | `dense` | Dense-only FAISS retrieval over `BAAI/bge-m3`, before BM25 fusion, RRF, de-duplication, enriched index text, weighted RRF, reranking, or query expansion. |
| After optimization | `hybrid` | Optimized retrieval using BM25+dense candidates, RRF fusion, source de-duplication/context packing, enriched index text, weighted dense RRF, strict source diversity, and selective local reranking. |
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
- Same SIST article IDs under different column paths, such as `c2863a1120270/page.htm` and
  `c7339a1120270/page.htm`, match after Fix 4.
- These URL/root/qrels canonicalization rules are evaluation hygiene. They apply to both before and after conditions and
  are not counted as retrieval optimizations.
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

Use the report-facing comparison below for retrieval-only report diagnostics. The timestamped rootfix directory is
retained as a control anchor, not as the final before/after result.

## Control Anchor Retrieval Results

Control anchor run:

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

The `remote_retrieve_full_rootfix_20260611` run above is the pre-Fix-1 control anchor. Each follow-up run changes one
factor and uses the same 100-question file, corpus snapshot, retrieval modes, and `top_k=5`.

Fix 1 through Fix 3 tables preserve their original remote summary JSON metrics. The Fix 4 and report-facing tables use
the shared same-article-ID evaluation alias; where needed, earlier immutable JSONL artifacts were rescored rather than
retrieved again.

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

### Fix 2: Enriched Index Text

Change:

- commit: `82768b2` (`Enrich retrieval index text`)
- branch: `codex/retrieval-control-url-canonicalization`
- behavior: BM25 tokenization and FAISS embeddings are built from enriched retrieval text:
  title, category, canonical URL, URL slug/path tokens, and raw chunk text.
- display snippets, packed context text, and source metadata still use the original SQLite chunk text.

Remote index build:

```text
remote worktree: /home/richard/cs290s-project3-RAG-retrieval-urlcanon
index artifacts:
  data/rag_enriched_20260611/bm25_enriched_20260611.pkl
  data/rag_enriched_20260611/faiss_bge_m3_enriched_20260611.index
  data/rag_enriched_20260611/chunk_index_enriched_20260611.jsonl
  data/rag_enriched_20260611/build_report_enriched_20260611.json
chunk_count: 35315
model: BAAI/bge-m3 local snapshot
```

Remote run:

```text
run_id: remote_retrieve_enriched_index_20260612
remote worktree: /home/richard/cs290s-project3-RAG-retrieval-urlcanon
remote artifacts:
  data/eval/run_remote_retrieve_enriched_index_20260612.jsonl
  data/eval/summary_remote_retrieve_enriched_index_20260612.json
  data/eval/review_queue_remote_retrieve_enriched_index_20260612.csv
  data/eval/gap_notes_remote_retrieve_enriched_index_20260612.md
  data/eval/results_before_after_remote_retrieve_enriched_index_20260612.xlsx
records: 200
status: dense 100 ok / 0 errors; hybrid 100 ok / 0 errors
```

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 | avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 0.65 | 0.85 | 0.813333 | 0.724333 | 0.721326 | 0.192 | 1.995500 |
| `hybrid` | 0.64 | 0.80 | 0.768333 | 0.705833 | 0.696288 | 0.178 | 2.948884 |

Delta versus Fix 1:

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | +0.14 | +0.11 | +0.090000 | +0.135833 | +0.112808 | +0.020 |
| `hybrid` | +0.08 | +0.09 | +0.075000 | +0.082666 | +0.070599 | +0.016 |

Per-question top-5 source-hit overlap after fix 2:

| dense hit@5 | hybrid hit@5 | questions |
| ---: | ---: | ---: |
| 0 | 0 | 14 |
| 0 | 1 | 1 |
| 1 | 0 | 6 |
| 1 | 1 | 79 |

Per-question top-1 source-hit overlap after fix 2:

| dense hit@1 | hybrid hit@1 | questions |
| ---: | ---: | ---: |
| 0 | 0 | 28 |
| 0 | 1 | 7 |
| 1 | 0 | 8 |
| 1 | 1 | 57 |

Interpretation:

- Enriched index text is the largest accepted retrieval gain so far, improving both top-rank quality and top-5 coverage
  for both retrieval conditions.
- The remaining both-missed top-5 set is 14 questions, down from 23 after Fix 1.
- Dense remains stronger on top-5 coverage, but hybrid keeps competitive top-rank quality after the same index enrichment.

### Fix 3: Weighted Hybrid RRF

Change:

- commit: `dedac87` (`Weight dense channel in hybrid RRF`)
- branch: `codex/retrieval-control-url-canonicalization`
- behavior: hybrid RRF now uses default channel weights `sparse_weight=1.0` and `dense_weight=1.5`.
- no candidate pool size, qrels canonicalization, diagnostic-depth, index text, reranker, or boost changes.

Remote run:

```text
run_id: remote_retrieve_weighted_rrf_dense15_20260612
remote worktree: /home/richard/cs290s-project3-RAG-retrieval-urlcanon
remote artifacts:
  data/eval/run_remote_retrieve_weighted_rrf_dense15_20260612.jsonl
  data/eval/summary_remote_retrieve_weighted_rrf_dense15_20260612.json
  data/eval/review_queue_remote_retrieve_weighted_rrf_dense15_20260612.csv
  data/eval/gap_notes_remote_retrieve_weighted_rrf_dense15_20260612.md
  data/eval/results_before_after_remote_retrieve_weighted_rrf_dense15_20260612.xlsx
records: 200
status: dense 100 ok / 0 errors; hybrid 100 ok / 0 errors
```

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 | avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 0.65 | 0.85 | 0.813333 | 0.724333 | 0.721326 | 0.192 | 2.076126 |
| `hybrid` | 0.64 | 0.82 | 0.788333 | 0.710833 | 0.709939 | 0.190 | 2.989514 |

Delta versus Fix 2:

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | +0.00 | +0.00 | +0.000000 | +0.000000 | +0.000000 | +0.000 |
| `hybrid` | +0.00 | +0.02 | +0.020000 | +0.005000 | +0.013651 | +0.012 |

Per-question top-5 source-hit overlap after fix 3:

| dense hit@5 | hybrid hit@5 | questions |
| ---: | ---: | ---: |
| 0 | 0 | 13 |
| 0 | 1 | 2 |
| 1 | 0 | 5 |
| 1 | 1 | 80 |

Per-question top-1 source-hit overlap after fix 3:

| dense hit@1 | hybrid hit@1 | questions |
| ---: | ---: | ---: |
| 0 | 0 | 28 |
| 0 | 1 | 7 |
| 1 | 0 | 8 |
| 1 | 1 | 57 |

Interpretation:

- Weighted RRF gives a small positive top-5 gain for hybrid without changing dense retrieval.
- Hybrid `source_hit@1` is unchanged, so the effect is deeper top-5 evidence ordering rather than first-rank correction.
- The remaining both-missed top-5 set is 13 questions, down from 14 after Fix 2.

### Fix 4: Strict Source Diversity and Same-Article Evaluation Alias

Change:

- local implementation: strict final source diversity in hybrid retrieval.
- behavior: hybrid final top-5 now defaults to `url_cap=1` and de-duplicates by normalized URL, document ID, and SIST
  article ID family.
- evaluation hygiene: source metrics now treat same SIST article IDs under different column paths as aliases. This
  canonicalization is applied to all conditions and is not counted as a retrieval optimization.
- no candidate pool size, index text, reranker, query expansion, or generation changes.

Remote run:

```text
run_id: remote_retrieve_strict_diversity_articlealias_20260612
remote worktree: /home/richard/cs290s-project3-RAG-retrieval-urlcanon
remote artifacts:
  data/eval/run_remote_retrieve_strict_diversity_articlealias_20260612.jsonl
  data/eval/summary_remote_retrieve_strict_diversity_articlealias_20260612.json
  data/eval/review_queue_remote_retrieve_strict_diversity_articlealias_20260612.csv
  data/eval/gap_notes_remote_retrieve_strict_diversity_articlealias_20260612.md
  data/eval/results_before_after_remote_retrieve_strict_diversity_articlealias_20260612.xlsx
records: 200
status: dense 100 ok / 0 errors; hybrid 100 ok / 0 errors
```

The table below uses the shared same-article evaluation alias for both rows. The Fix 3 row was rescored from its
existing immutable JSONL artifact; the retrieval itself was not rerun.

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 | avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fix 3 `hybrid`, rescored | 0.66 | 0.83 | 0.798333 | 0.727500 | 0.728041 | 0.192 | 2.989514 |
| Fix 4 `hybrid` | 0.66 | 0.85 | 0.811667 | 0.735333 | 0.716696 | 0.182 | 2.969989 |
| Delta | +0.00 | +0.02 | +0.013334 | +0.007833 | -0.011345 | -0.010 | -0.019525 |

Per-question top-5 source-hit overlap after Fix 4:

| dense hit@5 | hybrid hit@5 | questions |
| ---: | ---: | ---: |
| 0 | 0 | 11 |
| 0 | 1 | 4 |
| 1 | 0 | 4 |
| 1 | 1 | 81 |

Interpretation:

- Strict source diversity improves coverage: hybrid `source_hit@5` rises from 0.83 to 0.85 under the shared
  same-article evaluation policy, and `source_recall@5` rises from 0.798333 to 0.811667.
- The improvement comes from recovering `q039` and `q066` in hybrid top-5 by preventing repeated source variants from
  consuming final slots.
- This is a coverage-oriented optimization, not a uniform win: `ndcg@5` drops by 0.011345 and `precision@5` drops by
  0.010 because some repeated relevant hits are replaced by diverse but non-relevant sources.

### Fix 5: Local CrossEncoder Rerank over Top-50 Hybrid Candidates

Change:

- eval plumbing: expose hybrid candidate, fusion, rerank, RRF, and URL-cap knobs through `rag-evaluate`.
- local reranker runtime: cache local `CrossEncoder` instances per `Retriever` and resolved model path so a full eval
  does not reload the reranker for every hybrid query.
- control setting: hybrid sparse candidate depth 50, dense candidate depth 50, fused depth 50, rerank depth 50, final
  top 5, and strict source diversity unchanged.
- reranker model: local `BAAI/bge-reranker-v2-m3` snapshot at `/home/richard/models/bge-reranker-v2-m3`.
- no source-type priors, structured sidecar retrieval, query expansion, qrels edits, or index-text changes.

Remote run:

```text
run_id: remote_retrieve_hybrid_rerank_bge_v2_m3_retry_20260612
remote worktree: /home/richard/cs290s-project3-RAG-retrieval-urlcanon
commits:
  a5b20b0 Expose hybrid retrieval knobs in evaluation
  3a7fa37 Cache local hybrid reranker models
remote artifacts:
  data/eval/run_remote_retrieve_hybrid_rerank_bge_v2_m3_retry_20260612.jsonl
  data/eval/summary_remote_retrieve_hybrid_rerank_bge_v2_m3_retry_20260612.json
  data/eval/review_queue_remote_retrieve_hybrid_rerank_bge_v2_m3_retry_20260612.csv
  data/eval/gap_notes_remote_retrieve_hybrid_rerank_bge_v2_m3_retry_20260612.md
  data/eval/results_before_after_remote_retrieve_hybrid_rerank_bge_v2_m3_retry_20260612.xlsx
records: 200
status: dense 100 ok / 0 errors; hybrid 100 ok / 0 errors
```

The first attached-SSH attempt, `remote_retrieve_hybrid_rerank_bge_v2_m3_20260612`, is discarded because the SSH stdout
pipe broke during the run and produced 84 retrieval errors. The retry above ran detached with stdout redirected.

The Top-50 pool-only control was run before enabling the reranker:

| run | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 | avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fix 4 `hybrid` | 0.66 | 0.85 | 0.811667 | 0.735333 | 0.716696 | 0.182 | 2.969989 |
| Top-50 pool-only `hybrid` | 0.65 | 0.84 | 0.815000 | 0.727333 | 0.712832 | 0.180 | 2.925436 |
| Delta | -0.01 | -0.01 | +0.003333 | -0.008000 | -0.003864 | -0.002 | -0.044553 |

The reranker result compared with the Fix 4 anchor:

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 | avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fix 4 `hybrid` | 0.66 | 0.85 | 0.811667 | 0.735333 | 0.716696 | 0.182 | 2.969989 |
| Fix 5 `hybrid` + local reranker | 0.58 | 0.86 | 0.826667 | 0.685500 | 0.682113 | 0.188 | 85.482282 |
| Delta | -0.08 | +0.01 | +0.015000 | -0.049833 | -0.034583 | +0.006 | +82.512293 |

Interpretation:

- The local CrossEncoder directly improves top-5 coverage: `source_hit@5` rises from 0.85 to 0.86 and
  `source_recall@5` rises from 0.811667 to 0.826667.
- It is not a clean default optimization. Top-rank quality regresses sharply: `source_hit@1` drops by 0.08, `mrr@5`
  drops by 0.049833, and `ndcg@5` drops by 0.034583.
- CPU latency is too high for the current deployment path. Even with model caching, hybrid latency rises from 2.969989s
  to 85.482282s per query on the remote host, where no GPU was visible.
- The report-facing after-optimization condition should remain Fix 4 unless the report explicitly chooses top-5
  coverage as the only optimization target. Fix 5 is best treated as evidence that full CrossEncoder reranking needs a
  lighter or more selective variant before adoption.

### Fix 6: Selective Top-20 Local Rerank with Fused Top-2 Preservation

Change:

- local implementation: add `rerank_preserve_top_k` to hybrid retrieval so the first N fused candidates keep their RRF
  order and only candidates after that preserved prefix are passed to the local CrossEncoder.
- eval plumbing: expose `--rerank-preserve-top-k` through `rag-retrieve`, `rag-evaluate`, and `EvaluationConfig`.
- control setting: hybrid sparse candidate depth 50, dense candidate depth 50, fused depth 50, rerank depth 20,
  preserve fused top 2, final top 5, and strict source diversity unchanged.
- reranker model: local `BAAI/bge-reranker-v2-m3` snapshot at `/home/richard/models/bge-reranker-v2-m3`.
- no source-type priors, structured sidecar retrieval, query expansion, qrels edits, or index-text changes.

Remote run:

```text
run_id: remote_retrieve_hybrid_rerank_preserve2_top20_20260612
remote worktree: /home/richard/cs290s-project3-RAG-retrieval-urlcanon
commit:
  f260aa2 Add selective hybrid rerank prefix preservation
remote artifacts:
  data/eval/run_remote_retrieve_hybrid_rerank_preserve2_top20_20260612.jsonl
  data/eval/summary_remote_retrieve_hybrid_rerank_preserve2_top20_20260612.json
  data/eval/review_queue_remote_retrieve_hybrid_rerank_preserve2_top20_20260612.csv
  data/eval/gap_notes_remote_retrieve_hybrid_rerank_preserve2_top20_20260612.md
  data/eval/results_before_after_remote_retrieve_hybrid_rerank_preserve2_top20_20260612.xlsx
records: 200
status: dense 100 ok / 0 errors; hybrid 100 ok / 0 errors
```

The selective reranker result compared with the Fix 4 anchor:

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 | avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fix 4 `hybrid` | 0.66 | 0.85 | 0.811667 | 0.735333 | 0.716696 | 0.182 | 2.969989 |
| Fix 6 `hybrid` + selective reranker | 0.65 | 0.88 | 0.853333 | 0.748333 | 0.738153 | 0.192 | 33.603731 |
| Delta | -0.01 | +0.03 | +0.041666 | +0.013000 | +0.021457 | +0.010 | +30.633742 |

Compared with the full top-50 CrossEncoder reranker from Fix 5:

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 | avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fix 5 full top-50 rerank | 0.58 | 0.86 | 0.826667 | 0.685500 | 0.682113 | 0.188 | 85.482282 |
| Fix 6 selective top-20 rerank | 0.65 | 0.88 | 0.853333 | 0.748333 | 0.738153 | 0.192 | 33.603731 |
| Delta | +0.07 | +0.02 | +0.026666 | +0.062833 | +0.056040 | +0.004 | -51.878551 |

Per-question top-5 source-hit changes:

| comparison | gained | lost |
| --- | --- | --- |
| Fix 6 vs Fix 4 | `q021`, `q031`, `q037`, `q072` | `q013` |
| Fix 6 vs Fix 5 full top-50 rerank | `q031`, `q066`, `q069`, `q088` | `q013`, `q041` |

Interpretation:

- Selective reranking is the strongest retrieval-only result so far on top-5 coverage: `source_hit@5` reaches 0.88 and
  `source_recall@5` reaches 0.853333.
- Preserving the fused top 2 prevents most of the first-rank damage seen in full CrossEncoder reranking. `source_hit@1`
  is still 0.01 below Fix 4, but far above the full top-50 reranker at 0.58.
- Rank-quality diagnostics improve over Fix 4: `mrr@5` rises by 0.013000 and `ndcg@5` rises by 0.021457.
- Latency remains expensive on CPU at 33.603731s per hybrid query, but it is much lower than full top-50 reranking.
  This is a report-usable accuracy optimization only if the report clearly states the latency tradeoff.

### Fix 7: CUDA Local Reranker Runtime

Change:

- local implementation: add `reranker_device` to hybrid retrieval so the local CrossEncoder can run on `cuda` when a
  GPU is available.
- eval plumbing: expose `--reranker-device` through `rag-retrieve`, `rag-evaluate`, and `EvaluationConfig`.
- control setting: hybrid sparse candidate depth 50, dense candidate depth 50, fused depth 50, rerank depth 20,
  preserve fused top 2, final top 5, strict source diversity, and the local reranker model are unchanged.
- reranker model: local `BAAI/bge-reranker-v2-m3` snapshot at `/home/richard/models/bge-reranker-v2-m3`.
- no source-type priors, structured sidecar retrieval, query expansion, qrels edits, index-text changes, or generation
  changes.

Remote run:

```text
run_id: remote_retrieve_reranker_cuda_20260613
remote worktree: /home/richard/cs290s-project3-RAG-fix7-reranker-device
commit:
  ffa8b22 Allow CUDA hybrid reranking
remote artifacts:
  data/eval/run_remote_retrieve_reranker_cuda_20260613.jsonl
  data/eval/summary_remote_retrieve_reranker_cuda_20260613.json
  data/eval/review_queue_remote_retrieve_reranker_cuda_20260613.csv
  data/eval/gap_notes_remote_retrieve_reranker_cuda_20260613.md
  data/eval/results_before_after_remote_retrieve_reranker_cuda_20260613.xlsx
records: 200
status: dense 100 ok / 0 errors; hybrid 100 ok / 0 errors
```

The CUDA reranker result compared with the Fix 6 CPU reranker anchor:

| mode | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 | avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fix 6 `hybrid` + CPU selective reranker | 0.65 | 0.88 | 0.853333 | 0.748333 | 0.738153 | 0.192 | 33.603731 |
| Fix 7 `hybrid` + CUDA selective reranker | 0.65 | 0.88 | 0.853333 | 0.748333 | 0.738153 | 0.192 | 5.141354 |
| Delta | +0.00 | +0.00 | +0.000000 | +0.000000 | +0.000000 | +0.000 | -28.462377 |

Interpretation:

- CUDA reranking keeps the Fix 6 retrieval ranking exactly unchanged while removing the CPU latency penalty.
- Hybrid top-5 coverage remains the best accepted retrieval-only result so far: `source_hit@5` is 0.88 and
  `source_recall@5` is 0.853333.
- Hybrid average latency falls from 33.603731s to 5.141354s per query, a 6.5x speedup for the after-optimization
  condition on the remote runner with one CUDA device visible.
- This is a runtime optimization, not a retrieval-ranking optimization. It should be reported as preserving Fix 6
  accuracy while making the selective reranker practical for GPU deployment.

## Report-Facing Before/After Comparison

The official retrieval-only report comparison uses a shared evaluation-canonicalization policy for both conditions:

| condition | source_hit@1 | source_hit@5 | source_recall@5 | mrr@5 | ndcg@5 | precision@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before optimization: Fix 1 `dense`, rescored | 0.52 | 0.74 | 0.723333 | 0.594833 | 0.614946 | 0.172 |
| After optimization: Fix 7 `hybrid` CUDA selective reranker | 0.65 | 0.88 | 0.853333 | 0.748333 | 0.738153 | 0.192 |
| Delta | +0.13 | +0.14 | +0.130000 | +0.153500 | +0.123207 | +0.020 |

This is the clean report-facing retrieval comparison because Fix 1 applies the shared URL/qrels canonicalization without
changing retrieval ranking, while Fix 7 includes the accepted retrieval optimizations plus CUDA execution for the local
selective reranker. The main caveat is deployment-dependent runtime: Fix 7 preserves Fix 6 accuracy and reduces hybrid
latency from 33.603731s on CPU to 5.141354s on the remote CUDA runner, but a CPU-only deployment should still expect the
slower Fix 6 latency profile.

The Fix 7 `dense` row is a diagnostic ceiling, not the official before-optimization baseline. It uses the enriched
index text introduced as an optimization, so comparing latest `dense` directly against latest `hybrid` answers a
different diagnostic question: whether hybrid fusion and selective reranking beat dense after both modes receive the
same enriched index. On that diagnostic view, latest `hybrid` is ahead on `source_hit@5` 0.88 versus dense 0.85 and
`mrr@5` 0.748333 versus dense 0.742667, while dense is essentially tied on `ndcg@5` 0.738275 versus hybrid 0.738153.

## Remaining Failure Taxonomy

After Fix 7, the top-5 source-hit overlap is unchanged from Fix 6:

| result type | question count | question IDs |
| --- | ---: | --- |
| Both hit | 81 | not listed individually |
| Both miss | 8 | `q006`, `q022`, `q023`, `q024`, `q032`, `q033`, `q041`, `q046` |
| Dense-only hit | 4 | `q003`, `q013`, `q018`, `q050` |
| Hybrid-only hit | 7 | `q029`, `q031`, `q037`, `q039`, `q066`, `q072`, `q099` |

Both-missed questions are dominated by factual and comparative cases. The
expected URLs for checked disputed questions are present in the enriched chunk index, so the remaining issue is usually
ranking, sibling pages, or qrels narrowness rather than complete corpus absence.

Main observed causes:

- Selective reranking recovered several expected sources that previously sat below top 5, including `q021`, `q031`,
  `q037`, and `q072`, but did not resolve `q022`.
- Strict source diversity and selective reranking fixed some repeated-source failures, including `q039`, `q066`, and
  `q072`, but course-table and faculty profile cases still need stronger intent signals.
- Some official pages are answer-bearing siblings but fail the current qrels, such as Chinese/English sibling pages and
  faculty `main.htm` versus list/profile variants.
- Course-table and faculty/course join questions need more structured retrieval signals than chunk similarity alone.
- Hybrid CUDA selective reranking is the strongest accepted top-5 retrieval condition so far, but sparse terms and
  reranker scores can still lift sibling pages or old list pages above the exact expected URL.

## Next Optimization Candidates

Prioritize actual retrieval changes separately from metric/qrels cleanup:

1. Test local Query2doc or HyDE-style query expansion as a retrieval-only candidate generator. Expanded text must remain
   retrieval-only and must never become packed context or a citation source.
2. Try contextual chunk enrichment v2 in a separate generated index, focusing on breadcrumbs, source type, update date,
   entity aliases, and Chinese/English sibling hints while keeping displayed contexts as original official text.
3. Revisit source-type priors only with narrower intent-specific rules. A broad structured-sidecar prior trial on
   2026-06-13 regressed hybrid `source_hit@5` from 0.88 to 0.86, so the next version needs a more specific hypothesis.

Keep qrels cleanup separate from retrieval optimization. The next qrels audit should target Chinese/English official
siblings and answer-bearing faculty/list pages; same-article-ID aliases are already covered by the shared evaluation
canonicalization policy. Existing remote diagnostics suggest that larger hybrid candidate pools alone were worse,
`rerank_top_k=10` lost top-5 coverage, and the simple metadata boost trial matched the weighted RRF metrics exactly, so
none of those should be the next priority without a more specific hypothesis.

## Verification

The root URL matching fix, URL-canonicalization fix, and evaluation module were checked locally:

```bash
uv run --locked --no-sync --offline python -m pytest tests/unit/test_evaluate_core.py tests/integration/test_evaluate_phase5.py -q
uv run --locked --no-sync --offline ruff check src/evaluate tests/unit/test_evaluate_core.py tests/integration/test_evaluate_phase5.py
```

The eval-plumbing and reranker-cache changes were checked locally:

```bash
uv run --locked --no-sync --offline python -m pytest tests/integration/test_evaluate_phase5.py -q
uv run --locked --no-sync --offline ruff check src/evaluate/cli.py src/evaluate/runner.py tests/integration/test_evaluate_phase5.py
uv run --locked --no-sync --offline python -m pytest tests/integration/test_rag_ingest_index.py::test_hybrid_reranker_reorders_with_local_model tests/integration/test_rag_ingest_index.py::test_hybrid_reranker_reuses_local_model_for_same_retriever tests/integration/test_rag_ingest_index.py::test_hybrid_reranker_reports_missing_local_model -q
uv run --locked --no-sync --offline ruff check src/rag/retrieve.py tests/integration/test_rag_ingest_index.py
```

The reranker-cache patch was also checked on the remote worktree with the same focused reranker tests and ruff command.
The final reranker retry wrote 200 JSONL records, 201 review-queue CSV lines, 202 gap-note lines, a summary JSON, and
the Excel workbook listed above.

The selective reranker prefix-preservation change was checked locally and remotely:

```bash
uv run --locked --no-sync --offline python -m pytest tests/integration/test_rag_ingest_index.py::test_hybrid_reranker_can_preserve_fused_prefix tests/integration/test_rag_ingest_index.py::test_hybrid_reranker_skips_prediction_when_preserved_prefix_covers_window tests/integration/test_rag_ingest_index.py::test_hybrid_reranker_reorders_with_local_model tests/integration/test_rag_ingest_index.py::test_hybrid_cli_json_includes_hits_contexts_and_config tests/integration/test_evaluate_phase5.py::test_evaluate_retrieve_passes_hybrid_knobs_only_to_hybrid -q
uv run --locked --no-sync --offline ruff check src/rag/retrieve.py src/evaluate/cli.py src/evaluate/runner.py tests/integration/test_rag_ingest_index.py tests/integration/test_evaluate_phase5.py
```

The remote selective reranker run used one cached CrossEncoder load and 200 dense encoder loads, then wrote 200 JSONL
records, 201 review-queue CSV lines, 202 gap-note lines, a summary JSON, and the Excel workbook listed above.

The CUDA reranker device change was checked locally and remotely:

```bash
uv run --locked --no-sync --offline python -m pytest tests/integration/test_rag_ingest_index.py tests/integration/test_evaluate_phase5.py -q
uv run --locked --no-sync --offline ruff check src/rag/retrieve.py src/evaluate/cli.py src/evaluate/runner.py tests/integration/test_rag_ingest_index.py tests/integration/test_evaluate_phase5.py
```

The remote CUDA run used `PYTHONPATH=src` so the branch worktree source, rather than the shared virtualenv's editable
install path, supplied `evaluate.cli` and `rag.retrieve`. A one-question CUDA smoke run completed before the full
100-question retrieval evaluation.

The enriched index text change was checked locally:

```bash
uv run --locked --no-sync --offline python -m pytest tests/integration/test_rag_ingest_index.py tests/integration/test_evaluate_phase5.py tests/unit/test_evaluate_core.py -q
uv run --locked --no-sync --offline ruff check src/rag/index.py tests/integration/test_rag_ingest_index.py
```

The weighted RRF change was checked locally:

```bash
uv run --locked --no-sync --offline python -m pytest tests/integration/test_rag_ingest_index.py tests/integration/test_evaluate_phase5.py tests/unit/test_evaluate_core.py -q
uv run --locked --no-sync --offline ruff check src/rag/retrieve.py tests/integration/test_rag_ingest_index.py
```

The strict source diversity and same-article alias changes were checked locally and remotely:

```bash
uv run --locked --no-sync --offline python -m pytest tests/unit/test_evaluate_core.py tests/integration/test_rag_ingest_index.py tests/integration/test_evaluate_phase5.py -q
uv run --locked --no-sync --offline ruff check src/rag src/evaluate tests/unit/test_evaluate_core.py tests/integration/test_rag_ingest_index.py tests/integration/test_evaluate_phase5.py
```

Remote validation:

```text
run_remote_retrieve_full_rootfix_20260611.jsonl: 200 lines
review_queue_remote_retrieve_full_rootfix_20260611.csv: 201 lines including header
gap_notes_remote_retrieve_full_rootfix_20260611.md: 202 lines
run_remote_retrieve_urlcanon_20260611.jsonl: 200 lines
review_queue_remote_retrieve_urlcanon_20260611.csv: 201 lines including header
gap_notes_remote_retrieve_urlcanon_20260611.md: 202 lines
run_remote_retrieve_enriched_index_20260612.jsonl: 200 lines
review_queue_remote_retrieve_enriched_index_20260612.csv: 201 lines including header
gap_notes_remote_retrieve_enriched_index_20260612.md: 202 lines
run_remote_retrieve_weighted_rrf_dense15_20260612.jsonl: 200 lines
review_queue_remote_retrieve_weighted_rrf_dense15_20260612.csv: 201 lines including header
gap_notes_remote_retrieve_weighted_rrf_dense15_20260612.md: 202 lines
run_remote_retrieve_strict_diversity_articlealias_20260612.jsonl: 200 lines
review_queue_remote_retrieve_strict_diversity_articlealias_20260612.csv: 201 lines including header
gap_notes_remote_retrieve_strict_diversity_articlealias_20260612.md: 202 lines
run_remote_retrieve_hybrid_rerank_preserve2_top20_20260612.jsonl: 200 lines
review_queue_remote_retrieve_hybrid_rerank_preserve2_top20_20260612.csv: 201 lines including header
gap_notes_remote_retrieve_hybrid_rerank_preserve2_top20_20260612.md: 202 lines
run_remote_retrieve_reranker_cuda_20260613.jsonl: 200 lines
review_queue_remote_retrieve_reranker_cuda_20260613.csv: 201 lines including header
gap_notes_remote_retrieve_reranker_cuda_20260613.md: 202 lines
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
