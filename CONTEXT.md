# ShanghaiTech/SIST RAG Evaluation Context

This context defines report and evaluation language for the ShanghaiTech/SIST RAG system. It keeps retrieval,
generation, and assignment-result terms stable across docs, experiments, and the final report.

## Language

**Before Optimization**:
The official pre-optimization retrieval condition for report comparison: dense-only FAISS retrieval with
`BAAI/bge-m3`, without BM25 fusion, RRF, source de-duplication, enriched index text, weighted RRF, reranking, or
query expansion.
_Avoid_: BM25 baseline, latest dense diagnostic result

**After Optimization**:
The official post-optimization retrieval condition for report comparison: optimized hybrid retrieval after the
implemented retrieval changes, including BM25+dense candidates, RRF fusion, source de-duplication/context packing,
enriched index text, weighted dense RRF, and strict source diversity in the final top-5.
_Avoid_: reranked system, optimized answer

**Evaluation Canonicalization**:
A shared metric-normalization layer applied to all report retrieval conditions. It normalizes URL aliases, such as root
SIST URLs, `_tNNN` template path variants, and same SIST article IDs under different column paths, so before/after
differences reflect retrieval behavior rather than qrels format differences.
_Avoid_: retrieval optimization, after-only fix

**Diagnostic Baseline**:
A retrieval mode used to explain failures or tradeoffs, but not used as the official before/after condition.
_Avoid_: before optimization

**Hybrid Retrieval**:
The optimized retrieval mode that fuses BM25 and dense candidates with RRF, applies strict source diversity, then packs
contexts.
_Avoid_: dense retrieval, reranker

**Retrieval Pilot**:
A small retrieval-only validation run before generator and full evaluation work.
_Avoid_: final evaluation, Excel evaluation

**Expected Source Hit@5**:
A retrieval metric that is true when at least one top-5 retrieved URL matches a question's expected official source
URL prefix.
_Avoid_: accuracy, correctness

**Cited Expected Source Hit**:
An answer-generation diagnostic that is true when a generated answer cites an official source matching the question's
expected source URL.
_Avoid_: retrieval hit, answer correctness

**Packed Context**:
A final retrieved chunk prepared for generation or UI display with source metadata and trace linkage.
_Avoid_: answer, system response

**Generated Answer**:
A model-produced answer grounded in retrieved official-source contexts, with citations that map back to structured
sources. This is the answer type eligible for Phase 5 `sys_resp_before_opt` and `sys_resp_after_opt` fields.
_Avoid_: packed context, retrieved snippet

**Source-Derived Generated Answer**:
A deterministic generated-answer fallback synthesized only from the user query, packed contexts, and source metadata
when the local model draft is rejected. It must be concise, cite retrieved source numbers, and pass the same citation
and leakage validation as model text.
_Avoid_: answer key extraction, uncited snippet copy

**Evidence-Insufficient Answer**:
A generated-answer abstention used when the system cannot cite enough official-source evidence to answer safely.
_Avoid_: failed retrieval, wrong answer, manual review

**Answer Synthesis Miss**:
An answer-generation diagnostic where **Expected Source Hit@5** is true for retrieved sources, but the final generated
answer either abstains as **Evidence-Insufficient** or fails **Cited Expected Source Hit**.
_Avoid_: retrieval miss, source_hit@5

## Relationships

- **Before Optimization** and **After Optimization** are the two conditions compared in the final report.
- **Hybrid Retrieval** is the current **After Optimization** implementation.
- **Evaluation Canonicalization** applies to both **Before Optimization** and **After Optimization**.
- **Diagnostic Baseline** can help interpret results but does not define assignment Excel columns.
- **Retrieval Pilot** checks retrieved sources and **Packed Context** quality before generated answers exist.
- **Expected Source Hit@5** measures retrieval evidence quality, not answer correctness.
- **Cited Expected Source Hit** measures answer citation grounding, not answer correctness.
- **Generated Answer** is produced from **Packed Context** values and can fill the final assignment response columns.
- **Source-Derived Generated Answer** is allowed only after a rejected local-model draft and before repair or abstention;
  it is still a **Generated Answer** because it is synthesized and cited, not a raw packed context.
- **Evidence-Insufficient Answer** is a valid generated-answer outcome when the retrieved evidence is not usable, but it
  still maps to `is_correct=0` for final assignment labels.
- **Answer Synthesis Miss** separates retrieval success from generation failure when expected official sources were
  retrieved but the final answer did not cite them.
- Dense retrieval built on enriched index text is a diagnostic ceiling, not the official **Before Optimization** baseline,
  because enriched index text is one of the implemented optimization changes.

## Example Dialogue

> **Dev:** "Can I put the top hybrid snippet into `sys_resp_after_opt`?"
> **Domain expert:** "No. That is a **Packed Context**, not a generated answer. Use it in the **Retrieval Pilot**, then
> fill `sys_resp_after_opt` only after the generator exists."
>
> **Dev:** "The model returned text but no `[1]` citation. Can I keep it?"
> **Domain expert:** "Not directly. First try citation repair against the **Packed Context** values; if it cannot
> produce a cited **Generated Answer**, treat it as an **Evidence-Insufficient Answer**."

## Flagged Ambiguities

- "Baseline" can mean BM25, dense, or any unoptimized condition. Resolved: use **Before Optimization** for dense-only
  FAISS retrieval and **Diagnostic Baseline** for BM25 unless the report convention is explicitly reopened.
- "Fix" can mean evaluation cleanup or retrieval optimization. Resolved: **Evaluation Canonicalization** is shared metric
  hygiene, while enriched index text, weighted hybrid RRF, and strict source diversity are retrieval optimizations.
- "Hit" can mean source retrieval or answer correctness. Resolved: **Expected Source Hit@5** is source-level retrieval
  evidence, **Cited Expected Source Hit** is answer citation grounding, and final answer correctness belongs to the
  Phase 5 Excel evaluation.
- "Answer" can mean a generated response or a retrieved snippet. Resolved: use **Generated Answer** for model output and
  **Packed Context** for retrieved evidence.
- "source_hit@5" can be ambiguous in answer reports. Resolved: use **Expected Source Hit@5** or
  `retrieved_expected_source_hit@5` for retrieved sources, and **Cited Expected Source Hit** or
  `cited_expected_source_hit@5` for final answer citations.
