# ShanghaiTech/SIST RAG Evaluation Context

This context defines report and evaluation language for the ShanghaiTech/SIST RAG system. It keeps retrieval,
generation, and assignment-result terms stable across docs, experiments, and the final report.

## Language

**Before Optimization**:
The official pre-optimization retrieval condition for report comparison, currently dense FAISS retrieval with
`BAAI/bge-m3`.
_Avoid_: BM25 baseline, unoptimized system

**After Optimization**:
The official post-optimization retrieval condition for report comparison, currently hybrid retrieval.
_Avoid_: reranked system, optimized answer

**Diagnostic Baseline**:
A retrieval mode used to explain failures or tradeoffs, but not used as the official before/after condition.
_Avoid_: before optimization

**Hybrid Retrieval**:
The optimized retrieval mode that fuses BM25 and dense candidates with RRF, then de-duplicates and packs contexts.
_Avoid_: dense retrieval, reranker

**Retrieval Pilot**:
A small retrieval-only validation run before generator and full evaluation work.
_Avoid_: final evaluation, Excel evaluation

**Expected Source Hit@5**:
A retrieval metric that is true when at least one top-5 retrieved URL matches a question's expected official source
URL prefix.
_Avoid_: accuracy, correctness

**Packed Context**:
A final retrieved chunk prepared for generation or UI display with source metadata and trace linkage.
_Avoid_: answer, system response

**Generated Answer**:
A model-produced answer grounded in retrieved official-source contexts, with citations that map back to structured
sources. This is the answer type eligible for Phase 5 `sys_resp_before_opt` and `sys_resp_after_opt` fields.
_Avoid_: packed context, retrieved snippet

**Evidence-Insufficient Answer**:
A generated-answer status used when the system cannot cite enough official-source evidence to answer safely.
_Avoid_: failed retrieval, wrong answer

## Relationships

- **Before Optimization** and **After Optimization** are the two conditions compared in the final report.
- **Hybrid Retrieval** is the current **After Optimization** implementation.
- **Diagnostic Baseline** can help interpret results but does not define assignment Excel columns.
- **Retrieval Pilot** checks retrieved sources and **Packed Context** quality before generated answers exist.
- **Expected Source Hit@5** measures retrieval evidence quality, not answer correctness.
- **Generated Answer** is produced from **Packed Context** values and can fill the final assignment response columns.
- **Evidence-Insufficient Answer** is a valid generated-answer outcome when the retrieved evidence is not usable.

## Example Dialogue

> **Dev:** "Can I put the top hybrid snippet into `sys_resp_after_opt`?"
> **Domain expert:** "No. That is a **Packed Context**, not a generated answer. Use it in the **Retrieval Pilot**, then
> fill `sys_resp_after_opt` only after the generator exists."
>
> **Dev:** "The model returned text but no `[1]` citation. Can I keep it?"
> **Domain expert:** "No. Treat it as an **Evidence-Insufficient Answer** because the generated answer is not grounded
> in a cited official source."

## Flagged Ambiguities

- "Baseline" can mean BM25, dense, or any unoptimized condition. Resolved: use **Before Optimization** for dense and
  **Diagnostic Baseline** for BM25 unless the report convention is explicitly reopened.
- "Hit" can mean source retrieval or answer correctness. Resolved: **Expected Source Hit@5** is source-level retrieval
  evidence, while final answer correctness belongs to the Phase 5 Excel evaluation.
- "Answer" can mean a generated response or a retrieved snippet. Resolved: use **Generated Answer** for model output and
  **Packed Context** for retrieved evidence.
