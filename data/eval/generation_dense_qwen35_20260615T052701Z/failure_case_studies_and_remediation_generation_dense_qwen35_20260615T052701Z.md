# Failure Case Studies and Remediation Plan

Run: `generation_dense_qwen35_20260615T052701Z`

This note follows the manual review in
`manual_score_incorrect_generation_dense_qwen35_20260615T052701Z.md`.

## Executive Finding

The poor score is mostly an answer-generation problem, not an evaluator problem.
After manually reviewing the 28 script-level `incorrect` answers, only 3 are clear
evaluator false negatives. The adjusted result is therefore 57/100, not a large
hidden win masked by the evaluator.

The failure pattern is more specific:

- 15/28 `incorrect` answers retrieved and cited an expected source, but still missed required facts.
- 12/28 `incorrect` answers did not retrieve or cite the expected source.
- 1/28 retrieved an expected source but did not cite it in the final answer.

The fix should prioritize evidence extraction and answer validation over further
judge loosening.

## Case Study 1: Correct Source, Missing Numeric Table Facts

Representative question: `q004`

Question: 2025 EE undergraduate plan, total graduation credits and elective credits.

Expected answer: 145 total credits; 9 elective credits.

Observed answer: The model cited the 2025 EE curriculum page but said the source
did not contain the required numbers.

Signals:

- `retrieved_expected_source_hit@5 = 1.0`
- `cited_expected_source_hit@5 = 1.0`
- `answer_synthesis_miss = 0.0`
- Manual score: 0

Root cause:

The retrieval layer found the right page, but the answer context did not expose
the relevant table row strongly enough. The local evidence selector favored
query anchors and nearby text, while the required numeric facts lived in a
table-like section. The generator then made a conservative but wrong abstention.

Same family:

- `q009`: retrieved/cited 2025 EE, missed 45 and 32 credits.
- `q047`: retrieved/cited 2025 EE, missed 27 elective credits.
- `q067`: retrieved/cited 2025 CS, answered only 39 elective credits and missed 145/20/59.
- `q078`: retrieved/cited the program PDF, missed the 8-credit course-practice requirement.

Remediation:

1. Add table-aware evidence expansion for degree-program pages.
   - Preserve nearby table rows around headings such as `共计`, `学分`, `任选课`, `专业课程`, `必修`, `选修`, `专业实践`.
   - Do not compact these sections into sentence windows that can drop label-value bindings.

2. Add a required-slot validator before accepting an answer.
   - For credit questions, infer slots from the query: total credits, required credits, elective credits, course-practice credits.
   - If cited context contains those slot labels and numbers but the answer omits them, trigger repair or extractive fallback.

3. Add a deterministic extractive fallback for credit tables.
   - If the question asks for credits and cited context contains `学分`, return a compact answer assembled from matched label-value rows before asking the LLM to summarize.

## Case Study 2: Correct Source, Partial Multi-Field Answer

Representative question: `q065`

Question: PSIT lab research directions, quota, contact email, and PI.

Expected answer: directions include nondestructive testing, electromagnetic
measurement/imaging, electromagnetic field and circuit-system modeling; quota is
2-3 master or PhD students per year; email is `yechf@shanghaitech.edu.cn`; PI is
Professor Ye Chaofeng.

Observed answer: `email: 申请请联系yechf@shanghaitech.edu.cn [1].`

Signals:

- `retrieved_expected_source_hit@5 = 1.0`
- `cited_expected_source_hit@5 = 1.0`
- Manual score: 0

Root cause:

The context snippet itself contained all required fields, but the generator
returned only one slot. This is a multi-field completeness failure, not a
retrieval failure.

Same family:

- `q030`: correctly answered the phosphine procurement supplier, missed the dichlorosilane supplier.
- `q053`: correctly identified the speaker's institution, missed room `3-301`.
- `q077`: got the 1-year course-completion point, missed dual-advisor confirmation and used doctoral credits.

Remediation:

1. Add query-slot planning to generation.
   - Before generation, derive required fields from conjunctions and nouns in the question.
   - Example for `q065`: `research_directions`, `quota`, `email`, `pi`.

2. Change the prompt shape for multi-field questions.
   - Ask the model to answer in one bullet per requested field.
   - Forbid returning only a subset when other fields appear in cited context.

3. Validate field coverage after generation.
   - If the query asks for 3-4 fields and the answer has only one citation-bearing fact, reject and repair.
   - Existing rejection logic already handles some profile/contact cases; extend it to generic multi-slot questions.

## Case Study 3: Wrong Year or Wrong Program Routed Into Answer

Representative question: `q069`

Question: 2025 CS degree plan, two `二选一` course groups and how overflow credits count.

Expected answer: overflow credits count toward the 34-credit discipline elective
bucket; the same rule applies to `操作系统I` and `人工智能I`.

Observed answer: The model used 2020 CS and older EE plans, gave 31 credits, and
therefore produced a stale answer.

Signals:

- `retrieved_expected_source_hit@5 = 0.0`
- `cited_expected_source_hit@5 = 0.0`
- Manual score: 0

Root cause:

Dense retrieval matched semantically similar degree-plan pages, but not the exact
year/program page. The downstream generator then correctly summarized the wrong
source.

Same family:

- `q011`: 2025 EE question, but retrieved 2021-2024 plans and missed the 2025 answer.
- `q070`: needed 2025 CS plan plus 2025-2026 course table, but retrieved course-promotion and graduate schedule pages.
- `q077`: needed 2025 electronic-information master's PDF, but cited ordinary doctoral and older master's PDFs.

Remediation:

1. Add exact year/program routing before semantic ranking.
   - If the query contains `2025级` and `CS`/`EE`, require or heavily boost URLs/titles containing the same year and program.
   - Penalize older degree pages more aggressively when the target year is present.

2. Add graduate-program type filters.
   - Distinguish `硕士`, `普博`, `硕博`, `直博`, `改革专项`, and `企业联培`.
   - Penalize PDFs whose URL/title has the wrong program type even if the surrounding text is semantically similar.

3. Add a stale-source guard in answer generation.
   - If the question has a year and the cited source is older, the answer should either cite a target-year source or abstain.
   - For this project, a wrong-year answer is worse than an evidence-insufficient answer.

## Case Study 4: Retrieved Source Hit, Citation/Context Selection Miss

Representative question: `q045`

Question: Six discipline directions for the national first-class CS undergraduate
program.

Expected answer: artificial intelligence, data science, computer systems,
information security, robotics, bioinformatics.

Observed answer: The model listed concentration/course-bucket names such as CGVI,
robotics and automation, software and system, data science, AI, and circuit/system.

Signals:

- `retrieved_expected_source_hit@5 = 1.0`
- `cited_expected_source_hit@5 = 0.0`
- `answer_synthesis_miss = 1.0`
- Manual score: 0

Root cause:

The correct source appeared in retrieval, but the final answer relied on a
different context about course concentrations. The answer-context ordering and
local evidence selection selected a plausible but wrong interpretation of
`direction`.

Remediation:

1. Add task-anchor boosting for exact phrase matches.
   - For questions asking `六个学科方向`, prefer contexts containing `学科方向` and the six listed terms.
   - Penalize `专业选修课分类图` or `Concentration` when the question asks for admissions/professional-feature wording.

2. Add citation coverage validation.
   - If the expected source was retrieved but no final citation points to it, trigger repair with the expected/retrieved context included.

3. Make answer-context ordering explainable in review artifacts.
   - Keep `answer_context_order`, but add a short reason when a lower-ranked context is selected over a higher-ranked exact anchor.

## Case Study 5: Evaluator False Negatives Are Real but Small

Representative questions: `q034`, `q041`, `q043`

These were manually corrected:

- `q034`: answered `电力电子`, which is sufficient.
- `q041`: answered `2013` and `Cher Wang / 王雪红`, which is sufficient.
- `q043`: paraphrased the topic as the relationship/dialogue between money and the heart, which is sufficient.

Signals:

- Manual recovery: 3/28 incorrect records.
- Adjusted score: 57/100.

Root cause:

The deterministic judge still misses some paraphrase or bilingual alias cases.
However, this contributes only three points in this run, so continuing to loosen
the evaluator will not solve the main performance issue.

Remediation:

1. Add narrow aliases only for stable equivalences.
   - `Cher Wang` <-> `王雪红`
   - `Power Electronics` <-> `电力电子`
   - `Money与心` <-> `金钱与心`

2. Keep paired negative tests for each new acceptance rule.
   - Do not make a broad semantic judge that starts accepting partial answers like `q065` or stale-year answers like `q069`.

## Engineering Plan

Priority 1: table/PDF evidence extraction

- Target file: `src/rag/generate.py`.
- Add curriculum/degree-plan evidence patterns for `总学分`, `任选课`, `专业课程`, `必修`, `选修`, `合计`, `专业实践`, `课程实践`.
- When a query asks about credits, include table-like windows around these labels even if the anchor overlap is low.
- Expected impact: fixes many retrieved-and-cited failures such as `q004`, `q009`, `q047`, `q067`, `q078`.

Priority 2: multi-field completeness validation

- Target file: `src/rag/generate.py`.
- Add a lightweight slot detector for questions containing `分别`, `哪些`, `是什么`, `和`, `以及`.
- Reject answers that cover only one requested slot when the cited context contains multiple slots.
- Expected impact: fixes partial answers such as `q030`, `q053`, `q065`, and parts of `q077`.

Priority 3: source routing for year/program pages

- Target files: `src/rag/retrieve.py` and/or `src/rag/generate.py`.
- Add exact filters or strong boosts for year + program + document type.
- For graduate PDFs, distinguish master's, ordinary doctoral, direct doctoral, and enterprise-joint reform tracks.
- Expected impact: fixes wrong-source failures such as `q011`, `q069`, `q070`, `q077`.

Priority 4: citation coverage repair

- Target file: `src/rag/generate.py`.
- If an expected source is retrieved but not cited, run a repair pass that forces the model to inspect the retrieved expected-source context.
- Expected impact: fixes citation/context-selection misses like `q045`.

Priority 5: narrow evaluator aliases

- Target file: `src/evaluate/judge.py`.
- Add only the three confirmed false-negative alias families from this run.
- Expected impact: improves reported score from 54 to 57, but should not be treated as the main system improvement.

## Report-Ready Interpretation

The dense Qwen35 experiment shows that retrieval quality is necessary but not
sufficient for answer correctness. In many failed cases, the system retrieved
and cited the correct official page, yet the generator omitted key table values
or returned only one of several requested fields. This is especially visible on
curriculum and graduate-program questions, where critical facts are stored in
semi-structured tables and PDFs.

The next optimization should therefore be described as answer-side evidence
grounding: table-aware evidence extraction, multi-field completeness checking,
and stricter source routing for year/program-specific documents. Evaluator
repair is still useful, but it only recovers 3 percentage points on this run and
does not address the dominant failure mode.
