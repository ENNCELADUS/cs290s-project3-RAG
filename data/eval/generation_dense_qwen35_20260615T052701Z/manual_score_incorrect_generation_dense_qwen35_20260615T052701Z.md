# Manual Score for Incorrect Answers

Run: `generation_dense_qwen35_20260615T052701Z`

Scope: manually reviewed only the 28 records whose evaluator status was `incorrect`.
The 18 `evidence_insufficient` records were not regraded here.

## Summary

- Script score: 54 correct, 28 incorrect, 18 evidence insufficient.
- Manual recovery from `incorrect`: 3 clear false negatives.
- Manual adjusted score for this run: 57 correct, 25 incorrect, 18 evidence insufficient.
- Adjusted overall accuracy: 57/100 = 57.0%.
- Adjusted answered-only accuracy: 57/82 = 69.5%.

The main problem is not evaluator strictness. Most remaining `incorrect` answers are genuinely incomplete or wrong:

- 15/28 had both retrieved and cited expected-source hit@5, but the answer still missed required facts.
- 12/28 had neither retrieved nor cited expected-source hit@5.
- 1/28 had retrieved expected-source hit@5 but cited-source miss.
- 3/28 were judged incorrect by script but are acceptable under manual grading.

## Per-Question Manual Scores

| id | manual score | manual label | note |
| --- | ---: | --- | --- |
| q004 | 0 | wrong / insufficient | Refuses to answer; misses 145 total credits and 9 elective credits. |
| q005 | 0 | partial, not correct | Gets two journals, but expands the third as IEEE Transactions on Power Electronics instead of CPSS Transactions on Power Electronics and Applications. |
| q009 | 0 | wrong / insufficient | Refuses to answer; misses 45 and 32 credits. |
| q011 | 0 | wrong / insufficient | Does not find the 2025 source and misses Circuit Foundations plus 4+1 credits. |
| q015 | 0 | partial, not correct | Captures three-choose-two and overflow-to-elective rule, but omits the required 22-credit bucket. |
| q018 | 0 | partial, not correct | Identifies Zhang Xinyun and email, but misses office room 3-210. |
| q023 | 0 | wrong official name | Gives a close but not official English lab name; company/lab English wording is wrong. |
| q028 | 0 | wrong / insufficient | Refuses the office answer despite expected-source citation. |
| q030 | 0 | partial, not correct | Answers only the phosphine project; misses the dichlorosilane project. |
| q031 | 0 | wrong / insufficient | Refuses the English official full name. |
| q032 | 0 | partial, not correct | Kou Xufeng room is correct, but Zhou Pingqiang room is wrong. |
| q033 | 0 | wrong / insufficient | Fails to find Lian Lixiang's Fall EE150 record. |
| q034 | 1 | evaluator false negative | Correctly answers Power Electronics / 电力电子 as the recorded self-study course. |
| q037 | 0 | wrong / insufficient | Refuses the course code; misses EE111. |
| q039 | 0 | wrong / irrelevant | Returns unrelated training/news text. |
| q041 | 1 | evaluator false negative | Correctly gives 2013 and Cher Wang / Wang Xuehong. |
| q043 | 1 | evaluator false negative | Correctly paraphrases the topic as the relationship/dialogue between money and the heart. |
| q044 | 0 | partial, not correct | Mentions some research areas, but misses University of Maryland and several required directions. |
| q045 | 0 | wrong / partial | Lists different concentration buckets and misses information security and bioinformatics. |
| q047 | 0 | wrong / insufficient | Refuses the required 27 elective credits. |
| q051 | 0 | partial with hallucination | High school is correct, but Michigan College is missing and Antai is hallucinated. |
| q053 | 0 | partial, not correct | Institution is correct, but room 3-301 is missing. |
| q065 | 0 | partial, not correct | Only gives email; misses directions, quota, and PI. |
| q067 | 0 | partial, not correct | Only gives 39 elective credits; misses total 145, required 20, and total professional 59. |
| q069 | 0 | wrong source / stale answer | Uses 2020/31-credit information instead of 2025/34-credit information. |
| q070 | 0 | wrong / insufficient | Refuses the 2025 CS AI direction and 2025-2026 Deep Learning instructor facts. |
| q077 | 0 | wrong / partial | Gets 1 year, but misses dual-advisor confirmation and answers doctoral 15/14 instead of master's 33/32. |
| q078 | 0 | partial, not correct | Gets 5/7 years and 42/40 credits, but misses course-practice minimum of 8 credits. |

## Experiment Interpretation

This dense Qwen35 run is better than the earliest repaired dense runs, but it is not a clean win:

- Compared with `generation_dense_qwen35_20260615T025840Z`, script-correct dropped from 58 to 54, while evidence-insufficient rose from 10 to 18.
- Compared with `generation_dense_qwen35_20260615T042222Z`, this run is nearly flat: 54 correct vs 55, 28 incorrect vs 27, 18 evidence-insufficient vs 18.
- Manual adjustment recovers only 3 points, so evaluator false negatives explain a small part of the poor result, not the majority.

Failure attribution:

- The biggest issue is answer synthesis over table/PDF-like curriculum facts. Many records cite the correct source but claim the numeric facts are absent.
- The second issue is retrieval or source routing for exact official English names, old course schedules, and 2025 graduate-program PDFs.
- The third issue is partial-answer behavior: the model often answers one required fact, then abstains or hallucinates for the remaining facts.

For reporting, the fairest score for this artifact is the manual-adjusted 57/100 overall accuracy, with a note that 18/100 were abstentions and 25/100 remained substantively wrong or incomplete after manual review.
