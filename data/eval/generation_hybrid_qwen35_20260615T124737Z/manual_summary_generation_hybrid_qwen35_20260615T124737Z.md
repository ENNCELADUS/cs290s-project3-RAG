# Manual Review Summary

Run: `generation_hybrid_qwen35_20260615T124737Z`

Review policy: content-first loose grading with weighted manual scores. A response receives `1.0` when it substantially answers the requested entities, numbers, dates, or names, even if the deterministic judge missed aliases or the cited expected source was not hit. It receives `0.5` when it answers about half of the requested facts or identifies the correct object but misses the requested English official full name. Empty/insufficient-evidence answers, wrong core numbers, and answer-only titles remain `0`.

## Adjusted Counts

| Metric | Count |
| --- | ---: |
| automatic correct | 57 |
| incorrect manually scored 1.0 | 12 |
| incorrect manually scored 0.5 | 4 |
| weighted added score | 14.0 |
| adjusted weighted score | 71.0 |
| remaining zero-score incorrect | 12 |
| evidence insufficient | 14 |
| run errors | 1 |
| total records | 100 |

Adjusted weighted answer score is `71.0/100 = 71.0%` if the OOM record is counted in the denominator, or `71.0/99 = 71.7%` over completed records.

## Scored 1.0

`q005`, `q007`, `q010`, `q015`, `q020`, `q023`, `q032`, `q041`, `q043`, `q051`, `q053`, `q085`

## Scored 0.5

`q011`, `q031`, `q044`, `q046`

## Scored 0

`q004`, `q009`, `q016`, `q018`, `q027`, `q028`, `q045`, `q047`, `q052`, `q067`, `q070`, `q072`

## Notes

- `q005` is accepted only under a permissive alias rule: the response includes `TPEA`, but expands it imprecisely. Strict journal-title grading would leave it incorrect.
- `q023` is now accepted under the permissive English-name rule: the response is missing `Healthcare`, but otherwise gives the intended joint laboratory English name.
- `q031` gets half credit because it identifies the Chinese joint laboratory name but does not provide the requested English official full name.
- All `evidence_insufficient` records were reviewed at a high level; they are no-answer abstentions and remain not correct.
- `q068` remains an execution error caused by CUDA OOM, not an answer-quality failure.
