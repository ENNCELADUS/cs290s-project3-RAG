# Data Collection Run Record

This document records the independent official-source data collection runs completed by 2026-05-27 for the ShanghaiTech/SIST RAG system. The collection layer is append-only: each crawl or reparse run is preserved under `data/collection_runs/`, and accepted outputs are merged into a separate downstream indexing layer under `data/merged/`.

## Data Collection Summary

The dataset was built from official ShanghaiTech and SIST domains through reproducible collection runs. The final clean merged output for downstream indexing is:

```text
data/merged/all-collection-runs-clean-2026-05-27
```

Final merged JSONL totals:

| artifact | rows |
| --- | ---: |
| `documents.jsonl` | 7190 |
| `chunks.jsonl` | 28481 |
| `courses.jsonl` | 707 |
| `faculty_members.jsonl` | 503 |
| `program_requirements.jsonl` | 1777 |
| `events.jsonl` | 5199 |

## Source Strategy

The collection scope used official ShanghaiTech/SIST sources only. It includes the SIST main site, the SIST faculty subsite, SIST research center subdomains, and official ShanghaiTech subdomains used for university-level academic and administrative facts, including OAA, open information, admissions, graduate admissions, library services, and jobs.

Chinese and English pages are preserved as separate documents when both versions are available. Each document keeps source metadata such as canonical URL, host, category, language, content type, parser, OCR usage, raw path, text path, and content hash.

The parser handles:

- HTML pages with link discovery and page text extraction.
- PDFs with text extraction first; OCR is used only as a fallback for low-text PDFs.
- Office files where supported by local extractors.
- Unsupported binaries as non-indexable source artifacts rather than retrieval documents.

## Collection Run Inventory

| run | purpose | documents | chunks | top categories | parser status | quality note |
| --- | --- | ---: | ---: | --- | --- | --- |
| `2026-05-26-fetch` | Initial official-source crawl across ShanghaiTech/SIST seeds. | 1127 | 3828 | program, school_info, program_requirements, admission, courses | Reparsed/parser-fixed | Low-quality ratio 0.44%; mostly short pages and a small number of unsupported files. |
| `2026-05-27-extra-balanced-fetch` | Balanced supplement for official university-level and thin categories. | 3000 | 5546 | school_info, admission, research, career, program | Reparsed/parser-fixed | Low-quality ratio 1.57%; many short informational pages, plus some legacy Office files. |
| `2026-05-27-sist-targeted-refresh` | SIST-focused refresh for courses, research, events, admissions, requirements, and faculty. | 1273 | 4214 | courses, research, events, admission, program_requirements | Reparsed/parser-fixed | Low-quality ratio 0.47%; mainly short pages and a few unsupported legacy files. |
| `2026-05-27-sist-nav-deep-fetch` | Deep SIST navigation crawl with pagination and SIST/faculty allowlist. | 1500 | 9276 | news, faculty, admission, research, career | Reparsed with current parser | Low-quality ratio 1.40%; old parser output was replaced by the current-parser reparse. |
| `2026-05-27-sist-nav-deep-saturation-pass2` | Second SIST navigation pass with skip-known to collect remaining discoverable pages. | 1422 | 8944 | events, news, faculty, research, career | Collected after parser/crawler fixes | Low-quality ratio 5.91%; mostly unsupported videos, old Office files, OCR PDFs, and stale archive links. |

## Final Merged Dataset Profile

The clean merged dataset was built from the five accepted collection runs after parser cleanup and merge-time filtering of non-indexable documents.

Category distribution:

| category | documents |
| --- | ---: |
| news | 1506 |
| school_info | 1439 |
| research | 960 |
| admission | 790 |
| events | 717 |
| courses | 550 |
| faculty | 533 |
| career | 346 |
| program_requirements | 189 |
| program | 160 |

Language distribution:

| language | documents |
| --- | ---: |
| zh | 4862 |
| en | 1250 |
| unknown | 1078 |

Top source hosts:

| host | documents |
| --- | ---: |
| `sist.shanghaitech.edu.cn` | 3155 |
| `openinfo.shanghaitech.edu.cn` | 720 |
| `faculty.sist.shanghaitech.edu.cn` | 661 |
| `oaa.shanghaitech.edu.cn` | 466 |
| `cts.shanghaitech.edu.cn` | 272 |
| `www.shanghaitech.edu.cn` | 219 |
| `bme.shanghaitech.edu.cn` | 169 |
| `yanzhao.shanghaitech.edu.cn` | 122 |
| `library.shanghaitech.edu.cn` | 101 |
| `smirc.sist.shanghaitech.edu.cn` | 96 |
| `admission.shanghaitech.edu.cn` | 92 |
| `ims.shanghaitech.edu.cn` | 91 |
| `ihuman.shanghaitech.edu.cn` | 89 |
| `siais.shanghaitech.edu.cn` | 88 |
| `slst.shanghaitech.edu.cn` | 84 |

## Quality And Parser Notes

Parser cleanup was applied before the final clean merge:

- `2026-05-26-fetch`, `2026-05-27-extra-balanced-fetch`, and `2026-05-27-sist-targeted-refresh` are reparsed/parser-fixed runs.
- `2026-05-27-sist-nav-deep-fetch` was reparsed with the current parser and replaced its pre-current-parser output in the active run list.
- `2026-05-27-sist-nav-deep-saturation-pass2` was collected after parser and crawler fixes.

Quality audit conclusions:

- Unsupported videos, binary attachments, and old Office `.doc` files may appear in raw run artifacts, but they are filtered or marked non-indexable during merge.
- OCR is fallback-only for low-text PDFs, not the default PDF path.
- `source_manifest.csv` and `quality_report.md` should be used to audit low-quality rows, duplicate content hashes, OCR usage, and stale URLs.
- Structured JSONL files are evidence-bearing extraction candidates. They are useful for retrieval, evaluation seed discovery, and manual review, but final QA/evaluation ground truth should still verify the supporting evidence text.

## Reproducibility Notes

Representative environment check:

```bash
uv run collect-data doctor \
  --seeds config/official_seed_urls_sist_nav_deep.csv
```

Representative crawl pattern:

```bash
uv run collect-data collect \
  --seeds config/official_seed_urls_sist_nav_deep.csv \
  --skip-known \
  --existing-jsonl data/merged/all-collection-runs-clean-2026-05-27 \
  --run-name <run-name> \
  --max-pages <page-budget> \
  --delay 0.5 \
  --timeout 20 \
  --retries 1 \
  --allowed-hosts sist.shanghaitech.edu.cn,faculty.sist.shanghaitech.edu.cn,vdi.sist.shanghaitech.edu.cn,nice.sist.shanghaitech.edu.cn,pmicc.sist.shanghaitech.edu.cn,cipes.sist.shanghaitech.edu.cn,ssc.sist.shanghaitech.edu.cn,smirc.sist.shanghaitech.edu.cn,star-center.shanghaitech.edu.cn \
  --expand-list-pages
```

Representative reparse pattern:

```bash
uv run collect-data reparse \
  --source-run data/collection_runs/<source-run> \
  --seeds config/official_seed_urls_sist_nav_deep.csv \
  --run-name <reparse-run>
```

Representative merge pattern:

```bash
uv run collect-data merge \
  --existing-jsonl <existing-jsonl-dir> \
  --run-jsonl data/collection_runs/<run-name>/jsonl \
  --output data/merged/<merged-output-name>
```

Each collection run keeps the following audit artifacts:

- `source_manifest.csv`: one row per collected source, including URL, category, parser, OCR usage, paths, hash, and quality flags.
- `quality_report.md`: coverage, low-quality ratio, duplicate counts, OCR usage, and failed URLs.
- `jsonl/*.jsonl`: normalized documents, chunks, and structured extraction candidates.
