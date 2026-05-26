# Data Collection Pipeline

This pipeline adds an append-only collection layer for official ShanghaiTech/SIST sources. It does not overwrite the existing `data/raw`, `data/texts`, `data/jsonl`, or `data/sist_kb.sqlite` snapshot.

## Commands

Check seeds and local PDF/OCR tools:

```bash
python3 scripts/collect_data.py doctor
```

Create a non-network dry-run scaffold:

```bash
python3 scripts/collect_data.py collect --dry-run --run-name 2026-05-26-dry-run
```

Run a conservative bounded crawl:

```bash
python3 scripts/collect_data.py collect --max-pages 50 --delay 1.0 --retries 1
```

Merge an accepted run with the existing snapshot for downstream RAG indexing:

```bash
python3 scripts/collect_data.py merge --run-jsonl data/collection_runs/<run-name>/jsonl
```

Reparse saved raw files from a prior run without re-fetching:

```bash
python3 scripts/collect_data.py reparse \
  --source-run data/collection_runs/<run-name> \
  --only-flag empty_text \
  --only-flag possibly_garbled \
  --run-name <run-name>-reparse
```

If you already know the affected URLs, put one URL per line in a text file and use:

```bash
python3 scripts/collect_data.py reparse \
  --source-run data/collection_runs/<run-name> \
  --url-file data/reparse_urls.txt \
  --run-name <run-name>-targeted-reparse
```

## Outputs

Each run writes:

- `raw/`: fetched bytes keyed by SHA-256.
- `texts/`: extracted UTF-8 text.
- `jsonl/documents.jsonl`: normalized source records.
- `jsonl/chunks.jsonl`: retrieval-ready chunks.
- `jsonl/courses.jsonl`, `jsonl/faculty_members.jsonl`, `jsonl/program_requirements.jsonl`, `jsonl/events.jsonl`: conservative structured extracts with evidence and confidence.
- `source_manifest.csv`: source audit table.
- `quality_report.md`: coverage, quality, and manual audit notes.
- `eval_seed_candidates.jsonl`: evidence-backed question themes for the later evaluation set.

Reparse runs have the same structure and add `reparsed_at`, `reparsed_from_run`, and
`reparsed_from_document_id` to `documents.jsonl`. They are still append-only and can
be merged the same way as collection runs.

## OCR Requirement

PDF text extraction uses `pdftotext` first. OCR is only a fallback for low-text PDFs. Chinese OCR requires Tesseract language data `chi_sim`; without it, the pipeline records `tesseract_chi_sim_missing` and keeps any text extracted by `pdftotext`.
