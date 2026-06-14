#!/usr/bin/env bash
set -euo pipefail
cd /home/richard/cs290s-project3-RAG
uv run --locked --no-sync --offline python -m evaluate.cli \
  --runner retrieve \
  --modes dense \
  --top-k 5 \
  --output-dir "data/eval/20260613T080811Z_rerun_baseline_dense" \
  --timestamp "rerun_baseline_dense_20260613T080811Z"
