#!/usr/bin/env bash
set -euo pipefail
cd /home/richard/cs290s-project3-RAG
uv run --locked --no-sync --offline python /home/richard/generate_query_expansions.py \
  --questions data/test/question_final_structured_100.csv \
  --model /home/richard/models/Qwen3-0.6B \
  --output "data/eval/20260613T080811Z_rerun_fix8_query_expansion/expanded_queries_fix8.jsonl"
uv run --locked --no-sync --offline python -m evaluate.cli \
  --runner retrieve \
  --modes hybrid \
  --top-k 5 \
  --sparse-top-k 50 \
  --dense-top-k 50 \
  --fused-top-k 50 \
  --rerank-top-k 20 \
  --rerank-preserve-top-k 2 \
  --reranker-model /home/richard/models/bge-reranker-v2-m3 \
  --reranker-device cuda \
  --expanded-queries-jsonl "data/eval/20260613T080811Z_rerun_fix8_query_expansion/expanded_queries_fix8.jsonl" \
  --output-dir "data/eval/20260613T080811Z_rerun_fix8_query_expansion" \
  --timestamp "rerun_fix8_query_expansion_20260613T080811Z"
