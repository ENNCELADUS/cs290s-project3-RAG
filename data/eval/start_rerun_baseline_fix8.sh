#!/usr/bin/env bash
set -euo pipefail

cd /home/richard/cs290s-project3-RAG

ts="$(date -u +%Y%m%dT%H%M%SZ)"
base_id="rerun_baseline_dense_${ts}"
fix_id="rerun_fix8_query_expansion_${ts}"
base_dir="data/eval/${ts}_rerun_baseline_dense"
fix_dir="data/eval/${ts}_rerun_fix8_query_expansion"

mkdir -p "$base_dir" "$fix_dir"

cat >"$base_dir/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /home/richard/cs290s-project3-RAG
uv run --locked --no-sync --offline python -m evaluate.cli \\
  --runner retrieve \\
  --modes dense \\
  --top-k 5 \\
  --output-dir "$base_dir" \\
  --timestamp "$base_id"
EOF
chmod +x "$base_dir/run.sh"

nohup bash "$base_dir/run.sh" >"$base_dir/job.out" 2>"$base_dir/job.err" &
base_pid="$!"
echo "$base_pid" >"$base_dir/job.pid"

cat >"$fix_dir/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /home/richard/cs290s-project3-RAG
uv run --locked --no-sync --offline python /home/richard/generate_query_expansions.py \\
  --questions data/test/question_final_structured_100.csv \\
  --model /home/richard/models/Qwen3-0.6B \\
  --output "$fix_dir/expanded_queries_fix8.jsonl"
uv run --locked --no-sync --offline python -m evaluate.cli \\
  --runner retrieve \\
  --modes hybrid \\
  --top-k 5 \\
  --sparse-top-k 50 \\
  --dense-top-k 50 \\
  --fused-top-k 50 \\
  --rerank-top-k 20 \\
  --rerank-preserve-top-k 2 \\
  --reranker-model /home/richard/models/bge-reranker-v2-m3 \\
  --reranker-device cuda \\
  --expanded-queries-jsonl "$fix_dir/expanded_queries_fix8.jsonl" \\
  --output-dir "$fix_dir" \\
  --timestamp "$fix_id"
EOF
chmod +x "$fix_dir/run.sh"

nohup bash "$fix_dir/run.sh" >"$fix_dir/job.out" 2>"$fix_dir/job.err" &
fix_pid="$!"
echo "$fix_pid" >"$fix_dir/job.pid"

printf "BASE_ID=%s\nBASE_DIR=%s\nBASE_PID=%s\nFIX_ID=%s\nFIX_DIR=%s\nFIX_PID=%s\n" \
  "$base_id" "$base_dir" "$base_pid" "$fix_id" "$fix_dir" "$fix_pid"
