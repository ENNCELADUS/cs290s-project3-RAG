#!/bin/bash
#SBATCH -J RAG-HYBRID-QWEN35
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIATITANRTX:1
#SBATCH --exclude=ai_gpu31
#SBATCH --output=data/eval/slurm_%j.out
#SBATCH --error=data/eval/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$REPO_ROOT"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync --locked --dev' before running hybrid Qwen3.5 generation evaluation."
  exit 1
fi

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

RUN_ID="${RUN_ID:-generation_hybrid_qwen35_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-data/eval/${RUN_ID}}"
QUESTIONS_PATH="${QUESTIONS_PATH:-data/test/question_final_structured_100.csv}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/public/home/wangar2023/models/Qwen3.5-9B}"
DENSE_MODEL_PATH="${DENSE_MODEL_PATH:-/public/home/wangar2023/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181}"

DB_PATH="${DB_PATH:-data/rag/sist_merged_2026-05-27.sqlite}"
BM25_PATH="${BM25_PATH:-data/rag/bm25_2026-05-27.pkl}"
FAISS_PATH="${FAISS_PATH:-data/rag/faiss_bge_m3_2026-05-27.index}"
CHUNK_INDEX_PATH="${CHUNK_INDEX_PATH:-data/rag/chunk_index_2026-05-27.jsonl}"
REPORT_PATH="${REPORT_PATH:-data/rag/build_report_2026-05-27.json}"

TOP_K="${TOP_K:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DEVICE="${DEVICE:-cuda}"
ANSWER_RERANKER_DEVICE="${ANSWER_RERANKER_DEVICE:-cpu}"

DEFAULT_HYBRID_RERANKER_MODEL_PATH="/public/home/wangar2023/models/bge-reranker-v2-m3"
if [ ! -e "$DEFAULT_HYBRID_RERANKER_MODEL_PATH" ] && [ -e "/home/richard/models/bge-reranker-v2-m3" ]; then
  DEFAULT_HYBRID_RERANKER_MODEL_PATH="/home/richard/models/bge-reranker-v2-m3"
fi
HYBRID_SPARSE_TOP_K="${HYBRID_SPARSE_TOP_K:-50}"
HYBRID_DENSE_TOP_K="${HYBRID_DENSE_TOP_K:-50}"
HYBRID_FUSED_TOP_K="${HYBRID_FUSED_TOP_K:-50}"
HYBRID_RERANK_TOP_K="${HYBRID_RERANK_TOP_K:-10}"
HYBRID_RERANK_PRESERVE_TOP_K="${HYBRID_RERANK_PRESERVE_TOP_K:-2}"
HYBRID_RERANKER_MODEL_PATH="${HYBRID_RERANKER_MODEL_PATH:-$DEFAULT_HYBRID_RERANKER_MODEL_PATH}"
HYBRID_RERANKER_DEVICE="${HYBRID_RERANKER_DEVICE:-cuda}"
EXPANDED_QUERIES_PATH="${EXPANDED_QUERIES_PATH:-data/eval/20260613T080811Z_rerun_fix8_query_expansion/expanded_queries_fix8.jsonl}"

mkdir -p "$OUTPUT_DIR"

for required_path in \
  "$QUESTIONS_PATH" \
  "$QWEN_MODEL_PATH" \
  "$DENSE_MODEL_PATH" \
  "$DB_PATH" \
  "$BM25_PATH" \
  "$FAISS_PATH" \
  "$CHUNK_INDEX_PATH" \
  "$REPORT_PATH" \
  "$HYBRID_RERANKER_MODEL_PATH" \
  "$EXPANDED_QUERIES_PATH"; do
  if [ ! -e "$required_path" ]; then
    echo "Missing required path: $required_path"
    exit 1
  fi
done

ARGS=(
  --runner answer
  --modes hybrid
  --questions "$QUESTIONS_PATH"
  --output-dir "$OUTPUT_DIR"
  --timestamp "$RUN_ID"
  --top-k "$TOP_K"
  --model-path "$QWEN_MODEL_PATH"
  --device "$DEVICE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --dense-model "$DENSE_MODEL_PATH"
  --db "$DB_PATH"
  --bm25 "$BM25_PATH"
  --faiss "$FAISS_PATH"
  --chunk-index "$CHUNK_INDEX_PATH"
  --report "$REPORT_PATH"
  --sparse-top-k "$HYBRID_SPARSE_TOP_K"
  --dense-top-k "$HYBRID_DENSE_TOP_K"
  --fused-top-k "$HYBRID_FUSED_TOP_K"
  --rerank-top-k "$HYBRID_RERANK_TOP_K"
  --rerank-preserve-top-k "$HYBRID_RERANK_PRESERVE_TOP_K"
  --reranker-model "$HYBRID_RERANKER_MODEL_PATH"
  --reranker-device "$HYBRID_RERANKER_DEVICE"
  --expanded-queries-jsonl "$EXPANDED_QUERIES_PATH"
)

if [ -n "${ANSWER_RERANKER_MODEL:-}" ]; then
  if [ ! -e "$ANSWER_RERANKER_MODEL" ]; then
    echo "Missing ANSWER_RERANKER_MODEL path: $ANSWER_RERANKER_MODEL"
    exit 1
  fi
  ARGS+=(--answer-reranker-model "$ANSWER_RERANKER_MODEL")
  ARGS+=(--answer-reranker-device "$ANSWER_RERANKER_DEVICE")
fi

if [ -n "${REVIEW_DECISIONS_PATH:-}" ]; then
  if [ ! -e "$REVIEW_DECISIONS_PATH" ]; then
    echo "Missing REVIEW_DECISIONS_PATH path: $REVIEW_DECISIONS_PATH"
    exit 1
  fi
  ARGS+=(--review-decisions "$REVIEW_DECISIONS_PATH")
fi

if [ "${REQUIRE_FINAL_LABELS:-0}" = "1" ]; then
  ARGS+=(--require-final-labels)
fi

echo "Running hybrid Qwen3.5 generation evaluation"
echo "repo: $REPO_ROOT"
echo "run_id: $RUN_ID"
echo "output_dir: $OUTPUT_DIR"
echo "questions: $QUESTIONS_PATH"
echo "qwen_model: $QWEN_MODEL_PATH"
echo "dense_model: $DENSE_MODEL_PATH"
echo "device: $DEVICE"
echo "top_k: $TOP_K"
echo "hybrid_sparse_top_k: $HYBRID_SPARSE_TOP_K"
echo "hybrid_dense_top_k: $HYBRID_DENSE_TOP_K"
echo "hybrid_fused_top_k: $HYBRID_FUSED_TOP_K"
echo "hybrid_rerank_top_k: $HYBRID_RERANK_TOP_K"
echo "hybrid_rerank_preserve_top_k: $HYBRID_RERANK_PRESERVE_TOP_K"
echo "hybrid_reranker_model: $HYBRID_RERANKER_MODEL_PATH"
echo "hybrid_reranker_device: $HYBRID_RERANKER_DEVICE"
echo "expanded_queries: $EXPANDED_QUERIES_PATH"
echo "max_new_tokens: $MAX_NEW_TOKENS"
echo "temperature: $TEMPERATURE"

srun uv run --locked --no-sync --offline python -m evaluate.cli "${ARGS[@]}"
