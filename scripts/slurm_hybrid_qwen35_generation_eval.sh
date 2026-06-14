#!/bin/bash
#SBATCH -J RAG-HYBRID-QWEN35
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
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

mkdir -p "$OUTPUT_DIR"

for required_path in \
  "$QUESTIONS_PATH" \
  "$QWEN_MODEL_PATH" \
  "$DENSE_MODEL_PATH" \
  "$DB_PATH" \
  "$BM25_PATH" \
  "$FAISS_PATH" \
  "$CHUNK_INDEX_PATH" \
  "$REPORT_PATH"; do
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
echo "max_new_tokens: $MAX_NEW_TOKENS"
echo "temperature: $TEMPERATURE"

srun uv run --locked --no-sync --offline python -m evaluate.cli "${ARGS[@]}"
