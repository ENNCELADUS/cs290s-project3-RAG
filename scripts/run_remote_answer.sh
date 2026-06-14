#!/usr/bin/env bash
set -euo pipefail

cd /home/richard/cs290s-project3-RAG

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec uv run --locked --no-sync --offline rag-answer \
  --model-path "${RAG_MODEL_PATH:?set RAG_MODEL_PATH in .env}" \
  --device "${RAG_DEVICE:-cuda}" \
  "$@"
