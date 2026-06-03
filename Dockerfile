FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md project3.md ./
COPY src ./src

RUN uv sync --locked --no-dev

VOLUME ["/app/data/rag"]

CMD ["rag-retrieve", "--help"]
