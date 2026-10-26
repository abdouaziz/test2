FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/hf \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first, so this layer is cached when only the code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Download Whisper small into the image so containers don't fetch it on startup.
RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-small')"

# Cloud Run provides the port in $PORT.
CMD ["sh", "-c", "uvicorn ggapi.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
