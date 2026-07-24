FROM python:3.12-slim

# ffmpeg — libav для PyAV + удобная отладка видео
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY stego ./stego

# --frozen: ставим строго версии из uv.lock (без пере-резолва) — воспроизводимая сборка.
# Пакет в /app/.venv, venv в PATH — stego/python/streamlit доступны напрямую.
# extra lfvsn тянет torch (CPU) — чтобы LF-VSN работал в образе. Для GPU — Dockerfile.cuda.
RUN uv sync --extra ui --extra lfvsn --frozen
ENV PATH="/app/.venv/bin:$PATH"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/entrypoint.sh"]
