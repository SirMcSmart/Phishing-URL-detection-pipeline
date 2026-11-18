# syntax=docker/dockerfile:1
# Serves the trained models in models/. Train first (make train) or use the committed ones.
FROM python:3.11.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml config.yaml ./
COPY src/ src/
COPY models/ models/
RUN pip install --no-cache-dir --no-deps . \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser

USER 10001
ENV PHISHDETECT_MODELS_DIR=/app/models
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "phishdetect.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-server-header"]
