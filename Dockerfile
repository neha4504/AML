FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


FROM python:3.10-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


COPY --from=builder /install /usr/local

COPY FastAPI/       ./FastAPI/
COPY src/           ./src/
COPY scripts/       ./scripts/
COPY data/test/     ./data/test/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Environment variables ──────────────────────────────────────────────────────
# Never hardcode secrets — inject at runtime via:
#   docker run -e DAGSHUB_TOKEN=xxx -e DAGSHUB_USERNAME=xxx
# or via Kubernetes secrets
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DAGSHUB_TOKEN="" \
    DAGSHUB_USERNAME=""

EXPOSE 8000

# 60s grace period for MLflow model download on startup before health checks begin
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "FastAPI.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
