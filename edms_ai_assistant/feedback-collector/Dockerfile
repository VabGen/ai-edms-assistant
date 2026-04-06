# feedback-collector/Dockerfile
FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
RUN groupadd --gid 1001 appuser && useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/false appuser
COPY --chown=appuser:appuser . .
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs
USER appuser
EXPOSE 8003
HEALTHCHECK --interval=20s --timeout=5s --retries=3 --start-period=30s \
    CMD curl -fsS http://localhost:${FEEDBACK_PORT:-8003}/health || exit 1
CMD ["python", "feedback_api.py"]
