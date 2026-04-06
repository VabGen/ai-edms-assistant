# orchestrator/Dockerfile
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/false appuser

COPY --chown=appuser:appuser . .
RUN chmod +x entrypoint.sh \
    && mkdir -p /tmp/edms_ai_uploads /app/logs \
    && chown -R appuser:appuser /tmp/edms_ai_uploads /app/logs

USER appuser

EXPOSE 8002

HEALTHCHECK --interval=20s --timeout=10s --retries=5 --start-period=60s \
    CMD curl -fsS http://localhost:${API_PORT:-8002}/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
