# Dockerfile
# Точка запуска передаётся через `command` в docker-compose.yml.
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY requirements.txt* ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/false appuser

COPY --chown=appuser:appuser . .

RUN mkdir -p /tmp/edms_uploads /app/logs \
    && chown -R appuser:appuser /tmp/edms_uploads /app/logs

RUN pip install --no-cache-dir -e . 2>/dev/null || true

USER appuser

# PYTHONPATH `from edms_ai_assistant.config import settings`
ENV PYTHONPATH=/app

EXPOSE 8000 8001 8002