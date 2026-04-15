#!/bin/bash
set -e

echo "[entrypoint] Waiting for PostgreSQL..."
until pg_isready -h "${POSTGRES_HOST:-postgres}" -U "${POSTGRES_USER:-edms}" -d "${POSTGRES_DB:-edms_ai}" 2>/dev/null; do
  echo "[entrypoint] PostgreSQL not ready — retrying in 2s..."
  sleep 2
done
echo "[entrypoint] PostgreSQL is ready."

# Запускаем Alembic миграции (спецификация: alembic upgrade head)
echo "[entrypoint] Running alembic upgrade head..."
cd /app
alembic upgrade head
echo "[entrypoint] Alembic migrations complete."

echo "[entrypoint] Starting orchestrator on port ${API_PORT:-8002}..."
exec python main.py
