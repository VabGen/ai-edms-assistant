#!/bin/bash
set -e

echo "[entrypoint] Waiting for PostgreSQL..."
until pg_isready -h "${POSTGRES_HOST:-postgres}" -U "${POSTGRES_USER:-edms}" -d "${POSTGRES_DB:-edms_ai}" 2>/dev/null; do
  echo "[entrypoint] PostgreSQL not ready — retrying in 2s..."
  sleep 2
done
echo "[entrypoint] PostgreSQL is ready."

# Инициализация схемы БД (idempotent — IF NOT EXISTS)
if [ -f "/app/db_init_sql/01_init.sql" ]; then
  echo "[entrypoint] Running DB init SQL..."
  PGPASSWORD="${POSTGRES_PASSWORD:-edms_secret}" psql \
    -h "${POSTGRES_HOST:-postgres}" \
    -U "${POSTGRES_USER:-edms}" \
    -d "${POSTGRES_DB:-edms_ai}" \
    -f /app/db_init_sql/01_init.sql \
    --on-error-stop 2>&1 | grep -E "ERROR|NOTICE|CREATE|INSERT" || true
  echo "[entrypoint] DB init complete."
fi

echo "[entrypoint] Starting orchestrator on port ${API_PORT:-8002}..."
exec python main.py
