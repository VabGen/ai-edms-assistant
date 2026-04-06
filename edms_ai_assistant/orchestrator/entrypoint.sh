#!/bin/bash
# orchestrator/entrypoint.sh
set -eo pipefail

GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"
log()  { echo -e "${GREEN}[entrypoint]${NC} $*"; }
warn() { echo -e "${YELLOW}[entrypoint]${NC} $*"; }
err()  { echo -e "${RED}[entrypoint]${NC} $*" >&2; }

DB_HOST="${POSTGRES_HOST:-postgres}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_USER="${POSTGRES_USER:-edms}"
DB_NAME="${POSTGRES_DB:-edms_ai}"

log "Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT}..."
for i in $(seq 1 30); do
    if pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -q 2>/dev/null; then
        log "PostgreSQL ready (attempt ${i}/30)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        err "PostgreSQL not ready after 60s. Exiting."
        exit 1
    fi
    warn "Retrying in 2s (${i}/30)..."
    sleep 2
done

log "Running Alembic migrations..."
if alembic upgrade head; then
    log "Migrations applied successfully"
else
    err "Alembic migration failed!"
    exit 1
fi

APP_PORT="${API_PORT:-8002}"
WORKERS="${ORCHESTRATOR_WORKERS:-1}"
LOG_LVL=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')

log "Starting orchestrator on port ${APP_PORT} (workers: ${WORKERS})"
exec uvicorn edms_ai_assistant.main:app \
    --host "0.0.0.0" \
    --port "${APP_PORT}" \
    --workers "${WORKERS}" \
    --log-level "${LOG_LVL}" \
    --no-access-log \
    --timeout-keep-alive 75
