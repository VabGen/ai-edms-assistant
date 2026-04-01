#!/usr/bin/env bash
# scripts/start.sh — Запуск EDMS AI Assistant

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ── Цвета ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Проверки ───────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || error "Docker не установлен"
command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1 || error "Docker Compose не установлен"

if [ ! -f ".env" ]; then
    warn ".env не найден, создаю из .env.example..."
    cp .env.example .env
    warn "Заполни .env перед запуском (особенно EDMS_API_URL и LLM URLs)"
fi

# ── Ollama check ──────────────────────────────────────────────────────────
LIGHT_MODEL=$(grep '^LLM_LIGHT_MODEL' .env | cut -d= -f2 | tr -d '"' || echo "llama3.2:3b")
EMBED_MODEL=$(grep '^LLM_EMBEDDING_MODEL' .env | cut -d= -f2 | tr -d '"' || echo "nomic-embed-text")

if command -v ollama >/dev/null 2>&1; then
    info "Проверяю наличие моделей Ollama..."
    if ! ollama list 2>/dev/null | grep -q "$LIGHT_MODEL"; then
        warn "Модель $LIGHT_MODEL не найдена. Запускаю загрузку..."
        ollama pull "$LIGHT_MODEL" &
    else
        ok "Модель $LIGHT_MODEL доступна"
    fi
    if ! ollama list 2>/dev/null | grep -q "$EMBED_MODEL"; then
        warn "Embedding модель $EMBED_MODEL не найдена. Загружаю..."
        ollama pull "$EMBED_MODEL" &
    else
        ok "Embedding модель $EMBED_MODEL доступна"
    fi
fi

# ── Запуск ────────────────────────────────────────────────────────────────
info "Запускаю EDMS AI Assistant..."

COMPOSE_CMD="docker compose"
command -v docker-compose >/dev/null 2>&1 && COMPOSE_CMD="docker-compose"

$COMPOSE_CMD up -d --build

info "Ожидаю запуска сервисов..."
sleep 5

# ── Проверка здоровья ─────────────────────────────────────────────────────
ORCH_PORT=$(grep '^ORCHESTRATOR_PORT' .env | cut -d= -f2 || echo "8002")
FEED_PORT=$(grep '^FEEDBACK_PORT' .env | cut -d= -f2 || echo "8003")
MCP_PORT=$(grep '^MCP_PORT' .env | cut -d= -f2 || echo "8001")

check_service() {
    local name=$1; local url=$2
    if curl -sf "$url" >/dev/null 2>&1; then
        ok "$name запущен: $url"
    else
        warn "$name недоступен: $url (может ещё стартует)"
    fi
}

sleep 10
check_service "MCP Server"        "http://localhost:${MCP_PORT}/health"
check_service "Orchestrator"      "http://localhost:${ORCH_PORT}/health"
check_service "Feedback Collector" "http://localhost:${FEED_PORT}/health"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  EDMS AI Assistant запущен!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}API:${NC}         http://localhost:${ORCH_PORT}"
echo -e "  ${BLUE}API Docs:${NC}    http://localhost:${ORCH_PORT}/docs"
echo -e "  ${BLUE}Feedback:${NC}    http://localhost:${FEED_PORT}"
echo -e "  ${BLUE}Prometheus:${NC}  http://localhost:$(grep '^PROMETHEUS_PORT' .env | cut -d= -f2 || echo 9090)"
echo -e "  ${BLUE}Grafana:${NC}     http://localhost:$(grep '^GRAFANA_PORT' .env | cut -d= -f2 || echo 3001)"
echo ""
echo -e "  Логи:        docker compose logs -f orchestrator"
echo -e "  Остановить:  docker compose down"
echo ""
