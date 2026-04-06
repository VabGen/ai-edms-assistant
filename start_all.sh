#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# EDMS AI Assistant — Единый скрипт запуска всех сервисов
# Использует: Ollama (gpt-oss:120b-cloud) + PostgreSQL + Redis + Qdrant
#
# chmod +x start_all.sh
# Использование:
#   ./start_all.sh            — запуск всех сервисов
#   ./start_all.sh --dev      — dev режим (с hot-reload)
#   ./start_all.sh --stop     — остановка
#   ./start_all.sh --status   — статус сервисов
#   ./start_all.sh --logs     — просмотр логов
#   ./start_all.sh --reset-db — сброс БД и перезапуск
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Цвета ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()     { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
err()     { echo -e "${RED}[✗]${NC} $*" >&2; }
info()    { echo -e "${CYAN}[→]${NC} $*"; }
section() { echo -e "\n${BOLD}${BLUE}══ $* ══${NC}"; }

# ── Параметры ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
LOG_DIR="${SCRIPT_DIR}/logs"

DEV_MODE=false
CMD="up"

for arg in "$@"; do
    case "$arg" in
        --dev)      DEV_MODE=true ;;
        --stop)     CMD="stop" ;;
        --status)   CMD="status" ;;
        --logs)     CMD="logs" ;;
        --reset-db) CMD="reset-db" ;;
        --help|-h)  CMD="help" ;;
    esac
done

# ── Загружаем .env ────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    warn ".env не найден, создаём из .env.example..."
    if [ -f "${SCRIPT_DIR}/.env.example" ]; then
        cp "${SCRIPT_DIR}/.env.example" "$ENV_FILE"
        warn "Отредактируйте .env перед продакшеном!"
    else
        err ".env.example тоже не найден. Создайте .env вручную."
        exit 1
    fi
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Значения по умолчанию
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
MODEL_NAME="${MODEL_NAME:-gpt-oss:120b-cloud}"
MODEL_PLANNER="${MODEL_PLANNER:-$MODEL_NAME}"
MODEL_RESEARCHER="${MODEL_RESEARCHER:-$MODEL_NAME}"
MODEL_EXECUTOR="${MODEL_EXECUTOR:-$MODEL_NAME}"
MODEL_REVIEWER="${MODEL_REVIEWER:-$MODEL_NAME}"
MODEL_EXPLAINER="${MODEL_EXPLAINER:-$MODEL_NAME}"
API_PORT="${API_PORT:-8000}"
MCP_PORT="${MCP_PORT:-8001}"
FEEDBACK_PORT="${FEEDBACK_PORT:-8002}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"

mkdir -p "$LOG_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# КОМАНДЫ
# ═══════════════════════════════════════════════════════════════════════════

cmd_help() {
    echo -e "${BOLD}EDMS AI Assistant — управление сервисами${NC}"
    echo ""
    echo "Использование: $0 [ОПЦИЯ]"
    echo ""
    echo "  (без опций)   Запустить все сервисы"
    echo "  --dev         Dev режим (hot-reload для orchestrator)"
    echo "  --stop        Остановить все сервисы"
    echo "  --status      Показать статус сервисов"
    echo "  --logs        Показать логи (tail -f)"
    echo "  --reset-db    Сбросить БД Postgres и перезапустить"
    echo "  --help        Эта справка"
    echo ""
    echo "Текущая конфигурация:"
    echo "  OLLAMA_BASE_URL = $OLLAMA_BASE_URL"
    echo "  MODEL_NAME      = $MODEL_NAME"
    echo "  API_PORT        = $API_PORT"
    echo "  MCP_PORT        = $MCP_PORT"
}

cmd_stop() {
    section "Остановка сервисов"
    if command -v docker &>/dev/null && [ -f "$COMPOSE_FILE" ]; then
        info "Останавливаем Docker Compose..."
        docker compose -f "$COMPOSE_FILE" down --remove-orphans || true
    fi
    # Останавливаем локальные процессы
    for pidfile in "${LOG_DIR}"/*.pid; do
        [ -f "$pidfile" ] || continue
        pid=$(cat "$pidfile")
        name=$(basename "$pidfile" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && log "Остановлен $name (pid=$pid)"
        fi
        rm -f "$pidfile"
    done
    log "Все сервисы остановлены"
}

cmd_status() {
    section "Статус сервисов"
    printf "%-30s %-10s %-20s\n" "Сервис" "Статус" "URL"
    printf "%-30s %-10s %-20s\n" "──────────────────────────────" "──────────" "────────────────────"

    check_http() {
        local name="$1" url="$2"
        if curl -fsS --max-time 3 "$url" &>/dev/null; then
            printf "${GREEN}%-30s %-10s${NC} %-20s\n" "$name" "✓ online" "$url"
        else
            printf "${RED}%-30s %-10s${NC} %-20s\n" "$name" "✗ offline" "$url"
        fi
    }

    check_tcp() {
        local name="$1" host="$2" port="$3"
        if nc -z -w 2 "$host" "$port" 2>/dev/null; then
            printf "${GREEN}%-30s %-10s${NC} %s:%s\n" "$name" "✓ online" "$host" "$port"
        else
            printf "${RED}%-30s %-10s${NC} %s:%s\n" "$name" "✗ offline" "$host" "$port"
        fi
    }

    check_http  "Orchestrator (API)"      "http://localhost:${API_PORT}/health"
    check_http  "MCP Server"              "http://localhost:${MCP_PORT}/health"
    check_http  "Feedback Collector"      "http://localhost:${FEEDBACK_PORT}/health"
    check_tcp   "PostgreSQL"              "$POSTGRES_HOST" "$POSTGRES_PORT"
    check_tcp   "Redis"                   "$REDIS_HOST" "$REDIS_PORT"
    check_http  "Qdrant"                  "http://localhost:${QDRANT_PORT}/healthz"
    check_http  "Prometheus"              "http://localhost:${PROMETHEUS_PORT}/-/ready"
    check_http  "Grafana"                 "http://localhost:${GRAFANA_PORT}/api/health"

    echo ""
    info "Ollama: $OLLAMA_BASE_URL"
    if curl -fsS --max-time 3 "${OLLAMA_BASE_URL}/api/version" &>/dev/null; then
        log "Ollama доступен"
        model_info=$(curl -fsS "${OLLAMA_BASE_URL}/api/show" -d "{\"name\":\"$MODEL_NAME\"}" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('details',{}).get('parameter_size','?'))" 2>/dev/null || echo "?")
        log "Модель $MODEL_NAME (${model_info})"
    else
        warn "Ollama недоступен по адресу $OLLAMA_BASE_URL"
    fi
}

cmd_logs() {
    section "Логи сервисов"
    if command -v docker &>/dev/null && [ -f "$COMPOSE_FILE" ]; then
        docker compose -f "$COMPOSE_FILE" logs -f --tail=50
    else
        # Локальные логи
        log_files=("${LOG_DIR}"/orchestrator.log "${LOG_DIR}"/mcp-server.log "${LOG_DIR}"/feedback.log)
        existing=()
        for f in "${log_files[@]}"; do
            [ -f "$f" ] && existing+=("$f")
        done
        if [ ${#existing[@]} -gt 0 ]; then
            tail -f "${existing[@]}"
        else
            err "Лог-файлы не найдены в $LOG_DIR"
        fi
    fi
}

cmd_reset_db() {
    section "Сброс базы данных"
    warn "Это удалит ВСЕ данные в PostgreSQL! Продолжить? (y/N)"
    read -r confirm
    if [ "$confirm" != "y" ]; then
        info "Отменено"
        exit 0
    fi
    if command -v docker &>/dev/null && [ -f "$COMPOSE_FILE" ]; then
        docker compose -f "$COMPOSE_FILE" down -v postgres
        docker compose -f "$COMPOSE_FILE" up -d postgres
        sleep 5
    fi
    run_migrations
    log "БД сброшена и пересоздана"
}

# ── Проверки зависимостей ─────────────────────────────────────────────────

check_dependencies() {
    section "Проверка зависимостей"

    local missing=0

    check_cmd() {
        if command -v "$1" &>/dev/null; then
            log "$1 найден ($(command -v "$1"))"
        else
            warn "$1 не найден — $2"
            missing=$((missing + 1))
        fi
    }

    check_cmd python3      "установите Python 3.12+"
    check_cmd pip          "установите pip"
    check_cmd docker       "рекомендуется для инфраструктуры"
    check_cmd curl         "нужен для health checks"

    # Python версия
    py_ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    py_major=$(echo "$py_ver" | cut -d. -f1)
    py_minor=$(echo "$py_ver" | cut -d. -f2)
    if [ "$py_major" -ge 3 ] && [ "$py_minor" -ge 12 ]; then
        log "Python $py_ver (OK)"
    else
        warn "Python $py_ver — рекомендуется 3.12+, используйте на свой риск"
    fi

    # Ollama
    if curl -fsS --max-time 3 "${OLLAMA_BASE_URL}/api/version" &>/dev/null; then
        ollama_ver=$(curl -fsS "${OLLAMA_BASE_URL}/api/version" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','?'))" 2>/dev/null || echo "?")
        log "Ollama $ollama_ver доступен"
    else
        warn "Ollama не найден на $OLLAMA_BASE_URL"
        warn "Запустите Ollama: https://ollama.com/download"
        warn "Затем: ollama pull $MODEL_NAME"
    fi

    return $missing
}

check_ollama_model() {
    info "Проверяем наличие модели $MODEL_NAME в Ollama..."
    if curl -fsS "${OLLAMA_BASE_URL}/api/show" \
            -H "Content-Type: application/json" \
            -d "{\"name\":\"$MODEL_NAME\"}" &>/dev/null; then
        log "Модель $MODEL_NAME доступна"
    else
        warn "Модель $MODEL_NAME не найдена в Ollama"
        warn "Если это кастомная/корпоративная модель — убедитесь что Ollama имеет к ней доступ"
        warn "Для публичных моделей: ollama pull $MODEL_NAME"
    fi
}

# ── Инфраструктура (Docker) ───────────────────────────────────────────────

start_infrastructure() {
    section "Запуск инфраструктуры (Docker)"

    if ! command -v docker &>/dev/null; then
        warn "Docker не найден. Запустите PostgreSQL, Redis, Qdrant вручную."
        return 0
    fi

    if [ ! -f "$COMPOSE_FILE" ]; then
        warn "docker-compose.yml не найден в $SCRIPT_DIR"
        return 0
    fi

    # Запускаем только инфраструктуру (без приложений)
    info "Запускаем PostgreSQL, Redis, Qdrant, Prometheus, Grafana..."
    docker compose -f "$COMPOSE_FILE" up -d \
        postgres redis qdrant prometheus grafana \
        2>&1 | tee -a "${LOG_DIR}/docker.log"

    # Ждём готовности PostgreSQL
    info "Ждём готовности PostgreSQL..."
    local attempt=0
    until docker compose -f "$COMPOSE_FILE" exec -T postgres \
            pg_isready -U "${POSTGRES_USER:-edms}" -d "${POSTGRES_DB:-edms_ai}" -q 2>/dev/null \
          || [ $attempt -ge 30 ]; do
        attempt=$((attempt + 1))
        sleep 2
        printf "."
    done
    echo ""

    if [ $attempt -ge 30 ]; then
        err "PostgreSQL не запустился за 60 секунд"
        return 1
    fi
    log "PostgreSQL готов"

    # Ждём Redis
    info "Ждём готовности Redis..."
    local attempt=0
    until docker compose -f "$COMPOSE_FILE" exec -T redis \
            redis-cli ping 2>/dev/null | grep -q PONG \
          || [ $attempt -ge 15 ]; do
        attempt=$((attempt + 1))
        sleep 2
    done
    log "Redis готов"

    log "Инфраструктура запущена"
    info "Grafana: http://localhost:${GRAFANA_PORT} (admin / ${GRAFANA_ADMIN_PASSWORD:-change-me})"
    info "Prometheus: http://localhost:${PROMETHEUS_PORT}"
    info "Qdrant: http://localhost:${QDRANT_PORT}/dashboard"
}

# ── Миграции ─────────────────────────────────────────────────────────────

run_migrations() {
    section "Применение миграций Alembic"
    local orchestrator_dir="${SCRIPT_DIR}/edms_ai_assistant/orchestrator"

    if [ ! -d "$orchestrator_dir" ]; then
        warn "Директория orchestrator не найдена: $orchestrator_dir"
        return 0
    fi

    cd "$orchestrator_dir"
    if python3 -m alembic upgrade head 2>&1 | tee -a "${LOG_DIR}/alembic.log"; then
        log "Миграции применены"
    else
        warn "Ошибка миграций — сервисы могут работать нестабильно"
    fi
    cd "$SCRIPT_DIR"
}

# ── Установка Python-зависимостей ─────────────────────────────────────────

install_dependencies() {
    section "Установка Python-зависимостей"

    # Orchestrator
    if [ -f "${SCRIPT_DIR}/edms_ai_assistant/orchestrator/requirements.txt" ]; then
        info "Устанавливаем зависимости orchestrator..."
        pip install -q -r "${SCRIPT_DIR}/edms_ai_assistant/orchestrator/requirements.txt" \
            2>&1 | tail -5
        log "Зависимости orchestrator установлены"
    fi

    # MCP server
    if [ -f "${SCRIPT_DIR}/edms_ai_assistant/mcp-server/requirements.txt" ]; then
        info "Устанавливаем зависимости mcp-server..."
        pip install -q -r "${SCRIPT_DIR}/edms_ai_assistant/mcp-server/requirements.txt" \
            2>&1 | tail -5
        log "Зависимости mcp-server установлены"
    fi

    # Feedback collector
    if [ -f "${SCRIPT_DIR}/edms_ai_assistant/feedback-collector/requirements.txt" ]; then
        info "Устанавливаем зависимости feedback-collector..."
        pip install -q -r "${SCRIPT_DIR}/edms_ai_assistant/feedback-collector/requirements.txt" \
            2>&1 | tail -5
        log "Зависимости feedback-collector установлены"
    fi
}

# ── Запуск Python-сервисов ────────────────────────────────────────────────

start_mcp_server() {
    section "MCP Server"
    local mcp_dir="${SCRIPT_DIR}/edms_ai_assistant/mcp-server"
    local pidfile="${LOG_DIR}/mcp-server.pid"
    local logfile="${LOG_DIR}/mcp-server.log"

    if [ ! -d "$mcp_dir" ]; then
        warn "mcp-server директория не найдена: $mcp_dir"
        return 0
    fi

    # Убиваем старый процесс
    if [ -f "$pidfile" ]; then
        local old_pid
        old_pid=$(cat "$pidfile")
        kill "$old_pid" 2>/dev/null || true
        rm -f "$pidfile"
        sleep 1
    fi

    info "Запускаем MCP Server на порту $MCP_PORT..."
    cd "$mcp_dir"
    EDMS_API_URL="${EDMS_BASE_URL:-http://localhost:8098}/api" \
    MCP_HOST="0.0.0.0" \
    MCP_PORT="$MCP_PORT" \
    LOG_LEVEL="${LOG_LEVEL:-INFO}" \
        python3 run_server.py \
        >> "$logfile" 2>&1 &

    echo $! > "$pidfile"
    log "MCP Server запущен (pid=$(cat "$pidfile"), port=$MCP_PORT)"
    log "Логи: $logfile"
    cd "$SCRIPT_DIR"
}

start_orchestrator() {
    section "Orchestrator"
    local orch_dir="${SCRIPT_DIR}/edms_ai_assistant/orchestrator"
    local pidfile="${LOG_DIR}/orchestrator.pid"
    local logfile="${LOG_DIR}/orchestrator.log"

    if [ ! -d "$orch_dir" ]; then
        warn "orchestrator директория не найдена: $orch_dir"
        return 0
    fi

    # Убиваем старый процесс
    if [ -f "$pidfile" ]; then
        local old_pid
        old_pid=$(cat "$pidfile")
        kill "$old_pid" 2>/dev/null || true
        rm -f "$pidfile"
        sleep 1
    fi

    info "Ждём MCP Server на порту $MCP_PORT..."
    local attempt=0
    until curl -fsS --max-time 2 "http://localhost:${MCP_PORT}/health" &>/dev/null \
          || [ $attempt -ge 20 ]; do
        attempt=$((attempt + 1))
        sleep 2
        printf "."
    done
    echo ""

    local reload_flag=""
    if [ "$DEV_MODE" = true ]; then
        reload_flag="--reload"
        warn "Dev режим: hot-reload включён"
    fi

    info "Запускаем Orchestrator на порту $API_PORT..."
    cd "$orch_dir"

    # Экспортируем все MODEL_* переменные явно
    OLLAMA_BASE_URL="$OLLAMA_BASE_URL" \
    MODEL_NAME="$MODEL_NAME" \
    MODEL_PLANNER="$MODEL_PLANNER" \
    MODEL_RESEARCHER="$MODEL_RESEARCHER" \
    MODEL_EXECUTOR="$MODEL_EXECUTOR" \
    MODEL_REVIEWER="$MODEL_REVIEWER" \
    MODEL_EXPLAINER="$MODEL_EXPLAINER" \
    MCP_URL="http://localhost:${MCP_PORT}" \
    API_PORT="$API_PORT" \
    LOG_LEVEL="${LOG_LEVEL:-INFO}" \
        python3 -m uvicorn main:app \
            --host 0.0.0.0 \
            --port "$API_PORT" \
            --log-level "$(echo "${LOG_LEVEL:-INFO}" | tr '[:upper:]' '[:lower:]')" \
            --no-access-log \
            $reload_flag \
        >> "$logfile" 2>&1 &

    echo $! > "$pidfile"
    log "Orchestrator запущен (pid=$(cat "$pidfile"), port=$API_PORT)"
    log "Логи: $logfile"
    cd "$SCRIPT_DIR"
}

start_feedback_collector() {
    section "Feedback Collector"
    local fb_dir="${SCRIPT_DIR}/edms_ai_assistant/feedback-collector"
    local pidfile="${LOG_DIR}/feedback.pid"
    local logfile="${LOG_DIR}/feedback.log"

    if [ ! -d "$fb_dir" ]; then
        warn "feedback-collector директория не найдена: $fb_dir"
        return 0
    fi

    if [ -f "$pidfile" ]; then
        local old_pid
        old_pid=$(cat "$pidfile")
        kill "$old_pid" 2>/dev/null || true
        rm -f "$pidfile"
        sleep 1
    fi

    info "Запускаем Feedback Collector на порту $FEEDBACK_PORT..."
    cd "$fb_dir"

    DATABASE_URL="${DATABASE_URL}" \
    REDIS_URL="${REDIS_URL}" \
    ORCHESTRATOR_URL="http://localhost:${API_PORT}" \
    FEEDBACK_PORT="$FEEDBACK_PORT" \
    LOG_LEVEL="${LOG_LEVEL:-INFO}" \
        python3 feedback_api.py \
        >> "$logfile" 2>&1 &

    echo $! > "$pidfile"
    log "Feedback Collector запущен (pid=$(cat "$pidfile"), port=$FEEDBACK_PORT)"
    log "Логи: $logfile"
    cd "$SCRIPT_DIR"
}

# ── Health check после запуска ────────────────────────────────────────────

wait_for_services() {
    section "Ожидание готовности сервисов"

    local max_wait=60
    local services=(
        "Orchestrator|http://localhost:${API_PORT}/health"
        "MCP Server|http://localhost:${MCP_PORT}/health"
        "Feedback Collector|http://localhost:${FEEDBACK_PORT}/health"
    )

    for svc_spec in "${services[@]}"; do
        local name url
        name=$(echo "$svc_spec" | cut -d'|' -f1)
        url=$(echo "$svc_spec" | cut -d'|' -f2)

        local attempt=0
        printf "  Ждём %s" "$name"
        until curl -fsS --max-time 3 "$url" &>/dev/null || [ $attempt -ge $((max_wait / 2)) ]; do
            attempt=$((attempt + 1))
            sleep 2
            printf "."
        done

        if curl -fsS --max-time 3 "$url" &>/dev/null; then
            echo -e " ${GREEN}✓${NC}"
        else
            echo -e " ${RED}✗ (timeout)${NC}"
            warn "$name не ответил за ${max_wait}с — проверьте $LOG_DIR"
        fi
    done
}

print_summary() {
    echo ""
    echo -e "${BOLD}${GREEN}══════════════════════════════════════════${NC}"
    echo -e "${BOLD}${GREEN}  EDMS AI Assistant — запущен              ${NC}"
    echo -e "${BOLD}${GREEN}══════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}API:${NC}          http://localhost:${API_PORT}"
    echo -e "  ${CYAN}Docs:${NC}         http://localhost:${API_PORT}/docs"
    echo -e "  ${CYAN}MCP Server:${NC}   http://localhost:${MCP_PORT}"
    echo -e "  ${CYAN}Feedback:${NC}     http://localhost:${FEEDBACK_PORT}"
    echo -e "  ${CYAN}Grafana:${NC}      http://localhost:${GRAFANA_PORT}"
    echo -e "  ${CYAN}Prometheus:${NC}   http://localhost:${PROMETHEUS_PORT}"
    echo -e "  ${CYAN}Qdrant:${NC}       http://localhost:${QDRANT_PORT}/dashboard"
    echo ""
    echo -e "  ${YELLOW}LLM Backend:${NC}  Ollama → $MODEL_NAME"
    echo -e "  ${YELLOW}Ollama URL:${NC}   $OLLAMA_BASE_URL"
    echo ""
    echo -e "  Логи:         $LOG_DIR/"
    echo -e "  Остановка:    $0 --stop"
    echo -e "  Статус:       $0 --status"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ FLOW
# ═══════════════════════════════════════════════════════════════════════════

case "$CMD" in
    help)
        cmd_help
        exit 0
        ;;
    stop)
        cmd_stop
        exit 0
        ;;
    status)
        cmd_status
        exit 0
        ;;
    logs)
        cmd_logs
        exit 0
        ;;
    reset-db)
        cmd_reset_db
        exit 0
        ;;
    up)
        echo -e "${BOLD}${BLUE}"
        echo "  ╔═══════════════════════════════════════╗"
        echo "  ║   EDMS AI Assistant — запуск          ║"
        echo "  ╚═══════════════════════════════════════╝"
        echo -e "${NC}"

        # 1. Проверки
        check_dependencies || true

        # 2. Проверяем модель Ollama
        check_ollama_model || true

        # 3. Инфраструктура (Docker: Postgres, Redis, Qdrant, Prometheus, Grafana)
        start_infrastructure

        # 4. Python зависимости
        if [ "${SKIP_INSTALL:-false}" != "true" ]; then
            install_dependencies
        fi

        # 5. Миграции БД
        run_migrations || true

        # 6. Python сервисы
        start_mcp_server
        sleep 3

        start_orchestrator
        sleep 3

        start_feedback_collector
        sleep 2

        # 7. Проверяем готовность
        wait_for_services

        # 8. Итоговый вывод
        print_summary
        ;;
esac