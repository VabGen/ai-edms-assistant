# EDMS AI Assistant

ИИ-ассистент для корпоративной системы электронного документооборота на базе Claude (Anthropic) и MCP.

## 1. Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        Пользователь                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ POST /chat
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator :8000                            │
│                                                                  │
│  NLU Preprocessor ──► bypass? ──► прямой MCP вызов             │
│       │                    │                                     │
│       └──► MultiAgentCoordinator                                 │
│                │                                                 │
│    ┌───────────┼───────────────────┐                            │
│    ▼           ▼                   ▼                            │
│  Planner   Researcher          Executor                         │
│  (opus)    (haiku)             (sonnet)                         │
│    │           │                   │                            │
│    └───────────┼───────────────────┘                            │
│                ▼                                                 │
│           Reviewer (opus) ──► Explainer (haiku)                 │
│                                                                  │
│  MemoryManager: Short(buffer) + Medium(Redis) + Long(Postgres)  │
│  RAGModule: Qdrant (primary) / FAISS (fallback)                 │
└─────────────┬────────────────────────────────────────────────────┘
              │ MCP calls
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Server :8001                              │
│  8 инструментов: get/search/create/update/history/assign/...    │
└─────────────────────────────────────────────────────────────────┘
              │ REST API
              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Корпоративная EDMS система                          │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐     ┌──────────────────────┐
│  Feedback Collector  │     │     Monitoring        │
│  :8002               │     │  Prometheus :9090     │
│  APScheduler 03:00   │     │  Grafana :3000        │
│  RAG daily update    │     │                       │
└──────────────────────┘     └──────────────────────┘
```

## 2. Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd edms-ai

# 2. Настроить конфигурацию
cp .env.example .env
# Обязательно заполнить:
# ANTHROPIC_API_KEY=sk-ant-...
# EDMS_API_URL=http://your-edms/api/v1
# EDMS_API_KEY=your-key
# POSTGRES_PASSWORD=secure-password

# 3. Запустить
docker compose up -d

# 4. Проверить
curl http://localhost:8000/health
```

**Сервисы после запуска:**
| Сервис | URL |
|---|---|
| Orchestrator API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MCP Server | http://localhost:8001 |
| Feedback Collector | http://localhost:8002 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/change-me) |

## 3. Локальная разработка (без Docker)

```bash
# Зависимости
cd orchestrator
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# PostgreSQL и Redis должны быть запущены локально
# Или только инфраструктура через Docker:
docker compose up -d postgres redis qdrant

# Миграции
alembic upgrade head

# Запуск оркестратора
export $(cat ../.env | grep -v '^#' | xargs)
export DATABASE_URL=postgresql+asyncpg://edms:change-me@localhost:5432/edms_ai
export REDIS_URL=redis://localhost:6379/0
export QDRANT_URL=http://localhost:6333
python agent_orchestrator.py

# MCP сервер (в отдельном терминале)
cd ../mcp-server
pip install -r requirements.txt
python edms_mcp_server.py
```

## 4. Добавление нового MCP-инструмента

**Шаг 1:** Добавить описание в `mcp-server/tools_registry.json`:
```json
{
  "name": "my_new_tool",
  "description_ru": "Что делает инструмент и когда использовать",
  "input_schema": { "type": "object", "properties": {...} },
  "output_schema": { "type": "object", "properties": {...} },
  "tags": ["read"]
}
```

**Шаг 2:** Реализовать в `mcp-server/edms_mcp_server.py`:
```python

```

**Шаг 3:** Добавить в `orchestrator/multi_agent.py` в словарь `tool_schemas` класса `BaseAgent`.

**Шаг 4:** Если инструмент read-only — добавить в `ResearcherAgent.allowed_tools`.
           Если write — в `ExecutorAgent.allowed_tools`.

## 5. Обновление RAG-индекса

RAG обновляется автоматически ежедневно в 03:00 UTC.

Ручное обновление:
```bash
# Принудительный запуск
curl -X POST http://localhost:8002/rag/trigger-update

# Или через оркестратор (перестройка эмбеддингов)
curl -X POST http://localhost:8000/rag/rebuild

# Статистика RAG
curl http://localhost:8000/health | jq .components.qdrant
```

## 6. Мониторинг

**Grafana дашборды** (http://localhost:3000):
- EDMS Overview: запросы/сек, latency p95, error rate
- Agent Performance: использование по агентам и моделям
- RAG Quality: hit rate, avg similarity score
- Feedback: позитивные/негативные оценки по времени

**Ключевые метрики Prometheus:**
```
edms_requests_total{intent, model, status}
edms_latency_seconds_bucket{intent, model}
edms_tool_calls_total{tool_name, success}
edms_llm_tokens_total{model, type}
edms_cache_hits_total
edms_user_ratings_total{rating}
```

## 7. Управление пользователями и ролями

Профили хранятся в таблице `user_profiles`. Предпочтения задаются через поле `preferences` (JSONB):

```json
{
  "preferred_language": "ru",
  "default_page_size": 20,
  "notification_enabled": true
}
```

Доступ контролируется на уровне EDMS API (передаётся API-ключ пользователя).

## 8. Troubleshooting

**Проблема: MCP-сервер недоступен**
```bash
docker compose logs mcp-server --tail=50
curl http://localhost:8001/health
# Проверить EDMS_API_URL и EDMS_API_KEY в .env
```

**Проблема: Qdrant не запускается**
```bash
docker compose logs qdrant --tail=50
# Проверить права на директорию qdrantdata
docker volume inspect edms-ai_qdrantdata
# Fallback на FAISS происходит автоматически
```

**Проблема: Миграции не применяются**
```bash
docker compose exec orchestrator alembic current
docker compose exec orchestrator alembic upgrade head
docker compose exec orchestrator alembic history
```

**Проблема: Claude не отвечает**
```bash
# Проверить ключ
echo $ANTHROPIC_API_KEY
# Проверить лимиты в логах
docker compose logs orchestrator | grep "anthropic"
```

**Проблема: Кэш не работает**
```bash
docker compose exec redis redis-cli ping
docker compose exec redis redis-cli KEYS "edms:*" | head -20
```

**Проблема: RAG возвращает нерелевантные примеры**
```bash
# Запустить перестройку индекса
curl -X POST http://localhost:8000/rag/rebuild
# Проверить статистику
curl http://localhost:8000/health
```

**Проблема: Высокая latency**
```bash
# Посмотреть метрики
curl http://localhost:9090/api/v1/query?query=edms_latency_seconds_bucket
# Переключить агентов на более лёгкие модели в .env:
MODEL_RESEARCHER=claude-haiku-4-5
MODEL_EXECUTOR=claude-haiku-4-5
```

**Проблема: Агент выбирает неверный инструмент**
```bash
# Посмотреть NLU результаты в логах
docker compose logs orchestrator | grep "NLU result"
# Добавить few-shot пример для этого типа запроса через feedback (rating=1)
curl -X POST http://localhost:8002/feedback \
  -H "Content-Type: application/json" \
  -d '{"dialog_id": "...", "rating": 1}'
```

**Проблема: PostgreSQL соединения исчерпаны**
```bash
docker compose exec postgres psql -U edms -c "SELECT count(*) FROM pg_stat_activity;"
# Уменьшить pool_size в DATABASE_URL или перезапустить сервисы
docker compose restart orchestrator feedback-collector
```
