# EDMS AI Assistant — Продакшен-готовый ИИ Ассистент

Автономный ИИ ассистент для корпоративной системы электронного документооборота (EDMS).

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Chrome Extension / Web UI                 │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────┐
│              Orchestrator (FastAPI :8002)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │   NLU    │ │  ReAct   │ │ Plan+Exe │ │ Multi-Agent    │ │
│  │Fast-path │ │  Cycle   │ │  Cycle   │ │ Coordinator    │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │  Short   │ │  Medium  │ │  Long    │ │  RAG Module    │ │
│  │  Memory  │ │  Memory  │ │  Memory  │ │  (pgvector)    │ │
│  │(in-proc) │ │ (Redis)  │ │(Postgres)│ │                │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────┐
│              MCP Server (FastAPI :8001)                      │
│  15 инструментов: get_document, search_documents,           │
│  create_task, create_introduction, send_notification...      │
└───────────────────────────┬─────────────────────────────────┘
                            │ REST API
┌───────────────────────────▼─────────────────────────────────┐
│              EDMS Java API (:8098)                           │
└─────────────────────────────────────────────────────────────┘
```

## Быстрый старт

```bash
# 1. Клонируем и настраиваем
cp .env.example .env
nano .env   # EDMS_API_URL, LLM_*_URL, LLM_*_MODEL

# 2. Запускаем все сервисы
docker compose up -d

# 3. Проверяем
curl http://localhost:8002/health

# 4. Тестируем
./scripts/test_api.sh
```

## Порты сервисов

| Сервис | Порт | Описание |
|--------|------|----------|
| Orchestrator | 8002 | Главный API (FastAPI) |
| MCP Server | 8001 | EDMS инструменты |
| Feedback | 8003 | Сбор обратной связи |
| PostgreSQL | 5432 | Данные + pgvector |
| Redis | 6379 | Кэш + сессии |
| Prometheus | 9090 | Метрики |
| Grafana | 3001 | Дашборды |

## API Endpoints (Orchestrator)

```
POST /chat                    — основной диалог
POST /feedback                — оценить ответ (-1/0/1)
GET  /chat/history/{id}       — история диалога
POST /chat/new                — новый тред
POST /upload-file             — загрузить файл для анализа
POST /actions/summarize       — прямая суммаризация вложения
GET  /rag/stats               — статистика RAG индекса
POST /rag/rebuild             — пересобрать RAG из логов
GET  /health                  — статус всех компонентов
GET  /metrics                 — Prometheus метрики
```

## Добавление нового MCP инструмента

1. Добавьте функцию в `mcp-server/edms_mcp_server.py`:
```python
@mcp.tool()
async def my_new_tool(token: str, param: str) -> dict:
    """Описание инструмента на русском."""
    async with MyClient() as client:
        return await client.do_something(token, param)
```

2. Добавьте в `mcp-server/tools_registry.json`:
```json
{"name": "my_new_tool", "description": "...", "inputSchema": {...}}
```

3. Добавьте маппинг в `orchestrator/tools/router.py` для нужных интентов.

4. Перезапустите MCP сервер: `docker compose restart mcp-server`

## Конфигурация моделей

```env
# Лёгкая модель (простые запросы, fast-path)
LLM_LIGHT_URL=http://localhost:11434/v1
LLM_LIGHT_MODEL=llama3.2:3b

# Тяжёлая модель (сложные задачи, Plan+Execute)
LLM_HEAVY_URL=http://localhost:11434/v1
LLM_HEAVY_MODEL=gpt-oss:120b-cloud

# Порог сложности (0.0-1.0): выше → тяжёлая модель
COMPLEXITY_THRESHOLD=0.7
```

## RAG обновление

```bash
# Автоматически — каждые 24 часа (feedback-collector)

# Вручную:
curl -X POST http://localhost:8002/rag/rebuild

# Статистика:
curl http://localhost:8002/rag/stats
```

## Обратная связь

```bash
# Пользователь оценивает ответ:
curl -X POST http://localhost:8003/feedback \
  -H "Content-Type: application/json" \
  -d '{"dialog_id": "uuid", "rating": 1, "comment": "Отлично!"}'

# Негативные примеры → обновляют anti-examples блок промпта
# Позитивные примеры → попадают в RAG few-shot базу
```

## Мониторинг

- Grafana: http://localhost:3001 (admin/admin_change_me)
- Prometheus: http://localhost:9090
- Метрики: latency_ms, request_count, tool_call_success_rate, feedback_distribution

## Структура проекта

```
edms-ai-assistant/
├── mcp-server/              # MCP сервер (15 EDMS инструментов)
├── orchestrator/            # Главный агент
│   ├── agent_orchestrator.py  # ReAct + Plan+Execute + Multi-agent
│   ├── memory.py             # 3-уровневая память
│   ├── rag_module.py         # RAG с pgvector
│   ├── nlp_preprocessor.py   # NLU + fast-path
│   ├── tools/               # 15 LangChain инструментов
│   ├── services/            # Бизнес-логика
│   ├── clients/             # EDMS HTTP клиенты
│   └── prompts/             # Skills промпты
├── feedback-collector/      # RLHF сервис
├── monitoring/              # Prometheus + Grafana
└── docs/init_db.sql         # Схема PostgreSQL
```
