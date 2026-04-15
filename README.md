# EDMS AI Assistant

Продвинутый ИИ-ассистент для корпоративной системы электронного документооборота (EDMS/СЭД).

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Chrome Extension / API                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              Orchestrator (FastAPI, port 8002)               │
│  ┌─────────┐ ┌──────────┐ ┌─────┐ ┌───────┐ ┌──────────┐  │
│  │  NLU    │ │  ReAct   │ │ RAG │ │Memory │ │Multi-    │  │
│  │Preproc  │ │  Cycle   │ │     │ │3-level│ │Agent     │  │
│  └─────────┘ └──────────┘ └─────┘ └───────┘ └──────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │ MCP Protocol
┌──────────────────────────▼──────────────────────────────────┐
│              MCP Server (FastMCP, port 8001)                 │
│              15 инструментов для работы с EDMS              │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API
                    ┌──────▼──────┐
                    │  EDMS/СЭД  │
                    │ (port 8098) │
                    └─────────────┘
```

### Компоненты

| Сервис | Порт | Описание |
|--------|------|----------|
| MCP Server | 8001 | FastMCP с 15 EDMS-инструментами |
| Orchestrator | 8002 | ReAct/Plan+Execute агент + RAG + Память |
| Feedback Collector | 8003 | RLHF — сбор оценок, обновление RAG |
| PostgreSQL | 5432 | pgvector — диалоги, профили, RAG-индекс |
| Redis | 6379 | Кэш сессий (medium-term memory) |
| Prometheus | 9090 | Метрики |
| Grafana | 3000 | Дашборды |

## Быстрый старт

### Docker Compose (рекомендуется)

```bash
cp .env.example .env
# Отредактируйте .env — укажите EDMS_API_URL, LLM URLs

docker compose up -d
```

Сервисы будут доступны:
- Оркестратор: http://localhost:8002/docs
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### Локальный запуск

```bash
# 1. PostgreSQL + Redis
docker compose up -d postgres redis

# 2. MCP Server
cd mcp-server
pip install -r requirements.txt
python mcp_http_bridge.py

# 3. Orchestrator
cd orchestrator
pip install -r requirements.txt
alembic upgrade head
python main.py

# 4. Feedback Collector
cd feedback-collector
pip install -r requirements.txt
python feedback_api.py
```

## Использование API

### Отправить запрос ассистенту

```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Найди входящие документы за январь 2026",
    "user_token": "your-jwt-token",
    "session_id": "user123"
  }'
```

### Отправить обратную связь

```bash
curl -X POST http://localhost:8002/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "dialog_id": "uuid-from-chat-response",
    "rating": 1,
    "comment": "Отличный ответ!"
  }'
```

## Добавление новых MCP-инструментов

1. Добавить функцию в `mcp-server/edms_mcp_server.py`:
```python
@mcp.tool(description="Описание инструмента на русском")
async def my_new_tool(param1: str, token: str) -> dict:
    result = await _edms_request("GET", f"api/my-endpoint/{param1}", token=token)
    return {"success": True, "data": result["data"]}
```

2. Добавить запись в `mcp-server/tools_registry.json`:
```json
{
  "name": "my_new_tool",
  "description": "Описание",
  "category": "document_read",
  "inputSchema": {
    "type": "object",
    "properties": {
      "param1": {"type": "string"},
      "token": {"type": "string"}
    },
    "required": ["param1", "token"]
  }
}
```

3. Перезапустить MCP Server — оркестратор подхватит автоматически.

## Обновление RAG

RAG-индекс обновляется автоматически:
- **Ежедневно** — из всех диалогов с оценкой +1
- **При получении оценки +1** — немедленно добавляется в индекс
- **Вручную**: `POST http://localhost:8002/rag/rebuild`

Статистика RAG: `GET http://localhost:8002/rag/stats`

## Структура проекта

```
edms-ai-assistant/
├── mcp-server/
│   ├── edms_mcp_server.py    # FastMCP, 15 инструментов
│   ├── mcp_http_bridge.py    # HTTP обёртка для оркестратора
│   ├── tools_registry.json   # Реестр инструментов
│   └── Dockerfile
├── orchestrator/
│   ├── main.py               # FastAPI endpoints
│   ├── agent_orchestrator.py # ReAct + Plan+Execute + Multi-agent
│   ├── memory.py             # 3-уровневая память
│   ├── rag_module.py         # RAG с pgvector
│   ├── nlp_preprocessor.py  # NLU + fast-path
│   ├── mcp_client.py         # MCP HTTP клиент
│   ├── llm.py                # LLM routing (light/heavy)
│   ├── config.py             # Конфигурация
│   ├── prompts/              # Skills (системные промпты)
│   │   ├── system_prompt.txt
│   │   ├── planner_prompt.txt
│   │   ├── reviewer_prompt.txt
│   │   └── explainer_prompt.txt
│   ├── alembic/              # Миграции БД
│   ├── db/                   # SQLAlchemy модели
│   ├── entrypoint.sh         # Docker entrypoint
│   └── Dockerfile
├── feedback-collector/
│   ├── feedback_api.py       # RLHF сервис
│   └── Dockerfile
├── monitoring/
│   ├── prometheus.yml
│   └── grafana_datasource.yml
├── docker-compose.yml
├── .env.example
└── README.md
```

## Мониторинг

Метрики Prometheus доступны на `/metrics` каждого сервиса:
- `edms_requests_total` — всего запросов
- `edms_latency_avg_ms` — среднее время ответа
- `edms_tool_calls_total` — вызовы MCP-инструментов
- `edms_feedback_positive_total` — положительные оценки
- `edms_feedback_negative_total` — отрицательные оценки

Health checks: `/health` на каждом сервисе.
