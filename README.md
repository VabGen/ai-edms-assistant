# EDMS AI Assistant

ИИ-ассистент для корпоративной системы электронного документооборота.

## 1. Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                    Chrome Extension / Web UI                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │ POST /chat
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Orchestrator  :8002  (FastAPI)                      │
│                                                                   │
│  EdmsDocumentAgent                                               │
│  ├── Anthropic SDK (нативный tool_use)                          │
│  ├── AsyncPostgresSaver  ← CHECKPOINT_DB_URL                    │
│  ├── MCPClient (HTTP)    ← MCP_URL                              │
│  └── SemanticDispatcher  (NLU, маршрутизация моделей)           │
│                                                                   │
│  ModelRouter:                                                    │
│    haiku   → intent known + confidence>0.85 + readonly          │
│    sonnet  → write ops, moderate complexity                      │
│    opus    → planning, unknown intent, complex workflow          │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP POST /call
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              MCP Server  :8001  (FastMCP)                        │
│                                                                   │
│  @mcp.tool() декораторы — единственный правильный API:          │
│  get_document · search_documents · create_document               │
│  update_document_status · get_document_history                   │
│  assign_document · get_analytics · get_workflow_status           │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP
                          ▼
                   Java EDMS API (внешний)

  PostgreSQL :5432    Redis :6379    Qdrant :6333
  Prometheus :9090    Grafana :3000
  Feedback   :8003
```

**Ключевые архитектурные решения:**

- MCP-сервер использует `@mcp.tool()` (FastMCP) — **не** LangChain `@tool`
- Агент работает через нативный Anthropic SDK (`tool_use`), **не** через LangGraph граф
- История тредов хранится в PostgreSQL через `AsyncPostgresSaver` (персистентно)
- Инструменты вызываются по HTTP из агента через `MCPClient`, не импортируются напрямую

## 2. Быстрый старт

```bash
# 1. Клонировать и настроить
git clone https://github.com/your-org/edms-ai-assistant.git
cd edms-ai-assistant
cp .env.example .env
# Заполнить обязательные поля: ANTHROPIC_API_KEY, POSTGRES_PASSWORD,
# EDMS_API_URL, GRAFANA_ADMIN_PASSWORD

# 2. Запустить
docker compose up -d

# 3. Проверить
docker compose ps
curl http://localhost:8002/health
```

## 3. Локальная разработка

```bash
# Зависимости
uv sync

# Запустить инфраструктуру
docker compose up -d postgres redis qdrant

# MCP-сервер
cd mcp-server && python run_server.py &

# Миграции
cd orchestrator && alembic upgrade head

# Оркестратор
cd orchestrator && python main.py
```

## 4. Добавление нового MCP-инструмента

```python
# mcp-server/edms_mcp_server.py

@mcp.tool(description="Описание на русском что делает инструмент")
async def my_new_tool(
    document_id: str,
    param: str | None = None,
) -> dict[str, Any]:
    """
    Подробное описание аргументов.

    Args:
        document_id: UUID документа.
        param:       Опциональный параметр.
    """
    ts = _log_call("my_new_tool", {"document_id": document_id})

    # Валидация
    if not document_id:
        return _err("INVALID_REQUEST", "document_id обязателен")

    # HTTP-запрос к EDMS
    result = await _request(
        "POST",
        f"/documents/{document_id}/my-action",
        json_body={"param": param},
        tool_name="my_new_tool",
    )

    _log_result("my_new_tool", ts, result["success"])
    return result
```

FastMCP автоматически генерирует JSON Schema из аннотаций Python.
Перезапусти `mcp-server`: `docker compose restart mcp-server`.

## 5. Обновление RAG-индекса

Автоматически каждый день в `RAG_UPDATE_HOUR:00 UTC`.

Ручной запуск:
```bash
# Требует admin JWT
curl -X POST http://localhost:8003/rag/trigger-update \
  -H "Authorization: Bearer $ADMIN_JWT"
```

## 6. Мониторинг

- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Метрики оркестратора: http://localhost:8002/metrics
- Метрики feedback: http://localhost:8003/metrics

## 7. Troubleshooting

**`ImportError: No module named 'edms_ai_assistant'`**
```bash
cd orchestrator && pip install -e .
```

**MCP-сервер не отвечает**
```bash
docker compose logs mcp-server
curl http://localhost:8001/health
# Проверь EDMS_API_URL в .env
```

**`AsyncPostgresSaver` не инициализируется**
```bash
# Установить psycopg v3 (не psycopg2!)
pip install "psycopg[binary]>=3.2.0" langgraph-checkpoint-postgres
# Проверить CHECKPOINT_DB_URL или DATABASE_URL
```

**Alembic: таблицы уже существуют**
```bash
cd orchestrator
alembic stamp head  # пометить текущее состояние как применённое
```

**LLM timeout**
```bash
# Уменьшить AGENT_MAX_ITERATIONS и AGENT_TIMEOUT в .env
# Или переключиться на более быструю модель
```

**Redis недоступен**
```bash
redis-cli -h $REDIS_HOST ping
docker compose restart redis
```
