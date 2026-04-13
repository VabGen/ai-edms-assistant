# EDMS AI Assistant

## Запуск

### Локальная разработка

```bash
# Установить зависимости
pip install -r requirements.txt
pip install -e .   # чтобы `from edms_ai_assistant` работало

# Установить переменную пути
export PYTHONPATH=/path/to/project

# Запустить инфраструктуру
docker compose up -d postgres redis qdrant ollama

# Применить миграции
cd edms_ai_assistant/orchestrator
alembic upgrade head

# Запустить MCP-сервер
uvicorn edms_ai_assistant.mcp_server.run_server:app --port 8001

# Запустить оркестратор
uvicorn edms_ai_assistant.orchestrator.main:app --port 8000

# Запустить feedback collector
uvicorn edms_ai_assistant.feedback_collector.feedback_api:app --port 8002
```

### Docker

```bash
docker compose up -d
```

