# EDMS AI Assistant — Архитектура монорепо

## Структура проекта

```
edms-ai-assistant/                     ← корень монорепо
│
├── edms_ai_assistant/                 ← ЕДИНЫЙ Python-пакет
│   ├── __init__.py
│   ├── config.py                      ← ЕДИНСТВЕННЫЙ Settings для всего проекта
│   ├── llm_client.py                  ← ЕДИНСТВЕННЫЙ LLM-клиент (Ollama/Anthropic)
│   │
│   ├── shared/                        ← Общие утилиты (не дублировать!)
│   │   ├── __init__.py
│   │   ├── utils.py                   ← UUID_RE, CustomJSONEncoder, hash, format
│   │   └── retry.py                   ← async_retry decorator
│   │
│   ├── mcp_server/                    ← MCP-сервер (импортируется оркестратором)
│   │   ├── __init__.py
│   │   ├── edms_mcp_server.py         ← FastMCP приложение + 8 базовых tools
│   │   ├── run_server.py              ← uvicorn точка входа
│   │   ├── clients/                   ← HTTP-клиенты к Java EDMS API
│   │   │   ├── base_client.py         ← EdmsHttpClient (общий)
│   │   │   ├── document_client.py
│   │   │   ├── employee_client.py
│   │   │   ├── task_client.py
│   │   │   ├── attachment_client.py
│   │   │   ├── department_client.py
│   │   │   ├── group_client.py
│   │   │   ├── reference_client.py    ← использует llm_client для LLM
│   │   │   └── document_creator_client.py
│   │   ├── models/                    ← Pydantic модели для MCP
│   │   │   ├── appeal_fields.py
│   │   │   └── task_models.py
│   │   ├── services/                  ← Бизнес-логика MCP
│   │   │   ├── document_service.py    ← основной сервис документов
│   │   │   ├── document_enricher.py
│   │   │   ├── task_service.py
│   │   │   ├── introduction_service.py
│   │   │   ├── file_processor.py
│   │   │   └── appeal_extraction_service.py ← использует llm_client
│   │   └── tools/                     ← FastMCP @tool регистрация
│   │       ├── document_tools.py
│   │       ├── content_tools.py
│   │       ├── workflow_tools.py
│   │       └── appeal_tools.py
│   │
│   ├── orchestrator/                  ← Главный агент
│   │   ├── __init__.py
│   │   ├── main.py                    ← FastAPI приложение
│   │   ├── agent.py                   ← EdmsDocumentAgent (ReAct цикл)
│   │   ├── model.py                   ← Pydantic модели API
│   │   ├── memory.py                  ← 3-уровневая память
│   │   ├── rag_module.py              ← RAG (Qdrant/FAISS)
│   │   ├── nlp_preprocessor.py        ← NLU (без зависимостей)
│   │   ├── security.py               ← JWT decode
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   └── database.py            ← SQLAlchemy engine + SummarizationCache
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── nlp_service.py         ← SemanticDispatcher (адаптер NLU)
│   │   │   └── document_cache.py      ← Redis кэш ответов агента
│   │   └── alembic/
│   │       ├── env.py
│   │       └── versions/
│   │           └── 001_init.py
│   │
│   └── feedback_collector/            ← RLHF сервис
│       ├── __init__.py
│       └── feedback_api.py
│
├── infrastructure/
│   ├── prometheus.yml
│   └── grafana/
│       └── provisioning/
│           └── datasources/
│               └── prometheus.yml
│
├── Dockerfile                         ← ЕДИНСТВЕННЫЙ Dockerfile для монорепо
├── docker-compose.yml
├── requirements.txt                   ← ЕДИНСТВЕННЫЙ файл зависимостей
├── pyproject.toml
└── .env
```

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

## Переменные окружения

Все переменные описаны в `.env.example`. Ключевые:

```env
# LLM через Ollama (основной бэкенд)
OLLAMA_BASE_URL=http://127.0.0.1:11434
MODEL_NAME=gpt-oss:120b-cloud

# Или через Anthropic (если ключ задан, приоритет над Ollama)
ANTHROPIC_API_KEY=sk-ant-...

# EDMS Java API
EDMS_BASE_URL=http://127.0.0.1:8098

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://postgres:1234@localhost:5432/edms

# Redis
REDIS_URL=redis://localhost:6379/0
```
