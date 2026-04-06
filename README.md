# EDMS AI Assistant — Руководство по запуску

| Файл                          | Назначение                                                         |
|-------------------------------|--------------------------------------------------------------------|
| `orchestrator/llm.py`         | Единый LLM-адаптер: Ollama + Anthropic                             |
| `orchestrator/agent.py`       | Обновлён: использует `LLMClient` вместо `anthropic.AsyncAnthropic` |
| `orchestrator/multi_agent.py` | Обновлён: агенты используют `LLMClient`, модели из `.env`          |
| `start_all.sh`                | Единый скрипт запуска всех сервисов                                |
| `docker-compose.yml`          | Обновлён: добавлен сервис Ollama, MODEL_* переменные               |


## Быстрый старт

### Вариант 1: Docker Compose (рекомендуется)

```bash
# 1. Скопируйте конфиг
cp .env.example .env
# Отредактируйте .env — убедитесь что MODEL_NAME правильный

# 2. Запустите всё
docker compose up -d

# 3. Проверьте статус
docker compose ps
curl http://localhost:8000/health
```

### Вариант 2: Локальный запуск (shell-скрипт)

```bash
# Дайте права на выполнение
chmod +x start_all.sh

# Запустить все сервисы
./start_all.sh

# Dev режим (hot-reload)
./start_all.sh --dev

# Статус
./start_all.sh --status

# Логи
./start_all.sh --logs

# Остановка
./start_all.sh --stop
```

---

## Маппинг агентов → модели

| Агент           | Роль         | Переменная .env    |
|-----------------|--------------|--------------------|
| PlannerAgent    | `planner`    | `MODEL_PLANNER`    |
| ResearcherAgent | `researcher` | `MODEL_RESEARCHER` |
| ExecutorAgent   | `executor`   | `MODEL_EXECUTOR`   |
| ReviewerAgent   | `reviewer`   | `MODEL_REVIEWER`   |
| ExplainerAgent  | `explainer`  | `MODEL_EXPLAINER`  |

---

## URL сервисов

| Сервис       | URL                             |
|--------------|---------------------------------|
| API          | http://localhost:8000           |
| Swagger Docs | http://localhost:8000/docs      |
| MCP Server   | http://localhost:8001           |
| Feedback     | http://localhost:8002           |
| Grafana      | http://localhost:3000           |
| Prometheus   | http://localhost:9090           |
| Qdrant       | http://localhost:6333/dashboard |
| Ollama       | http://localhost:11434          |

