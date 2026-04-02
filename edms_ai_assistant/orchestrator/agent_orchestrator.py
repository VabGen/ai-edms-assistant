"""
orchestrator/agent_orchestrator.py — Главный FastAPI сервис оркестратора.

Endpoints:
  POST /chat    — диалог с ассистентом
  POST /rag/add — добавить запись в RAG (вызывается feedback-collector)
  GET  /health  — состояние компонентов
  GET  /metrics — Prometheus метрики

ReAct цикл с NLU bypass, RAG few-shot, мульти-агентами и Redis кэшированием.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from memory import MemoryManager
from multi_agent import MultiAgentCoordinator
from nlp_preprocessor import NLUResult, get_preprocessor
from rag_module import DialogRecord, RAGModule

log = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
MCP_URL: str = os.getenv("MCP_URL", "http://mcp-server:8001")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://edms:edms@postgres:5432/edms_ai"
)
CACHE_TTL_READ: int = int(os.getenv("CACHE_TTL_READ", "300"))
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Prometheus метрики
# ---------------------------------------------------------------------------
REQUESTS_TOTAL = Counter(
    "edms_requests_total",
    "Total chat requests",
    ["intent", "model", "status"],
)
LATENCY = Histogram(
    "edms_latency_seconds",
    "Request latency",
    ["intent", "model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
TOOL_CALLS = Counter(
    "edms_tool_calls_total",
    "MCP tool calls",
    ["tool_name", "success"],
)
LLM_TOKENS = Counter(
    "edms_llm_tokens_total",
    "LLM tokens used",
    ["model", "type"],
)
CACHE_HITS = Counter("edms_cache_hits_total", "Redis cache hits")
USER_RATINGS = Counter(
    "edms_user_ratings_total",
    "User ratings",
    ["rating"],
)

# ---------------------------------------------------------------------------
# Pydantic модели API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Входящий запрос к /chat."""
    user_id: str = Field(..., description="UUID пользователя")
    session_id: str = Field(..., description="UUID сессии")
    message: str = Field(..., min_length=1, max_length=5000, description="Текст запроса")
    stream: bool = Field(False, description="Стриминг ответа (не реализован в v1)")


class ChatResponse(BaseModel):
    """Ответ /chat."""
    dialog_id: str
    response: str
    intent: str
    confidence: float
    tool_calls: list[dict[str, Any]]
    model_used: str
    tokens_used: int
    latency_ms: int
    bypass_llm: bool
    cached: bool


class FeedbackRequest(BaseModel):
    """Запрос обратной связи."""
    dialog_id: str = Field(..., description="UUID диалога из ChatResponse")
    rating: int = Field(..., ge=-1, le=1, description="-1, 0 или 1")
    comment: str | None = Field(None, max_length=2000)


class FeedbackResponse(BaseModel):
    """Ответ на обратную связь."""
    success: bool
    message: str


class RAGAddRequest(BaseModel):
    """Запрос добавления записи в RAG (от feedback-collector)."""
    dialog_id: str
    user_query: str
    normalized_query: str
    intent: str
    tool_used: str
    tool_args: dict[str, Any]
    response: str
    rating: int = 1
    is_anti_example: bool = False


class HealthResponse(BaseModel):
    """Статус компонентов системы."""
    status: str
    components: dict[str, str]
    uptime_seconds: float
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Глобальное состояние приложения
# ---------------------------------------------------------------------------

class AppState:
    """Глобальное состояние FastAPI приложения."""
    start_time: float = time.monotonic()
    memory_manager: MemoryManager | None = None
    rag: RAGModule | None = None
    coordinator: MultiAgentCoordinator | None = None
    # Хранение диалогов для feedback (dialog_id → NLUResult + модель)
    dialog_registry: dict[str, dict[str, Any]] = {}


_state = AppState()

# ---------------------------------------------------------------------------
# Жизненный цикл приложения
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и завершение работы приложения."""
    log.info("Starting EDMS Orchestrator...")

    # Инициализируем компоненты
    _state.memory_manager = MemoryManager(redis_url=REDIS_URL)
    _state.rag = RAGModule()
    await _state.rag.initialize()
    _state.coordinator = MultiAgentCoordinator(mcp_url=MCP_URL)

    log.info("EDMS Orchestrator started, MCP=%s", MCP_URL)
    yield

    # Очистка
    if _state.memory_manager:
        await _state.memory_manager.close()
    log.info("EDMS Orchestrator stopped")


# ---------------------------------------------------------------------------
# FastAPI приложение
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EDMS AI Orchestrator",
    version="1.0.0",
    description="Оркестратор ИИ-ассистента для корпоративной EDMS",
    lifespan=lifespan,
)


def _require_state(attr: str) -> Any:
    """Получить компонент из состояния или выбросить 503."""
    val = getattr(_state, attr, None)
    if val is None:
        raise HTTPException(status_code=503, detail=f"Компонент {attr} не инициализирован")
    return val


def _cache_key(user_id: str, message: str) -> str:
    """Вычислить ключ кэша для запроса."""
    h = hashlib.sha256(f"{user_id}:{message}".encode()).hexdigest()
    return f"edms:chat_cache:{h}"


def _is_write_intent(intent: str) -> bool:
    """Является ли намерение операцией записи."""
    return intent in {"create_document", "update_status", "assign_document"}


# ---------------------------------------------------------------------------
# Основной ReAct цикл
# ---------------------------------------------------------------------------

async def _process_request(
    request: ChatRequest,
    memory: MemoryManager,
    rag: RAGModule,
    coordinator: MultiAgentCoordinator,
) -> tuple[ChatResponse, NLUResult]:
    """
    Основной ReAct-цикл обработки запроса.

    Шаги:
    1. NLU-анализ
    2. Проверка кэша (только для read-операций)
    3. Загрузка контекста памяти
    4. Загрузка few-shot и антипримеров из RAG
    5. Маршрутизация к агентам
    6. Обновление памяти
    7. Сохранение ответа

    Параметры:
        request: входящий ChatRequest
        memory: MemoryManager
        rag: RAGModule
        coordinator: MultiAgentCoordinator

    Возвращает:
        (ChatResponse, NLUResult)
    """
    start_ts = time.monotonic()
    dialog_id = str(uuid.uuid4())
    preprocessor = get_preprocessor()

    # --- 1. NLU ---
    nlu_result = preprocessor.preprocess(request.message)
    log.info(
        "NLU result: intent=%s confidence=%.2f bypass=%s",
        nlu_result.intent, nlu_result.confidence, nlu_result.bypass_llm,
    )

    # --- 2. Кэш (только read) ---
    cached = False
    if not _is_write_intent(nlu_result.intent):
        cache_key = _cache_key(request.user_id, nlu_result.normalized_query)
        cached_value = await memory.medium.get(request.session_id, cache_key)
        if cached_value:
            CACHE_HITS.inc()
            log.info("Cache hit for dialog_id=%s", dialog_id)
            cached_response = cached_value
            latency_ms = int((time.monotonic() - start_ts) * 1000)
            REQUESTS_TOTAL.labels(
                intent=nlu_result.intent,
                model="cache",
                status="success",
            ).inc()
            return ChatResponse(
                dialog_id=dialog_id,
                response=cached_response["response"],
                intent=nlu_result.intent,
                confidence=nlu_result.confidence,
                tool_calls=[],
                model_used="cache",
                tokens_used=0,
                latency_ms=latency_ms,
                bypass_llm=nlu_result.bypass_llm,
                cached=True,
            ), nlu_result

    # --- 3. Контекст памяти ---
    mem_context = await memory.build_context_for_prompt(
        user_id=request.user_id,
        session_id=request.session_id,
    )

    # --- 4. RAG few-shot ---
    few_shot = await rag.build_few_shot_block(request.message, top_k=3)
    anti_examples = await rag.build_anti_examples_block(request.message, top_k=2)

    # --- 5. Маршрутизация к агентам ---
    agent_context: dict[str, Any] = {
        "query": request.message,
        "user_id": request.user_id,
        "session_id": request.session_id,
        "user_profile": mem_context.get("user_profile", {}),
        "session_state": mem_context.get("session_state", {}),
        "recent_actions": mem_context.get("recent_actions", []),
        "few_shot_examples": few_shot,
        "anti_examples": anti_examples,
        "current_datetime": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    agent_result = await coordinator.route(nlu_result, agent_context)

    final_response = agent_result.output.get("final_response", "")
    if not final_response:
        final_response = agent_result.reasoning or "Произошла ошибка. Попробуйте переформулировать запрос."

    # --- 6. Обновление памяти ---
    await memory.record_exchange(
        user_id=request.user_id,
        session_id=request.session_id,
        user_message=request.message,
        assistant_message=final_response,
        tool_calls=agent_result.tool_calls,
    )

    # Обновляем сессионный контекст
    await memory.medium.set(
        request.session_id, "last_intent", nlu_result.intent
    )
    await memory.medium.set(
        request.session_id, "last_agent", agent_result.agent_name
    )

    # --- 7. Кэшируем read-ответы ---
    if not _is_write_intent(nlu_result.intent) and agent_result.success:
        cache_key = _cache_key(request.user_id, nlu_result.normalized_query)
        await memory.medium.set(
            request.session_id,
            cache_key,
            {"response": final_response, "intent": nlu_result.intent},
            ttl=CACHE_TTL_READ,
        )

    latency_ms = int((time.monotonic() - start_ts) * 1000)

    # Prometheus
    status = "success" if agent_result.success else "error"
    REQUESTS_TOTAL.labels(
        intent=nlu_result.intent,
        model=agent_result.model_used,
        status=status,
    ).inc()
    LATENCY.labels(
        intent=nlu_result.intent,
        model=agent_result.model_used,
    ).observe(latency_ms / 1000)

    for tc in agent_result.tool_calls:
        tool_name = tc.get("tool", "unknown")
        tc_success = tc.get("result", {}).get("success", False)
        TOOL_CALLS.labels(tool_name=tool_name, success=str(tc_success)).inc()

    return ChatResponse(
        dialog_id=dialog_id,
        response=final_response,
        intent=nlu_result.intent,
        confidence=nlu_result.confidence,
        tool_calls=agent_result.tool_calls,
        model_used=agent_result.model_used,
        tokens_used=0,  # Anthropic API не всегда возвращает токены в streaming
        latency_ms=latency_ms,
        bypass_llm=nlu_result.bypass_llm,
        cached=cached,
    ), nlu_result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, summary="Диалог с ассистентом")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Обработать запрос пользователя и вернуть ответ ассистента.

    Выполняет:
    1. NLU-анализ с возможным bypass LLM
    2. Загрузку контекста из памяти
    3. RAG few-shot примеры
    4. Маршрутизацию к агентам
    5. Обновление памяти и кэша
    """
    memory = _require_state("memory_manager")
    rag = _require_state("rag")
    coordinator = _require_state("coordinator")

    try:
        response, nlu_result = await _process_request(request, memory, rag, coordinator)

        # Сохраняем в реестр для feedback
        _state.dialog_registry[response.dialog_id] = {
            "user_id": request.user_id,
            "session_id": request.session_id,
            "user_query": request.message,
            "normalized_query": nlu_result.normalized_query,
            "intent": nlu_result.intent,
            "tool_used": response.tool_calls[0]["tool"] if response.tool_calls else "",
            "tool_args": response.tool_calls[0].get("args", {}) if response.tool_calls else {},
            "response": response.response,
            "model_used": response.model_used,
        }

        # Ограничиваем размер реестра
        if len(_state.dialog_registry) > 10000:
            oldest_keys = list(_state.dialog_registry.keys())[:1000]
            for k in oldest_keys:
                del _state.dialog_registry[k]

        return response

    except HTTPException:
        raise
    except Exception as exc:
        log.error("Chat error for user=%s: %s", request.user_id, exc, exc_info=True)
        REQUESTS_TOTAL.labels(
            intent="unknown", model="unknown", status="error"
        ).inc()
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/feedback", response_model=FeedbackResponse, summary="Оценить ответ")
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Принять оценку ответа от пользователя.

    При rating=1: добавляет диалог в RAG successful_dialogs
    При rating=-1: переносит в anti_examples
    """
    rag = _require_state("rag")

    # Обновляем рейтинг в Qdrant
    await rag.update_rating(request.dialog_id, request.rating)

    # Если положительный рейтинг — добавляем в RAG
    if request.rating == 1 and request.dialog_id in _state.dialog_registry:
        dialog_data = _state.dialog_registry[request.dialog_id]
        preprocessor = get_preprocessor()
        nlu = preprocessor.preprocess(dialog_data["user_query"])

        record = DialogRecord(
            id=request.dialog_id,
            user_query=dialog_data["user_query"],
            normalized_query=dialog_data["normalized_query"],
            intent=dialog_data["intent"],
            tool_used=dialog_data.get("tool_used", ""),
            tool_args=dialog_data.get("tool_args", {}),
            response=dialog_data["response"],
            rating=1,
            is_anti_example=False,
        )
        await rag.add_dialog(record)

    # Prometheus
    rating_label = {1: "positive", 0: "neutral", -1: "negative"}.get(request.rating, "neutral")
    USER_RATINGS.labels(rating=rating_label).inc()

    log.info(
        "Feedback received: dialog=%s rating=%d comment=%s",
        request.dialog_id, request.rating, bool(request.comment),
    )

    return FeedbackResponse(
        success=True,
        message=f"Оценка '{rating_label}' сохранена. Спасибо за обратную связь!",
    )


@app.post("/rag/add", summary="Добавить запись в RAG (внутренний)")
async def rag_add(request: RAGAddRequest) -> dict[str, str]:
    """
    Добавить запись диалога в RAG-индекс.

    Вызывается feedback-collector при ежедневном обновлении.
    """
    rag = _require_state("rag")

    record = DialogRecord(
        id=request.dialog_id,
        user_query=request.user_query,
        normalized_query=request.normalized_query,
        intent=request.intent,
        tool_used=request.tool_used,
        tool_args=request.tool_args,
        response=request.response,
        rating=request.rating,
        is_anti_example=request.is_anti_example,
    )
    await rag.add_dialog(record)
    log.info("RAG record added: dialog=%s intent=%s", request.dialog_id, request.intent)
    return {"status": "added", "dialog_id": request.dialog_id}


@app.post("/rag/rebuild", summary="Перестроить RAG индекс")
async def rag_rebuild() -> dict[str, Any]:
    """
    Запустить ежедневную перестройку RAG-индекса.

    Пересчитывает эмбеддинги для всех записей.
    """
    rag = _require_state("rag")
    stats = await rag.rebuild_index()
    return {"status": "completed", "stats": stats}


@app.get("/health", response_model=HealthResponse, summary="Проверка состояния")
async def health() -> HealthResponse:
    """
    Проверить доступность всех компонентов системы.

    Проверяет: MCP-сервер, Redis, PostgreSQL, Qdrant.
    """
    components: dict[str, str] = {}
    overall = "healthy"

    # MCP
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{MCP_URL}/health")
            components["mcp"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as exc:
        components["mcp"] = f"error: {exc!s:.50}"
        overall = "degraded"

    # Redis
    try:
        memory = _state.memory_manager
        if memory:
            await memory.medium.set("__health", "__check", True, ttl=5)
            val = await memory.medium.get("__health", "__check")
            components["redis"] = "ok" if val else "error"
        else:
            components["redis"] = "not_initialized"
    except Exception as exc:
        components["redis"] = f"error: {exc!s:.50}"
        overall = "degraded"

    # PostgreSQL
    try:
        from memory import _engine
        async with _engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        components["postgres"] = "ok"
    except Exception as exc:
        components["postgres"] = f"error: {exc!s:.50}"
        overall = "degraded"

    # Qdrant
    try:
        rag = _state.rag
        if rag and rag._qdrant_available:
            stats = await rag.get_stats()
            components["qdrant"] = f"ok (backend={stats.get('backend', '?')})"
        else:
            components["qdrant"] = "fallback_faiss"
    except Exception as exc:
        components["qdrant"] = f"error: {exc!s:.50}"

    return HealthResponse(
        status=overall,
        components=components,
        uptime_seconds=round(time.monotonic() - _state.start_time, 1),
    )


@app.get("/metrics", response_class=PlainTextResponse, summary="Prometheus метрики")
async def metrics() -> PlainTextResponse:
    """Экспортировать Prometheus метрики."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_orchestrator:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
