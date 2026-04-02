"""
feedback-collector/feedback_api.py — Сервис сбора обратной связи.

Функции:
  - Хранение диалогов с оценками пользователей
  - Ежедневное обновление RAG из положительных диалогов (APScheduler)
  - Формирование антипримеров из отрицательных оценок
  - Аналитика качества ответов
  - Prometheus метрики
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String, Text, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

log = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://edms:edms@postgres:5432/edms_ai"
)
ORCHESTRATOR_URL: str = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
API_PORT: int = int(os.getenv("FEEDBACK_PORT", "8002"))
RAG_REBUILD_HOUR: int = int(os.getenv("RAG_REBUILD_HOUR", "3"))  # 03:00 UTC
ANTI_EXAMPLE_REDIS_KEY: str = "edms:anti_examples_block"
ANTI_EXAMPLE_TTL: int = 86400  # 24 часа

# ---------------------------------------------------------------------------
# Prometheus метрики
# ---------------------------------------------------------------------------
DIALOGS_SAVED = Counter("edms_fb_dialogs_saved_total", "Total dialogs saved")
FEEDBACK_RECEIVED = Counter(
    "edms_fb_feedback_total", "Feedback received", ["rating"]
)
RAG_UPDATES = Counter("edms_fb_rag_updates_total", "RAG rebuild runs")
RAG_UPDATE_LATENCY = Histogram(
    "edms_fb_rag_update_seconds", "RAG rebuild duration"
)

# ---------------------------------------------------------------------------
# SQLAlchemy
# ---------------------------------------------------------------------------

class FBBase(DeclarativeBase):
    pass


_engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


class DialogLog(FBBase):
    """
    Лог диалога с оценкой пользователя.

    Хранит все необходимые данные для обновления RAG-индекса.
    """
    __tablename__ = "dialog_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    session_id: Mapped[str] = mapped_column(String(255), index=True)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_query: Mapped[str | None] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(100), index=True)
    confidence: Mapped[float | None] = mapped_column()
    entities: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    selected_tool: Mapped[str | None] = mapped_column(String(255))
    tool_args: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    tool_result: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    final_response: Mapped[str | None] = mapped_column(Text)
    model_used: Mapped[str | None] = mapped_column(String(100))
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    user_feedback: Mapped[int | None] = mapped_column(Integer)  # -1, 0, 1
    feedback_comment: Mapped[str | None] = mapped_column(Text)
    bypass_llm: Mapped[bool] = mapped_column(default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


async def _create_tables() -> None:
    """Создать таблицы если не существуют."""
    async with _engine.begin() as conn:
        await conn.run_sync(FBBase.metadata.create_all)


# ---------------------------------------------------------------------------
# Pydantic модели
# ---------------------------------------------------------------------------

class SaveDialogRequest(BaseModel):
    """Запрос сохранения диалога (от оркестратора)."""
    dialog_id: str
    user_id: str
    session_id: str
    user_query: str
    normalized_query: str | None = None
    intent: str | None = None
    confidence: float | None = None
    entities: dict[str, Any] = Field(default_factory=dict)
    selected_tool: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result: dict[str, Any] = Field(default_factory=dict)
    final_response: str | None = None
    model_used: str | None = None
    tokens_used: int = 0
    bypass_llm: bool = False
    latency_ms: int = 0


class FeedbackSubmitRequest(BaseModel):
    """Запрос обратной связи от пользователя."""
    dialog_id: str = Field(..., description="UUID диалога")
    rating: int = Field(..., ge=-1, le=1, description="-1, 0, или 1")
    comment: str | None = Field(None, max_length=2000)


class FeedbackStatsResponse(BaseModel):
    """Агрегированная статистика качества."""
    total_dialogs: int
    rated_dialogs: int
    positive_count: int
    negative_count: int
    neutral_count: int
    positive_rate: float
    negative_rate: float
    avg_latency_ms: float
    top_intents: list[dict[str, Any]]
    top_tools: list[dict[str, Any]]
    period_days: int


class HealthResponse(BaseModel):
    status: str
    postgres: str
    orchestrator: str
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Глобальное состояние
# ---------------------------------------------------------------------------

_start_time = time.monotonic()
_scheduler: AsyncIOScheduler | None = None


# ---------------------------------------------------------------------------
# Ежедневное обновление RAG
# ---------------------------------------------------------------------------

async def _daily_rag_update() -> None:
    """
    Ежедневное обновление RAG-индекса в 03:00 UTC.

    Шаги:
    1. Загрузить диалоги с rating=1 за последние 24 часа
    2. Отправить в orchestrator /rag/add
    3. Загрузить диалоги с rating=-1
    4. Сформировать блок антипримеров → записать в Redis
    5. Залогировать статистику
    """
    start = time.monotonic()
    log.info("Daily RAG update started")

    async with _session_factory() as session:
        # --- Позитивные диалоги ---
        stmt_pos = (
            select(DialogLog)
            .where(
                DialogLog.user_feedback == 1,
                DialogLog.created_at
                >= func.now() - __import__("sqlalchemy").text("INTERVAL '24 hours'"),
            )
            .limit(500)
        )
        positive_rows = (await session.execute(stmt_pos)).scalars().all()

        # --- Негативные диалоги ---
        stmt_neg = (
            select(DialogLog)
            .where(DialogLog.user_feedback == -1)
            .order_by(DialogLog.created_at.desc())
            .limit(200)
        )
        negative_rows = (await session.execute(stmt_neg)).scalars().all()

    # Отправляем позитивные в orchestrator
    added_count = 0
    async with httpx.AsyncClient(timeout=30) as client:
        for row in positive_rows:
            try:
                payload = {
                    "dialog_id": row.id,
                    "user_query": row.user_query,
                    "normalized_query": row.normalized_query or "",
                    "intent": row.intent or "unknown",
                    "tool_used": row.selected_tool or "",
                    "tool_args": row.tool_args,
                    "response": (row.final_response or "")[:500],
                    "rating": 1,
                    "is_anti_example": False,
                }
                r = await client.post(f"{ORCHESTRATOR_URL}/rag/add", json=payload)
                if r.status_code == 200:
                    added_count += 1
            except Exception as exc:
                log.warning("Failed to add positive dialog to RAG: %s", exc)

    # Формируем блок антипримеров
    anti_lines = ["=== АНТИПРИМЕРЫ (избегать подобного) ===\n"]
    for row in negative_rows[:30]:  # не более 30 антипримеров
        anti_lines.append(f"Запрос: {row.user_query}")
        if row.final_response:
            anti_lines.append(f"Неудачный ответ: {row.final_response[:200]}")
        if row.feedback_comment:
            anti_lines.append(f"Причина: {row.feedback_comment}")
        anti_lines.append("---")

    anti_block = "\n".join(anti_lines)

    # Сохраняем в Redis
    try:
        import redis.asyncio as redis_aio
        r_client = redis_aio.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        await r_client.setex(ANTI_EXAMPLE_REDIS_KEY, ANTI_EXAMPLE_TTL, anti_block)
        await r_client.aclose()
    except Exception as exc:
        log.warning("Failed to save anti-examples to Redis: %s", exc)

    elapsed = time.monotonic() - start
    RAG_UPDATES.inc()
    RAG_UPDATE_LATENCY.observe(elapsed)

    log.info(
        "Daily RAG update completed: positive=%d anti=%d elapsed=%.2fs",
        added_count, len(negative_rows), elapsed,
    )


# ---------------------------------------------------------------------------
# Жизненный цикл
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация БД и планировщика."""
    global _scheduler

    await _create_tables()
    log.info("Database tables created/verified")

    # Планировщик ежедневного обновления
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        _daily_rag_update,
        trigger=CronTrigger(hour=RAG_REBUILD_HOUR, minute=0, timezone="UTC"),
        id="daily_rag_update",
        name="Daily RAG Update",
        replace_existing=True,
    )
    _scheduler.start()
    log.info("Scheduler started, RAG update at %02d:00 UTC", RAG_REBUILD_HOUR)

    yield

    if _scheduler:
        _scheduler.shutdown(wait=False)
    await _engine.dispose()
    log.info("Feedback Collector stopped")


# ---------------------------------------------------------------------------
# FastAPI приложение
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EDMS Feedback Collector",
    version="1.0.0",
    description="Сервис сбора обратной связи и обновления RAG для EDMS AI",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/dialogs", summary="Сохранить диалог")
async def save_dialog(request: SaveDialogRequest) -> dict[str, str]:
    """
    Сохранить диалог в базу данных.

    Вызывается оркестратором после каждого ответа ассистента.
    """
    async with _session_factory() as session:
        async with session.begin():
            # Проверяем что dialog_id ещё не существует
            existing = await session.get(DialogLog, request.dialog_id)
            if existing:
                return {"status": "already_exists", "dialog_id": request.dialog_id}

            dialog = DialogLog(
                id=request.dialog_id,
                user_id=request.user_id,
                session_id=request.session_id,
                user_query=request.user_query,
                normalized_query=request.normalized_query,
                intent=request.intent,
                confidence=request.confidence,
                entities=request.entities,
                selected_tool=request.selected_tool,
                tool_args=request.tool_args,
                tool_result=request.tool_result,
                final_response=request.final_response,
                model_used=request.model_used,
                tokens_used=request.tokens_used,
                bypass_llm=request.bypass_llm,
                latency_ms=request.latency_ms,
            )
            session.add(dialog)

    DIALOGS_SAVED.inc()
    log.debug("Dialog saved: %s intent=%s", request.dialog_id, request.intent)
    return {"status": "saved", "dialog_id": request.dialog_id}


@app.post("/feedback", summary="Оценить ответ пользователем")
async def submit_feedback(request: FeedbackSubmitRequest) -> dict[str, Any]:
    """
    Принять оценку пользователя для диалога.

    Обновляет user_feedback и feedback_comment в базе данных.
    """
    async with _session_factory() as session:
        async with session.begin():
            dialog = await session.get(DialogLog, request.dialog_id)
            if not dialog:
                raise HTTPException(
                    status_code=404,
                    detail=f"Диалог {request.dialog_id} не найден",
                )

            dialog.user_feedback = request.rating
            dialog.feedback_comment = request.comment
            dialog.feedback_at = datetime.now(timezone.utc)

    rating_label = {1: "positive", 0: "neutral", -1: "negative"}.get(request.rating, "neutral")
    FEEDBACK_RECEIVED.labels(rating=rating_label).inc()

    log.info(
        "Feedback saved: dialog=%s rating=%d",
        request.dialog_id, request.rating,
    )
    return {
        "status": "saved",
        "dialog_id": request.dialog_id,
        "rating": request.rating,
        "message": "Спасибо за обратную связь!",
    }


@app.get("/feedback/stats", response_model=FeedbackStatsResponse, summary="Статистика")
async def get_feedback_stats(days: int = 30) -> FeedbackStatsResponse:
    """
    Получить агрегированную статистику качества ответов.

    Параметры:
        days: период анализа в днях (по умолчанию 30)
    """
    import sqlalchemy as sa

    async with _session_factory() as session:
        # Общая статистика за период
        cutoff_expr = func.now() - sa.text(f"INTERVAL '{days} days'")

        total_stmt = select(func.count()).where(
            DialogLog.created_at >= cutoff_expr
        )
        total = (await session.execute(total_stmt)).scalar() or 0

        rated_stmt = select(func.count()).where(
            DialogLog.created_at >= cutoff_expr,
            DialogLog.user_feedback.is_not(None),
        )
        rated = (await session.execute(rated_stmt)).scalar() or 0

        pos_stmt = select(func.count()).where(
            DialogLog.created_at >= cutoff_expr,
            DialogLog.user_feedback == 1,
        )
        positive = (await session.execute(pos_stmt)).scalar() or 0

        neg_stmt = select(func.count()).where(
            DialogLog.created_at >= cutoff_expr,
            DialogLog.user_feedback == -1,
        )
        negative = (await session.execute(neg_stmt)).scalar() or 0

        neutral = rated - positive - negative

        avg_latency_stmt = select(func.avg(DialogLog.latency_ms)).where(
            DialogLog.created_at >= cutoff_expr
        )
        avg_latency = float((await session.execute(avg_latency_stmt)).scalar() or 0)

        # Топ намерений
        intent_stmt = (
            select(DialogLog.intent, func.count().label("cnt"))
            .where(
                DialogLog.created_at >= cutoff_expr,
                DialogLog.intent.is_not(None),
            )
            .group_by(DialogLog.intent)
            .order_by(sa.desc("cnt"))
            .limit(10)
        )
        intent_rows = (await session.execute(intent_stmt)).all()
        top_intents = [{"intent": r[0], "count": r[1]} for r in intent_rows]

        # Топ инструментов
        tool_stmt = (
            select(DialogLog.selected_tool, func.count().label("cnt"))
            .where(
                DialogLog.created_at >= cutoff_expr,
                DialogLog.selected_tool.is_not(None),
            )
            .group_by(DialogLog.selected_tool)
            .order_by(sa.desc("cnt"))
            .limit(10)
        )
        tool_rows = (await session.execute(tool_stmt)).all()
        top_tools = [{"tool": r[0], "count": r[1]} for r in tool_rows]

    return FeedbackStatsResponse(
        total_dialogs=total,
        rated_dialogs=rated,
        positive_count=positive,
        negative_count=negative,
        neutral_count=neutral,
        positive_rate=positive / rated if rated > 0 else 0.0,
        negative_rate=negative / rated if rated > 0 else 0.0,
        avg_latency_ms=round(avg_latency, 2),
        top_intents=top_intents,
        top_tools=top_tools,
        period_days=days,
    )


@app.post("/rag/trigger-update", summary="Принудительный запуск обновления RAG")
async def trigger_rag_update() -> dict[str, str]:
    """
    Запустить обновление RAG вручную (не ждать 03:00 UTC).
    """
    asyncio.create_task(_daily_rag_update())
    return {"status": "triggered", "message": "Обновление RAG запущено в фоне"}


@app.get("/health", response_model=HealthResponse, summary="Проверка состояния")
async def health() -> HealthResponse:
    """Проверить доступность компонентов feedback-collector."""
    postgres_status = "ok"
    orchestrator_status = "ok"

    # PostgreSQL
    try:
        async with _session_factory() as session:
            await session.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as exc:
        postgres_status = f"error: {exc!s:.50}"

    # Orchestrator
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{ORCHESTRATOR_URL}/health")
            orchestrator_status = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as exc:
        orchestrator_status = f"error: {exc!s:.50}"

    overall = "healthy" if postgres_status == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        postgres=postgres_status,
        orchestrator=orchestrator_status,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
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
        "feedback_api:app",
        host="0.0.0.0",
        port=API_PORT,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
