# edms_ai_assistant/feedback_collector/feedback_api.py
"""
Сервис сбора обратной связи и обновления RAG.

ИСПРАВЛЕНИЯ:
  1. Добавлен endpoint POST /dialogs — ожидается AgentOrchestrator._log_dialog()
  2. APScheduler (ежедневно 03:00 UTC): rebuild RAG из позитивных диалогов
  3. Anti-examples из негативных диалогов → Redis key "anti_examples_block"
  4. Все зависимости через settings (не хардкод)

Endpoints:
  POST /dialogs        — сохранить диалог (вызывается оркестратором)
  POST /feedback       — оценка от пользователя (-1/0/1)
  GET  /feedback/stats — статистика оценок
  POST /rag/rebuild    — ручной запуск обновления RAG
  GET  /health
  GET  /metrics
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import httpx
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger("feedback")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Конфигурация из переменных окружения ──────────────────────────────────────
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8002")
FEEDBACK_PORT = int(os.getenv("FEEDBACK_PORT", "8003"))
REBUILD_INTERVAL_HOURS = int(os.getenv("RAG_REBUILD_INTERVAL_HOURS", "24"))

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://"
    f"{os.environ.get('POSTGRES_USER', 'edms')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'edms_secret')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'edms_ai')}",
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── SQLAlchemy ────────────────────────────────────────────────────────────────
engine = create_async_engine(_DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

_SCHEMA = "edms"  # ИСПРАВЛЕНО: единая схема для всего проекта


class Base(DeclarativeBase):
    pass


class DialogLog(Base):
    """ORM-модель лога диалога.

    Хранит полную историю взаимодействия: запрос, обработка, ответ, оценка.
    """
    __tablename__ = "dialog_logs"
    __table_args__ = {"schema": _SCHEMA}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    dialog_id = Column(PG_UUID(as_uuid=False), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    user_query = Column(Text, nullable=False)
    normalized_query = Column(Text, nullable=True)
    intent = Column(String(100), nullable=True, index=True)
    confidence = Column(Float, default=0.0)
    entities = Column(JSONB, default={})
    selected_tool = Column(String(255), nullable=True)
    tool_args = Column(JSONB, default={})
    tool_result = Column(JSONB, default={})
    final_response = Column(Text, nullable=True)
    model_used = Column(String(100), nullable=True)
    tokens_used = Column(Integer, default=0)
    user_feedback = Column(Integer, nullable=True)        # -1 / 0 / 1
    feedback_comment = Column(Text, nullable=True)
    bypass_llm = Column(Boolean, default=False)
    latency_ms = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    feedback_at = Column(DateTime(timezone=True), nullable=True)


# ── Pydantic модели ───────────────────────────────────────────────────────────

class DialogCreateRequest(BaseModel):
    """Запрос на сохранение диалога от оркестратора."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    user_query: str
    normalized_query: str = ""
    intent: str = ""
    confidence: float = 0.0
    entities: dict[str, Any] = {}
    selected_tool: str = ""
    tool_args: dict[str, Any] = {}
    tool_result: dict[str, Any] = {}
    final_response: str = ""
    model_used: str = ""
    tokens_used: int = 0
    bypass_llm: bool = False
    latency_ms: int = 0


class FeedbackRequest(BaseModel):
    """Запрос оценки диалога от пользователя."""
    dialog_id: str
    rating: int = Field(..., ge=-1, le=1, description="-1 (плохо), 0 (нейтрально), 1 (хорошо)")
    comment: str = Field(default="", max_length=1000)


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    rating: int
    negative_rate: float


# ── Метрики ───────────────────────────────────────────────────────────────────
_metrics: dict[str, Any] = {
    "feedback_received_total": 0,
    "feedback_positive": 0,
    "feedback_negative": 0,
    "feedback_neutral": 0,
    "dialogs_saved": 0,
    "rag_rebuilds_total": 0,
    "last_rebuild_timestamp": 0,
    "start_time": time.time(),
}

# ── Scheduler ─────────────────────────────────────────────────────────────────
_scheduler = AsyncIOScheduler(timezone="UTC")


async def _scheduled_rag_rebuild() -> None:
    """Ежедневная задача (03:00 UTC): обновление RAG из логов диалогов.

    Алгоритм:
    1. Берём диалоги с rating=1 за последние 24ч → добавляем в successful_dialogs
    2. Берём диалоги с rating=-1 → формируем блок anti-examples → Redis
    3. Логируем статистику
    """
    logger.info("Запуск ежедневного обновления RAG...")
    try:
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select, and_
            from datetime import timedelta

            cutoff = datetime.utcnow() - timedelta(hours=24)

            # Позитивные диалоги → RAG
            stmt_pos = select(DialogLog).where(
                and_(DialogLog.user_feedback == 1, DialogLog.created_at >= cutoff)
            )
            pos_rows = (await db.execute(stmt_pos)).scalars().all()

            if pos_rows:
                positive_dialogs = [
                    {
                        "user_query": r.user_query,
                        "intent": r.intent or "",
                        "selected_tool": r.selected_tool or "",
                        "final_response": r.final_response or "",
                        "user_feedback": 1,
                    }
                    for r in pos_rows
                ]
                async with httpx.AsyncClient(timeout=30) as client:
                    await client.post(f"{ORCHESTRATOR_URL}/rag/rebuild")
                logger.info("RAG обновлён: %d позитивных диалогов", len(positive_dialogs))

            # Негативные диалоги → anti-examples в Redis
            stmt_neg = select(DialogLog).where(
                and_(DialogLog.user_feedback == -1, DialogLog.created_at >= cutoff)
            ).limit(50)
            neg_rows = (await db.execute(stmt_neg)).scalars().all()

            if neg_rows:
                anti_block = "\n".join(
                    f"❌ Запрос: {r.user_query}\n   Плохой ответ: {(r.final_response or '')[:150]}"
                    for r in neg_rows
                )
                try:
                    import redis.asyncio as aioredis
                    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
                    await redis_client.setex("anti_examples_block", 86400 * 7, anti_block)
                    await redis_client.aclose()
                    logger.info("Anti-examples обновлены: %d диалогов", len(neg_rows))
                except Exception as redis_exc:
                    logger.error("Ошибка записи anti-examples в Redis: %s", redis_exc)

        _metrics["rag_rebuilds_total"] += 1
        _metrics["last_rebuild_timestamp"] = int(time.time())

    except Exception as exc:
        logger.error("Ошибка ежедневного обновления RAG: %s", exc, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и остановка сервиса."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database connection established and schema checked.")
    except Exception as e:
        logger.error(f"CRITICAL: Could not connect to database: {e}")
        logger.error("Application will start, but database features will be disabled.")

    _scheduler.add_job(
        _scheduled_rag_rebuild,
        trigger="cron",
        hour=3,
        minute=0,
        id="daily_rag_rebuild",
    )

    _scheduler.start()
    logger.info(
        "Feedback collector запущен, RAG rebuild interval=%dh",
        REBUILD_INTERVAL_HOURS,
    )

    yield

    _scheduler.shutdown(wait=False)
    await engine.dispose()


# ── FastAPI приложение ────────────────────────────────────────────────────────
app = FastAPI(
    title="EDMS Feedback Collector",
    version="1.1.0",
    description="Сервис сбора обратной связи, логирования диалогов и обновления RAG",
    lifespan=lifespan,
)


@app.post(
    "/dialogs",
    status_code=201,
    summary="Сохранить диалог (вызывается оркестратором)",
    tags=["Dialogs"],
)
async def save_dialog(req: DialogCreateRequest) -> dict:
    """Сохраняет полный лог диалога из оркестратора.

    Вызывается автоматически после каждого ответа агента.
    """
    try:
        async with AsyncSessionLocal() as db, db.begin():
            log = DialogLog(
                dialog_id=req.id,
                user_id=req.user_id,
                session_id=req.session_id,
                user_query=req.user_query,
                normalized_query=req.normalized_query,
                intent=req.intent,
                confidence=req.confidence,
                entities=req.entities,
                selected_tool=req.selected_tool,
                tool_args=req.tool_args,
                tool_result=req.tool_result,
                final_response=req.final_response,
                model_used=req.model_used,
                tokens_used=req.tokens_used,
                bypass_llm=req.bypass_llm,
                latency_ms=req.latency_ms,
            )
            db.add(log)
        _metrics["dialogs_saved"] += 1
        return {"success": True, "dialog_id": req.id}
    except Exception as exc:
        logger.error("Ошибка сохранения диалога %s: %s", req.id, exc, exc_info=True)
        return {"success": False, "error": str(exc)}


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def receive_feedback(
    req: FeedbackRequest,
    background_tasks: BackgroundTasks,
) -> FeedbackResponse:
    """Принимает оценку диалога от пользователя.

    При rating=1 немедленно добавляет диалог в RAG (фоновая задача).
    При rating=-1 диалог попадёт в anti-examples при следующем ежедневном rebuild.
    """
    _metrics["feedback_received_total"] += 1
    if req.rating == 1:
        _metrics["feedback_positive"] += 1
    elif req.rating == -1:
        _metrics["feedback_negative"] += 1
    else:
        _metrics["feedback_neutral"] += 1

    # Обновляем оценку в БД
    ok = False
    try:
        async with AsyncSessionLocal() as db, db.begin():
            from sqlalchemy import update as sa_update
            stmt = (
                sa_update(DialogLog)
                .where(DialogLog.dialog_id == req.dialog_id)
                .values(
                    user_feedback=req.rating,
                    feedback_comment=req.comment,
                    feedback_at=datetime.utcnow(),
                )
            )
            await db.execute(stmt)
            ok = True
    except Exception as exc:
        logger.error("Ошибка обновления оценки %s: %s", req.dialog_id, exc)

    # Немедленный RAG update при положительной оценке
    if req.rating == 1 and ok:
        background_tasks.add_task(_trigger_rag_update, req.dialog_id)

    total = max(_metrics["feedback_received_total"], 1)
    neg_rate = round(_metrics["feedback_negative"] / total, 3)
    labels = {1: "положительная", 0: "нейтральная", -1: "отрицательная"}

    return FeedbackResponse(
        success=ok,
        message=f"Оценка '{labels.get(req.rating, '')}' {'принята' if ok else 'не удалось сохранить'}",
        rating=req.rating,
        negative_rate=neg_rate,
    )


async def _trigger_rag_update(dialog_id: str) -> None:
    """Фоновая задача: уведомить оркестратор об обновлении RAG."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                f"{ORCHESTRATOR_URL}/rag/rebuild",
                params={"dialog_id": dialog_id},
            )
    except Exception as exc:
        logger.error("Ошибка фонового обновления RAG: %s", exc)


@app.get("/feedback/stats", tags=["Feedback"])
async def feedback_stats() -> dict:
    """Возвращает агрегированную статистику оценок."""
    total = max(_metrics["feedback_received_total"], 1)
    return {
        "total": _metrics["feedback_received_total"],
        "positive": _metrics["feedback_positive"],
        "negative": _metrics["feedback_negative"],
        "neutral": _metrics["feedback_neutral"],
        "positive_rate": round(_metrics["feedback_positive"] / total, 3),
        "negative_rate": round(_metrics["feedback_negative"] / total, 3),
        "dialogs_saved": _metrics["dialogs_saved"],
        "rag_rebuilds": _metrics["rag_rebuilds_total"],
        "last_rebuild": _metrics["last_rebuild_timestamp"],
    }


@app.post("/rag/rebuild", tags=["RAG"])
async def trigger_rebuild(background_tasks: BackgroundTasks) -> dict:
    """Вручную запускает перестройку RAG-индекса."""
    background_tasks.add_task(_scheduled_rag_rebuild)
    return {"status": "started", "message": "Перестройка RAG запущена в фоне"}


@app.get("/health", tags=["System"])
async def health() -> dict:
    return {
        "status": "ok",
        "service": "feedback-collector",
        "version": "1.1.0",
        "feedback_received": _metrics["feedback_received_total"],
        "dialogs_saved": _metrics["dialogs_saved"],
        "rag_rebuilds": _metrics["rag_rebuilds_total"],
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["System"])
async def prometheus_metrics() -> str:
    """Prometheus-метрики в text/plain формате."""
    uptime = int(time.time() - _metrics["start_time"])
    lines = [
        f'edms_feedback_total {_metrics["feedback_received_total"]}',
        f'edms_feedback_positive_total {_metrics["feedback_positive"]}',
        f'edms_feedback_negative_total {_metrics["feedback_negative"]}',
        f'edms_dialogs_saved_total {_metrics["dialogs_saved"]}',
        f'edms_rag_rebuilds_total {_metrics["rag_rebuilds_total"]}',
        f"edms_feedback_uptime_seconds {uptime}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    uvicorn.run("feedback_api:app", host="0.0.0.0", port=FEEDBACK_PORT, log_level="info")