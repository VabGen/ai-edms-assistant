"""
feedback-collector/feedback_api.py — Сервис сбора обратной связи.

Архитектура:
• FastAPI + SQLAlchemy (async)
• Structlog для контекстного логирования
• Pydantic Settings для конфигурации
• Prometheus метрики + OpenTelemetry трассировка
• Интеграция с MCP (опционально)

Функции:
• Хранение диалогов с оценками пользователей
• Ежедневное обновление RAG из положительных диалогов (APScheduler)
• Формирование антипримеров из отрицательных оценок
• Аналитика качества ответов
"""
# ✅ ИСПРАВЛЕНО: __future__ с двумя подчёркиваниями
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncGenerator, List, Dict

import structlog
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String, Text, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ✅ Импорты из packages.core
from edms_ai_assistant.packages.core.settings import settings
from edms_ai_assistant.packages.core.logging.config import setup_logging_from_settings
from edms_ai_assistant.packages.core.logging import get_logger
from edms_ai_assistant.packages.core.security.jwt import verify_jwt_token

# ✅ Инициализация логирования (ОДИН РАЗ при старте)
setup_logging_from_settings(settings)

# ✅ ИСПРАВЛЕНО: __name__ вместо name
log = get_logger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# Конфигурация (через centralized settings)
# ───────────────────────────────────────────────────────────────────────────
DATABASE_URL: str = settings.DATABASE_URL
ORCHESTRATOR_URL: str = str(settings.ORCHESTRATOR_URL)
REDIS_URL: str = settings.REDIS_URL_BUILD
API_PORT: int = settings.FEEDBACK_PORT
RAG_REBUILD_HOUR: int = settings.RAG_UPDATE_HOUR
ANTI_EXAMPLE_REDIS_KEY: str = "edms:anti_examples_block"
ANTI_EXAMPLE_TTL: int = 86400  # 24 часа

# Security
security = HTTPBearer(auto_error=False)

# ───────────────────────────────────────────────────────────────────────────
# Prometheus метрики
# ───────────────────────────────────────────────────────────────────────────
DIALOGS_SAVED = Counter("edms_fb_dialogs_saved_total", "Total dialogs saved")
FEEDBACK_RECEIVED = Counter(
    "edms_fb_feedback_total", "Feedback received", ["rating"]
)
RAG_UPDATES = Counter("edms_fb_rag_updates_total", "RAG rebuild runs")
RAG_UPDATE_LATENCY = Histogram(
    "edms_fb_rag_update_seconds", "RAG rebuild duration"
)
FEEDBACK_PROCESSING_TIME = Histogram("edms_fb_processing_seconds", "Feedback processing time")

# ───────────────────────────────────────────────────────────────────────────
# SQLAlchemy
# ───────────────────────────────────────────────────────────────────────────
class FBBase(DeclarativeBase):
    pass

_engine = create_async_engine(DATABASE_URL, pool_pre_ping=True, echo=settings.DEBUG)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency для получения сессии БД."""
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ✅ ИСПРАВЛЕНО: __tablename__ с двумя подчёркиваниями
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
    entities: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    selected_tool: Mapped[str | None] = mapped_column(String(255))
    tool_args: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    tool_result: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
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
    log.info("Database tables created/verified", port=settings.FEEDBACK_PORT)


# ───────────────────────────────────────────────────────────────────────────
# Pydantic модели
# ───────────────────────────────────────────────────────────────────────────
class SaveDialogRequest(BaseModel):
    """Запрос сохранения диалога (от оркестратора)."""
    dialog_id: str
    user_id: str
    session_id: str
    user_query: str
    normalized_query: str | None = None
    intent: str | None = None
    confidence: float | None = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    selected_tool: str | None = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_result: Dict[str, Any] = Field(default_factory=dict)
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
    top_intents: List[Dict[str, Any]]
    top_tools: List[Dict[str, Any]]
    period_days: int


class HealthResponse(BaseModel):
    status: str
    postgres: str
    orchestrator: str
    uptime_seconds: float


# ───────────────────────────────────────────────────────────────────────────
# Глобальное состояние
# ───────────────────────────────────────────────────────────────────────────
_start_time = time.monotonic()
_scheduler: Any = None


# ───────────────────────────────────────────────────────────────────────────
# Ежедневное обновление RAG
# ───────────────────────────────────────────────────────────────────────────
async def _daily_rag_update() -> None:
    """
    Ежедневное обновление RAG-индекса в 03:00 UTC.
    """
    start = time.monotonic()
    log.info("Daily RAG update started", port=settings.FEEDBACK_PORT)

    async with _session_factory() as session:
        # --- Позитивные диалоги ---
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        stmt_pos = (
            select(DialogLog)
            .where(
                DialogLog.user_feedback == 1,
                DialogLog.created_at >= cutoff,
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
                # ✅ ИСПРАВЛЕНО: удалены пробелы в ключах
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
                log.warning("Failed to add positive dialog to RAG", error=str(exc), exc_info=True)

    # Формируем блок антипримеров
    anti_lines = ["=== АНТИПРИМЕРЫ (избегать подобного) ===\n"]
    for row in negative_rows[:30]:
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
        log.info("Anti-examples saved to Redis", count=len(negative_rows[:30]))
    except Exception as exc:
        log.warning("Failed to save anti-examples to Redis", error=str(exc))

    elapsed = time.monotonic() - start
    RAG_UPDATES.inc()
    RAG_UPDATE_LATENCY.observe(elapsed)

    log.info(
        "Daily RAG update completed",
        positive_added=added_count,
        anti_examples=len(negative_rows),
        elapsed_seconds=f"{elapsed:.2f}",
    )


# ───────────────────────────────────────────────────────────────────────────
# Жизненный цикл приложения
# ───────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Инициализация БД и планировщика."""
    global _scheduler

    await _create_tables()

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
    log.info("Scheduler started", rag_update_hour=f"{RAG_REBUILD_HOUR:02d}:00 UTC")

    yield

    if _scheduler:
        _scheduler.shutdown(wait=False)
    await _engine.dispose()
    log.info("Feedback Collector stopped", port=settings.FEEDBACK_PORT)


# ───────────────────────────────────────────────────────────────────────────
# FastAPI приложение
# ───────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EDMS Feedback Collector",
    version="1.0.0",
    description="Сервис сбора обратной связи и обновления RAG для EDMS AI",
    lifespan=lifespan,
    openapi_url="/openapi.json" if settings.DEBUG else None,  # Отключаем в prod
)


# ───────────────────────────────────────────────────────────────────────────
# Security dependency
# ───────────────────────────────────────────────────────────────────────────
async def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> Dict[str, Any]:
    """Проверка JWT токена для защищённых эндпоинтов."""
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    try:
        token_data = verify_jwt_token(
            credentials.credentials,
            settings.JWT_SECRET_KEY.get_secret_value()
        )
        return {
            "user_id": token_data.user_id,
            "role": token_data.role,
            "permissions": token_data.permissions,
        }
    except Exception as e:
        log.warning("Auth failed", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid token")


# ───────────────────────────────────────────────────────────────────────────
# Endpoints
# ───────────────────────────────────────────────────────────────────────────
@app.post("/dialogs", summary="Сохранить диалог", status_code=201)
@FEEDBACK_PROCESSING_TIME.time()
async def save_dialog(
    request: SaveDialogRequest,
    db: AsyncSession = Depends(get_db),
    # ✅ Рекомендуется включить для production
    # auth: Dict = Depends(require_auth),
) -> Dict[str, str]:
    """
    Сохранить диалог в базу данных.
    Вызывается оркестратором после каждого ответа ассистента.
    """
    bound_log = log.bind(
        dialog_id=request.dialog_id,
        user_id=request.user_id,
        intent=request.intent,
    )

    bound_log.debug("Saving dialog request received")

    async with db.begin():
        # ✅ ИСПРАВЛЕНО: удалены пробелы в строках
        existing = await db.get(DialogLog, request.dialog_id)
        if existing:
            bound_log.info("Dialog already exists", status="already_exists")
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
        db.add(dialog)

    DIALOGS_SAVED.inc()
    bound_log.info("Dialog saved successfully", port=settings.FEEDBACK_PORT)
    return {"status": "saved", "dialog_id": request.dialog_id}


@app.post("/feedback", summary="Оценить ответ пользователем")
@FEEDBACK_PROCESSING_TIME.time()
async def submit_feedback(
    request: FeedbackSubmitRequest,
    db: AsyncSession = Depends(get_db),
    # ✅ Опционально: раскомментировать для защиты
    # auth: Dict = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Принять оценку пользователя для диалога.
    Обновляет user_feedback и feedback_comment в базе данных.
    """
    bound_log = log.bind(dialog_id=request.dialog_id, rating=request.rating)

    async with db.begin():
        dialog = await db.get(DialogLog, request.dialog_id)
        if not dialog:
            bound_log.warning("Dialog not found for feedback")
            raise HTTPException(
                status_code=404,
                detail=f"Диалог {request.dialog_id} не найден",
            )

        dialog.user_feedback = request.rating
        dialog.feedback_comment = request.comment
        dialog.feedback_at = datetime.now(timezone.utc)

    rating_label = {1: "positive", 0: "neutral", -1: "negative"}.get(request.rating, "neutral")
    FEEDBACK_RECEIVED.labels(rating=rating_label).inc()

    bound_log.info("Feedback saved successfully", port=settings.FEEDBACK_PORT)
    return {
        "status": "saved",
        "dialog_id": request.dialog_id,
        "rating": request.rating,
        "message": "Спасибо за обратную связь!",
    }


@app.get("/feedback/stats", response_model=FeedbackStatsResponse, summary="Статистика")
async def get_feedback_stats(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    # ✅ Рекомендуется включить защиту для аналитики
    # auth: Dict = Depends(require_auth),
) -> FeedbackStatsResponse:
    """
    Получить агрегированную статистику качества ответов.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    total = (await db.execute(
        select(func.count()).where(DialogLog.created_at >= cutoff)
    )).scalar() or 0

    rated = (await db.execute(
        select(func.count()).where(
            DialogLog.created_at >= cutoff,
            DialogLog.user_feedback.is_not(None),
        )
    )).scalar() or 0

    positive = (await db.execute(
        select(func.count()).where(
            DialogLog.created_at >= cutoff,
            DialogLog.user_feedback == 1,
        )
    )).scalar() or 0

    negative = (await db.execute(
        select(func.count()).where(
            DialogLog.created_at >= cutoff,
            DialogLog.user_feedback == -1,
        )
    )).scalar() or 0

    neutral = rated - positive - negative

    avg_latency = (await db.execute(
        select(func.avg(DialogLog.latency_ms)).where(DialogLog.created_at >= cutoff)
    )).scalar() or 0

    intent_rows = (await db.execute(
        select(DialogLog.intent, func.count().label("cnt"))
        .where(
            DialogLog.created_at >= cutoff,
            DialogLog.intent.is_not(None),
        )
        .group_by(DialogLog.intent)
        .order_by(text("cnt DESC"))
        .limit(10)
    )).all()
    top_intents = [{"intent": r[0], "count": r[1]} for r in intent_rows]

    tool_rows = (await db.execute(
        select(DialogLog.selected_tool, func.count().label("cnt"))
        .where(
            DialogLog.created_at >= cutoff,
            DialogLog.selected_tool.is_not(None),
        )
        .group_by(DialogLog.selected_tool)
        .order_by(text("cnt DESC"))
        .limit(10)
    )).all()
    top_tools = [{"tool": r[0], "count": r[1]} for r in tool_rows]

    return FeedbackStatsResponse(
        total_dialogs=total,
        rated_dialogs=rated,
        positive_count=positive,
        negative_count=negative,
        neutral_count=neutral,
        positive_rate=positive / rated if rated > 0 else 0.0,
        negative_rate=negative / rated if rated > 0 else 0.0,
        avg_latency_ms=round(float(avg_latency), 2),
        top_intents=top_intents,
        top_tools=top_tools,
        period_days=days,
    )


@app.post("/rag/trigger-update", summary="Принудительный запуск обновления RAG")
async def trigger_rag_update(
    # ✅ Только для админов
    auth: Dict = Depends(require_auth),
) -> Dict[str, str]:
    """Запустить обновление RAG вручную."""
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    asyncio.create_task(_daily_rag_update())
    log.info("Manual RAG update triggered", triggered_by=auth.get("user_id"))
    return {"status": "triggered", "message": "Обновление RAG запущено в фоне"}


@app.get("/health", response_model=HealthResponse, summary="Проверка состояния")
async def health() -> HealthResponse:
    """Проверить доступность компонентов feedback-collector."""
    postgres_status = "ok"
    orchestrator_status = "ok"

    # PostgreSQL
    try:
        async with _session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        postgres_status = f"error: {str(exc)[:50]}"
        log.warning("Health check: PostgreSQL failed", error=postgres_status)

    # Orchestrator
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{ORCHESTRATOR_URL}/health")
            orchestrator_status = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as exc:
        orchestrator_status = f"error: {str(exc)[:50]}"
        log.warning("Health check: Orchestrator unreachable", error=orchestrator_status)

    overall = "healthy" if postgres_status == "ok" and orchestrator_status == "ok" else "degraded"

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


# ───────────────────────────────────────────────────────────────────────────
# Запуск
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "feedback_api:app",
        host="0.0.0.0",
        port=API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG,
    )