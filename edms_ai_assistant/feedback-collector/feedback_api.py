# feedback-collector/feedback_api.py
"""
Feedback Collector — FastAPI сервис для RLHF-loop.

Endpoints:
    POST /dialogs         — сохранить диалог (вызывается оркестратором)
    POST /feedback        — оценить диалог (-1 | 0 | 1)
    GET  /feedback/stats  — агрегированная статистика
    GET  /health
    GET  /metrics

APScheduler (ежедневно в RAG_UPDATE_HOUR UTC):
    1. Диалоги rating=1 → POST orchestrator /rag/add (successful_dialogs)
    2. Диалоги rating=-1 → Redis key="anti_examples_block" (текст антипримеров)
    3. Логирование статистики обновления
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal

import httpx
import redis.asyncio as aioredis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String, Text, Boolean, func, select, update
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8002")
FEEDBACK_PORT = int(os.getenv("FEEDBACK_PORT", "8003"))
RAG_UPDATE_HOUR = int(os.getenv("RAG_UPDATE_HOUR", "3"))
RAG_UPDATE_MINUTE = int(os.getenv("RAG_UPDATE_MINUTE", "0"))

_engine = create_async_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)
_Session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    _engine, expire_on_commit=False, autoflush=False
)
_redis: aioredis.Redis | None = None
_scheduler = AsyncIOScheduler(timezone="UTC")


# ── ORM ───────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class DialogLog(Base):
    """Лог диалога с обратной связью."""

    __tablename__ = "dialog_logs"
    __table_args__ = {"schema": "edms"}

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    session_id: Mapped[str] = mapped_column(String(255), index=True)
    user_query: Mapped[str] = mapped_column(Text)
    normalized_query: Mapped[str | None] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(100), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    entities: Mapped[dict] = mapped_column(JSONB(astext_type=Text()), server_default="{}")
    selected_tool: Mapped[str | None] = mapped_column(String(255))
    tool_args: Mapped[dict] = mapped_column(JSONB(astext_type=Text()), server_default="{}")
    tool_result: Mapped[dict] = mapped_column(JSONB(astext_type=Text()), server_default="{}")
    final_response: Mapped[str | None] = mapped_column(Text)
    model_used: Mapped[str | None] = mapped_column(String(100))
    tokens_used: Mapped[int] = mapped_column(Integer, server_default="0")
    user_feedback: Mapped[int | None] = mapped_column(Integer)
    feedback_comment: Mapped[str | None] = mapped_column(Text)
    bypass_llm: Mapped[bool] = mapped_column(Boolean, server_default="false")
    latency_ms: Mapped[int] = mapped_column(Integer, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ── Pydantic схемы ────────────────────────────────────────────────────────


class DialogCreate(BaseModel):
    id: str
    user_id: str
    session_id: str
    user_query: str
    normalized_query: str | None = None
    intent: str | None = None
    confidence: float | None = None
    entities: dict = Field(default_factory=dict)
    selected_tool: str | None = None
    tool_args: dict = Field(default_factory=dict)
    tool_result: dict = Field(default_factory=dict)
    final_response: str | None = None
    model_used: str | None = None
    tokens_used: int = 0
    bypass_llm: bool = False
    latency_ms: int = 0


class FeedbackRequest(BaseModel):
    dialog_id: str
    rating: Literal[-1, 0, 1]
    comment: str | None = Field(None, max_length=2000)


# ── Lifespan ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _redis

    try:
        _redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        await _redis.ping()
        logger.info("Redis connected")
    except Exception as exc:
        logger.warning("Redis unavailable: %s", exc)
        _redis = None

    _scheduler.add_job(
        _daily_rlhf_update,
        trigger=CronTrigger(hour=RAG_UPDATE_HOUR, minute=RAG_UPDATE_MINUTE),
        id="rlhf_update",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "Feedback collector started — RLHF update scheduled at %02d:%02d UTC",
        RAG_UPDATE_HOUR, RAG_UPDATE_MINUTE,
    )

    yield

    _scheduler.shutdown(wait=False)
    if _redis:
        await _redis.aclose()
    await _engine.dispose()
    logger.info("Feedback collector stopped")


app = FastAPI(
    title="EDMS Feedback Collector",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.post("/dialogs", status_code=201, summary="Сохранить диалог")
async def save_dialog(data: DialogCreate) -> dict:
    async with _Session() as session:
        async with session.begin():
            stmt = pg_insert(DialogLog).values(
                id=data.id,
                user_id=data.user_id,
                session_id=data.session_id,
                user_query=data.user_query,
                normalized_query=data.normalized_query,
                intent=data.intent,
                confidence=data.confidence,
                entities=data.entities,
                selected_tool=data.selected_tool,
                tool_args=data.tool_args,
                tool_result=data.tool_result,
                final_response=data.final_response,
                model_used=data.model_used,
                tokens_used=data.tokens_used,
                bypass_llm=data.bypass_llm,
                latency_ms=data.latency_ms,
            ).on_conflict_do_nothing(index_elements=["id"])
            await session.execute(stmt)
    return {"status": "saved", "dialog_id": data.id}


@app.post("/feedback", summary="Оценить диалог")
async def submit_feedback(req: FeedbackRequest) -> dict:
    async with _Session() as session:
        async with session.begin():
            result = await session.execute(
                select(DialogLog).where(DialogLog.id == req.dialog_id)
            )
            dialog = result.scalar_one_or_none()
            if not dialog:
                raise HTTPException(404, f"Диалог {req.dialog_id} не найден")

            await session.execute(
                update(DialogLog)
                .where(DialogLog.id == req.dialog_id)
                .values(
                    user_feedback=req.rating,
                    feedback_comment=req.comment,
                    feedback_at=datetime.now(timezone.utc),
                )
            )

    logger.info("Feedback: dialog=%s rating=%d", req.dialog_id, req.rating)
    return {"status": "recorded", "dialog_id": req.dialog_id, "rating": req.rating}


@app.get("/feedback/stats", summary="Агрегированная статистика обратной связи")
async def feedback_stats() -> dict:
    async with _Session() as session:
        total_r = await session.execute(select(func.count(DialogLog.id)))
        total = total_r.scalar_one()

        rated_r = await session.execute(
            select(func.count(DialogLog.id)).where(DialogLog.user_feedback.isnot(None))
        )
        rated = rated_r.scalar_one()

        pos_r = await session.execute(
            select(func.count(DialogLog.id)).where(DialogLog.user_feedback == 1)
        )
        neg_r = await session.execute(
            select(func.count(DialogLog.id)).where(DialogLog.user_feedback == -1)
        )

        avg_lat_r = await session.execute(select(func.avg(DialogLog.latency_ms)))
        avg_lat = avg_lat_r.scalar_one()

        bypass_r = await session.execute(
            select(func.count(DialogLog.id)).where(DialogLog.bypass_llm.is_(True))
        )

    return {
        "total_dialogs": total,
        "rated_dialogs": rated,
        "positive": pos_r.scalar_one(),
        "negative": neg_r.scalar_one(),
        "neutral": rated - pos_r.scalar_one() - neg_r.scalar_one(),
        "rating_rate": round(rated / total, 3) if total else 0.0,
        "avg_latency_ms": round(avg_lat or 0, 1),
        "bypass_llm_count": bypass_r.scalar_one(),
    }


@app.get("/health")
async def health() -> dict:
    redis_ok = False
    if _redis:
        try:
            await _redis.ping()
            redis_ok = True
        except Exception:
            pass
    return {
        "status": "ok",
        "redis": "ok" if redis_ok else "unavailable",
        "scheduler_running": _scheduler.running,
    }


@app.get("/metrics")
async def metrics() -> dict:
    return {"status": "ok", "note": "Install prometheus_client for full metrics"}


# ── RLHF Background Job ───────────────────────────────────────────────────


async def _daily_rlhf_update() -> None:
    """
    Ежедневный RLHF-цикл.

    1. Диалоги rating=1 за последние 24ч → POST к оркестратору /rag/add
    2. Диалоги rating=-1 → сохранить блок антипримеров в Redis
    3. Логировать статистику
    """
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    logger.info("RLHF daily update started (since %s)", since.isoformat())

    positive_count = 0
    negative_count = 0

    async with _Session() as session:
        # Позитивные диалоги → RAG
        pos_r = await session.execute(
            select(DialogLog).where(
                DialogLog.user_feedback == 1,
                DialogLog.created_at >= since,
            )
        )
        positive_dialogs = list(pos_r.scalars().all())

        # Негативные диалоги → anti_examples
        neg_r = await session.execute(
            select(DialogLog).where(
                DialogLog.user_feedback == -1,
                DialogLog.created_at >= since,
            )
        )
        negative_dialogs = list(neg_r.scalars().all())

    # Отправляем позитивные в RAG
    if positive_dialogs:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for dialog in positive_dialogs:
                try:
                    await client.post(
                        f"{ORCHESTRATOR_URL}/rag/add",
                        json={
                            "dialog_id": dialog.id,
                            "user_query": dialog.user_query,
                            "normalized_query": dialog.normalized_query or dialog.user_query,
                            "intent": dialog.intent or "unknown",
                            "tool_used": dialog.selected_tool or "",
                            "tool_args": dialog.tool_args or {},
                            "response": dialog.final_response or "",
                            "rating": 1,
                        },
                    )
                    positive_count += 1
                except Exception as exc:
                    logger.warning("RAG add failed for dialog %s: %s", dialog.id, exc)

    # Сохраняем негативные как антипримеры в Redis
    if negative_dialogs and _redis:
        lines = ["=== ЧЕГО НЕЛЬЗЯ ДЕЛАТЬ ==="]
        for i, d in enumerate(negative_dialogs[:20], 1):
            lines.append(
                f"[{i}] Запрос: {d.user_query}\n"
                f"    Инструмент: {d.selected_tool or '—'}\n"
                f"    Плохой ответ: {(d.final_response or '')[:200]}\n"
                f"    Комментарий: {d.feedback_comment or '—'}"
            )
            negative_count += 1
        try:
            await _redis.setex("anti_examples_block", 86400 * 2, "\n\n".join(lines))
        except Exception as exc:
            logger.warning("Redis anti_examples write failed: %s", exc)

    logger.info(
        "RLHF update completed: positive_added=%d negative_stored=%d",
        positive_count, negative_count,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=FEEDBACK_PORT, log_level="info")
