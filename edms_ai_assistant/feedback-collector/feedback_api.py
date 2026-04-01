"""
Feedback Collector API — Сервис сбора обратной связи и ежедневного обновления RAG.

Функции:
- Приём оценок от пользователей (/feedback)
- Ежедневное задание: rebuild RAG из актуальных логов
- Анализ негативных паттернов и генерация анти-примеров
- Дашборд с аналитикой качества ответов
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feedback_collector")

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8002")
PG_DSN = os.getenv(
    "DATABASE_URL", "postgresql://edms:edms@localhost:5432/edms_ai"
)
REBUILD_INTERVAL_HOURS = int(os.getenv("RAG_REBUILD_INTERVAL_HOURS", "24"))

# ── Metrics ───────────────────────────────────────────────────────────────────
_metrics: dict[str, float] = {
    "feedback_received_total": 0,
    "feedback_positive": 0,
    "feedback_negative": 0,
    "feedback_neutral": 0,
    "rag_rebuilds_total": 0,
    "last_rebuild_timestamp": 0,
}

# ── DB Pool ───────────────────────────────────────────────────────────────────
_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool | None:
    global _pool
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=5)
        except Exception as exc:
            logger.error("DB connection failed: %s", exc)
    return _pool


# ── Background task ───────────────────────────────────────────────────────────


async def _daily_rag_rebuild() -> None:
    """Ежедневное обновление RAG — запрашиваем оркестратор."""
    while True:
        await asyncio.sleep(REBUILD_INTERVAL_HOURS * 3600)
        logger.info("Starting scheduled RAG rebuild...")
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(f"{ORCHESTRATOR_URL}/rag/rebuild")
                if resp.status_code == 200:
                    data = resp.json()
                    _metrics["rag_rebuilds_total"] += 1
                    _metrics["last_rebuild_timestamp"] = datetime.now().timestamp()
                    logger.info("RAG rebuild complete: %s", data.get("stats", {}))
                else:
                    logger.error("RAG rebuild failed: HTTP %d", resp.status_code)
        except Exception as exc:
            logger.error("RAG rebuild error: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_pool()
    task = asyncio.create_task(_daily_rag_rebuild())
    logger.info("Feedback Collector started, daily rebuild scheduled every %dh", REBUILD_INTERVAL_HOURS)
    yield
    task.cancel()
    if _pool:
        await _pool.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EDMS Feedback Collector",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    dialog_id: str
    rating: int = Field(..., ge=-1, le=1)
    comment: str = Field("", max_length=1000)
    user_id: str = ""


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    analysis: dict | None = None


class AnalyticsResponse(BaseModel):
    period_days: int
    total_dialogs: int
    rated_dialogs: int
    positive_rate: float
    negative_rate: float
    top_intents: list[dict]
    top_failing_intents: list[dict]
    avg_latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Принять оценку пользователя и проксировать в оркестратор."""
    _metrics["feedback_received_total"] += 1
    if req.rating == 1:
        _metrics["feedback_positive"] += 1
    elif req.rating == -1:
        _metrics["feedback_negative"] += 1
    else:
        _metrics["feedback_neutral"] += 1

    # Проксируем в оркестратор
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{ORCHESTRATOR_URL}/feedback",
                json={"dialog_id": req.dialog_id, "rating": req.rating, "comment": req.comment},
            )
            data = resp.json()

        # Анализ негативного фидбека
        analysis = None
        if req.rating == -1 and req.comment:
            analysis = await _analyze_negative_feedback(req.dialog_id, req.comment)

        return FeedbackResponse(
            success=data.get("success", False),
            message=data.get("message", ""),
            analysis=analysis,
        )
    except Exception as exc:
        logger.error("Feedback proxy error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(days: int = 7) -> AnalyticsResponse:
    """Аналитика качества ответов ассистента за период."""
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        async with pool.acquire() as conn:
            # Общие метрики
            totals = await conn.fetchrow("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE user_feedback IS NOT NULL) AS rated,
                    COUNT(*) FILTER (WHERE user_feedback = 1) AS positive,
                    COUNT(*) FILTER (WHERE user_feedback = -1) AS negative,
                    AVG(latency_ms) AS avg_latency
                FROM edms_ai.dialog_logs
                WHERE created_at > NOW() - INTERVAL '1 day' * $1
            """, days)

            # Топ успешных намерений
            top_intents = await conn.fetch("""
                SELECT intent, COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE user_feedback = 1) AS positive
                FROM edms_ai.dialog_logs
                WHERE created_at > NOW() - INTERVAL '1 day' * $1
                  AND intent IS NOT NULL
                GROUP BY intent ORDER BY total DESC LIMIT 10
            """, days)

            # Топ проблемных намерений
            failing_intents = await conn.fetch("""
                SELECT intent, COUNT(*) FILTER (WHERE user_feedback = -1) AS negative,
                       COUNT(*) AS total
                FROM edms_ai.dialog_logs
                WHERE created_at > NOW() - INTERVAL '1 day' * $1
                  AND intent IS NOT NULL AND user_feedback IS NOT NULL
                GROUP BY intent
                HAVING COUNT(*) FILTER (WHERE user_feedback = -1) > 0
                ORDER BY negative DESC LIMIT 5
            """, days)

        total = totals["total"] or 0
        rated = totals["rated"] or 0
        positive = totals["positive"] or 0
        negative = totals["negative"] or 0

        return AnalyticsResponse(
            period_days=days,
            total_dialogs=total,
            rated_dialogs=rated,
            positive_rate=positive / rated if rated > 0 else 0.0,
            negative_rate=negative / rated if rated > 0 else 0.0,
            top_intents=[
                {"intent": r["intent"], "total": r["total"], "positive": r["positive"]}
                for r in top_intents
            ],
            top_failing_intents=[
                {"intent": r["intent"], "negative": r["negative"], "total": r["total"]}
                for r in failing_intents
            ],
            avg_latency_ms=float(totals["avg_latency"] or 0),
        )
    except Exception as exc:
        logger.error("Analytics error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/analytics/negative-patterns")
async def get_negative_patterns(limit: int = 20) -> dict:
    """Паттерны запросов с негативными оценками для улучшения промптов."""
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT intent, selected_tool,
                       COUNT(*) AS count,
                       array_agg(feedback_comment) FILTER (WHERE feedback_comment != '') AS comments,
                       array_agg(user_query) AS sample_queries
                FROM edms_ai.dialog_logs
                WHERE user_feedback = -1
                GROUP BY intent, selected_tool
                ORDER BY count DESC
                LIMIT $1
            """, limit)

        patterns = []
        for row in rows:
            patterns.append({
                "intent": row["intent"],
                "tool": row["selected_tool"],
                "failure_count": row["count"],
                "user_comments": (row["comments"] or [])[:3],
                "sample_queries": (row["sample_queries"] or [])[:3],
            })

        return {"patterns": patterns, "total_negative": sum(p["failure_count"] for p in patterns)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/rag/trigger-rebuild")
async def trigger_rebuild() -> dict:
    """Ручной запуск обновления RAG."""
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{ORCHESTRATOR_URL}/rag/rebuild")
            data = resp.json()
        _metrics["rag_rebuilds_total"] += 1
        _metrics["last_rebuild_timestamp"] = datetime.now().timestamp()
        return {"success": True, "result": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health() -> dict:
    pool = await get_pool()
    db_ok = False
    if pool:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
        except Exception:
            pass
    return {
        "status": "healthy",
        "database": "healthy" if db_ok else "unavailable",
        "last_rag_rebuild": datetime.fromtimestamp(_metrics["last_rebuild_timestamp"]).isoformat()
        if _metrics["last_rebuild_timestamp"] else "never",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    lines = [
        f"feedback_received_total {int(_metrics['feedback_received_total'])}",
        f"feedback_positive_total {int(_metrics['feedback_positive'])}",
        f"feedback_negative_total {int(_metrics['feedback_negative'])}",
        f"rag_rebuilds_total {int(_metrics['rag_rebuilds_total'])}",
    ]
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _analyze_negative_feedback(dialog_id: str, comment: str) -> dict | None:
    """Простой анализ паттернов в негативном фидбеке."""
    keywords = {
        "неправильно": "incorrect_answer",
        "ошибка": "tool_error",
        "не то": "wrong_intent",
        "не понял": "intent_mismatch",
        "медленно": "performance",
        "долго": "performance",
    }
    pattern = None
    for kw, pat in keywords.items():
        if kw.lower() in comment.lower():
            pattern = pat
            break
    return {"pattern": pattern, "comment_length": len(comment)} if pattern else None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "feedback_api:app",
        host="0.0.0.0",
        port=int(os.getenv("FEEDBACK_PORT", "8003")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
