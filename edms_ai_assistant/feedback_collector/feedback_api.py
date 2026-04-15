"""
feedback_api.py — Сервис сбора обратной связи и обновления RAG.

Endpoints:
  POST /feedback       — приём оценки от пользователя
  GET  /feedback/stats — статистика оценок
  POST /rag/rebuild    — ручной запуск перестройки RAG
  GET  /health         — состояние сервиса

Ежедневная задача: rebuild RAG из актуальных логов.
Негативные диалоги → анти-примеры в RAG.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("feedback")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8002")
REBUILD_INTERVAL_HOURS = int(os.getenv("RAG_REBUILD_INTERVAL_HOURS", "24"))
API_PORT = int(os.getenv("FEEDBACK_PORT", "8003"))

_metrics: dict = {
    "feedback_received_total": 0,
    "feedback_positive": 0,
    "feedback_negative": 0,
    "feedback_neutral": 0,
    "rag_rebuilds_total": 0,
    "last_rebuild_timestamp": 0,
    "start_time": time.time(),
}


async def _daily_rag_rebuild() -> None:
    """Ежедневная задача: перестройка RAG-индекса."""
    await asyncio.sleep(60)  # первый запуск через минуту
    while True:
        try:
            logger.info("Daily RAG rebuild starting...")
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(f"{ORCHESTRATOR_URL}/rag/rebuild")
                if resp.is_success:
                    _metrics["rag_rebuilds_total"] += 1
                    _metrics["last_rebuild_timestamp"] = int(time.time())
                    logger.info("Daily RAG rebuild complete")
                else:
                    logger.warning("RAG rebuild failed: %s", resp.text[:200])
        except Exception as exc:
            logger.error("Daily RAG rebuild error: %s", exc)

        await asyncio.sleep(REBUILD_INTERVAL_HOURS * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_daily_rag_rebuild())
    logger.info("Feedback collector started, RAG rebuild interval=%dh", REBUILD_INTERVAL_HOURS)
    yield
    task.cancel()


app = FastAPI(
    title="EDMS Feedback Collector",
    version="1.0.0",
    description="Сервис сбора обратной связи и обновления RAG",
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    dialog_id: str = Field(..., description="UUID диалога из /chat ответа")
    rating: int = Field(..., ge=-1, le=1, description="-1 (плохо) / 0 (нейтрально) / 1 (хорошо)")
    comment: str = Field(default="", max_length=1000, description="Комментарий пользователя")


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    rating: int
    negative_rate: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse)
async def receive_feedback(
    req: FeedbackRequest,
    background_tasks: BackgroundTasks,
) -> FeedbackResponse:
    """
    Принять оценку пользователя.

    rating=1  → диалог успешный → добавляется в RAG как few-shot пример
    rating=-1 → диалог неудачный → добавляется как анти-пример
    """
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
            if resp.is_success:
                result = resp.json()
                success = result.get("success", True)
            else:
                logger.warning("Orchestrator feedback error: %d", resp.status_code)
                success = False
    except Exception as exc:
        logger.error("Failed to forward feedback: %s", exc)
        success = False

    # При позитивной оценке — немедленно добавляем в RAG
    if req.rating == 1 and success:
        background_tasks.add_task(_trigger_rag_update, req.dialog_id)

    total = max(_metrics["feedback_received_total"], 1)
    neg_rate = round(_metrics["feedback_negative"] / total, 3)

    labels = {1: "положительная", 0: "нейтральная", -1: "отрицательная"}
    return FeedbackResponse(
        success=success,
        message=f"Оценка '{labels.get(req.rating, '')}' {'принята' if success else 'не удалось сохранить'}",
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
        logger.error("RAG trigger update failed: %s", exc)


@app.get("/feedback/stats")
async def feedback_stats() -> dict:
    """Статистика оценок пользователей."""
    total = max(_metrics["feedback_received_total"], 1)
    return {
        "total": _metrics["feedback_received_total"],
        "positive": _metrics["feedback_positive"],
        "negative": _metrics["feedback_negative"],
        "neutral": _metrics["feedback_neutral"],
        "positive_rate": round(_metrics["feedback_positive"] / total, 3),
        "negative_rate": round(_metrics["feedback_negative"] / total, 3),
        "rag_rebuilds": _metrics["rag_rebuilds_total"],
        "last_rebuild": _metrics["last_rebuild_timestamp"],
    }


@app.post("/rag/rebuild")
async def trigger_rebuild(background_tasks: BackgroundTasks) -> dict:
    """Вручную запустить перестройку RAG."""
    background_tasks.add_task(_run_manual_rebuild)
    return {"status": "started", "message": "Перестройка RAG запущена"}


async def _run_manual_rebuild() -> None:
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{ORCHESTRATOR_URL}/rag/rebuild")
            if resp.is_success:
                _metrics["rag_rebuilds_total"] += 1
                _metrics["last_rebuild_timestamp"] = int(time.time())
                logger.info("Manual RAG rebuild triggered")
    except Exception as exc:
        logger.error("Manual RAG rebuild failed: %s", exc)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "feedback-collector",
        "version": "1.0.0",
        "feedback_received": _metrics["feedback_received_total"],
        "rag_rebuilds": _metrics["rag_rebuilds_total"],
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    uptime = int(time.time() - _metrics["start_time"])
    lines = [
        f'edms_feedback_total {_metrics["feedback_received_total"]}',
        f'edms_feedback_positive_total {_metrics["feedback_positive"]}',
        f'edms_feedback_negative_total {_metrics["feedback_negative"]}',
        f'edms_rag_rebuilds_total {_metrics["rag_rebuilds_total"]}',
        f"edms_feedback_uptime_seconds {uptime}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    uvicorn.run("feedback_api:app", host="0.0.0.0", port=API_PORT, log_level="info")
