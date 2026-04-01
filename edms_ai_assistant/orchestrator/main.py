"""
EDMS AI Orchestrator — FastAPI приложение (полная версия).

Эндпоинты из старого проекта + новая архитектура MCP/ReAct:

Chat & Feedback:
  POST /chat                        — диалог с ассистентом
  POST /feedback                    — оценка ответа
  GET  /chat/history/{thread_id}    — история диалога
  POST /chat/new                    — новый тред

File operations:
  POST /upload-file                 — загрузка файла для анализа

Summarization (direct action):
  POST /actions/summarize           — прямая суммаризация вложения

RAG:
  GET  /rag/stats                   — статистика RAG
  POST /rag/rebuild                 — пересборка RAG

System:
  GET  /health                      — состояние компонентов
  GET  /metrics                     — Prometheus метрики

API routes (mounted):
  /api/settings/*                   — runtime настройки агента
  /api/cache/*                      — управление кэшем суммаризации
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import aiofiles
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy import select

from agent_orchestrator import EdmsAgentOrchestrator, OrchestratorConfig
from api.routes.cache import router as cache_router
from api.routes.settings import router as settings_router
from config import settings
from db.database import AsyncSessionLocal, SummarizationCache, init_db_schema
from db_init import init_db
from models.api_models import (
    ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse,
    FileUploadResponse, NewChatRequest,
)
from security import extract_user_id_from_token, get_user_id_safe
from utils.hash_utils import get_file_hash
from utils.regex_utils import UUID_RE

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

# ── Upload dir ────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_uploads"

# ── Prometheus counters ───────────────────────────────────────────────────────
_metrics: dict[str, float] = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "fast_path_total": 0,
    "avg_latency_ms": 0,
    "feedback_positive": 0,
    "feedback_negative": 0,
}


def _update_metrics(latency_ms: int, success: bool, fast_path: bool = False) -> None:
    _metrics["requests_total"] += 1
    if success:
        _metrics["requests_success"] += 1
    else:
        _metrics["requests_error"] += 1
    if fast_path:
        _metrics["fast_path_total"] += 1
    prev = _metrics["avg_latency_ms"]
    n = _metrics["requests_total"]
    _metrics["avg_latency_ms"] = prev + (latency_ms - prev) / n


# ── App lifecycle ─────────────────────────────────────────────────────────────
_orchestrator: EdmsAgentOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Init asyncpg schema (pgvector tables)
    try:
        await init_db(settings.DATABASE_URL_ASYNCPG)
        logger.info("asyncpg schema initialized")
    except Exception as exc:
        logger.error("asyncpg init failed (continuing): %s", exc)

    # 2. Init SQLAlchemy schema (SummarizationCache)
    try:
        await init_db_schema()
        logger.info("SQLAlchemy schema initialized")
    except Exception as exc:
        logger.error("SQLAlchemy init failed (continuing): %s", exc)

    # 3. Init Redis (for DocumentService)
    from services.document_service import init_redis
    try:
        await init_redis()
    except Exception as exc:
        logger.warning("Redis init failed: %s", exc)

    # 4. Init orchestrator
    try:
        config = OrchestratorConfig()
        _orchestrator = EdmsAgentOrchestrator(
            config=config,
            prompts_dir=settings.PROMPTS_DIR,
        )
        await _orchestrator.init()
        logger.info("Orchestrator ready")
    except Exception as exc:
        logger.critical("Orchestrator init FAILED — /chat will return 503: %s", exc)

    yield

    if _orchestrator:
        await _orchestrator.close()

    from services.document_service import close_redis
    await close_redis()

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        logger.info("Upload dir cleaned up")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EDMS AI Assistant API",
    version="2.0.0",
    description="AI-powered assistant for EDMS document management",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if isinstance(settings.ALLOWED_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings_router)
app.include_router(cache_router)


# ── Dependency ────────────────────────────────────────────────────────────────
def get_orchestrator() -> EdmsAgentOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="ИИ-агент не инициализирован")
    return _orchestrator


# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_uuid(s: str | None) -> bool:
    return bool(s and UUID_RE.match(str(s).strip()))


def _cleanup_file(path: str) -> None:
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception as exc:
        logger.warning("File cleanup failed: %s — %s", path, exc)


async def _resolve_user_context(token: str, user_id: str) -> dict:
    """Попытка загрузить профиль из EDMS, fallback на минимум."""
    try:
        from clients.employee_client import EmployeeClient
        async with EmployeeClient() as emp:
            data = await emp.get_current_user(token)
            if data:
                return data
    except Exception:
        pass
    return {"firstName": "Коллега", "id": user_id}


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    user_input: ChatRequest,
    background_tasks: BackgroundTasks,
    orch: Annotated[EdmsAgentOrchestrator, Depends(get_orchestrator)],
) -> ChatResponse:
    """Основной эндпоинт диалога с ассистентом."""
    user_id = get_user_id_safe(user_input.token, default=user_input.user_id)
    session_id = user_input.session_id or f"user_{user_id}_doc_{user_input.context.get('document_id', 'general')}"

    user_ctx = await _resolve_user_context(user_input.token, user_id)

    start = time.monotonic()
    try:
        result = await orch.process(
            user_message=user_input.message,
            user_id=user_id,
            session_id=session_id,
            token=user_input.token,
            context={**user_input.context, "user_context": user_ctx},
        )
        _update_metrics(result.latency_ms, success=True,
                        fast_path=(result.model_used == "fast_path"))

        # Cleanup temp file if not a continuing disambiguation
        if user_input.context.get("file_path") and not _is_uuid(str(user_input.context.get("file_path"))):
            if result.model_used != "fast_path":
                background_tasks.add_task(_cleanup_file, str(user_input.context["file_path"]))

        return ChatResponse(
            status="success",
            content=result.content,
            response=result.content,
            session_id=result.session_id,
            dialog_id=result.dialog_id,
            intent=result.intent,
            tools_used=result.tools_used,
            model_used=result.model_used,
            latency_ms=result.latency_ms,
            requires_clarification=result.requires_clarification,
            clarification_question=result.clarification_question,
            requires_reload=result.metadata.get("requires_reload", False),
            navigate_url=result.metadata.get("navigate_url"),
            metadata=result.metadata,
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        _update_metrics(latency_ms, success=False)
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {exc!s}")


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(
    req: FeedbackRequest,
    orch: Annotated[EdmsAgentOrchestrator, Depends(get_orchestrator)],
) -> FeedbackResponse:
    """Оценка ответа ассистента."""
    try:
        ok = await orch.save_feedback(req.dialog_id, req.rating, req.comment)
        if req.rating == 1:
            _metrics["feedback_positive"] += 1
        elif req.rating == -1:
            _metrics["feedback_negative"] += 1
        return FeedbackResponse(
            success=ok,
            message="Спасибо за оценку!" if ok else "Диалог не найден",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/chat/history/{thread_id}", tags=["Chat"])
async def get_history(
    thread_id: str,
    orch: Annotated[EdmsAgentOrchestrator, Depends(get_orchestrator)],
) -> dict:
    """История диалога по thread_id."""
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        state = await orch.state_manager.get_state(thread_id)
        messages = state.values.get("messages", [])
        filtered = []
        for m in messages:
            if not isinstance(m, (HumanMessage, AIMessage)):
                continue
            if isinstance(m, AIMessage) and not m.content:
                continue
            filtered.append({
                "type": "human" if isinstance(m, HumanMessage) else "ai",
                "content": m.content,
            })
        return {"messages": filtered}
    except Exception as exc:
        logger.error("History retrieval failed for thread %s: %s", thread_id, exc)
        return {"messages": []}


@app.post("/chat/new", tags=["Chat"])
async def create_new_thread(req: NewChatRequest) -> dict:
    """Создать новый тред диалога."""
    try:
        user_id = extract_user_id_from_token(req.user_token)
        thread_id = f"chat_{user_id}_{uuid.uuid4().hex[:8]}"
        return {"status": "success", "thread_id": thread_id}
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid token")


# ═══════════════════════════════════════════════════════════════════════════════
# FILE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/upload-file", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    user_token: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> FileUploadResponse:
    """Загрузить файл для анализа в чате."""
    try:
        extract_user_id_from_token(user_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Имя файла не указано")

    original = Path(file.filename)
    suffix = original.suffix.lower()
    if not suffix:
        ct = file.content_type or ""
        suffix = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/msword": ".doc",
            "text/plain": ".txt",
        }.get(ct, "")

    safe_stem = re.sub(r"[^\w\-.]", "_", original.stem[:80])
    safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")
    dest = UPLOAD_DIR / f"{safe_stem}{suffix}"

    try:
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                await out.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла: {exc}")

    logger.info("File uploaded: %s → %s", file.filename, dest)
    return FileUploadResponse(file_path=str(dest), file_name=file.filename)


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECT SUMMARIZATION ACTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/actions/summarize", response_model=ChatResponse, tags=["Actions"])
async def direct_summarize(
    user_input: ChatRequest,
    background_tasks: BackgroundTasks,
    orch: Annotated[EdmsAgentOrchestrator, Depends(get_orchestrator)],
) -> ChatResponse:
    """
    Прямая суммаризация вложения (кнопка ✦ на вложении в EDMS UI).

    Проверяет кэш суммаризации по file_identifier, при промахе —
    запускает агент и кэширует результат в SummarizationCache.
    """
    user_id = get_user_id_safe(user_input.token, default=user_input.user_id)
    new_thread = f"action_{user_id}_{uuid.uuid4().hex[:8]}"
    summary_type = user_input.human_choice or "extractive"

    file_path = str(user_input.context.get("file_path", "")).strip()
    doc_id = str(user_input.context.get("document_id", "")).strip() or None
    is_uuid_path = _is_uuid(file_path)

    # Determine cache key
    file_identifier: str | None = None
    if is_uuid_path:
        file_identifier = file_path
    elif file_path and Path(file_path).exists():
        try:
            file_identifier = get_file_hash(file_path)
        except Exception:
            file_identifier = None

    # ── Cache lookup ──────────────────────────────────────────────────────────
    if file_identifier:
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(SummarizationCache).where(
                    SummarizationCache.file_identifier == file_identifier,
                    SummarizationCache.summary_type == summary_type,
                )
                row = (await db.execute(stmt)).scalar_one_or_none()
                if row:
                    logger.info("Summarization cache HIT: %s / %s", file_identifier[:8], summary_type)
                    return ChatResponse(
                        status="success",
                        content=row.content,
                        response=row.content,
                        session_id=new_thread,
                        metadata={"cache_file_identifier": file_identifier,
                                  "cache_summary_type": summary_type, "from_cache": True},
                    )
        except Exception as exc:
            logger.warning("Cache lookup error: %s", exc)

    # ── Run agent ─────────────────────────────────────────────────────────────
    type_map = {
        "extractive": "ключевые факты, даты, суммы",
        "abstractive": "краткое изложение своими словами",
        "thesis": "структурированный тезисный план",
    }
    label = type_map.get(summary_type, summary_type)
    instr = f"Работай с вложением {file_path}. " if is_uuid_path else ""
    agent_msg = f"{instr}Проанализируй этот файл и выдели {label}."

    user_ctx = await _resolve_user_context(user_input.token, user_id)
    start = time.monotonic()
    try:
        result = await orch.process(
            user_message=agent_msg,
            user_id=user_id,
            session_id=new_thread,
            token=user_input.token,
            context={
                "file_path": file_path,
                "document_id": doc_id or "",
                "user_context": user_ctx,
                "human_choice": summary_type,
            },
        )
        _update_metrics(result.latency_ms, success=True)
    except Exception as exc:
        _update_metrics(int((time.monotonic() - start) * 1000), success=False)
        raise HTTPException(status_code=500, detail=str(exc))

    response_text = result.content or "Анализ завершён."

    # ── Cache save ────────────────────────────────────────────────────────────
    if file_identifier and response_text and result.model_used != "error":
        try:
            async with AsyncSessionLocal() as db, db.begin():
                db.add(SummarizationCache(
                    id=str(uuid.uuid4()),
                    file_identifier=file_identifier,
                    summary_type=summary_type,
                    content=response_text,
                ))
            logger.info("Summarization cached: %s / %s", file_identifier[:8], summary_type)
        except Exception as exc:
            logger.warning("Cache save error: %s", exc)

    # Cleanup local file
    if file_path and not is_uuid_path and Path(file_path).exists():
        background_tasks.add_task(_cleanup_file, file_path)

    return ChatResponse(
        status="success",
        content=response_text,
        response=response_text,
        session_id=new_thread,
        dialog_id=result.dialog_id,
        intent=result.intent,
        tools_used=result.tools_used,
        model_used=result.model_used,
        latency_ms=result.latency_ms,
        metadata={
            "cache_file_identifier": file_identifier,
            "cache_summary_type": summary_type,
            "from_cache": False,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RAG ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/rag/stats", tags=["RAG"])
async def rag_stats(
    orch: Annotated[EdmsAgentOrchestrator, Depends(get_orchestrator)],
) -> dict:
    """Статистика RAG индекса."""
    return await orch.rag.get_stats()


@app.post("/rag/rebuild", tags=["RAG"])
async def rag_rebuild(
    orch: Annotated[EdmsAgentOrchestrator, Depends(get_orchestrator)],
) -> dict:
    """Принудительное обновление RAG из логов диалогов."""
    try:
        positive = await orch.memory.long.get_positive_dialogs(limit=200)
        negative = await orch.memory.long.get_negative_dialogs(limit=100)
        stats = await orch.rag.rebuild_from_logs(positive, negative)
        return {"success": True, "stats": stats}
    except Exception as exc:
        logger.error("RAG rebuild error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health(
    orch: Annotated[EdmsAgentOrchestrator, Depends(get_orchestrator)],
) -> dict:
    """Проверка состояния всех компонентов."""
    try:
        return await orch.get_health()
    except HTTPException:
        return {"status": "not_initialized"}


@app.get("/metrics", response_class=PlainTextResponse, tags=["System"])
async def metrics() -> str:
    """Метрики Prometheus."""
    lines = [
        "# HELP edms_requests_total Total chat requests",
        "# TYPE edms_requests_total counter",
        f"edms_requests_total {int(_metrics['requests_total'])}",
        "",
        "# HELP edms_requests_success Successful requests",
        "# TYPE edms_requests_success counter",
        f"edms_requests_success {int(_metrics['requests_success'])}",
        "",
        "# HELP edms_requests_error Failed requests",
        "# TYPE edms_requests_error counter",
        f"edms_requests_error {int(_metrics['requests_error'])}",
        "",
        "# HELP edms_fast_path_total Requests without LLM",
        "# TYPE edms_fast_path_total counter",
        f"edms_fast_path_total {int(_metrics['fast_path_total'])}",
        "",
        "# HELP edms_avg_latency_ms Average latency ms",
        "# TYPE edms_avg_latency_ms gauge",
        f"edms_avg_latency_ms {_metrics['avg_latency_ms']:.2f}",
        "",
        "# HELP edms_feedback_positive Positive ratings",
        "# TYPE edms_feedback_positive counter",
        f"edms_feedback_positive {int(_metrics['feedback_positive'])}",
        "",
        "# HELP edms_feedback_negative Negative ratings",
        "# TYPE edms_feedback_negative counter",
        f"edms_feedback_negative {int(_metrics['feedback_negative'])}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
