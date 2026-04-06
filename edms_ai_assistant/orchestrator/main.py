# orchestrator/main.py
"""
EDMS AI Assistant — главный FastAPI сервис оркестратора.

Endpoints:
    POST /chat              — диалог с ИИ-ассистентом
    GET  /chat/history/{id} — история треда
    POST /chat/new          — создать новый тред
    POST /upload-file       — загрузить файл для анализа
    POST /actions/summarize — суммаризация вложения
    GET  /health            — статус компонентов
    GET  /metrics           — Prometheus метрики
    GET/PATCH/DELETE /api/settings — управление настройками
    GET/DELETE /api/cache   — управление кэшом суммаризаций
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import aiofiles
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from starlette.middleware.cors import CORSMiddleware

from edms_ai_assistant.orchestrator.agent import EdmsDocumentAgent
from edms_ai_assistant.orchestrator.api.routes.cache import router as cache_router
from edms_ai_assistant.orchestrator.api.routes.settings import router as settings_router
from edms_ai_assistant.orchestrator.config import settings
from edms_ai_assistant.orchestrator.db.database import init_db
from edms_ai_assistant.orchestrator.model import (
    AssistantResponse,
    FileUploadResponse,
    NewChatRequest,
    UserInput,
)
from edms_ai_assistant.orchestrator.security import extract_user_id_from_token
from edms_ai_assistant.orchestrator.services.document_service import close_redis, init_redis
from edms_ai_assistant.orchestrator.utils.regex_utils import UUID_RE

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

_agent: EdmsDocumentAgent | None = None


def get_agent() -> EdmsDocumentAgent:
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="ИИ-Агент не инициализирован. Повторите попытку позже.",
        )
    return _agent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _agent

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing database...")
    await init_db()

    logger.info("Initializing Redis...")
    await init_redis()

    logger.info("Initializing agent...")
    try:
        _agent = EdmsDocumentAgent()
        await _agent.initialize()
        logger.info(
            "EDMS AI Assistant started",
            extra={"health": _agent.health_check()},
        )
    except Exception:
        logger.critical(
            "Agent initialization failed — all /chat requests will return 503",
            exc_info=True,
        )

    yield

    # Graceful shutdown
    if _agent:
        await _agent.close()
    await close_redis()
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    logger.info("EDMS AI Assistant stopped")


app = FastAPI(
    title="EDMS AI Assistant API",
    version="2.0.0",
    description="AI-powered assistant for EDMS document management workflows.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings_router)
app.include_router(cache_router)


def _is_system_attachment(file_path: str | None) -> bool:
    return bool(file_path and UUID_RE.match(str(file_path)))


def _cleanup_file(file_path: str) -> None:
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
    except Exception as exc:
        logger.warning("Failed to remove temp file %s: %s", file_path, exc)


@app.post(
    "/chat",
    response_model=AssistantResponse,
    summary="Send a message to the EDMS AI assistant",
    tags=["Chat"],
)
async def chat_endpoint(
    user_input: UserInput,
    background_tasks: BackgroundTasks,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    user_id = extract_user_id_from_token(user_input.user_token)
    thread_id = (
        user_input.thread_id
        or f"user_{user_id}_doc_{user_input.context_ui_id or 'general'}"
    )

    # Получаем контекст пользователя
    user_context: dict = {}
    if user_input.context:
        user_context = user_input.context.model_dump(exclude_none=True)

    result = await agent.chat(
        message=user_input.message,
        user_token=user_input.user_token,
        context_ui_id=user_input.context_ui_id,
        thread_id=thread_id,
        user_context=user_context,
        file_path=user_input.file_path,
        file_name=user_input.file_name,
        human_choice=user_input.human_choice,
    )

    # Планируем очистку временного файла
    if (
        user_input.file_path
        and not _is_system_attachment(user_input.file_path)
        and result.get("status") == "success"
        and not result.get("requires_reload")
    ):
        background_tasks.add_task(_cleanup_file, user_input.file_path)

    return AssistantResponse(
        status=result.get("status", "success"),
        response=result.get("content"),
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
        requires_reload=result.get("requires_reload", False),
        navigate_url=result.get("navigate_url"),
        metadata=result.get("metadata", {}),
    )


@app.get(
    "/chat/history/{thread_id}",
    summary="Get conversation history for a thread",
    tags=["Chat"],
)
async def get_history(
    thread_id: str,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    try:
        history = await agent._load_thread_history(thread_id)
        return {"messages": history}
    except Exception as exc:
        logger.error("History retrieval failed for thread %s: %s", thread_id, exc)
        return {"messages": []}


@app.post(
    "/chat/new",
    summary="Create a new conversation thread",
    tags=["Chat"],
)
async def create_new_thread(request: NewChatRequest) -> dict:
    try:
        user_id = extract_user_id_from_token(request.user_token)
        new_thread_id = f"chat_{user_id}_{uuid.uuid4().hex[:8]}"
        return {"status": "success", "thread_id": new_thread_id}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


@app.post(
    "/upload-file",
    response_model=FileUploadResponse,
    summary="Upload a file for in-chat analysis",
    tags=["Files"],
)
async def upload_file(
    user_token: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> FileUploadResponse:
    try:
        extract_user_id_from_token(user_token)

        if not file.filename:
            raise HTTPException(status_code=400, detail="Имя файла не указано")

        original_path = Path(file.filename)
        suffix = original_path.suffix.lower()
        if not suffix:
            ct = file.content_type or ""
            suffix = {
                "application/pdf": ".pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                "application/msword": ".doc",
                "text/plain": ".txt",
            }.get(ct, "")

        safe_stem = re.sub(r"[^\w\-.]", "_", original_path.stem[:80])
        safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")
        dest_path = UPLOAD_DIR / f"{safe_stem}{suffix}"

        async with aiofiles.open(dest_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)

        logger.info("File uploaded: %s → %s", file.filename, dest_path)
        return FileUploadResponse(file_path=str(dest_path), file_name=file.filename)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File upload failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла") from exc


@app.get(
    "/health",
    summary="Agent and service health check",
    tags=["System"],
)
async def health_check(
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "components": agent.health_check(),
    }


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    tags=["System"],
)
async def metrics() -> dict:
    """Базовые метрики. В production подключить prometheus_client."""
    return {"status": "ok", "note": "Install prometheus_client for full metrics"}


if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.orchestrator.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOGGING_LEVEL.lower(),
    )
