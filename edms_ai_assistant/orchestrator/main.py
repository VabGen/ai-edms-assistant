# edms_ai_assistant/orchestrator/main.py
"""
EDMS AI Assistant — главный FastAPI сервис оркестратора.

ИСПРАВЛЕНИЯ (архитектурное согласование):
  1. Импорты rag_module / memory / mcp_client теперь с полным пакетным путём
  2. Вызов агента через agent.process() (новый API) — убран несуществующий agent.chat()
  3. Добавлен endpoint POST /appeal/autofill (Chrome extension ожидает его)
  4. health_check() заменён на agent.get_health()
  5. Убран мёртвый импорт get_mcp_client (не используется в новом main)
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

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
from sqlalchemy import select
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse

from edms_ai_assistant.config import settings

# ── БД ────────────────────────────────────────────────────────────────────────
from edms_ai_assistant.orchestrator.db.database import AsyncSessionLocal, SummarizationCache

# ── Инфраструктура ────────────────────────────────────────────────────────────
from edms_ai_assistant.infrastructure.redis_client import close_redis, init_redis

# ── Клиенты (через __init__.py пакета) ───────────────────────────────────────
from edms_ai_assistant.orchestrator.clients.document_client import DocumentClient
from edms_ai_assistant.orchestrator.clients.employee_client import EmployeeClient

# ── Агент (ИСПРАВЛЕНО: используем единственный AgentOrchestrator) ─────────────
from edms_ai_assistant.orchestrator.agent import (
    AgentOrchestrator,
    AgentOrchestratorConfig,
    AgentResponse,
)

# ── API-роуты ─────────────────────────────────────────────────────────────────
from edms_ai_assistant.orchestrator.api.routes.cache import router as cache_router
from edms_ai_assistant.orchestrator.api.routes.settings import router as settings_router

# ── Pydantic-модели запросов/ответов ──────────────────────────────────────────
from edms_ai_assistant.orchestrator.model import (
    AssistantResponse,
    FileUploadResponse,
    NewChatRequest,
    UserInput,
)

# ── Утилиты ───────────────────────────────────────────────────────────────────
from edms_ai_assistant.orchestrator.security import extract_user_id_from_token
from edms_ai_assistant.shared.utils.utils import UUID_RE, get_file_hash

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

# ── Метрики (in-process, до Prometheus) ───────────────────────────────────────
_metrics: dict[str, Any] = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "latency_sum_ms": 0,
    "tool_calls_total": 0,
    "feedback_positive": 0,
    "feedback_negative": 0,
    "start_time": time.time(),
}

# ── Глобальные синглтоны (инициализируются в lifespan) ────────────────────────
_agent: AgentOrchestrator | None = None


def get_agent() -> AgentOrchestrator:
    """FastAPI dependency: возвращает инициализированный агент.

    Raises:
        HTTPException 503: Агент не инициализирован.
    """
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="ИИ-Агент не инициализирован. Повторите попытку позже.",
        )
    return _agent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Управление жизненным циклом приложения (инициализация / shutdown)."""
    global _agent

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Инициализация Redis...")
    await init_redis()

    logger.info("Инициализация агента...")
    try:
        cfg = AgentOrchestratorConfig()
        _agent = AgentOrchestrator(config=cfg)
        await _agent.initialize()
        health = await _agent.health_check()
        logger.info("EDMS AI Assistant запущен", extra={"health": health})
    except Exception:
        logger.critical("Инициализация агента провалилась", exc_info=True)

    yield

    if _agent:
        await _agent.close()
    await close_redis()
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(
    title="EDMS AI Assistant API",
    version="2.3.0",
    description=(
        "AI-powered assistant for EDMS document management workflows.\n\n"
        "## Архитектура\n"
        "- **AgentOrchestrator** — единый агент с ReAct-циклом и Plan+Execute\n"
        "- **MCP-сервер** — все инструменты для работы с EDMS API\n"
        "- **RAG** — few-shot примеры из успешных диалогов\n"
        "- **NLU** — предобработка + fast-path без LLM\n"
    ),
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


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _is_system_attachment(file_path: str | None) -> bool:
    """Проверяет, является ли путь UUID вложения EDMS (не локальный файл)."""
    return bool(file_path and UUID_RE.match(str(file_path)))


def _cleanup_file(file_path: str) -> None:
    """Удаляет временный файл. Вызывается как background task."""
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
    except Exception as exc:
        logger.warning("Не удалось удалить временный файл: %s — %s", file_path, exc)


async def _resolve_user_context(user_input: UserInput, user_id: str) -> dict:
    """Получает контекст сотрудника из EDMS API или берёт из запроса."""
    if user_input.context:
        return user_input.context.model_dump(exclude_none=True)
    try:
        async with EmployeeClient() as emp_client:
            ctx = await emp_client.get_employee(user_input.user_token, user_id)
            if ctx:
                return ctx
    except Exception as exc:
        logger.warning("Не удалось получить контекст сотрудника %s: %s", user_id, exc)
    return {"firstName": "Коллега"}


async def _update_rag_on_feedback(dialog_id: str, agent: AgentOrchestrator) -> None:
    """Фоновое обновление RAG после положительной оценки."""
    try:
        await agent.update_rag_from_feedback(dialog_id)
    except Exception as exc:
        logger.error("Ошибка обновления RAG для диалога %s: %s", dialog_id, exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=AssistantResponse,
    summary="Отправить сообщение ИИ-ассистенту",
    tags=["Chat"],
)
async def chat_endpoint(
    user_input: UserInput,
    background_tasks: BackgroundTasks,
    agent: Annotated[AgentOrchestrator, Depends(get_agent)],
) -> AssistantResponse:
    """Основной endpoint чата.

    Принимает сообщение пользователя, прогоняет через ReAct-цикл агента,
    возвращает ответ с метаданными (intent, model_used, dialog_id).
    """
    _metrics["requests_total"] += 1
    start = time.time()

    user_id = extract_user_id_from_token(user_input.user_token)
    thread_id = (
        user_input.thread_id
        or f"user_{user_id}_doc_{user_input.context_ui_id or 'general'}"
    )

    user_context = await _resolve_user_context(user_input, user_id)
    context_data = user_context.copy()
    if user_input.file_path:
        context_data["file_path"] = user_input.file_path
    if user_input.file_name:
        context_data["file_name"] = user_input.file_name
    if user_input.context_ui_id:
        context_data["document_id"] = user_input.context_ui_id

    try:
        # ИСПРАВЛЕНО: agent.process() — правильный метод нового AgentOrchestrator
        agent_response: AgentResponse = await agent.process(
            user_message=user_input.message,
            user_id=user_id,
            session_id=thread_id,
            token=user_input.user_token,
            context=context_data,
        )

        latency = int((time.time() - start) * 1000)
        _metrics["requests_success"] += 1
        _metrics["latency_sum_ms"] += latency
        _metrics["tool_calls_total"] += len(agent_response.tools_used)

        if user_input.file_path and not _is_system_attachment(user_input.file_path):
            background_tasks.add_task(_cleanup_file, user_input.file_path)

        return AssistantResponse(
            status="success",
            response=agent_response.content,
            thread_id=agent_response.session_id,
            action_type=agent_response.metadata.get("action_type"),
            requires_reload=agent_response.metadata.get("requires_reload", False),
            navigate_url=agent_response.metadata.get("navigate_url"),
            metadata={
                **agent_response.metadata,
                "intent": agent_response.intent,
                "model_used": agent_response.model_used,
                "dialog_id": agent_response.dialog_id,
            },
        )
    except Exception as exc:
        _metrics["requests_error"] += 1
        logger.error("Chat error для user=%s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера") from exc


@app.post(
    "/actions/summarize",
    response_model=AssistantResponse,
    summary="Суммаризация вложения документа",
    tags=["Actions"],
)
async def api_direct_summarize(
    user_input: UserInput,
    background_tasks: BackgroundTasks,
    agent: Annotated[AgentOrchestrator, Depends(get_agent)],
) -> AssistantResponse:
    """Суммаризация вложения с кэшированием результата в PostgreSQL.

    Порядок обработки:
    1. Определить file_identifier (UUID или hash файла)
    2. Проверить кэш (SummarizationCache)
    3. Если не в кэше — вызвать агент с инструкцией суммаризации
    4. Сохранить результат в кэш
    """
    _metrics["requests_total"] += 1
    start = time.time()

    current_path = (user_input.file_path or "").strip()
    is_uuid = _is_system_attachment(current_path)

    try:
        user_id = extract_user_id_from_token(user_input.user_token)
        new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:8]}"
        summary_type = user_input.human_choice or "extractive"
        file_identifier: str | None = None

        # ── Определяем идентификатор файла ───────────────────────────────────
        if is_uuid:
            file_identifier = current_path
        elif current_path and Path(current_path).exists():
            file_identifier = get_file_hash(current_path)
        elif user_input.context_ui_id:
            # Пытаемся найти вложение по имени в документе EDMS
            try:
                async with DocumentClient() as doc_client:
                    doc_data = await doc_client.get_document_metadata(
                        user_input.user_token, user_input.context_ui_id
                    )
                    attachments: list = (
                        doc_data.get("attachmentDocument", []) if isinstance(doc_data, dict) else []
                    )
                    if attachments:
                        def _normalize(s: str) -> str:
                            return re.sub(r"[^a-zA-Zа-яА-Я0-9]", "", s.lower()) if s else ""

                        clean_input = _normalize(current_path)
                        if clean_input:
                            for att in attachments:
                                att_name = att.get("name", "") if isinstance(att, dict) else ""
                                att_id = str(att.get("id", "") if isinstance(att, dict) else "")
                                if clean_input in _normalize(att_name):
                                    file_identifier = att_id
                                    break

                        if not file_identifier and attachments:
                            first = attachments[0]
                            file_identifier = str(first.get("id", "") if isinstance(first, dict) else "")

                        if file_identifier:
                            current_path = file_identifier
                            is_uuid = True
            except Exception as exc:
                logger.error("Ошибка резолва вложений EDMS: %s", exc)

        # ── Проверяем кэш ─────────────────────────────────────────────────────
        if file_identifier:
            try:
                async with AsyncSessionLocal() as db:
                    stmt = select(SummarizationCache).where(
                        SummarizationCache.file_identifier == str(file_identifier),
                        SummarizationCache.summary_type == summary_type,
                    )
                    cached_row = (await db.execute(stmt)).scalar_one_or_none()
                    if cached_row:
                        return AssistantResponse(
                            status="success",
                            response=cached_row.content,
                            thread_id=new_thread_id,
                            metadata={
                                "cache_file_identifier": file_identifier,
                                "cache_summary_type": summary_type,
                                "from_cache": True,
                            },
                        )
            except Exception as db_err:
                logger.error("Ошибка чтения кэша суммаризации: %s", db_err)

        # ── Вызываем агент ────────────────────────────────────────────────────
        _type_labels = {
            "extractive": "ключевые факты, даты и суммы",
            "abstractive": "краткое изложение своими словами",
            "thesis": "структурированный тезисный план",
        }
        type_label = _type_labels.get(summary_type, summary_type)
        instruction = f"Сделай {type_label} для файла/документа."
        if current_path:
            instruction += f" Идентификатор файла: {current_path}."

        agent_response = await agent.process(
            user_message=instruction,
            user_id=user_id,
            session_id=new_thread_id,
            token=user_input.user_token,
            context={"document_id": user_input.context_ui_id, "file_path": current_path},
        )
        response_text = agent_response.content

        _metrics["requests_success"] += 1
        _metrics["latency_sum_ms"] += int((time.time() - start) * 1000)

        # ── Сохраняем в кэш ───────────────────────────────────────────────────
        if file_identifier and response_text and response_text.strip():
            try:
                async with AsyncSessionLocal() as db, db.begin():
                    db.add(SummarizationCache(
                        id=str(uuid.uuid4()),
                        file_identifier=str(file_identifier),
                        summary_type=summary_type,
                        content=response_text,
                    ))
            except Exception as db_exc:
                logger.error("Ошибка записи кэша суммаризации: %s", db_exc)

        if current_path and not is_uuid:
            background_tasks.add_task(_cleanup_file, current_path)

        return AssistantResponse(
            status="success",
            response=response_text or "Анализ завершён.",
            thread_id=new_thread_id,
            metadata={
                "cache_file_identifier": file_identifier,
                "cache_summary_type": summary_type,
                "from_cache": False,
            },
        )

    except Exception as exc:
        _metrics["requests_error"] += 1
        logger.error("Ошибка /actions/summarize: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка суммаризации") from exc


@app.post(
    "/appeal/autofill",
    response_model=AssistantResponse,
    summary="Автозаполнение карточки обращения гражданина",
    tags=["Actions"],
)
async def appeal_autofill(
    user_input: UserInput,
    agent: Annotated[AgentOrchestrator, Depends(get_agent)],
) -> AssistantResponse:
    """Автозаполнение полей APPEAL-документа через LLM-анализ вложения.

    Endpoint ожидается Chrome-расширением (background.ts → case 'autofillAppeal').
    Делегирует выполнение агенту с инструментом autofill_appeal_document.
    """
    try:
        user_id = extract_user_id_from_token(user_input.user_token)
        thread_id = f"autofill_{user_id}_{uuid.uuid4().hex[:8]}"

        context: dict[str, Any] = {
            "document_id": user_input.context_ui_id,
            "file_path": user_input.file_path,
        }

        instruction = (
            f"Автозаполни карточку обращения гражданина. "
            f"ID документа: {user_input.context_ui_id or 'из контекста'}. "
            f"{'ID вложения: ' + user_input.file_path if user_input.file_path else ''}"
        ).strip()

        agent_response = await agent.process(
            user_message=instruction,
            user_id=user_id,
            session_id=thread_id,
            token=user_input.user_token,
            context=context,
        )

        return AssistantResponse(
            status="success",
            response=agent_response.content,
            thread_id=thread_id,
            requires_reload=agent_response.metadata.get("requires_reload", True),
            metadata=agent_response.metadata,
        )
    except Exception as exc:
        logger.error("Ошибка /appeal/autofill: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка автозаполнения") from exc


@app.post(
    "/feedback",
    summary="Сохранить оценку диалога",
    tags=["Feedback"],
)
async def submit_feedback(
    dialog_id: str,
    rating: int,
    agent: Annotated[AgentOrchestrator, Depends(get_agent)],
    comment: str = "",
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict:
    """Принимает оценку (-1/0/1) и обновляет RAG при положительной."""
    if rating == 1:
        _metrics["feedback_positive"] += 1
    elif rating == -1:
        _metrics["feedback_negative"] += 1

    ok = await agent.save_feedback(dialog_id, rating, comment)

    if ok and rating == 1:
        background_tasks.add_task(_update_rag_on_feedback, dialog_id, agent)

    label = {1: "положительная", 0: "нейтральная", -1: "отрицательная"}.get(rating, "")
    return {
        "success": ok,
        "message": f"Оценка '{label}' {'сохранена' if ok else 'не удалось сохранить'}",
    }


@app.get(
    "/chat/history/{thread_id}",
    summary="Получить историю треда",
    tags=["Chat"],
)
async def get_history(
    thread_id: str,
    agent: Annotated[AgentOrchestrator, Depends(get_agent)],
) -> dict:
    """Возвращает историю сообщений треда в формате [{type, content}]."""
    try:
        messages = await agent.get_thread_history(thread_id)
        filtered = []
        for m in messages:
            role = m.get("role") or m.get("type", "")
            content = m.get("content", "")
            if role in ("human", "user"):
                filtered.append({"type": "human", "content": content})
            elif role in ("ai", "assistant"):
                filtered.append({"type": "ai", "content": content})
        return {"messages": filtered}
    except Exception as exc:
        logger.error("Ошибка истории треда %s: %s", thread_id, exc)
        return {"messages": []}


@app.post(
    "/chat/new",
    summary="Создать новый тред диалога",
    tags=["Chat"],
)
async def create_new_thread(request: NewChatRequest) -> dict:
    """Генерирует новый thread_id для чистого диалога."""
    try:
        user_id = extract_user_id_from_token(request.user_token)
        new_thread_id = f"chat_{user_id}_{uuid.uuid4().hex[:8]}"
        return {"status": "success", "thread_id": new_thread_id}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Неверный токен") from exc


@app.post(
    "/upload-file",
    response_model=FileUploadResponse,
    summary="Загрузить файл для анализа",
    tags=["Files"],
)
async def upload_file(
    user_token: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> FileUploadResponse:
    """Сохраняет загруженный файл во временную директорию.

    Возвращает file_path, который затем передаётся в /chat или /actions/summarize.
    Файл автоматически удаляется как background task после обработки.
    """
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

        return FileUploadResponse(file_path=str(dest_path), file_name=file.filename)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ошибка загрузки файла: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла") from exc


@app.post("/rag/rebuild", tags=["RAG"])
async def rag_rebuild(
    background_tasks: BackgroundTasks,
    agent: Annotated[AgentOrchestrator, Depends(get_agent)],
) -> dict:
    """Запускает перестройку RAG-индекса из логов диалогов."""
    background_tasks.add_task(agent.rebuild_rag)
    return {"status": "started", "message": "Перестройка RAG запущена в фоне"}


@app.get(
    "/health",
    summary="Проверка состояния всех компонентов",
    tags=["System"],
)
async def health_check(
    agent: Annotated[AgentOrchestrator, Depends(get_agent)],
) -> dict:
    """Возвращает статус MCP, Redis, PostgreSQL, RAG."""
    # ИСПРАВЛЕНО: agent.health_check() — правильный метод
    health = await agent.health_check()
    return {"status": "ok", "version": app.version, "components": health}


@app.get("/metrics", response_class=PlainTextResponse, tags=["System"])
async def metrics() -> str:
    """Prometheus-метрики в text/plain формате."""
    uptime = int(time.time() - _metrics["start_time"])
    avg_latency = _metrics["latency_sum_ms"] / max(_metrics["requests_success"], 1)
    lines = [
        f'edms_requests_total {_metrics["requests_total"]}',
        f'edms_requests_success_total {_metrics["requests_success"]}',
        f'edms_requests_error_total {_metrics["requests_error"]}',
        f"edms_latency_avg_ms {avg_latency:.1f}",
        f'edms_tool_calls_total {_metrics["tool_calls_total"]}',
        f'edms_feedback_positive_total {_metrics["feedback_positive"]}',
        f'edms_feedback_negative_total {_metrics["feedback_negative"]}',
        f"edms_uptime_seconds {uptime}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.orchestrator.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOGGING_LEVEL.lower(),
    )