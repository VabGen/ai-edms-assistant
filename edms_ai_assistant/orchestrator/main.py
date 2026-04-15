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
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select
from starlette.middleware.cors import CORSMiddleware

from edms_ai_assistant.config import settings
from edms_ai_assistant.orchestrator.db.database import AsyncSessionLocal, SummarizationCache
from edms_ai_assistant.infrastructure.redis_client import close_redis, init_redis
from edms_ai_assistant.orchestrator.clients.document_client import DocumentClient
from edms_ai_assistant.orchestrator.clients.employee_client import EmployeeClient
from edms_ai_assistant.orchestrator.agent_orchestrator import EdmsAgentOrchestrator, OrchestratorConfig
from edms_ai_assistant.orchestrator.api.routes.cache import router as cache_router
from edms_ai_assistant.orchestrator.api.routes.settings import router as settings_router
from fastapi.responses import PlainTextResponse
from edms_ai_assistant.orchestrator.model import (
    AssistantResponse,
    FileUploadResponse,
    NewChatRequest,
    UserInput,
)
from edms_ai_assistant.orchestrator.security import extract_user_id_from_token
from edms_ai_assistant.shared.utils.utils import UUID_RE, get_file_hash

from rag_module import RAGModule
from memory import MemoryManager
from mcp_client import get_mcp_client

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

_metrics: dict[str, Any] = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "latency_sum_ms": 0,
    "tool_calls_total": 0,
    "tool_calls_error": 0,
    "feedback_positive": 0,
    "feedback_negative": 0,
    "start_time": time.time(),
}

_agent: EdmsAgentOrchestrator | None = None
_rag: RAGModule | None = None
_memory: MemoryManager | None = None


def get_agent() -> EdmsAgentOrchestrator:
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="ИИ-Агент не инициализирован. Повторите попытку позже.",
        )
    return _agent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _agent, _rag, _memory

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Инициализация Redis...")
    await init_redis()

    logger.info("Инициализация Memory и RAG модулей...")
    try:
        _memory = MemoryManager(
            redis_url=getattr(settings, 'REDIS_URL', "redis://localhost"),
            postgres_dsn=getattr(settings, 'POSTGRES_DSN', ""),
            max_context_tokens=getattr(settings, 'MAX_CONTEXT_TOKENS', 4000),
        )
        await _memory.initialize()

        _rag = RAGModule(
            postgres_dsn=getattr(settings, 'POSTGRES_DSN', ""),
            embedding_url=getattr(settings, 'LLM_EMBEDDING_URL', ""),
            embedding_model=getattr(settings, 'LLM_EMBEDDING_MODEL', ""),
        )
        await _rag.initialize()
    except Exception as e:
        logger.warning(f"Не удалось инициализировать RAG/Memory: {e}")

    logger.info("Инициализация агента...")
    try:
        _agent = EdmsAgentOrchestrator()
        await _agent.initialize()
        logger.info("EDMS AI Assistant запущен", extra={"health": _agent.health_check()})
    except Exception:
        logger.critical("Инициализация агента провалилась", exc_info=True)

    yield

    if _agent:
        await _agent.close()
    if _rag:
        await _rag.close()
    if _memory:
        await _memory.close()
    await close_redis()
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(
    title="EDMS AI Assistant API",
    version="2.2.0",
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
            logger.debug("Временный файл удалён", extra={"path": file_path})
    except Exception as exc:
        logger.warning("Не удалось удалить временный файл", extra={"path": file_path, "error": str(exc)})


async def _resolve_user_context(
        user_input: UserInput,
        user_id: str,
) -> dict:
    if user_input.context:
        return user_input.context.model_dump(exclude_none=True)

    try:
        async with EmployeeClient() as emp_client:
            ctx = await emp_client.get_employee(user_input.user_token, user_id)
            if ctx:
                return ctx
    except Exception as exc:
        logger.warning("Не удалось получить контекст сотрудника", extra={"user_id": user_id, "error": str(exc)})

    return {"firstName": "Коллега"}


async def _update_rag_from_feedback(dialog_id: str) -> None:
    if not _memory or not _rag:
        return
    try:
        dialogs = await _memory.long.get_positive_dialogs(limit=1)
        if dialogs:
            await _rag.rebuild_from_logs(dialogs)
    except Exception as exc:
        logger.error("RAG update from feedback failed: %s", exc)


async def _run_rag_rebuild() -> None:
    if not _memory or not _rag:
        return
    try:
        logs = await _memory.long.get_positive_dialogs(limit=500)
        added = await _rag.rebuild_from_logs(logs)
        logger.info("RAG rebuild complete: %d entries", added)
    except Exception as exc:
        logger.error("RAG rebuild failed: %s", exc)


@app.post(
    "/chat",
    response_model=AssistantResponse,
    summary="Отправить сообщение ИИ-ассистенту",
    tags=["Chat"],
)
async def chat_endpoint(
        user_input: UserInput,
        background_tasks: BackgroundTasks,
        agent: Annotated[EdmsAgentOrchestrator, Depends(get_agent)],
) -> AssistantResponse:
    _metrics["requests_total"] += 1
    start = time.time()

    user_id = extract_user_id_from_token(user_input.user_token)
    thread_id = (
            user_input.thread_id
            or f"user_{user_id}_doc_{user_input.context_ui_id or 'general'}"
    )

    user_context = await _resolve_user_context(user_input, user_id)

    if (
            user_input.preferred_summary_format
            and user_input.preferred_summary_format != "ask"
    ):
        user_context["preferred_summary_format"] = user_input.preferred_summary_format

    try:
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

        latency = int((time.time() - start) * 1000)
        _metrics["requests_success"] += 1
        _metrics["latency_sum_ms"] += latency
        _metrics["tool_calls_total"] += len(result.get("tool_calls_made", []))

        _FILE_OPERATION_KEYWORDS = (
            "сравни", "сравнение", "сравн", "compare", "отличи", "анализ",
            "проанализируй", "суммаризир", "прочит", "содержим", "прочти",
            "что в файл", "читай", "изучи",
        )
        _is_file_operation = any(
            kw in (user_input.message or "").lower() for kw in _FILE_OPERATION_KEYWORDS
        )
        _is_continuation = bool(user_input.human_choice)

        if user_input.file_path and not _is_system_attachment(user_input.file_path):
            _is_disambiguation = result.get("action_type") in (
                "requires_disambiguation", "summarize_selection",
            )
            _should_cleanup = (
                    result.get("status") not in ("requires_action",)
                    and not _is_file_operation
                    and not _is_continuation
                    and not _is_disambiguation
                    and result.get("requires_reload", False)
            )
            if _should_cleanup:
                background_tasks.add_task(_cleanup_file, user_input.file_path)

        final_response_text = result.get("content") or result.get("message")

        return AssistantResponse(
            status=result.get("status") or "success",
            response=final_response_text,
            action_type=result.get("action_type"),
            message=result.get("message"),
            thread_id=thread_id,
            requires_reload=result.get("requires_reload", False),
            navigate_url=result.get("navigate_url"),
            metadata=result.get("metadata", {}),
        )
    except Exception as exc:
        latency = int((time.time() - start) * 1000)
        _metrics["requests_error"] += 1
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/feedback",
    tags=["Feedback"],
)
async def submit_feedback(
        dialog_id: str,
        rating: int,
        comment: str = "",
        background_tasks: BackgroundTasks | None = None,
) -> dict:
    if rating == 1:
        _metrics["feedback_positive"] += 1
    elif rating == -1:
        _metrics["feedback_negative"] += 1

    ok = False
    if _memory:
        ok = await _memory.update_feedback(dialog_id, rating, comment)

    if ok and rating == 1 and background_tasks:
        background_tasks.add_task(_update_rag_from_feedback, dialog_id)

    label = {1: "положительная", 0: "нейтральная", -1: "отрицательная"}.get(rating, "")
    return {"success": ok, "message": f"Оценка '{label}' сохранена" if ok else "Не удалось сохранить оценку"}


@app.post(
    "/actions/summarize",
    response_model=AssistantResponse,
    summary="Суммаризация вложения",
    tags=["Actions"],
)
async def api_direct_summarize(
        user_input: UserInput,
        background_tasks: BackgroundTasks,
        agent: Annotated[EdmsAgentOrchestrator, Depends(get_agent)],
) -> AssistantResponse:
    _metrics["requests_total"] += 1
    start = time.time()

    current_path = (user_input.file_path or "").strip()
    is_uuid = _is_system_attachment(current_path)

    try:
        user_id = extract_user_id_from_token(user_input.user_token)
        new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:8]}"
        summary_type = user_input.human_choice or "extractive"
        file_identifier: str | None = None

        if is_uuid:
            file_identifier = current_path
        elif current_path and Path(current_path).exists():
            file_identifier = get_file_hash(current_path)
        elif user_input.context_ui_id:
            try:
                async with DocumentClient() as doc_client:
                    doc_dto = await doc_client.get_document_metadata(
                        user_input.user_token, user_input.context_ui_id
                    )

                    attachments: list = []
                    if hasattr(doc_dto, "attachmentDocument"):
                        attachments = doc_dto.attachmentDocument or []
                    elif isinstance(doc_dto, dict):
                        attachments = doc_dto.get("attachmentDocument") or []

                    if attachments:
                        def _normalize(s: str) -> str:
                            return (re.sub(r"[^a-zA-Zа-яА-Я0-9]", "", s.lower()) if s else "")

                        clean_input = _normalize(current_path)
                        if clean_input:
                            for att in attachments:
                                att_name = (att.get("name", "") if isinstance(att, dict) else getattr(att, "name",
                                                                                                      "")) or ""
                                att_id = str(att.get("id", "") if isinstance(att, dict) else getattr(att, "id", ""))
                                if clean_input in _normalize(att_name):
                                    file_identifier = att_id
                                    break

                        if not file_identifier and attachments:
                            first = attachments[0]
                            file_identifier = str(
                                first.get("id", "") if isinstance(first, dict) else getattr(first, "id", ""))

                        if file_identifier:
                            current_path = file_identifier
                            is_uuid = True
            except Exception as exc:
                logger.error("Ошибка резолва вложений EDMS: %s", exc)

        if file_identifier:
            try:
                async with AsyncSessionLocal() as db:
                    stmt = select(SummarizationCache).where(
                        SummarizationCache.file_identifier == str(file_identifier),
                        SummarizationCache.summary_type == summary_type,
                    )
                    result_row = await db.execute(stmt)
                    cached_row = result_row.scalar_one_or_none()

                    if cached_row:
                        return AssistantResponse(
                            status="success",
                            response=cached_row.content,
                            thread_id=new_thread_id,
                            metadata={
                                "cache_file_identifier": file_identifier,
                                "cache_summary_type": summary_type,
                                "cache_context_ui_id": user_input.context_ui_id,
                                "from_cache": True,
                            },
                        )
            except Exception as db_err:
                logger.error("Ошибка чтения кэша: %s", db_err)

        _type_labels = {
            "extractive": "ключевые факты, даты, суммы",
            "abstractive": "краткое из ложение своими словами",
            "thesis": "структурированный тезисный план",
        }
        type_label = _type_labels.get(summary_type, summary_type)
        user_context = await _resolve_user_context(user_input, user_id)

        instructions = f"Работай с вложением {current_path}. " if is_uuid else ""
        agent_msg = f"{instructions}Проанализируй этот файл и выдели {type_label}."

        agent_result = await agent.chat(
            message=agent_msg,
            user_token=user_input.user_token,
            context_ui_id=user_input.context_ui_id,
            thread_id=new_thread_id,
            user_context=user_context,
            file_path=current_path,
            human_choice=summary_type,
        )

        response_text = agent_result.get("content") or agent_result.get("response")

        latency = int((time.time() - start) * 1000)
        _metrics["requests_success"] += 1
        _metrics["latency_sum_ms"] += latency
        _metrics["tool_calls_total"] += len(agent_result.get("tool_calls_made", []))

        if file_identifier and response_text and response_text.strip():
            if agent_result.get("status") == "success":
                try:
                    async with AsyncSessionLocal() as db:
                        async with db.begin():
                            new_cache = SummarizationCache(
                                id=str(uuid.uuid4()),
                                file_identifier=str(file_identifier),
                                summary_type=summary_type,
                                content=response_text,
                            )
                            db.add(new_cache)
                except Exception as db_exc:
                    logger.error("Ошибка записи кэша: %s", db_exc)

        if current_path and not is_uuid:
            background_tasks.add_task(_cleanup_file, current_path)

        return AssistantResponse(
            status=agent_result.get("status", "success"),
            response=response_text or "Анализ завершён.",
            thread_id=new_thread_id,
            message=agent_result.get("message"),
            requires_reload=agent_result.get("requires_reload", False),
            metadata={
                **agent_result.get("metadata", {}),
                "cache_file_identifier": file_identifier,
                "cache_summary_type": summary_type,
                "cache_context_ui_id": user_input.context_ui_id,
                "from_cache": False,
            },
        )

    except Exception as exc:
        _metrics["requests_error"] += 1
        logger.error("Ошибка /actions/summarize: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/chat/history/{thread_id}",
    summary="Получить историю треда",
    tags=["Chat"],
)
async def get_history(
        thread_id: str,
        agent: Annotated[EdmsAgentOrchestrator, Depends(get_agent)],
) -> dict:
    try:
        state = await agent.state_manager.get_state(thread_id)
        messages = state.values.get("messages", []) if state else []

        filtered = []
        for m in messages:
            if not isinstance(m, (HumanMessage, AIMessage)):
                continue
            if isinstance(m, AIMessage) and not m.content:
                continue
            filtered.append(
                {
                    "type": "human" if isinstance(m, HumanMessage) else "ai",
                    "content": m.content,
                }
            )

        return {"messages": filtered}

    except Exception as exc:
        logger.error("Ошибка получения истории треда", extra={"thread_id": thread_id, "error": str(exc)})
        return {"messages": []}


@app.post(
    "/chat/new",
    summary="Создать новый тред диалога",
    tags=["Chat"],
)
async def create_new_thread(request: NewChatRequest) -> dict:
    try:
        user_id = extract_user_id_from_token(request.user_token)
        new_thread_id = f"chat_{user_id}_{uuid.uuid4().hex[:8]}"
        return {"status": "success", "thread_id": new_thread_id}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Неверный токен") from exc


@app.post(
    "/upload-file",
    response_model=FileUploadResponse,
    summary="Загрузить файл для анализа в чате",
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

        original_path = Path(file.filename or "file")
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

        logger.info("Файл загружен", extra={"orig_filename": file.filename, "dest": str(dest_path)})
        return FileUploadResponse(
            file_path=str(dest_path),
            file_name=file.filename,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ошибка загрузки файла", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла") from exc


@app.get("/rag/stats", tags=["RAG"])
async def rag_stats() -> dict:
    if not _rag:
        raise HTTPException(status_code=503, detail="RAG не инициализирован")
    return await _rag.get_stats()


@app.post("/rag/rebuild", tags=["RAG"])
async def rag_rebuild(background_tasks: BackgroundTasks) -> dict:
    if not _memory or not _rag:
        raise HTTPException(status_code=503, detail="Компоненты не инициализированы")
    background_tasks.add_task(_run_rag_rebuild)
    return {"status": "started", "message": "Перестройка RAG запущена в фоне"}


@app.get(
    "/health",
    summary="Проверка состояния агента и сервисов",
    tags=["System"],
)
async def health_check(
        agent: Annotated[EdmsAgentOrchestrator, Depends(get_agent)],
) -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "components": agent.health_check(),
        "rag": _rag is not None,
        "memory": _memory is not None,
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["System"])
async def metrics() -> str:
    uptime = int(time.time() - _metrics["start_time"])
    avg_latency = (
            _metrics["latency_sum_ms"] / max(_metrics["requests_success"], 1)
    )
    lines = [
        "# HELP edms_requests_total Всего запросов",
        "# TYPE edms_requests_total counter",
        f'edms_requests_total {_metrics["requests_total"]}',
        f'edms_requests_success_total {_metrics["requests_success"]}',
        f'edms_requests_error_total {_metrics["requests_error"]}',
        "# HELP edms_latency_avg_ms Среднее время ответа (мс)",
        "# TYPE edms_latency_avg_ms gauge",
        f"edms_latency_avg_ms {avg_latency:.1f}",
        "# HELP edms_tool_calls_total Вызовы MCP-инструментов",
        "# TYPE edms_tool_calls_total counter",
        f'edms_tool_calls_total {_metrics["tool_calls_total"]}',
        f'edms_tool_calls_error_total {_metrics["tool_calls_error"]}',
        "# HELP edms_feedback Оценки пользователей",
        "# TYPE edms_feedback counter",
        f'edms_feedback_positive_total {_metrics["feedback_positive"]}',
        f'edms_feedback_negative_total {_metrics["feedback_negative"]}',
        "# HELP edms_uptime_seconds Время работы сервиса",
        "# TYPE edms_uptime_seconds gauge",
        f"edms_uptime_seconds {uptime}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.orchestrator.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        reload_excludes=[".venv", "*.pyc", "__pycache__"],
        log_level=settings.LOGGING_LEVEL.lower(),
    )
