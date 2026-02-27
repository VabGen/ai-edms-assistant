# src/ai_edms_assistant/interfaces/api/routes/agent_routes.py
"""Agent API routes: chat, new-chat, history, summarize, file-upload."""

from __future__ import annotations

import re
import uuid
import aiofiles
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from langchain_core.messages import AIMessage, HumanMessage

from ..dependencies import AgentDep, UPLOAD_DIR, get_http_client
from ..schemas.agent_schema import (
    ChatRequest,
    ChatResponse,
    FileUploadResponse,
    NewChatRequest,
    NewChatResponse,
    SummarizeRequest,
)
from ....application.dto import AgentRequest
from ....infrastructure.edms_api.http_client import EdmsHttpClient
from ....infrastructure.edms_api.repositories.edms_employee_repository import (
    EdmsEmployeeRepository,
)
from ....shared.security.auth import extract_user_id_from_token

log = structlog.get_logger(__name__)

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

router = APIRouter(tags=["agent"])


def _cleanup_file(file_path: str) -> None:
    """Safely remove a temporary file (BackgroundTask).

    Args:
        file_path: Absolute path to the temporary file to remove.
    """
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
    except Exception:
        pass  # Never crash on cleanup


async def _resolve_user_context(
    user_id: str,
    token: str,
    http_client: EdmsHttpClient,
) -> dict:
    """Resolve user context from EDMS Employee API.

    Uses the shared http_client from DI — no new client instantiation.
    Graceful degradation: returns {"firstName": "Коллега"} on any error.

    Args:
        user_id: Employee UUID extracted from JWT.
        token: JWT bearer token for API auth.
        http_client: Shared EdmsHttpClient from DI container.

    Returns:
        Dict with employee fields (firstName, lastName, etc.) or fallback.
    """
    try:
        from uuid import UUID

        emp_uuid = UUID(user_id)
        repo = EdmsEmployeeRepository(http_client=http_client)
        employee = await repo.get_by_id(entity_id=emp_uuid, token=token)
        if employee:
            return {
                "firstName": employee.first_name,
                "lastName": employee.last_name,
                "middleName": employee.middle_name,
                "departmentName": employee.department_name,
                "postName": employee.post_name,
            }
    except Exception as exc:
        log.warning(
            "employee_context_fetch_failed",
            user_id=user_id,
            error=str(exc),
        )
    return {"firstName": "Коллега"}


@router.post("/chat", response_model=ChatResponse, summary="Отправить сообщение агенту")
async def chat(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    agent: AgentDep,
    http_client: Annotated[EdmsHttpClient, Depends(get_http_client)],
) -> ChatResponse:
    """Process user message via EdmsDocumentAgent.

    Pipeline:
        1. Extract user_id from JWT
        2. Resolve user_context (body.context → Employee API → fallback)
        3. Build AgentRequest and call agent.chat()
        4. Schedule temp-file cleanup (BackgroundTask)

    Args:
        body: Validated ChatRequest.
        background_tasks: FastAPI background task manager.
        agent: Injected EdmsDocumentAgent.
        http_client: Shared EdmsHttpClient (injected, not created here).

    Returns:
        ChatResponse with status, response text, thread_id.
    """
    user_id = extract_user_id_from_token(body.user_token)
    thread_id = (
        body.thread_id or f"user_{user_id}_doc_{body.context_ui_id or 'general'}"
    )

    # ── Resolve user_context ──────────────────────────────────────────────────
    user_context: dict = {}

    if body.context:
        try:
            user_context = body.context.model_dump(exclude_none=True)
        except AttributeError:
            user_context = (
                dict(body.context) if hasattr(body.context, "__iter__") else {}
            )

    if not user_context:
        user_context = await _resolve_user_context(
            user_id, body.user_token, http_client
        )

    # ── Call agent ────────────────────────────────────────────────────────────
    agent_request = AgentRequest(
        message=body.message,
        user_token=body.user_token,
        context_ui_id=body.context_ui_id,
        thread_id=thread_id,
        user_context=user_context,
        file_path=body.file_path,
        human_choice=body.human_choice,
    )

    result = await agent.chat(agent_request)

    # ── Schedule file cleanup ─────────────────────────────────────────────────
    if body.file_path and not UUID_RE.match(str(body.file_path)):
        if result.status not in ("requires_action", "requires_choice"):
            background_tasks.add_task(_cleanup_file, body.file_path)

    # ── choice_thread_id для human-in-the-loop ────────────────────────────────
    choice_thread_id: str | None = None
    if result.metadata:
        choice_thread_id = result.metadata.get("choice_thread_id")

    return ChatResponse(
        status=result.status,
        response=result.content,
        action_type=result.action_type,
        message=result.message,
        thread_id=thread_id,
        choice_thread_id=choice_thread_id,
        options=result.metadata.get("options") if result.metadata else None,
    )


@router.post(
    "/chat/new",
    response_model=NewChatResponse,
    summary="Создать новый чат-тред",
)
async def new_chat(body: NewChatRequest) -> NewChatResponse:
    """Generate a fresh thread_id for a new conversation session.

    Args:
        body: NewChatRequest with user_token.

    Returns:
        NewChatResponse with new thread_id.

    Raises:
        HTTPException 401: On invalid JWT.
    """
    try:
        user_id = extract_user_id_from_token(body.user_token)
        return NewChatResponse(
            status="success",
            thread_id=f"chat_{user_id}_{uuid.uuid4().hex[:6]}",
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Неверный токен")


@router.get("/chat/history/{thread_id}", summary="История сообщений треда")
async def chat_history(thread_id: str, agent: AgentDep) -> dict:
    """Return human/AI messages for a conversation thread.

    Filters:
        - ToolMessage (внутренние вызовы инструментов)
        - AIMessage без content
        - AIMessage с tool_calls (planning steps)

    Args:
        thread_id: LangGraph thread identifier.
        agent: Injected EdmsDocumentAgent.

    Returns:
        Dict with ``messages`` list of ``{type, content}`` objects.
    """
    try:
        state = await agent.state_manager.get_state(thread_id)
        messages = state.values.get("messages", [])

        filtered = []
        for m in messages:
            if not isinstance(m, (HumanMessage, AIMessage)):
                continue
            if isinstance(m, AIMessage) and not m.content:
                continue
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                continue
            content = m.content
            if not content:
                continue
            filtered.append(
                {
                    "type": "human" if isinstance(m, HumanMessage) else "ai",
                    "content": str(content),
                }
            )

        return {"messages": filtered}

    except Exception as exc:
        log.warning("chat_history_failed", thread_id=thread_id, error=str(exc))
        return {"messages": []}


@router.post(
    "/actions/summarize",
    response_model=ChatResponse,
    summary="Суммаризация документа / вложения",
)
async def summarize(
    body: SummarizeRequest,
    background_tasks: BackgroundTasks,
    agent: AgentDep,
) -> ChatResponse:
    """Direct summarize action.

    Uses a fresh thread_id to avoid polluting the main conversation history.

    Args:
        body: SummarizeRequest with file_path / context_ui_id.
        background_tasks: FastAPI background task manager.
        agent: Injected EdmsDocumentAgent.

    Returns:
        ChatResponse with summary or requires_action for format selection.
    """
    user_id = extract_user_id_from_token(body.user_token)
    thread_id = f"action_{user_id}_{uuid.uuid4().hex[:6]}"

    user_message = (
        body.message or "Проанализируй вложение документа и дай краткую сводку"
    )

    agent_request = AgentRequest(
        message=user_message,
        user_token=body.user_token,
        context_ui_id=body.context_ui_id,
        thread_id=thread_id,
        file_path=body.file_path,
        human_choice=body.human_choice,
    )

    result = await agent.chat(agent_request)

    if body.file_path and not UUID_RE.match(str(body.file_path)):
        if result.status not in ("requires_action", "requires_choice"):
            background_tasks.add_task(_cleanup_file, body.file_path)

    choice_thread_id: str | None = None
    if result.metadata:
        choice_thread_id = result.metadata.get("choice_thread_id")

    return ChatResponse(
        status=result.status,
        response=result.content or result.message or "Анализ готов.",
        action_type=result.action_type,
        message=result.message,
        thread_id=thread_id,
        choice_thread_id=choice_thread_id,
        options=result.metadata.get("options") if result.metadata else None,
    )


@router.post(
    "/upload-file",
    response_model=FileUploadResponse,
    summary="Загрузить файл для анализа",
)
async def upload_file(
    user_token: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> FileUploadResponse:
    """Upload a temporary file for subsequent chat/summarize use.

    Args:
        user_token: JWT token (Form field).
        file: Uploaded file (multipart/form-data).

    Returns:
        FileUploadResponse with file_path and file_name.

    Raises:
        HTTPException 500: On file save failure.
    """
    try:
        user_id = extract_user_id_from_token(user_token)
        original_name = file.filename or "upload"
        suffix = Path(original_name).suffix or ".tmp"
        file_id = f"{user_id}_{uuid.uuid4().hex}{suffix}"

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = UPLOAD_DIR / file_id

        async with aiofiles.open(dest_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)

        log.debug("file_uploaded", file_id=file_id, original=original_name)
        return FileUploadResponse(
            file_path=str(dest_path),
            file_name=original_name,
        )

    except Exception as exc:
        log.error("upload_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки файла: {exc}")
