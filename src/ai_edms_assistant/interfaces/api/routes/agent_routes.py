# src/ai_edms_assistant/interfaces/api/routes/agent_routes.py
"""
Agent API routes: chat, new-chat, history, summarize, file-upload.

Extracted from monolithic main.py. Each route handles only HTTP
translation — all business logic stays in EdmsDocumentAgent.
"""

from __future__ import annotations

import re
import uuid
import aiofiles
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from langchain_core.messages import HumanMessage, AIMessage

from ..dependencies import AgentDep, UPLOAD_DIR
from ..schemas.agent_schema import (
    ChatRequest,
    ChatResponse,
    NewChatRequest,
    NewChatResponse,
    FileUploadResponse,
    SummarizeRequest,
)
from ....application.dto import AgentRequest
from ....shared.security.auth import extract_user_id_from_token
from ....infrastructure.edms_api.clients.employee_client import EdmsEmployeeClient

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

router = APIRouter(tags=["agent"])


def _cleanup_file(file_path: str) -> None:
    """Safely remove a temp file. Used as a BackgroundTask."""
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
    except Exception:
        pass


@router.post("/chat", response_model=ChatResponse, summary="Отправить сообщение агенту")
async def chat(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    agent: AgentDep,
) -> ChatResponse:
    """
    - Extracts user_id from JWT
    - Resolves user context (from body or EdmsEmployeeClient)
    - Delegates to agent.chat()
    - Schedules temp-file cleanup as a BackgroundTask
    """
    user_id = extract_user_id_from_token(body.user_token)
    thread_id = (
        body.thread_id or f"user_{user_id}_doc_{body.context_ui_id or 'general'}"
    )

    user_context = body.context.model_dump() if body.context else None
    if not user_context:
        try:
            async with EdmsEmployeeClient() as emp:
                user_context = await emp.get_by_id(user_id, body.user_token)
        except Exception:
            user_context = {"firstName": "Коллега"}

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

    # Удаляем временный файл после ответа (если это не UUID системного вложения)
    if body.file_path and not UUID_RE.match(str(body.file_path)):
        if result.get("status") not in ("requires_action", "requires_choice"):
            background_tasks.add_task(_cleanup_file, body.file_path)

    return ChatResponse(
        status=result.get("status", "success"),
        response=result.get("content"),
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
    )


@router.post(
    "/chat/new", response_model=NewChatResponse, summary="Создать новый чат-тред"
)
async def new_chat(body: NewChatRequest) -> NewChatResponse:
    """Generate a fresh thread_id for a new conversation session."""
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
    """
    Return human/AI messages for a thread.

    Filters out ToolMessage and empty AIMessage entries.
    """
    try:
        state = await agent.agent.aget_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
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
    except Exception:
        return {"messages": []}


@router.post(
    "/actions/summarize", response_model=ChatResponse, summary="Суммаризация документа"
)
async def summarize(
    body: SummarizeRequest,
    background_tasks: BackgroundTasks,
    agent: AgentDep,
) -> ChatResponse:
    """
    Direct summarize action.

    Uses a fresh thread_id so it doesn't pollute the main conversation history.
    """
    user_id = extract_user_id_from_token(body.user_token)
    thread_id = f"action_{user_id}_{uuid.uuid4().hex[:6]}"

    agent_request = AgentRequest(
        message=f"Проанализируй вложение: {body.message}",
        user_token=body.user_token,
        context_ui_id=body.context_ui_id,
        thread_id=thread_id,
        file_path=body.file_path,
        human_choice=body.human_choice,
    )

    result = await agent.chat(agent_request)

    if body.file_path and not UUID_RE.match(str(body.file_path)):
        background_tasks.add_task(_cleanup_file, body.file_path)

    return ChatResponse(
        status="success",
        response=result.get("content") or "Анализ готов.",
        thread_id=thread_id,
    )


@router.post(
    "/upload-file", response_model=FileUploadResponse, summary="Загрузить файл"
)
async def upload_file(
    user_token: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> FileUploadResponse:
    """
    Upload a temporary file for subsequent chat/summarize usage.

    File is scoped to the user and removed after use by BackgroundTask.
    """
    try:
        user_id = extract_user_id_from_token(user_token)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "file").suffix
        dest = UPLOAD_DIR / f"{user_id}_{uuid.uuid4().hex}{suffix}"
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                await out.write(chunk)
        return FileUploadResponse(file_path=str(dest), file_name=file.filename or "")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки файла: {exc}")
