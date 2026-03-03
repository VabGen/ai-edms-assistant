# src/ai_edms_assistant/interfaces/api/routes/agent_routes.py
"""Agent API routes: chat, new-chat, history, summarize, file-upload.
"""

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
from ....infrastructure.adapters.user_context_normalizer import (
    UserContextNormalizer,
)
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

_ATTACHMENT_FILENAME_RE_A = re.compile(
    r"(?:Анализ файла|файл):\s*(.+?)(?:\s*$)",
    re.IGNORECASE,
)
_ATTACHMENT_FILENAME_RE_B = re.compile(
    r"^([^/\\<>:\"|?*\r\n]+\.(?:docx?|pdf|txt|rtf|html?|xlsx?|pptx?))$",
    re.IGNORECASE,
)


def _parse_attachment_filename(message: str | None) -> str | None:
    """Extract attachment file name from frontend message.
    """
    if not message:
        return None
    stripped = message.strip()

    m = _ATTACHMENT_FILENAME_RE_A.search(stripped)
    if m:
        return m.group(1).strip()

    m = _ATTACHMENT_FILENAME_RE_B.match(stripped)
    if m:
        return m.group(1).strip()

    return None

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
        pass


async def _resolve_user_context(
    user_id: str,
    token: str,
    http_client: EdmsHttpClient,
) -> dict:
    """Resolve user context from EDMS Employee API and normalize keys.

    Args:
        user_id: Employee UUID extracted from JWT.
        token: JWT bearer token for API auth.
        http_client: Shared EdmsHttpClient from DI container.

    Returns:
        Normalized dict with canonical camelCase keys:
            firstName, lastName, middleName, departmentName, postName.
    """
    try:
        from uuid import UUID

        emp_uuid = UUID(user_id)
        repo = EdmsEmployeeRepository(http_client=http_client)
        employee = await repo.get_by_id(entity_id=emp_uuid, token=token)
        if employee:
            raw_ctx = {
                "firstName": employee.first_name,
                "lastName": employee.last_name,
                "middleName": employee.middle_name,
                "departmentName": employee.department_name,
                "postName": employee.post_name,
            }
            return UserContextNormalizer.normalize(raw_ctx)
    except Exception as exc:
        log.warning("employee_context_fetch_failed", user_id=user_id, error=str(exc))
    return {"firstName": "Коллега"}


@router.post("/chat", response_model=ChatResponse, summary="Отправить сообщение агенту")
async def chat(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    agent: AgentDep,
    http_client: Annotated[EdmsHttpClient, Depends(get_http_client)],
) -> ChatResponse:
    """Process user message via EdmsDocumentAgent.
    """
    user_id = extract_user_id_from_token(body.user_token)
    thread_id = (
        body.thread_id or f"user_{user_id}_doc_{body.context_ui_id or 'general'}"
    )

    raw_context: dict = {}
    if body.context:
        try:
            raw_context = body.context.model_dump(exclude_none=True)
        except AttributeError:
            raw_context = dict(body.context) if hasattr(body.context, "__iter__") else {}

    if not raw_context:
        raw_context = await _resolve_user_context(user_id, body.user_token, http_client)

    user_context = UserContextNormalizer.normalize(raw_context)

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

    if body.file_path and not UUID_RE.match(str(body.file_path)):
        if result.get("status") not in ("requires_action", "requires_choice"):
            background_tasks.add_task(_cleanup_file, body.file_path)

    choice_thread_id: str | None = None
    if result.get("metadata"):
        choice_thread_id = result.get("metadata", {}).get("choice_thread_id")

    return ChatResponse(
        status=result.get("status", "success"),
        response=result.get("content"),
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
        choice_thread_id=choice_thread_id,
        options=result.get("metadata", {}).get("options") if result.get("metadata") else None,
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



async def _resolve_attachment_by_name(
    document_id: str,
    file_name: str,
    token: str
) -> str | None:
    """Resolve attachment UUID by file name using same pipeline as AttachmentTool.
    """
    try:
        from uuid import UUID
        from ....infrastructure.edms_api.http_client import EdmsHttpClient as _Client
        from ....infrastructure.edms_api.repositories.edms_document_repository import (
            EdmsDocumentRepository,
        )

        repo = EdmsDocumentRepository(http_client=_Client())
        doc = await repo.get_with_attachments(
            document_id=UUID(document_id),
            token=token,
        )

        if not doc:
            log.warning("attachment_resolve_no_doc", document_id=document_id[:8])
            return None

        attachments = doc.attachments or []

        if not attachments:
            log.warning(
                "attachment_resolve_no_attachments",
                document_id=document_id[:8],
                file_name=file_name,
            )
            return None

        needle = file_name.strip().lower()
        needle_stem = needle.rsplit(".", 1)[0]

        for att in attachments:
            if att.file_name.strip().lower() == needle:
                log.info(
                    "attachment_resolved_by_name",
                    file_name=att.file_name,
                    attachment_id=str(att.id)[:8],
                )
                return str(att.id)

        for att in attachments:
            if needle_stem and needle_stem in att.file_name.strip().lower():
                log.info(
                    "attachment_resolved_fuzzy",
                    file_name=att.file_name,
                    attachment_id=str(att.id)[:8],
                )
                return str(att.id)

        log.warning(
            "attachment_resolve_not_found",
            file_name=file_name,
            available=[a.file_name for a in attachments],
        )
        return None

    except Exception as exc:
        log.warning(
            "attachment_resolve_failed",
            document_id=document_id[:8],
            file_name=file_name,
            error=str(exc),
        )
        return None



@router.post(
    "/actions/summarize",
    response_model=ChatResponse,
    summary="Суммаризация документа / вложения",
)
async def summarize(
    body: SummarizeRequest,
    background_tasks: BackgroundTasks,
    agent: AgentDep,
    http_client: Annotated[EdmsHttpClient, Depends(get_http_client)],
) -> ChatResponse:
    """Direct summarize action.
    """
    user_id = extract_user_id_from_token(body.user_token)
    thread_id = body.thread_id or f"summarize_{user_id}_{uuid.uuid4().hex[:6]}"
    file_path = body.file_path

    log.debug(
        "summarize_request_received",
        file_path=str(file_path or ""),
        message_preview=str(body.message or "")[:80],
        human_choice=body.human_choice or "",
        context_ui_id=body.context_ui_id or "",
    )

    attachment_file_name: str | None = _parse_attachment_filename(body.message)

    if attachment_file_name and not file_path and body.context_ui_id:
        resolved_id = await _resolve_attachment_by_name(
            document_id=body.context_ui_id,
            file_name=attachment_file_name,
            token=body.user_token,
            http_client=http_client,
        )
        if resolved_id:
            file_path = resolved_id
            log.info(
                "attachment_pre_resolved",
                file_name=attachment_file_name,
                attachment_id=resolved_id[:8],
                document_id=body.context_ui_id,
            )

    if body.human_choice:
        user_message = body.message or "Выполни анализ файла"
    elif file_path and UUID_RE.match(str(file_path)):
        user_message = (
            body.message
            or f"Проанализируй вложение. "
               f"Используй doc_get_file_content(attachment_id='{file_path}') "
               f"для получения текста."
        )
    elif attachment_file_name:
        user_message = (
            f"Проанализируй вложение документа с именем «{attachment_file_name}»."
        )
    else:
        user_message = (
            body.message or "Проанализируй вложение документа и дай краткую сводку"
        )

    agent_request = AgentRequest(
        message=user_message,
        user_token=body.user_token,
        context_ui_id=body.context_ui_id,
        thread_id=thread_id,
        user_context={},
        file_path=file_path,
        human_choice=body.human_choice,
    )

    result = await agent.chat(agent_request)

    if file_path and not UUID_RE.match(str(file_path)):
        if result.get("status") not in ("requires_action", "requires_choice"):
            background_tasks.add_task(_cleanup_file, str(file_path))

    choice_thread_id: str | None = None
    if result.get("metadata"):
        choice_thread_id = result.get("metadata", {}).get("choice_thread_id")

    return ChatResponse(
        status=result.get("status", "success"),
        response=result.get("content") or result.get("message") or "Анализ готов.",
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
        choice_thread_id=choice_thread_id,
        options=result.get("metadata", {}).get("options") if result.get("metadata") else None,
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