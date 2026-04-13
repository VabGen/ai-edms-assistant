# edms_ai_assistant/mcp_server/edms_mcp_server.py
"""
EDMS MCP Server — FastMCP сервер со всеми инструментами.
"""

from __future__ import annotations

import logging
import time
import uuid

import httpx
import structlog
from fastmcp import FastMCP
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from edms_ai_assistant.config import settings
from edms_ai_assistant.mcp_server.tools.appeal_tools import register_appeal_tools
from edms_ai_assistant.mcp_server.tools.content_tools import register_content_tools
from edms_ai_assistant.mcp_server.tools.document_tools import register_document_tools
from edms_ai_assistant.mcp_server.tools.workflow_tools import register_workflow_tools

# ── Логирование ───────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(__import__("logging"), settings.LOG_LEVEL, 20)
    ),
    logger_factory=structlog.WriteLoggerFactory(
        file=open("edms_mcp.log", "a", encoding="utf-8")
    ),
)
log = structlog.get_logger()
logger = logging.getLogger(__name__)

# ── FastMCP приложение ─────────────────────────────────────────────────────

import inspect as _inspect

_fastmcp_init_params = set(_inspect.signature(FastMCP.__init__).parameters.keys())

_mcp_kwargs: dict = {"name": "EDMS Document Management"}
_mcp_description = (
    "Полный набор инструментов для работы с корпоративной СЭД: "
    "документы, поручения, ознакомления, обращения граждан, файлы, аналитика."
)
if "instructions" in _fastmcp_init_params:
    _mcp_kwargs["instructions"] = _mcp_description  # fastmcp >= 2.0
elif "description" in _fastmcp_init_params:
    _mcp_kwargs["description"] = _mcp_description  # fastmcp < 2.0

mcp = FastMCP(**_mcp_kwargs)

# ── Регистрация всех групп инструментов ───────────────────────────────────

register_document_tools(mcp)
register_content_tools(mcp)
register_workflow_tools(mcp)
register_appeal_tools(mcp)

log.info("tool_groups_registered", groups=["document", "content", "workflow", "appeal"])


# ── Вспомогательные функции исходных инструментов ─────────────────────────

EDMS_API_URL: str = settings.EDMS_BASE_URL.rstrip("/")
REQUEST_TIMEOUT: float = 30.0
MAX_RETRY_ATTEMPTS: int = 3


def _build_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Request-ID": str(uuid.uuid4()),
    }


def _ok(data: object) -> dict:
    return {"success": True, "data": data, "error": None}


def _err(code: str, message: str) -> dict:
    return {"success": False, "data": None, "error": {"code": code, "message": message}}


async def _request(
    method: str,
    endpoint: str,
    *,
    token: str | None = None,
    params: dict | None = None,
    json_body: dict | None = None,
    tool_name: str = "unknown",
) -> dict:
    """HTTP-запрос к EDMS API с retry."""
    url = f"{EDMS_API_URL}/{endpoint.lstrip('/')}"
    start_ts = time.monotonic()
    attempt_num = 0

    headers = _build_headers()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    _HTTP_ERRORS: dict[int, tuple[str, str]] = {
        400: ("INVALID_REQUEST", "Некорректные параметры запроса"),
        401: ("UNAUTHORIZED", "Ошибка авторизации"),
        403: ("FORBIDDEN", "Недостаточно прав"),
        404: ("NOT_FOUND", "Ресурс не найден"),
        409: ("CONFLICT", "Конфликт данных"),
        422: ("VALIDATION_ERROR", "Ошибка валидации"),
        429: ("RATE_LIMITED", "Превышен лимит запросов"),
        500: ("SERVER_ERROR", "Внутренняя ошибка сервера"),
        503: ("SERVICE_UNAVAILABLE", "Сервис временно недоступен"),
    }

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                attempt_num += 1
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        params=params,
                        json=json_body,
                    )

                elapsed_ms = round((time.monotonic() - start_ts) * 1000, 2)

                if response.status_code >= 400:
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = {"raw": response.text[:500]}
                    log.warning(
                        "edms_api_error",
                        tool=tool_name,
                        status=response.status_code,
                        error=error_body,
                        elapsed_ms=elapsed_ms,
                    )
                    code, msg = _HTTP_ERRORS.get(
                        response.status_code,
                        ("HTTP_ERROR", f"HTTP {response.status_code}"),
                    )
                    return _err(code, msg)

                log.info(
                    "edms_request_ok",
                    tool=tool_name,
                    method=method,
                    endpoint=endpoint,
                    status=response.status_code,
                    elapsed_ms=elapsed_ms,
                    attempts=attempt_num,
                )

                if response.status_code == 204 or not response.content:
                    return _ok({})

                return _ok(response.json())

    except Exception as exc:
        elapsed_ms = round((time.monotonic() - start_ts) * 1000, 2)
        log.error(
            "edms_request_failed", tool=tool_name, error=str(exc), elapsed_ms=elapsed_ms
        )
        return _err("REQUEST_FAILED", f"Ошибка запроса: {exc!s}")

    return _err("MAX_RETRIES", "Превышено количество попыток")


# ── 8 базовых инструментов ────────────────────────────────────────────────


@mcp.tool(
    description=(
        "Получить документ по UUID или номеру (DOC-XXXX). "
        "Возвращает метаданные, статус, вложения, ответственных. Только чтение."
    )
)
async def get_document(
    document_id: str,
    token: str,
    include_history: bool = False,
    include_attachments: bool = True,
) -> dict:
    if not document_id or not document_id.strip():
        return _err("INVALID_REQUEST", "document_id не может быть пустым")
    params: dict = {}
    if include_history:
        params["include_history"] = "true"
    if not include_attachments:
        params["include_attachments"] = "false"
    return await _request(
        "GET",
        f"/api/document/{document_id.strip()}",
        token=token,
        params=params,
        tool_name="get_document",
    )


@mcp.tool(
    description=(
        "Поиск документов по фильтрам: тип, статус, автор, дата, текст. "
        "Поддерживает пагинацию через page/page_size."
    )
)
async def search_documents(
    token: str,
    query: str | None = None,
    status: list[str] | None = None,
    document_type: list[str] | None = None,
    author_id: str | None = None,
    assignee_id: str | None = None,
    department: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "updated_at",
    sort_order: str = "desc",
) -> dict:
    if not 1 <= page_size <= 100:
        return _err("INVALID_REQUEST", "page_size должен быть от 1 до 100")
    params: dict = {
        "page": page,
        "page_size": page_size,
        "sort_by": sort_by,
        "sort_order": sort_order,
    }
    if query:
        params["q"] = query
    if status:
        params["status"] = ",".join(status)
    if document_type:
        params["type"] = ",".join(document_type)
    if author_id:
        params["author_id"] = author_id
    if assignee_id:
        params["assignee_id"] = assignee_id
    if department:
        params["department"] = department
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return await _request(
        "GET",
        "/api/documents",
        token=token,
        params=params,
        tool_name="search_documents",
    )


@mcp.tool(
    description=(
        "Создать новый документ. ВНИМАНИЕ: необратимая операция. "
        "Использовать только по явному запросу пользователя."
    )
)
async def create_document(
    token: str,
    title: str,
    document_type: str,
    content: str | None = None,
    assignees: list[str] | None = None,
    department: str | None = None,
    due_date: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    if not title or len(title.strip()) < 3:
        return _err(
            "INVALID_REQUEST", "Название документа должно содержать минимум 3 символа"
        )
    _VALID_TYPES = {"договор", "приказ", "акт", "счёт", "протокол", "спецификация"}
    if document_type not in _VALID_TYPES:
        return _err(
            "INVALID_REQUEST",
            f"Тип должен быть одним из: {', '.join(sorted(_VALID_TYPES))}",
        )
    body: dict = {"title": title.strip(), "type": document_type}
    if content:
        body["content"] = content
    if assignees:
        body["assignees"] = assignees
    if department:
        body["department"] = department
    if due_date:
        body["due_date"] = due_date
    if tags:
        body["tags"] = tags
    return await _request(
        "POST",
        "/api/documents",
        token=token,
        json_body=body,
        tool_name="create_document",
    )


@mcp.tool(
    description=(
        "Изменить статус документа. ВНИМАНИЕ: переход в rejected/archived необратим."
    )
)
async def update_document_status(
    document_id: str,
    token: str,
    new_status: str,
    comment: str | None = None,
    notify_assignees: bool = True,
) -> dict:
    _VALID_STATUSES = {"draft", "review", "approved", "rejected", "signed", "archived"}
    if new_status not in _VALID_STATUSES:
        return _err(
            "INVALID_REQUEST",
            f"Статус должен быть одним из: {', '.join(sorted(_VALID_STATUSES))}",
        )
    if new_status in {"rejected", "archived"} and not comment:
        return _err(
            "COMMENT_REQUIRED", f"Для статуса '{new_status}' необходим комментарий"
        )
    body: dict = {"status": new_status, "notify_assignees": notify_assignees}
    if comment:
        body["comment"] = comment.strip()
    return await _request(
        "PATCH",
        f"/api/documents/{document_id}/status",
        token=token,
        json_body=body,
        tool_name="update_document_status",
    )


@mcp.tool(description="История изменений документа: кто, когда и что менял.")
async def get_document_history(
    document_id: str,
    token: str,
    event_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
) -> dict:
    if not 1 <= limit <= 500:
        return _err("INVALID_REQUEST", "limit должен быть от 1 до 500")
    params: dict = {"limit": limit}
    if event_types:
        params["event_types"] = ",".join(event_types)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return await _request(
        "GET",
        f"/api/documents/{document_id}/history",
        token=token,
        params=params,
        tool_name="get_document_history",
    )


@mcp.tool(
    description="Назначить ответственных за документ с ролями: reviewer, approver, signer, observer."
)
async def assign_document(
    document_id: str,
    token: str,
    assignees: list[dict],
    replace_existing: bool = False,
    message: str | None = None,
) -> dict:
    if not assignees:
        return _err("INVALID_REQUEST", "assignees не может быть пустым")
    _VALID_ROLES = {"reviewer", "approver", "signer", "observer"}
    for a in assignees:
        if not a.get("user_id"):
            return _err("INVALID_REQUEST", "Каждый assignee должен содержать user_id")
        if a.get("role") not in _VALID_ROLES:
            return _err("INVALID_REQUEST", f"Роль '{a.get('role')}' недопустима")
    body: dict = {"assignees": assignees, "replace_existing": replace_existing}
    if message:
        body["message"] = message[:1000]
    return await _request(
        "POST",
        f"/api/documents/{document_id}/assignees",
        token=token,
        json_body=body,
        tool_name="assign_document",
    )


@mcp.tool(
    description="Аналитика документооборота: статистика, нагрузка, просрочки, коэффициент одобрения."
)
async def get_analytics(
    token: str,
    metric_type: str,
    date_from: str | None = None,
    date_to: str | None = None,
    group_by: str = "month",
    department: str | None = None,
    document_type: str | None = None,
) -> dict:
    _VALID_METRICS = {
        "status_distribution",
        "volume_by_type",
        "processing_time",
        "workload_by_user",
        "workload_by_department",
        "overdue_documents",
        "approval_rate",
    }
    if metric_type not in _VALID_METRICS:
        return _err(
            "INVALID_REQUEST",
            f"metric_type должен быть одним из: {', '.join(sorted(_VALID_METRICS))}",
        )
    params: dict = {"metric": metric_type, "group_by": group_by}
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    if department:
        params["department"] = department
    if document_type:
        params["type"] = document_type
    return await _request(
        "GET", "/api/analytics", token=token, params=params, tool_name="get_analytics"
    )


@mcp.tool(
    description="Статус рабочего процесса: кто должен действовать, просрочки, прогресс."
)
async def get_workflow_status(
    document_id: str,
    token: str,
    include_completed: bool = False,
) -> dict:
    params: dict = {}
    if include_completed:
        params["include_completed"] = "true"
    return await _request(
        "GET",
        f"/api/documents/{document_id}/workflow",
        token=token,
        params=params,
        tool_name="get_workflow_status",
    )


log.info("mcp_server_ready", total_tools="all_registered")
