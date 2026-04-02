"""
EDMS MCP Server — Сервер инструментов для работы с корпоративной системой документооборота.

Реализует 8 MCP-инструментов для управления документами через FastMCP.
Все вызовы логируются через structlog, HTTP-запросы к EDMS API выполняются
через httpx.AsyncClient с retry-логикой (tenacity, exponential backoff).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from typing import Any

import httpx
import structlog
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Конфигурация из переменных окружения
# ---------------------------------------------------------------------------
EDMS_API_URL: str = os.getenv("EDMS_API_URL", "http://localhost:8080/api/v1")
EDMS_API_KEY: str = os.getenv("EDMS_API_KEY", "")
MCP_HOST: str = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT: int = int(os.getenv("MCP_PORT", "8001"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
REQUEST_TIMEOUT: float = 30.0
MAX_RETRY_ATTEMPTS: int = 3

# ---------------------------------------------------------------------------
# Настройка structlog
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(__import__("logging"), LOG_LEVEL, 20)
    ),
    logger_factory=structlog.WriteLoggerFactory(
        file=open("edms_mcp.log", "a", encoding="utf-8")  # noqa: WPS515
    ),
)
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# MCP приложение
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="EDMS Document Management",
    version="1.0.0",
    description="Инструменты для работы с корпоративной системой электронного документооборота",
)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    """Сформировать стандартные заголовки HTTP-запроса к EDMS API."""
    return {
        "Authorization": f"Bearer {EDMS_API_KEY}",
        "Content-Type": "application/json",
        "X-Request-ID": str(uuid.uuid4()),
    }


def _success_response(data: Any) -> dict[str, Any]:
    """Сформировать успешный ответ инструмента."""
    return {"success": True, "data": data, "error": None}


def _error_response(code: str, message: str) -> dict[str, Any]:
    """Сформировать ответ с ошибкой."""
    return {"success": False, "data": None, "error": {"code": code, "message": message}}


async def _edms_request(
    method: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    tool_name: str = "unknown",
) -> dict[str, Any]:
    """
    Выполнить HTTP-запрос к EDMS API с retry-логикой.

    Параметры:
        method: HTTP-метод (GET, POST, PATCH, DELETE)
        endpoint: путь относительно EDMS_API_URL
        params: query-параметры
        json_body: тело запроса для POST/PATCH
        tool_name: имя вызвавшего инструмента (для логов)

    Возвращает:
        Словарь с ответом API

    Исключения:
        Любые httpx-исключения пробрасываются после исчерпания retry
    """
    url = f"{EDMS_API_URL}/{endpoint.lstrip('/')}"
    start_ts = time.monotonic()
    attempt_num = 0

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        reraise=True,
    ):
        with attempt:
            attempt_num += 1
            log.debug(
                "edms_api_request",
                tool=tool_name,
                method=method,
                url=url,
                attempt=attempt_num,
            )

            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=_build_headers(),
                    params=params,
                    json=json_body,
                )

            elapsed = round((time.monotonic() - start_ts) * 1000, 2)

            if response.status_code >= 400:
                error_body = {}
                try:
                    error_body = response.json()
                except Exception:
                    error_body = {"raw": response.text[:500]}

                log.warning(
                    "edms_api_error",
                    tool=tool_name,
                    status_code=response.status_code,
                    error=error_body,
                    elapsed_ms=elapsed,
                )
                error_map = {
                    400: ("INVALID_REQUEST", "Некорректные параметры запроса"),
                    401: ("UNAUTHORIZED", "Ошибка авторизации"),
                    403: ("FORBIDDEN", "Недостаточно прав доступа"),
                    404: ("NOT_FOUND", "Документ не найден"),
                    409: ("CONFLICT", "Конфликт данных: документ был изменён"),
                    422: ("VALIDATION_ERROR", error_body.get("detail", "Ошибка валидации")),
                    429: ("RATE_LIMITED", "Превышен лимит запросов"),
                    500: ("SERVER_ERROR", "Внутренняя ошибка сервера EDMS"),
                    503: ("SERVICE_UNAVAILABLE", "Сервис EDMS временно недоступен"),
                }
                code, msg = error_map.get(
                    response.status_code,
                    ("HTTP_ERROR", f"HTTP {response.status_code}"),
                )
                return _error_response(code, msg)

            log.info(
                "edms_api_success",
                tool=tool_name,
                method=method,
                endpoint=endpoint,
                status_code=response.status_code,
                elapsed_ms=elapsed,
                attempts=attempt_num,
            )

            if response.status_code == 204 or not response.content:
                return _success_response({})

            return _success_response(response.json())

    return _error_response("MAX_RETRIES", "Превышено количество попыток запроса")


def _log_tool_call(tool_name: str, params: dict[str, Any]) -> float:
    """Залогировать вызов инструмента, вернуть timestamp начала."""
    ts = time.monotonic()
    # Маскируем чувствительные поля перед логированием
    safe_params = {
        k: "***" if k in {"token", "password", "secret", "api_key"} else v
        for k, v in params.items()
    }
    log.info("tool_called", tool=tool_name, params=safe_params)
    return ts


def _log_tool_result(tool_name: str, start_ts: float, success: bool) -> None:
    """Залогировать результат вызова инструмента."""
    elapsed = round((time.monotonic() - start_ts) * 1000, 2)
    log.info(
        "tool_completed",
        tool=tool_name,
        elapsed_ms=elapsed,
        success=success,
    )


# ---------------------------------------------------------------------------
# Pydantic-модели входных параметров
# ---------------------------------------------------------------------------

class GetDocumentInput(BaseModel):
    """Параметры получения документа по ID."""

    document_id: str = Field(
        ...,
        description="UUID документа или номер вида DOC-12345",
        pattern=r"^(DOC-\d+|[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$",
    )
    include_history: bool = Field(False, description="Включить историю изменений")
    include_attachments: bool = Field(True, description="Включить список вложений")


class SearchDocumentsInput(BaseModel):
    """Параметры поиска документов."""

    query: str | None = Field(None, description="Полнотекстовый поиск", max_length=500)
    status: list[str] | None = Field(None, description="Фильтр по статусам")
    document_type: list[str] | None = Field(None, description="Фильтр по типам")
    author_id: str | None = Field(None, description="UUID автора")
    assignee_id: str | None = Field(None, description="UUID ответственного")
    department: str | None = Field(None, description="Отдел/подразделение")
    date_from: str | None = Field(None, description="Дата создания от (ISO 8601)")
    date_to: str | None = Field(None, description="Дата создания до (ISO 8601)")
    page: int = Field(1, description="Номер страницы", ge=1)
    page_size: int = Field(20, description="Записей на странице", ge=1, le=100)
    sort_by: str = Field("updated_at", description="Поле сортировки")
    sort_order: str = Field("desc", description="Направление сортировки")


class CreateDocumentInput(BaseModel):
    """Параметры создания нового документа."""

    title: str = Field(..., description="Название документа", min_length=3, max_length=500)
    document_type: str = Field(
        ...,
        description="Тип документа",
        pattern=r"^(договор|приказ|акт|счёт|протокол|спецификация)$",
    )
    content: str | None = Field(None, description="Содержимое/описание")
    assignees: list[str] | None = Field(None, description="UUID ответственных")
    department: str | None = Field(None, description="Отдел-владелец")
    due_date: str | None = Field(None, description="Срок исполнения (ISO 8601)")
    tags: list[str] | None = Field(None, description="Теги")
    metadata: dict[str, Any] | None = Field(None, description="Дополнительные метаданные")


class UpdateDocumentStatusInput(BaseModel):
    """Параметры изменения статуса документа."""

    document_id: str = Field(..., description="UUID документа")
    new_status: str = Field(
        ...,
        description="Новый статус",
        pattern=r"^(draft|review|approved|rejected|signed|archived)$",
    )
    comment: str | None = Field(None, description="Комментарий", max_length=2000)
    notify_assignees: bool = Field(True, description="Уведомить ответственных")


class GetDocumentHistoryInput(BaseModel):
    """Параметры получения истории документа."""

    document_id: str = Field(..., description="UUID документа")
    event_types: list[str] | None = Field(None, description="Фильтр по типам событий")
    date_from: str | None = Field(None, description="Дата от (ISO 8601)")
    date_to: str | None = Field(None, description="Дата до (ISO 8601)")
    limit: int = Field(50, description="Максимальное количество событий", ge=1, le=500)


class AssigneeSpec(BaseModel):
    """Спецификация назначаемого сотрудника."""

    user_id: str = Field(..., description="UUID пользователя")
    role: str = Field(
        ...,
        description="Роль",
        pattern=r"^(reviewer|approver|signer|observer)$",
    )
    due_date: str | None = Field(None, description="Срок выполнения роли (ISO 8601)")


class AssignDocumentInput(BaseModel):
    """Параметры назначения ответственных за документ."""

    document_id: str = Field(..., description="UUID документа")
    assignees: list[AssigneeSpec] = Field(
        ..., description="Список назначаемых с ролями", min_length=1
    )
    replace_existing: bool = Field(False, description="Заменить существующих")
    message: str | None = Field(None, description="Сообщение назначаемым", max_length=1000)


class GetAnalyticsInput(BaseModel):
    """Параметры запроса аналитики."""

    metric_type: str = Field(
        ...,
        description="Тип метрики",
        pattern=r"^(status_distribution|volume_by_type|processing_time|workload_by_user|workload_by_department|overdue_documents|approval_rate)$",
    )
    date_from: str | None = Field(None, description="Дата от (ISO 8601)")
    date_to: str | None = Field(None, description="Дата до (ISO 8601)")
    group_by: str = Field("month", description="Группировка")
    department: str | None = Field(None, description="Фильтр по отделу")
    document_type: str | None = Field(None, description="Фильтр по типу")


class GetWorkflowStatusInput(BaseModel):
    """Параметры запроса статуса рабочего процесса."""

    document_id: str = Field(..., description="UUID документа")
    include_completed: bool = Field(False, description="Включить завершённые шаги")


# ---------------------------------------------------------------------------
# Инструменты MCP
# ---------------------------------------------------------------------------

@mcp.tool(
    description="Получить документ по ID. Возвращает метаданные, статус, ответственных и вложения."
)
async def get_document(
    document_id: str,
    include_history: bool = False,
    include_attachments: bool = True,
) -> dict[str, Any]:
    """
    Получить документ по его UUID или номеру.

    Используется когда пользователь называет конкретный документ
    (например, «покажи договор DOC-12345» или «что с документом?»).
    """
    params_in = {"document_id": document_id, "include_history": include_history}
    ts = _log_tool_call("get_document", params_in)

    try:
        validated = GetDocumentInput(
            document_id=document_id,
            include_history=include_history,
            include_attachments=include_attachments,
        )
    except Exception as exc:
        _log_tool_result("get_document", ts, False)
        return _error_response("VALIDATION_ERROR", str(exc))

    params: dict[str, Any] = {}
    if validated.include_history:
        params["include_history"] = "true"
    if not validated.include_attachments:
        params["include_attachments"] = "false"

    result = await _edms_request(
        "GET",
        f"/documents/{validated.document_id}",
        params=params,
        tool_name="get_document",
    )
    _log_tool_result("get_document", ts, result["success"])
    return result


@mcp.tool(
    description="Поиск документов по фильтрам: статус, тип, автор, дата, текст. Поддерживает пагинацию."
)
async def search_documents(
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
) -> dict[str, Any]:
    """
    Поиск документов по набору фильтров.

    Используется для запросов вида «найди все договоры за прошлый месяц»,
    «покажи документы на согласовании у Иванова».
    """
    params_in = {k: v for k, v in locals().items() if v is not None and k != "self"}
    ts = _log_tool_call("search_documents", params_in)

    try:
        validated = SearchDocumentsInput(
            query=query,
            status=status,
            document_type=document_type,
            author_id=author_id,
            assignee_id=assignee_id,
            department=department,
            date_from=date_from,
            date_to=date_to,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
    except Exception as exc:
        _log_tool_result("search_documents", ts, False)
        return _error_response("VALIDATION_ERROR", str(exc))

    query_params: dict[str, Any] = {
        "page": validated.page,
        "page_size": validated.page_size,
        "sort_by": validated.sort_by,
        "sort_order": validated.sort_order,
    }
    if validated.query:
        query_params["q"] = validated.query
    if validated.status:
        query_params["status"] = ",".join(validated.status)
    if validated.document_type:
        query_params["type"] = ",".join(validated.document_type)
    if validated.author_id:
        query_params["author_id"] = validated.author_id
    if validated.assignee_id:
        query_params["assignee_id"] = validated.assignee_id
    if validated.department:
        query_params["department"] = validated.department
    if validated.date_from:
        query_params["date_from"] = validated.date_from
    if validated.date_to:
        query_params["date_to"] = validated.date_to

    result = await _edms_request(
        "GET",
        "/documents",
        params=query_params,
        tool_name="search_documents",
    )
    _log_tool_result("search_documents", ts, result["success"])
    return result


@mcp.tool(
    description="Создать новый документ. ВНИМАНИЕ: необратимая операция. Требует явного подтверждения параметров."
)
async def create_document(
    title: str,
    document_type: str,
    content: str | None = None,
    assignees: list[str] | None = None,
    department: str | None = None,
    due_date: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Создать новый документ в системе EDMS.

    Использовать только при явном запросе пользователя.
    Перед созданием обязательно подтвердить название, тип и параметры.
    """
    params_in = {"title": title, "document_type": document_type}
    ts = _log_tool_call("create_document", params_in)

    try:
        validated = CreateDocumentInput(
            title=title,
            document_type=document_type,
            content=content,
            assignees=assignees,
            department=department,
            due_date=due_date,
            tags=tags,
            metadata=metadata,
        )
    except Exception as exc:
        _log_tool_result("create_document", ts, False)
        return _error_response("VALIDATION_ERROR", str(exc))

    body: dict[str, Any] = {
        "title": validated.title,
        "type": validated.document_type,
    }
    if validated.content:
        body["content"] = validated.content
    if validated.assignees:
        body["assignees"] = validated.assignees
    if validated.department:
        body["department"] = validated.department
    if validated.due_date:
        body["due_date"] = validated.due_date
    if validated.tags:
        body["tags"] = validated.tags
    if validated.metadata:
        body["metadata"] = validated.metadata

    result = await _edms_request(
        "POST",
        "/documents",
        json_body=body,
        tool_name="create_document",
    )
    _log_tool_result("create_document", ts, result["success"])
    return result


@mcp.tool(
    description="Изменить статус документа (согласование, подписание, архивирование). Требует прав доступа."
)
async def update_document_status(
    document_id: str,
    new_status: str,
    comment: str | None = None,
    notify_assignees: bool = True,
) -> dict[str, Any]:
    """
    Изменить статус документа.

    Деструктивная операция — переход в 'rejected' или 'archived'
    требует комментария и подтверждения от пользователя.
    """
    ts = _log_tool_call(
        "update_document_status",
        {"document_id": document_id, "new_status": new_status},
    )

    if new_status in ("rejected", "archived") and not comment:
        _log_tool_result("update_document_status", ts, False)
        return _error_response(
            "COMMENT_REQUIRED",
            f"Для перевода в статус '{new_status}' необходим комментарий с обоснованием",
        )

    try:
        validated = UpdateDocumentStatusInput(
            document_id=document_id,
            new_status=new_status,
            comment=comment,
            notify_assignees=notify_assignees,
        )
    except Exception as exc:
        _log_tool_result("update_document_status", ts, False)
        return _error_response("VALIDATION_ERROR", str(exc))

    body: dict[str, Any] = {
        "status": validated.new_status,
        "notify_assignees": validated.notify_assignees,
    }
    if validated.comment:
        body["comment"] = validated.comment

    result = await _edms_request(
        "PATCH",
        f"/documents/{validated.document_id}/status",
        json_body=body,
        tool_name="update_document_status",
    )
    _log_tool_result("update_document_status", ts, result["success"])
    return result


@mcp.tool(
    description="Получить историю изменений документа: кто, когда и что менял. Для аудита и проверки."
)
async def get_document_history(
    document_id: str,
    event_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Получить журнал событий по документу.

    Использовать при запросах об аудите, хронологии изменений,
    проверке кто и когда выполнял действия с документом.
    """
    ts = _log_tool_call("get_document_history", {"document_id": document_id})

    try:
        validated = GetDocumentHistoryInput(
            document_id=document_id,
            event_types=event_types,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
    except Exception as exc:
        _log_tool_result("get_document_history", ts, False)
        return _error_response("VALIDATION_ERROR", str(exc))

    params: dict[str, Any] = {"limit": validated.limit}
    if validated.event_types:
        params["event_types"] = ",".join(validated.event_types)
    if validated.date_from:
        params["date_from"] = validated.date_from
    if validated.date_to:
        params["date_to"] = validated.date_to

    result = await _edms_request(
        "GET",
        f"/documents/{validated.document_id}/history",
        params=params,
        tool_name="get_document_history",
    )
    _log_tool_result("get_document_history", ts, result["success"])
    return result


@mcp.tool(
    description="Назначить ответственных за документ с ролями: проверяющий, согласующий, подписант, наблюдатель."
)
async def assign_document(
    document_id: str,
    assignees: list[dict[str, Any]],
    replace_existing: bool = False,
    message: str | None = None,
) -> dict[str, Any]:
    """
    Назначить ответственных за документ.

    Использовать при запросах «передай документ», «назначь ответственным»,
    «добавь в согласующие». Поддерживает роли: reviewer, approver, signer, observer.
    """
    ts = _log_tool_call("assign_document", {"document_id": document_id})

    try:
        assignee_specs = [AssigneeSpec(**a) for a in assignees]
        validated = AssignDocumentInput(
            document_id=document_id,
            assignees=assignee_specs,
            replace_existing=replace_existing,
            message=message,
        )
    except Exception as exc:
        _log_tool_result("assign_document", ts, False)
        return _error_response("VALIDATION_ERROR", str(exc))

    body: dict[str, Any] = {
        "assignees": [a.model_dump(exclude_none=True) for a in validated.assignees],
        "replace_existing": validated.replace_existing,
    }
    if validated.message:
        body["message"] = validated.message

    result = await _edms_request(
        "POST",
        f"/documents/{validated.document_id}/assignees",
        json_body=body,
        tool_name="assign_document",
    )
    _log_tool_result("assign_document", ts, result["success"])
    return result


@mcp.tool(
    description="Аналитика по документам: статистика, нагрузка, просрочки, коэффициент одобрения."
)
async def get_analytics(
    metric_type: str,
    date_from: str | None = None,
    date_to: str | None = None,
    group_by: str = "month",
    department: str | None = None,
    document_type: str | None = None,
) -> dict[str, Any]:
    """
    Получить аналитические данные по документообороту.

    Использовать при запросах об отчётах, дашбордах, метриках эффективности.
    Поддерживает: распределение статусов, объём по типам, время обработки,
    нагрузку на сотрудников/отделы, просроченные документы, коэффициент одобрения.
    """
    ts = _log_tool_call("get_analytics", {"metric_type": metric_type})

    try:
        validated = GetAnalyticsInput(
            metric_type=metric_type,
            date_from=date_from,
            date_to=date_to,
            group_by=group_by,
            department=department,
            document_type=document_type,
        )
    except Exception as exc:
        _log_tool_result("get_analytics", ts, False)
        return _error_response("VALIDATION_ERROR", str(exc))

    params: dict[str, Any] = {
        "metric": validated.metric_type,
        "group_by": validated.group_by,
    }
    if validated.date_from:
        params["date_from"] = validated.date_from
    if validated.date_to:
        params["date_to"] = validated.date_to
    if validated.department:
        params["department"] = validated.department
    if validated.document_type:
        params["type"] = validated.document_type

    result = await _edms_request(
        "GET",
        "/analytics",
        params=params,
        tool_name="get_analytics",
    )
    _log_tool_result("get_analytics", ts, result["success"])
    return result


@mcp.tool(
    description="Статус рабочего процесса документа: кто должен действовать, просрочки, прогресс согласования."
)
async def get_workflow_status(
    document_id: str,
    include_completed: bool = False,
) -> dict[str, Any]:
    """
    Получить текущий статус рабочего процесса документа.

    Использовать при вопросах «где застрял документ», «кто ещё не согласовал»,
    «сколько ждёт подписи» — показывает очередь ожидающих действий с просрочками.
    """
    ts = _log_tool_call("get_workflow_status", {"document_id": document_id})

    try:
        validated = GetWorkflowStatusInput(
            document_id=document_id,
            include_completed=include_completed,
        )
    except Exception as exc:
        _log_tool_result("get_workflow_status", ts, False)
        return _error_response("VALIDATION_ERROR", str(exc))

    params: dict[str, Any] = {}
    if validated.include_completed:
        params["include_completed"] = "true"

    result = await _edms_request(
        "GET",
        f"/documents/{validated.document_id}/workflow",
        params=params,
        tool_name="get_workflow_status",
    )
    _log_tool_result("get_workflow_status", ts, result["success"])
    return result


# ---------------------------------------------------------------------------
# Запуск сервера
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("edms_mcp_server_starting", host=MCP_HOST, port=MCP_PORT)
    mcp.run(host=MCP_HOST, port=MCP_PORT)
