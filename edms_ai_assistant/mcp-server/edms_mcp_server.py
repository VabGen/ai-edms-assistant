# mcp-server/edms_mcp_server.py
"""
EDMS MCP Server — FastMCP сервер для работы с корпоративной СЭД.

ВСЕ инструменты регистрируются через @mcp.tool() — это единственный
правильный способ использования FastMCP. LangChain @tool здесь НЕ используется.

Инструменты:
    get_document            — получить документ по UUID
    search_documents        — поиск по фильтрам
    create_document         — создать документ (необратимо)
    update_document_status  — изменить статус (необратимо)
    get_document_history    — история изменений
    assign_document         — назначить ответственных
    get_analytics           — аналитика и статистика
    get_workflow_status     — статус рабочего процесса

Конфигурация из .env:
    EDMS_API_URL   — базовый URL Java EDMS API
    EDMS_API_KEY   — ключ авторизации
    MCP_HOST       — хост сервера (default: 0.0.0.0)
    MCP_PORT       — порт сервера (default: 8001)
    LOG_LEVEL      — уровень логирования
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import httpx
import structlog
from fastmcp import FastMCP
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ── Конфигурация из переменных окружения ──────────────────────────────────
EDMS_API_URL: str = os.getenv("EDMS_API_URL", "http://localhost:8080/api")
EDMS_API_KEY: str = os.getenv("EDMS_API_KEY", "")
MCP_HOST: str = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT: int = int(os.getenv("MCP_PORT", "8001"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
REQUEST_TIMEOUT: float = 30.0
MAX_RETRY_ATTEMPTS: int = 3

# ── Структурированное логирование ─────────────────────────────────────────
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
        file=open("edms_mcp.log", "a", encoding="utf-8")
    ),
)
log = structlog.get_logger()

# ── FastMCP приложение ─────────────────────────────────────────────────────
mcp = FastMCP(
    name="EDMS Document Management",
    version="2.0.0",
    description="Инструменты для работы с корпоративной системой электронного документооборота",
)


# ── Вспомогательные функции ───────────────────────────────────────────────

def _build_headers() -> dict[str, str]:
    """Строит стандартные заголовки HTTP-запроса."""
    return {
        "Authorization": f"Bearer {EDMS_API_KEY}",
        "Content-Type": "application/json",
        "X-Request-ID": str(uuid.uuid4()),
    }


def _ok(data: Any) -> dict[str, Any]:
    """Стандартный успешный ответ."""
    return {"success": True, "data": data, "error": None}


def _err(code: str, message: str) -> dict[str, Any]:
    """Стандартный ответ с ошибкой."""
    return {"success": False, "data": None, "error": {"code": code, "message": message}}


async def _request(
    method: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    tool_name: str = "unknown",
) -> dict[str, Any]:
    """
    Выполняет HTTP-запрос к EDMS API с retry-логикой (exponential backoff).

    Повторяет при: ConnectError, TimeoutException.
    Не повторяет при: 4xx ответах.

    Args:
        method:    HTTP-метод (GET, POST, PATCH, DELETE).
        endpoint:  Путь относительно EDMS_API_URL.
        params:    Query-параметры.
        json_body: Тело запроса для POST/PATCH.
        tool_name: Имя вызвавшего инструмента (для логов).

    Returns:
        Стандартизированный ответ {success, data, error}.
    """
    url = f"{EDMS_API_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    start_ts = time.monotonic()
    attempt_num = 0

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                attempt_num += 1
                log.debug("edms_request", tool=tool_name, method=method, url=url, attempt=attempt_num)

                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers=_build_headers(),
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
                    _HTTP_ERRORS: dict[int, tuple[str, str]] = {
                        400: ("INVALID_REQUEST", "Некорректные параметры запроса"),
                        401: ("UNAUTHORIZED", "Ошибка авторизации"),
                        403: ("FORBIDDEN", "Недостаточно прав доступа"),
                        404: ("NOT_FOUND", "Ресурс не найден"),
                        409: ("CONFLICT", "Конфликт данных"),
                        422: ("VALIDATION_ERROR", error_body.get("detail", "Ошибка валидации")),
                        429: ("RATE_LIMITED", "Превышен лимит запросов"),
                        500: ("SERVER_ERROR", "Внутренняя ошибка сервера EDMS"),
                        503: ("SERVICE_UNAVAILABLE", "Сервис EDMS временно недоступен"),
                    }
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
        log.error("edms_request_failed", tool=tool_name, error=str(exc), elapsed_ms=elapsed_ms)
        return _err("REQUEST_FAILED", f"Ошибка запроса: {exc!s}")

    return _err("MAX_RETRIES", "Превышено количество попыток запроса")


def _log_call(tool_name: str, params: dict[str, Any]) -> float:
    """Логирует вызов инструмента. Возвращает timestamp начала."""
    safe_params = {
        k: "***" if k in {"token", "password", "secret", "api_key", "authorization"} else v
        for k, v in params.items()
    }
    log.info("tool_called", tool=tool_name, params=safe_params)
    return time.monotonic()


def _log_result(tool_name: str, start_ts: float, success: bool) -> None:
    """Логирует результат вызова инструмента."""
    log.info(
        "tool_completed",
        tool=tool_name,
        elapsed_ms=round((time.monotonic() - start_ts) * 1000, 2),
        success=success,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MCP ИНСТРУМЕНТЫ
# Все инструменты используют @mcp.tool() — стандарт FastMCP.
# Параметры аннотированы типами Python (FastMCP строит схему автоматически).
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool(
    description=(
        "Получить документ по UUID или номеру (DOC-NNNN). "
        "Используй когда пользователь упоминает конкретный документ. "
        "Возвращает метаданные, статус, вложения и ответственных."
    )
)
async def get_document(
    document_id: str,
    include_history: bool = False,
    include_attachments: bool = True,
) -> dict[str, Any]:
    """
    Получить документ по ID.

    Args:
        document_id:         UUID документа или номер вида DOC-12345.
        include_history:     Включить историю изменений.
        include_attachments: Включить список вложений (по умолчанию True).
    """
    ts = _log_call("get_document", {"document_id": document_id})

    if not document_id or not document_id.strip():
        _log_result("get_document", ts, False)
        return _err("INVALID_REQUEST", "document_id не может быть пустым")

    params: dict[str, Any] = {}
    if include_history:
        params["include_history"] = "true"
    if not include_attachments:
        params["include_attachments"] = "false"

    result = await _request("GET", f"/documents/{document_id.strip()}", params=params, tool_name="get_document")
    _log_result("get_document", ts, result["success"])
    return result


@mcp.tool(
    description=(
        "Поиск документов по фильтрам: статус, тип, автор, дата, текст. "
        "Используй для запросов 'найди договоры за прошлый месяц', "
        "'покажи документы на согласовании у Иванова'. "
        "Поддерживает пагинацию через page/page_size."
    )
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
    Поиск документов.

    Args:
        query:         Полнотекстовый поиск по заголовку и содержимому.
        status:        Список статусов: draft, review, approved, rejected, signed, archived.
        document_type: Список типов: договор, приказ, акт, счёт, протокол, спецификация.
        author_id:     UUID автора.
        assignee_id:   UUID ответственного.
        department:    Отдел/подразделение.
        date_from:     Дата создания от (ISO 8601).
        date_to:       Дата создания до (ISO 8601).
        page:          Номер страницы (от 1).
        page_size:     Записей на странице (1–100).
        sort_by:       Поле сортировки: created_at, updated_at, title, status.
        sort_order:    Направление: asc, desc.
    """
    ts = _log_call("search_documents", {"query": query, "status": status})

    if page < 1:
        return _err("INVALID_REQUEST", "page должен быть >= 1")
    if not 1 <= page_size <= 100:
        return _err("INVALID_REQUEST", "page_size должен быть от 1 до 100")

    params: dict[str, Any] = {"page": page, "page_size": page_size, "sort_by": sort_by, "sort_order": sort_order}
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

    result = await _request("GET", "/documents", params=params, tool_name="search_documents")
    _log_result("search_documents", ts, result["success"])
    return result


@mcp.tool(
    description=(
        "Создать новый документ. "
        "ВНИМАНИЕ: необратимая операция. "
        "Используй ТОЛЬКО при явном запросе пользователя. "
        "Обязательно подтверди параметры перед вызовом."
    )
)
async def create_document(
    title: str,
    document_type: str,
    content: str | None = None,
    assignees: list[str] | None = None,
    department: str | None = None,
    due_date: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Создать новый документ в системе EDMS.

    Args:
        title:         Название документа (3–500 символов).
        document_type: Тип: договор, приказ, акт, счёт, протокол, спецификация.
        content:       Содержимое/описание документа.
        assignees:     UUID ответственных сотрудников.
        department:    Отдел-владелец.
        due_date:      Срок исполнения (ISO 8601).
        tags:          Теги для категоризации.
    """
    ts = _log_call("create_document", {"title": title, "document_type": document_type})

    if not title or len(title.strip()) < 3:
        _log_result("create_document", ts, False)
        return _err("INVALID_REQUEST", "Название документа должно содержать минимум 3 символа")

    _VALID_TYPES = {"договор", "приказ", "акт", "счёт", "протокол", "спецификация"}
    if document_type not in _VALID_TYPES:
        _log_result("create_document", ts, False)
        return _err("INVALID_REQUEST", f"Тип документа должен быть одним из: {', '.join(sorted(_VALID_TYPES))}")

    body: dict[str, Any] = {"title": title.strip(), "type": document_type}
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

    result = await _request("POST", "/documents", json_body=body, tool_name="create_document")
    _log_result("create_document", ts, result["success"])
    return result


@mcp.tool(
    description=(
        "Изменить статус документа: согласование, подписание, архивирование. "
        "ВНИМАНИЕ: переход в rejected или archived необратим. "
        "Требует явного подтверждения при деструктивных переходах."
    )
)
async def update_document_status(
    document_id: str,
    new_status: str,
    comment: str | None = None,
    notify_assignees: bool = True,
) -> dict[str, Any]:
    """
    Изменить статус документа.

    Args:
        document_id:      UUID документа.
        new_status:       Новый статус: draft, review, approved, rejected, signed, archived.
        comment:          Комментарий (обязателен для rejected и archived).
        notify_assignees: Уведомить ответственных об изменении.
    """
    ts = _log_call("update_document_status", {"document_id": document_id, "new_status": new_status})

    _VALID_STATUSES = {"draft", "review", "approved", "rejected", "signed", "archived"}
    if new_status not in _VALID_STATUSES:
        _log_result("update_document_status", ts, False)
        return _err("INVALID_REQUEST", f"Статус должен быть одним из: {', '.join(sorted(_VALID_STATUSES))}")

    if new_status in {"rejected", "archived"} and not comment:
        _log_result("update_document_status", ts, False)
        return _err(
            "COMMENT_REQUIRED",
            f"Для перевода в статус '{new_status}' необходим комментарий с обоснованием",
        )

    body: dict[str, Any] = {"status": new_status, "notify_assignees": notify_assignees}
    if comment:
        body["comment"] = comment.strip()

    result = await _request(
        "PATCH",
        f"/documents/{document_id}/status",
        json_body=body,
        tool_name="update_document_status",
    )
    _log_result("update_document_status", ts, result["success"])
    return result


@mcp.tool(
    description=(
        "Получить историю изменений документа: кто, когда и что менял. "
        "Используй для аудита, проверки действий, "
        "восстановления хронологии событий."
    )
)
async def get_document_history(
    document_id: str,
    event_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Получить журнал событий документа.

    Args:
        document_id:  UUID документа.
        event_types:  Типы событий: status_change, edit, comment, assignment, view, download.
        date_from:    Дата от (ISO 8601).
        date_to:      Дата до (ISO 8601).
        limit:        Максимальное количество событий (1–500).
    """
    ts = _log_call("get_document_history", {"document_id": document_id})

    if not 1 <= limit <= 500:
        return _err("INVALID_REQUEST", "limit должен быть от 1 до 500")

    params: dict[str, Any] = {"limit": limit}
    if event_types:
        params["event_types"] = ",".join(event_types)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to

    result = await _request(
        "GET",
        f"/documents/{document_id}/history",
        params=params,
        tool_name="get_document_history",
    )
    _log_result("get_document_history", ts, result["success"])
    return result


@mcp.tool(
    description=(
        "Назначить ответственных за документ с ролями. "
        "Роли: reviewer (проверяющий), approver (согласующий), "
        "signer (подписант), observer (наблюдатель). "
        "Используй для запросов 'передай документ', 'добавь в согласующие'."
    )
)
async def assign_document(
    document_id: str,
    assignees: list[dict[str, Any]],
    replace_existing: bool = False,
    message: str | None = None,
) -> dict[str, Any]:
    """
    Назначить ответственных за документ.

    Args:
        document_id:      UUID документа.
        assignees:        Список [{user_id: str, role: str, due_date?: str}].
                          role: reviewer | approver | signer | observer.
        replace_existing: Заменить существующих назначенных.
        message:          Сообщение назначаемым (до 1000 символов).
    """
    ts = _log_call("assign_document", {"document_id": document_id, "assignees_count": len(assignees)})

    if not assignees:
        _log_result("assign_document", ts, False)
        return _err("INVALID_REQUEST", "Список assignees не может быть пустым")

    _VALID_ROLES = {"reviewer", "approver", "signer", "observer"}
    for a in assignees:
        if not a.get("user_id"):
            return _err("INVALID_REQUEST", "Каждый assignee должен содержать user_id")
        if a.get("role") not in _VALID_ROLES:
            return _err(
                "INVALID_REQUEST",
                f"Роль '{a.get('role')}' недопустима. Допустимые: {', '.join(sorted(_VALID_ROLES))}",
            )

    body: dict[str, Any] = {
        "assignees": assignees,
        "replace_existing": replace_existing,
    }
    if message:
        body["message"] = message[:1000]

    result = await _request(
        "POST",
        f"/documents/{document_id}/assignees",
        json_body=body,
        tool_name="assign_document",
    )
    _log_result("assign_document", ts, result["success"])
    return result


@mcp.tool(
    description=(
        "Аналитика по документам: статистика, нагрузка, просрочки, "
        "коэффициент одобрения. "
        "Используй для запросов об отчётах, дашбордах, метриках эффективности."
    )
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
    Получить аналитические данные.

    Args:
        metric_type:   Тип метрики: status_distribution, volume_by_type,
                       processing_time, workload_by_user, workload_by_department,
                       overdue_documents, approval_rate.
        date_from:     Начало периода (ISO 8601).
        date_to:       Конец периода (ISO 8601).
        group_by:      Группировка: day, week, month, quarter.
        department:    Фильтр по отделу.
        document_type: Фильтр по типу документа.
    """
    ts = _log_call("get_analytics", {"metric_type": metric_type})

    _VALID_METRICS = {
        "status_distribution", "volume_by_type", "processing_time",
        "workload_by_user", "workload_by_department",
        "overdue_documents", "approval_rate",
    }
    if metric_type not in _VALID_METRICS:
        _log_result("get_analytics", ts, False)
        return _err(
            "INVALID_REQUEST",
            f"metric_type должен быть одним из: {', '.join(sorted(_VALID_METRICS))}",
        )

    _VALID_GROUP_BY = {"day", "week", "month", "quarter"}
    if group_by not in _VALID_GROUP_BY:
        return _err("INVALID_REQUEST", f"group_by должен быть: {', '.join(sorted(_VALID_GROUP_BY))}")

    params: dict[str, Any] = {"metric": metric_type, "group_by": group_by}
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    if department:
        params["department"] = department
    if document_type:
        params["type"] = document_type

    result = await _request("GET", "/analytics", params=params, tool_name="get_analytics")
    _log_result("get_analytics", ts, result["success"])
    return result


@mcp.tool(
    description=(
        "Статус рабочего процесса документа: кто должен действовать, "
        "просрочки, прогресс согласования. "
        "Используй для 'где застрял документ', 'кто ещё не согласовал'."
    )
)
async def get_workflow_status(
    document_id: str,
    include_completed: bool = False,
) -> dict[str, Any]:
    """
    Получить текущий статус рабочего процесса.

    Args:
        document_id:        UUID документа.
        include_completed:  Включить завершённые шаги.
    """
    ts = _log_call("get_workflow_status", {"document_id": document_id})

    params: dict[str, Any] = {}
    if include_completed:
        params["include_completed"] = "true"

    result = await _request(
        "GET",
        f"/documents/{document_id}/workflow",
        params=params,
        tool_name="get_workflow_status",
    )
    _log_result("get_workflow_status", ts, result["success"])
    return result


# ── Health endpoint (FastMCP встроенный) ──────────────────────────────────
# FastMCP автоматически добавляет /health эндпоинт
