"""
edms_mcp_server.py — MCP-сервер для EDMS AI Ассистента.

Предоставляет 15 инструментов для работы с корпоративной СЭД через FastMCP.
Все вызовы проксируются к REST API EDMS.

Логирование: edms_mcp.log
Конфигурация: EDMS_API_URL, EDMS_API_KEY из переменных окружения.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("edms_mcp.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("edms_mcp")

# ── Config ────────────────────────────────────────────────────────────────────
EDMS_BASE_URL = os.getenv("EDMS_API_URL", "http://localhost:8098")
EDMS_API_KEY = os.getenv("EDMS_API_KEY", "")
HTTP_TIMEOUT = int(os.getenv("EDMS_TIMEOUT", "30"))

# ── FastMCP instance ──────────────────────────────────────────────────────────
mcp = FastMCP(
    name="EDMS MCP Server"
)


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _make_headers(token: str | None = None) -> dict[str, str]:
    """Формирует заголовки авторизации для EDMS API."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif EDMS_API_KEY:
        headers["Authorization"] = f"Bearer {EDMS_API_KEY}"
    return headers


async def _edms_request(
    method: str,
    endpoint: str,
    token: str | None = None,
    *,
    params: dict | None = None,
    json: dict | None = None,
    is_json_response: bool = True,
) -> dict[str, Any]:
    """
    Выполняет HTTP-запрос к EDMS API с обработкой ошибок.

    Returns:
        {"success": bool, "data": ..., "error": str | None, "status_code": int}
    """
    url = f"{EDMS_BASE_URL}/{endpoint.lstrip('/')}"
    headers = _make_headers(token)

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.request(
                method, url, headers=headers, params=params, json=json
            )
        logger.info("[EDMS] %s %s → %d", method, endpoint, resp.status_code)

        if resp.status_code == 204 or not resp.content:
            return {"success": True, "data": None, "status_code": resp.status_code}

        if not resp.is_success:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text[:500]
            logger.warning("[EDMS] Error %d: %s", resp.status_code, err_body)
            return {
                "success": False,
                "error": f"EDMS вернул {resp.status_code}: {err_body}",
                "status_code": resp.status_code,
            }

        if is_json_response:
            return {"success": True, "data": resp.json(), "status_code": resp.status_code}
        return {"success": True, "data": resp.text, "status_code": resp.status_code}

    except httpx.TimeoutException:
        logger.error("[EDMS] Timeout: %s %s", method, url)
        return {"success": False, "error": f"Таймаут при обращении к EDMS ({HTTP_TIMEOUT}с)"}
    except httpx.ConnectError:
        logger.error("[EDMS] Connection error: %s", url)
        return {"success": False, "error": "Не удалось подключиться к EDMS API"}
    except Exception as exc:
        logger.error("[EDMS] Unexpected error: %s", exc, exc_info=True)
        return {"success": False, "error": f"Внутренняя ошибка: {exc}"}


# ═══════════════════════════════════════════════════════════════════════════════
# ИНСТРУМЕНТЫ
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    description=(
        "Получить документ по UUID. "
        "Используй когда пользователь указал конкретный ID документа и хочет просмотреть его данные. "
        "Возвращает метаданные: номер, статус, автора, даты, тип, краткое содержание. "
        "Ограничение: требует валидный UUID в формате xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx."
    )
)
async def get_document(document_id: str, token: str) -> dict[str, Any]:
    """
    Получить документ по UUID.

    Args:
        document_id: UUID документа в формате xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        token: JWT токен авторизации пользователя
    """
    logger.info("[get_document] document_id=%s", document_id)
    result = await _edms_request("GET", f"api/document/{document_id}", token=token)
    if not result["success"]:
        return {"error": result["error"], "document_id": document_id}
    return {"success": True, "document": result["data"]}


@mcp.tool(
    description=(
        "Поиск документов по фильтрам. "
        "Используй для поиска документов по номеру, статусу, автору, дате, категории. "
        "Поддерживает пагинацию. Возвращает список документов с основными полями. "
        "Параметры фильтра передаются как JSON-объект."
    )
)
async def search_documents(
    token: str,
    filters: dict[str, Any] | None = None,
    page: int = 0,
    size: int = 10,
) -> dict[str, Any]:
    """
    Поиск документов.

    Args:
        token: JWT токен авторизации
        filters: Фильтры поиска: {"status": "REGISTERED", "regNumber": "ВХ-2026-001",
                 "dateFrom": "2026-01-01", "dateTo": "2026-12-31",
                 "category": "INCOMING", "authorLastName": "Иванов"}
        page: Номер страницы (0-based)
        size: Размер страницы (max 50)
    """
    logger.info("[search_documents] filters=%s page=%d size=%d", filters, page, size)
    params: dict[str, Any] = {"page": page, "size": min(size, 50)}
    if filters:
        params.update(filters)
    result = await _edms_request("GET", "api/document", token=token, params=params)
    if not result["success"]:
        return {"error": result["error"]}
    data = result["data"] or {}
    return {
        "success": True,
        "documents": data.get("content", []),
        "total": data.get("totalElements", 0),
        "page": data.get("number", page),
        "total_pages": data.get("totalPages", 1),
    }


@mcp.tool(
    description=(
        "Получить историю движения документа. "
        "Показывает все этапы согласования, подписания, отправки — кто, когда, что сделал. "
        "Используй когда пользователь спрашивает о статусе прохождения документа."
    )
)
async def get_document_history(document_id: str, token: str) -> dict[str, Any]:
    """
    Получить историю движения документа.

    Args:
        document_id: UUID документа
        token: JWT токен авторизации
    """
    logger.info("[get_document_history] document_id=%s", document_id)
    result = await _edms_request("GET", f"api/document/{document_id}/history/v2", token=token)
    if not result["success"]:
        return {"error": result["error"]}
    return {"success": True, "history": result["data"] or []}


@mcp.tool(
    description=(
        "Получить список вложений документа. "
        "Возвращает имена файлов, размеры, даты загрузки, типы вложений. "
        "Используй перед скачиванием или анализом вложений."
    )
)
async def get_document_attachments(document_id: str, token: str) -> dict[str, Any]:
    """
    Получить список вложений документа.

    Args:
        document_id: UUID документа
        token: JWT токен авторизации
    """
    logger.info("[get_document_attachments] document_id=%s", document_id)
    result = await _edms_request("GET", f"api/document/{document_id}", token=token)
    if not result["success"]:
        return {"error": result["error"]}
    doc = result["data"] or {}
    attachments = doc.get("attachmentDocument", [])
    return {
        "success": True,
        "document_id": document_id,
        "attachments": [
            {
                "id": str(a.get("id", "")),
                "name": a.get("name") or a.get("originalName", ""),
                "size_bytes": a.get("size", 0),
                "upload_date": str(a.get("uploadDate", "")),
                "type": str(a.get("attachmentDocumentType", {}).get("name", "") if isinstance(a.get("attachmentDocumentType"), dict) else ""),
            }
            for a in (attachments or [])
        ],
    }


@mcp.tool(
    description=(
        "Создать поручение (задачу) по документу. "
        "Назначает исполнителя и устанавливает срок. "
        "Используй когда пользователь хочет создать задачу или поручение для сотрудника. "
        "Требует: UUID документа, текст поручения, UUID исполнителя, дату окончания (ISO 8601)."
    )
)
async def create_task(
    document_id: str,
    token: str,
    task_text: str,
    executor_employee_id: str,
    planned_end_date: str,
    task_type: str = "GENERAL",
) -> dict[str, Any]:
    """
    Создать поручение по документу.

    Args:
        document_id: UUID документа
        token: JWT токен авторизации
        task_text: Текст поручения
        executor_employee_id: UUID сотрудника-исполнителя
        planned_end_date: Дата окончания в формате ISO 8601 (2026-04-30T23:59:59Z)
        task_type: Тип: GENERAL | PROJECT | CONTROL
    """
    logger.info("[create_task] document_id=%s executor=%s", document_id, executor_employee_id)
    payload = {
        "taskText": task_text,
        "planedDateEnd": planned_end_date,
        "type": task_type,
        "periodTask": False,
        "endless": False,
        "executors": [{"employeeId": executor_employee_id, "responsible": True}],
    }
    result = await _edms_request(
        "POST", f"api/document/{document_id}/task/batch", token=token,
        json=[payload], is_json_response=False,
    )
    if not result["success"]:
        return {"error": result["error"]}
    return {"success": True, "message": "Поручение успешно создано"}


@mcp.tool(
    description=(
        "Обновить статус документа. "
        "Используй для изменения статуса (REGISTERED, IN_PROGRESS, COMPLETED и др.). "
        "Требует наличия прав у пользователя на изменение статуса."
    )
)
async def update_document_status(
    document_id: str,
    token: str,
    new_status: str,
    comment: str = "",
) -> dict[str, Any]:
    """
    Обновить статус документа.

    Args:
        document_id: UUID документа
        token: JWT токен авторизации
        new_status: Новый статус: REGISTERED | IN_PROGRESS | COMPLETED | CANCELLED | ARCHIVE
        comment: Комментарий к изменению статуса
    """
    logger.info("[update_document_status] document_id=%s status=%s", document_id, new_status)
    payload = [{"operationType": "CHANGE_STATUS", "body": {"status": new_status, "comment": comment}}]
    result = await _edms_request(
        "POST", f"api/document/{document_id}/execute", token=token,
        json=payload, is_json_response=False,
    )
    if not result["success"]:
        return {"error": result["error"]}
    return {"success": True, "message": f"Статус изменён на {new_status}"}


@mcp.tool(
    description=(
        "Поиск сотрудников по фамилии, имени или отделу. "
        "Используй для нахождения UUID сотрудника перед созданием поручений или ознакомлений. "
        "Возвращает: UUID, ФИО, должность, отдел."
    )
)
async def search_employees(
    token: str,
    last_name: str | None = None,
    first_name: str | None = None,
    department_name: str | None = None,
    page: int = 0,
    size: int = 20,
) -> dict[str, Any]:
    """
    Поиск сотрудников.

    Args:
        token: JWT токен авторизации
        last_name: Фамилия (частичный поиск)
        first_name: Имя (частичный поиск)
        department_name: Название отдела
        page: Страница результатов
        size: Количество результатов
    """
    logger.info("[search_employees] last_name=%s first_name=%s", last_name, first_name)
    params: dict[str, Any] = {
        "page": page, "size": min(size, 50),
        "includes": ["POST", "DEPARTMENT"],
    }
    if last_name:
        params["lastName"] = last_name
    if first_name:
        params["firstName"] = first_name

    result = await _edms_request("GET", "api/employee", token=token, params=params)
    if not result["success"]:
        return {"error": result["error"]}

    data = result["data"] or {}
    items = data.get("content", []) if isinstance(data, dict) else data
    employees = []
    for emp in items:
        employees.append({
            "id": str(emp.get("id", "")),
            "last_name": emp.get("lastName", ""),
            "first_name": emp.get("firstName", ""),
            "middle_name": emp.get("middleName", ""),
            "post": emp.get("post", {}).get("postName", "") if isinstance(emp.get("post"), dict) else "",
            "department": emp.get("department", {}).get("name", "") if isinstance(emp.get("department"), dict) else "",
            "active": emp.get("active", True),
        })
    return {"success": True, "employees": employees, "total": len(employees)}


@mcp.tool(
    description=(
        "Добавить сотрудников в лист ознакомления с документом. "
        "Используй когда пользователь хочет ознакомить сотрудников с документом. "
        "Требует список UUID сотрудников."
    )
)
async def create_introduction(
    document_id: str,
    token: str,
    employee_ids: list[str],
    comment: str = "",
) -> dict[str, Any]:
    """
    Добавить в лист ознакомления.

    Args:
        document_id: UUID документа
        token: JWT токен авторизации
        employee_ids: Список UUID сотрудников
        comment: Комментарий к ознакомлению
    """
    logger.info("[create_introduction] document_id=%s employees=%d", document_id, len(employee_ids))
    payload = {"executorListIds": employee_ids, "comment": comment}
    result = await _edms_request(
        "POST", f"api/document/{document_id}/introduction",
        token=token, json=payload, is_json_response=False,
    )
    if not result["success"]:
        return {"error": result["error"]}
    return {"success": True, "message": f"Добавлено {len(employee_ids)} сотрудников в лист ознакомления"}


@mcp.tool(
    description=(
        "Поставить документ на контроль. "
        "Устанавливает контрольный срок и назначает контролёра. "
        "Используй когда пользователь хочет взять документ под контроль."
    )
)
async def set_document_control(
    document_id: str,
    token: str,
    control_date_end: str,
    controller_employee_id: str | None = None,
) -> dict[str, Any]:
    """
    Поставить документ на контроль.

    Args:
        document_id: UUID документа
        token: JWT токен авторизации
        control_date_end: Срок контроля в формате ISO 8601
        controller_employee_id: UUID контролёра (опционально)
    """
    logger.info("[set_document_control] document_id=%s date=%s", document_id, control_date_end)
    payload: dict[str, Any] = {"dateControlEnd": control_date_end}
    if controller_employee_id:
        payload["controlEmployeeId"] = controller_employee_id
    result = await _edms_request(
        "POST", f"api/document/{document_id}/control",
        token=token, json=payload,
    )
    if not result["success"]:
        return {"error": result["error"]}
    return {"success": True, "message": "Документ поставлен на контроль", "control": result["data"]}


@mcp.tool(
    description=(
        "Снять документ с контроля. "
        "Используй когда контроль исполнения документа завершён."
    )
)
async def remove_document_control(document_id: str, token: str) -> dict[str, Any]:
    """
    Снять документ с контроля.

    Args:
        document_id: UUID документа
        token: JWT токен авторизации
    """
    logger.info("[remove_document_control] document_id=%s", document_id)
    result = await _edms_request(
        "PUT", "api/document/control", token=token,
        json={"id": document_id}, is_json_response=False,
    )
    if not result["success"]:
        return {"error": result["error"]}
    return {"success": True, "message": "Документ снят с контроля"}


@mcp.tool(
    description=(
        "Запустить документ в работу (начать маршрут согласования). "
        "Используй когда документ готов и нужно запустить процесс согласования. "
        "После запуска документ поступает на согласование первому участнику маршрута."
    )
)
async def start_document(document_id: str, token: str) -> dict[str, Any]:
    """
    Запустить документ в маршрут согласования.

    Args:
        document_id: UUID документа
        token: JWT токен авторизации
    """
    logger.info("[start_document] document_id=%s", document_id)
    result = await _edms_request(
        "POST", "api/document/start", token=token,
        json={"id": document_id}, is_json_response=False,
    )
    if not result["success"]:
        return {"error": result["error"]}
    return {"success": True, "message": "Документ запущен в маршрут согласования"}


@mcp.tool(
    description=(
        "Получить статистику по документам текущего пользователя. "
        "Показывает количество документов в работе, на контроле, созданных. "
        "Используй для общего обзора нагрузки пользователя."
    )
)
async def get_user_document_stats(token: str) -> dict[str, Any]:
    """
    Получить статистику документов пользователя.

    Args:
        token: JWT токен авторизации
    """
    logger.info("[get_user_document_stats]")
    import asyncio
    executor_res, control_res, author_res = await asyncio.gather(
        _edms_request("GET", "api/document/stat/user-executor", token=token),
        _edms_request("GET", "api/document/stat/user-control", token=token),
        _edms_request("GET", "api/document/stat/user-author", token=token),
    )
    return {
        "success": True,
        "stats": {
            "executor": executor_res.get("data") if executor_res["success"] else None,
            "control": control_res.get("data") if control_res["success"] else None,
            "author": author_res.get("data") if author_res["success"] else None,
        },
    }


@mcp.tool(
    description=(
        "Получить версии документа. "
        "Показывает историю версий с датами и авторами изменений. "
        "Используй когда пользователь интересуется изменениями в документе."
    )
)
async def get_document_versions(document_id: str, token: str) -> dict[str, Any]:
    """
    Получить версии документа.

    Args:
        document_id: UUID документа
        token: JWT токен авторизации
    """
    logger.info("[get_document_versions] document_id=%s", document_id)
    result = await _edms_request("GET", f"api/document/{document_id}/version", token=token)
    if not result["success"]:
        return {"error": result["error"]}
    return {"success": True, "versions": result["data"] or []}


@mcp.tool(
    description=(
        "Обновить поле документа (краткое содержание, примечание и др.). "
        "Используй для редактирования метаданных документа. "
        "Допустимые поля: shortSummary (заголовок ≤80 символов), note, pages."
    )
)
async def update_document_field(
    document_id: str,
    token: str,
    field_name: str,
    field_value: str,
) -> dict[str, Any]:
    """
    Обновить поле документа.

    Args:
        document_id: UUID документа
        token: JWT токен авторизации
        field_name: Имя поля: shortSummary | note | pages | additionalPages
        field_value: Новое значение поля
    """
    logger.info("[update_document_field] document_id=%s field=%s", document_id, field_name)
    allowed = {"shortSummary", "note", "pages", "additionalPages", "exemplarCount"}
    if field_name not in allowed:
        return {"error": f"Поле '{field_name}' не поддерживается. Допустимые: {sorted(allowed)}"}
    value: Any = field_value
    if field_name == "shortSummary" and len(field_value) > 80:
        value = field_value[:80]
    if field_name in ("pages", "additionalPages", "exemplarCount"):
        try:
            value = int(field_value)
        except ValueError:
            return {"error": f"Поле {field_name} должно быть числом"}
    payload = [{"operationType": "DOCUMENT_MAIN_FIELDS_UPDATE", "body": {field_name: value}}]
    result = await _edms_request(
        "POST", f"api/document/{document_id}/execute",
        token=token, json=payload, is_json_response=False,
    )
    if not result["success"]:
        return {"error": result["error"]}
    return {"success": True, "message": f"Поле {field_name} обновлено", "new_value": value}


@mcp.tool(
    description=(
        "Получить информацию о текущем пользователе (профиль из EDMS). "
        "Возвращает: ID, ФИО, должность, отдел. "
        "Используй для персонализации ответов и подстановки данных пользователя."
    )
)
async def get_current_user(token: str) -> dict[str, Any]:
    """
    Получить информацию о текущем пользователе.

    Args:
        token: JWT токен авторизации
    """
    logger.info("[get_current_user]")
    result = await _edms_request("GET", "api/employee/me", token=token)
    if not result["success"]:
        return {"error": result["error"]}
    data = result["data"] or {}
    return {
        "success": True,
        "user": {
            "id": str(data.get("id", "")),
            "last_name": data.get("lastName", ""),
            "first_name": data.get("firstName", ""),
            "middle_name": data.get("middleName", ""),
            "post": data.get("post", {}).get("postName", "") if isinstance(data.get("post"), dict) else "",
            "department": data.get("department", {}).get("name", "") if isinstance(data.get("department"), dict) else "",
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EDMS MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    logger.info("Starting EDMS MCP Server (transport=%s)", args.transport)
    if args.transport == "http":
        mcp.run(transport="http", host="0.0.0.0", port=args.port)
    else:
        mcp.run(transport="stdio")
