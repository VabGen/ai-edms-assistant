"""
EDMS MCP Server — Model Context Protocol server for EDMS document management.

Provides standardized tools for AI agents to interact with the EDMS REST API.
All tools follow MCP spec and return structured, human-readable results.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
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

# ── Configuration ─────────────────────────────────────────────────────────────
EDMS_BASE_URL = os.getenv("EDMS_API_URL", "http://localhost:8098")
EDMS_API_KEY = os.getenv("EDMS_API_KEY", "")
REQUEST_TIMEOUT = int(os.getenv("EDMS_TIMEOUT", "30"))

# ── FastMCP instance ──────────────────────────────────────────────────────────
mcp = FastMCP(
    name="EDMS Assistant MCP Server",
    version="1.0.0",
    description="Инструменты для работы с системой электронного документооборота (EDMS/СЭД). "
                "Предоставляет доступ к документам, сотрудникам, поручениям и рабочим процессам.",
)


# ── HTTP Client helper ────────────────────────────────────────────────────────

def _build_headers(token: str | None = None) -> dict[str, str]:
    """Build authorization headers for EDMS API requests."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif EDMS_API_KEY:
        headers["Authorization"] = f"Bearer {EDMS_API_KEY}"
    return headers


def _format_error(exc: Exception, operation: str) -> str:
    """Format an exception into a user-friendly error message."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        msg_map = {
            400: "Некорректные параметры запроса",
            401: "Ошибка авторизации — токен недействителен или истёк",
            403: "Недостаточно прав доступа для выполнения операции",
            404: "Запрошенный ресурс не найден в системе",
            422: "Ошибка валидации данных",
            500: "Внутренняя ошибка сервера EDMS",
            503: "Сервис EDMS временно недоступен",
        }
        detail = msg_map.get(code, f"HTTP ошибка {code}")
        return json.dumps({
            "success": False,
            "error": detail,
            "operation": operation,
            "http_status": code,
        }, ensure_ascii=False)
    if isinstance(exc, httpx.ConnectError):
        return json.dumps({
            "success": False,
            "error": f"Не удалось подключиться к EDMS API: {EDMS_BASE_URL}",
            "operation": operation,
        }, ensure_ascii=False)
    if isinstance(exc, httpx.TimeoutException):
        return json.dumps({
            "success": False,
            "error": "Превышено время ожидания ответа от EDMS API",
            "operation": operation,
        }, ensure_ascii=False)
    return json.dumps({
        "success": False,
        "error": f"Непредвиденная ошибка: {exc!s}",
        "operation": operation,
    }, ensure_ascii=False)


async def _get(path: str, params: dict | None = None, token: str | None = None) -> Any:
    """Perform authenticated GET request to EDMS API."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.get(
            f"{EDMS_BASE_URL}/{path.lstrip('/')}",
            params=params,
            headers=_build_headers(token),
        )
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()


async def _post(path: str, body: dict, token: str | None = None) -> Any:
    """Perform authenticated POST request to EDMS API."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{EDMS_BASE_URL}/{path.lstrip('/')}",
            json=body,
            headers=_build_headers(token),
        )
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return {"success": True}
        return resp.json()


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def get_document(
    document_id: str,
    token: str = "",
) -> str:
    """
    Получить полную информацию о документе по его UUID.

    Используй когда:
    - Пользователь спрашивает «что за документ», «покажи документ»
    - Нужны метаданные: регномер, дата, статус, автор, вложения
    - Перед выполнением операций с документом (согласование, поручение)

    Параметры:
    - document_id: UUID документа в формате xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    - token: JWT токен авторизации пользователя

    Возвращает: JSON с полными метаданными документа
    """
    logger.info("get_document called: doc_id=%s...", document_id[:8])
    try:
        data = await _get(
            f"api/document/{document_id}",
            params={"includes": ["DOCUMENT_TYPE", "CORRESPONDENT", "REGISTRATION_JOURNAL"]},
            token=token,
        )
        return json.dumps({"success": True, "document": data}, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("get_document error: %s", exc)
        return _format_error(exc, "get_document")


@mcp.tool()
async def search_documents(
    query: str = "",
    category: str = "",
    status: str = "",
    author_last_name: str = "",
    date_from: str = "",
    date_to: str = "",
    page: int = 0,
    size: int = 10,
    token: str = "",
) -> str:
    """
    Поиск документов в EDMS по различным критериям.

    Используй когда:
    - Пользователь просит найти документы («найди договоры», «входящие за март»)
    - Нужен список документов по фильтрам
    - Поиск по содержимому или атрибутам

    Параметры:
    - query: Текст для поиска в содержимом и заголовке
    - category: Категория (INCOMING, OUTGOING, INTERN, APPEAL, CONTRACT, MEETING)
    - status: Статус (FORMING, REGISTRATION, IN_PROGRESS, COMPLETED, CANCELLED)
    - author_last_name: Фамилия автора
    - date_from: Начало периода регистрации (YYYY-MM-DD)
    - date_to: Конец периода регистрации (YYYY-MM-DD)
    - page: Номер страницы (начиная с 0)
    - size: Размер страницы (max 50)
    - token: JWT токен

    Возвращает: JSON список документов с пагинацией
    """
    logger.info("search_documents: query='%s' category='%s'", query, category)
    try:
        params: dict[str, Any] = {"page": page, "size": min(size, 50)}
        if query:
            params["shortSummary"] = query
        if category:
            params["categoryConstants"] = [category]
        if status:
            params["status"] = [status]
        if author_last_name:
            params["authorLastName"] = author_last_name
        if date_from:
            params["dateRegStart"] = f"{date_from}T00:00:00"
        if date_to:
            params["dateRegEnd"] = f"{date_to}T23:59:59"

        data = await _get("api/document", params=params, token=token)
        items = data.get("content", data if isinstance(data, list) else [])
        return json.dumps({
            "success": True,
            "total": data.get("totalElements", len(items)),
            "page": page,
            "documents": items,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("search_documents error: %s", exc)
        return _format_error(exc, "search_documents")


@mcp.tool()
async def get_document_history(
    document_id: str,
    token: str = "",
) -> str:
    """
    Получить историю движения документа (кто, что и когда сделал).

    Используй когда:
    - Пользователь спрашивает «что происходило с документом»
    - Нужно отследить этапы согласования или исполнения
    - Требуется аудит действий по документу

    Параметры:
    - document_id: UUID документа
    - token: JWT токен
    """
    logger.info("get_document_history: doc_id=%s...", document_id[:8])
    try:
        data = await _get(f"api/document/{document_id}/history/v2", token=token)
        items = data if isinstance(data, list) else data.get("content", [])
        return json.dumps({"success": True, "history": items}, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("get_document_history error: %s", exc)
        return _format_error(exc, "get_document_history")


@mcp.tool()
async def get_document_versions(
    document_id: str,
    token: str = "",
) -> str:
    """
    Получить все версии документа и сравнить изменения между ними.

    Используй когда:
    - Пользователь спрашивает об изменениях в документе
    - Нужно сравнить версии («что изменилось», «история изменений»)

    Параметры:
    - document_id: UUID документа
    - token: JWT токен
    """
    logger.info("get_document_versions: doc_id=%s...", document_id[:8])
    try:
        data = await _get(f"api/document/{document_id}/version", token=token)
        return json.dumps({"success": True, "versions": data}, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("get_document_versions error: %s", exc)
        return _format_error(exc, "get_document_versions")


@mcp.tool()
async def get_document_statistics(
    token: str = "",
) -> str:
    """
    Получить статистику документов текущего пользователя.

    Используй когда:
    - Пользователь спрашивает сводку («сколько документов», «что у меня на исполнении»)
    - Нужна дашборд-информация

    Параметры:
    - token: JWT токен пользователя
    """
    logger.info("get_document_statistics called")
    try:
        executor, control, author = await asyncio.gather(
            _get("api/document/stat/user-executor", token=token),
            _get("api/document/stat/user-control", token=token),
            _get("api/document/stat/user-author", token=token),
            return_exceptions=True,
        )
        return json.dumps({
            "success": True,
            "stats": {
                "executor": executor if not isinstance(executor, Exception) else None,
                "control": control if not isinstance(control, Exception) else None,
                "author": author if not isinstance(author, Exception) else None,
            },
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("get_document_statistics error: %s", exc)
        return _format_error(exc, "get_document_statistics")


# ═══════════════════════════════════════════════════════════════════════════════
# EMPLOYEE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def search_employees(
    last_name: str = "",
    first_name: str = "",
    department_name: str = "",
    post_name: str = "",
    active_only: bool = True,
    page: int = 0,
    size: int = 20,
    token: str = "",
) -> str:
    """
    Поиск сотрудников в организации.

    Используй когда:
    - Нужно найти сотрудника для назначения исполнителем
    - Пользователь называет фамилию («найди Иванова», «кто такой Петров»)
    - Нужен список сотрудников отдела

    Параметры:
    - last_name: Фамилия (поддерживает частичное совпадение)
    - first_name: Имя
    - department_name: Название отдела/подразделения
    - post_name: Название должности
    - active_only: Только активные сотрудники (True по умолчанию)
    - page, size: Пагинация
    - token: JWT токен
    """
    logger.info("search_employees: last_name='%s' dept='%s'", last_name, department_name)
    try:
        body: dict[str, Any] = {
            "includes": ["POST", "DEPARTMENT"],
            "page": page,
            "size": min(size, 50),
        }
        if last_name:
            body["lastName"] = last_name
        if first_name:
            body["firstName"] = first_name
        if active_only:
            body["active"] = True

        data = await _post("api/employee/search", body=body, token=token)
        items = data.get("content", data if isinstance(data, list) else [])
        return json.dumps({
            "success": True,
            "total": data.get("totalElements", len(items)),
            "employees": items,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("search_employees error: %s", exc)
        return _format_error(exc, "search_employees")


@mcp.tool()
async def get_current_user(token: str = "") -> str:
    """
    Получить информацию о текущем авторизованном пользователе.

    Используй когда:
    - Нужно узнать кто вошёл в систему
    - Требуется UUID текущего пользователя для операций
    - Пользователь говорит «мои документы», «добавь меня»

    Параметры:
    - token: JWT токен
    """
    logger.info("get_current_user called")
    try:
        data = await _get("api/employee/me", token=token)
        return json.dumps({"success": True, "user": data}, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("get_current_user error: %s", exc)
        return _format_error(exc, "get_current_user")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK / ASSIGNMENT TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def create_task(
    document_id: str,
    task_text: str,
    executor_ids: list[str],
    deadline: str = "",
    task_type: str = "GENERAL",
    token: str = "",
) -> str:
    """
    Создать поручение по документу.

    Используй когда:
    - Пользователь говорит «создай поручение», «поручи выполнить», «поставь задачу»
    - Нужно назначить исполнителя на задачу по документу

    Параметры:
    - document_id: UUID документа
    - task_text: Текст поручения (обязательно, не пустой)
    - executor_ids: Список UUID сотрудников-исполнителей (минимум 1)
    - deadline: Срок выполнения в формате YYYY-MM-DDTHH:MM:SSZ (необязательно)
    - task_type: Тип поручения (GENERAL, PROJECT, CONTROL)
    - token: JWT токен

    Ограничения:
    - Минимум 1 исполнитель
    - Первый исполнитель автоматически становится ответственным
    """
    logger.info("create_task: doc_id=%s... executors=%d", document_id[:8], len(executor_ids))
    if not task_text.strip():
        return json.dumps({"success": False, "error": "Текст поручения не может быть пустым"}, ensure_ascii=False)
    if not executor_ids:
        return json.dumps({"success": False, "error": "Необходимо указать хотя бы одного исполнителя"}, ensure_ascii=False)

    try:
        executors = [
            {"employeeId": eid, "responsible": (i == 0)}
            for i, eid in enumerate(executor_ids)
        ]
        task = {
            "taskText": task_text,
            "type": task_type,
            "executors": executors,
            "periodTask": False,
            "endless": not bool(deadline),
        }
        if deadline:
            task["planedDateEnd"] = deadline

        data = await _post(f"api/document/{document_id}/task/batch", body=[task], token=token)
        return json.dumps({
            "success": True,
            "message": f"Поручение успешно создано для {len(executor_ids)} сотрудника(-ов)",
            "result": data,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("create_task error: %s", exc)
        return _format_error(exc, "create_task")


@mcp.tool()
async def create_introduction(
    document_id: str,
    employee_ids: list[str],
    comment: str = "",
    token: str = "",
) -> str:
    """
    Добавить сотрудников в список ознакомления с документом.

    Используй когда:
    - Пользователь говорит «ознакомь», «добавь в ознакомление», «отправь на ознакомление»
    - Нужно разослать документ для ознакомления

    Параметры:
    - document_id: UUID документа
    - employee_ids: Список UUID сотрудников для ознакомления
    - comment: Комментарий к ознакомлению (необязательно)
    - token: JWT токен
    """
    logger.info("create_introduction: doc_id=%s... employees=%d", document_id[:8], len(employee_ids))
    if not employee_ids:
        return json.dumps({"success": False, "error": "Необходимо указать сотрудников для ознакомления"}, ensure_ascii=False)
    try:
        body = {"executorListIds": employee_ids, "comment": comment or ""}
        await _post(f"api/document/{document_id}/introduction", body=body, token=token)
        return json.dumps({
            "success": True,
            "message": f"Добавлено {len(employee_ids)} сотрудника(-ов) в список ознакомления",
        }, ensure_ascii=False)
    except Exception as exc:
        logger.error("create_introduction error: %s", exc)
        return _format_error(exc, "create_introduction")


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW / LIFECYCLE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def execute_document_operation(
    document_id: str,
    operation_type: str,
    comment: str = "",
    token: str = "",
) -> str:
    """
    Выполнить операцию над документом (согласование, подписание, отклонение и др.).

    Используй когда:
    - Пользователь говорит «согласуй», «подпиши», «отклони», «одобри»
    - Нужно изменить состояние документа в рамках процесса

    Параметры:
    - document_id: UUID документа
    - operation_type: Тип операции:
        AGREE — согласование
        SIGN — подписание
        REJECT — отклонение
        REVIEW — рассмотрение
        REGISTER — регистрация
        INTRODUCE — ознакомление
        APPROVE — утверждение
        EXECUTE — исполнение
    - comment: Комментарий к операции (рекомендуется при отклонении)
    - token: JWT токен

    Ограничения:
    - Пользователь должен иметь право на выполнение операции
    - Операция должна соответствовать текущему этапу процесса
    """
    logger.info("execute_document_operation: doc_id=%s... op=%s", document_id[:8], operation_type)
    valid_ops = {"AGREE", "SIGN", "REJECT", "REVIEW", "REGISTER", "INTRODUCE", "APPROVE", "EXECUTE", "CANCEL"}
    if operation_type not in valid_ops:
        return json.dumps({
            "success": False,
            "error": f"Недопустимая операция '{operation_type}'. Допустимые: {', '.join(sorted(valid_ops))}",
        }, ensure_ascii=False)
    try:
        op: dict[str, Any] = {"operationType": operation_type}
        if comment:
            op["comment"] = comment
        await _post(f"api/document/{document_id}/execute", body=[op], token=token)
        op_names = {
            "AGREE": "Документ согласован",
            "SIGN": "Документ подписан",
            "REJECT": "Документ отклонён",
            "REVIEW": "Документ рассмотрен",
            "REGISTER": "Документ зарегистрирован",
            "INTRODUCE": "Ознакомление выполнено",
            "APPROVE": "Документ утверждён",
            "EXECUTE": "Документ исполнен",
            "CANCEL": "Документ отменён",
        }
        return json.dumps({
            "success": True,
            "message": op_names.get(operation_type, f"Операция {operation_type} выполнена"),
        }, ensure_ascii=False)
    except Exception as exc:
        logger.error("execute_document_operation error: %s", exc)
        return _format_error(exc, "execute_document_operation")


@mcp.tool()
async def start_document_routing(
    document_id: str,
    token: str = "",
) -> str:
    """
    Запустить маршрут документа (отправить на согласование/подписание).

    Используй когда:
    - Пользователь говорит «запусти документ», «отправь на согласование», «запусти маршрут»
    - Документ в статусе FORMING и готов к отправке

    Параметры:
    - document_id: UUID документа
    - token: JWT токен
    """
    logger.info("start_document_routing: doc_id=%s...", document_id[:8])
    try:
        await _post("api/document/start", body={"id": document_id}, token=token)
        return json.dumps({
            "success": True,
            "message": "Документ отправлен на маршрут согласования/подписания",
        }, ensure_ascii=False)
    except Exception as exc:
        logger.error("start_document_routing error: %s", exc)
        return _format_error(exc, "start_document_routing")


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def set_document_control(
    document_id: str,
    control_type_id: str,
    deadline: str,
    controller_id: str = "",
    token: str = "",
) -> str:
    """
    Поставить документ на контроль.

    Используй когда:
    - Пользователь говорит «поставь на контроль», «установи контроль»
    - Нужно отслеживать исполнение документа

    Параметры:
    - document_id: UUID документа
    - control_type_id: UUID типа контроля
    - deadline: Дата снятия с контроля (YYYY-MM-DDTHH:MM:SS)
    - controller_id: UUID сотрудника-контролёра (необязательно)
    - token: JWT токен
    """
    logger.info("set_document_control: doc_id=%s...", document_id[:8])
    try:
        body: dict[str, Any] = {
            "controlTypeId": control_type_id,
            "dateControlEnd": deadline,
        }
        if controller_id:
            body["controlEmployeeId"] = controller_id
        data = await _post(f"api/document/{document_id}/control", body=body, token=token)
        return json.dumps({
            "success": True,
            "message": f"Документ поставлен на контроль до {deadline[:10]}",
            "control": data,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("set_document_control error: %s", exc)
        return _format_error(exc, "set_document_control")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTIFICATION TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def send_notification(
    document_id: str,
    recipient_ids: list[str],
    message: str,
    notification_type: str = "REMINDER",
    deadline: str = "",
    token: str = "",
) -> str:
    """
    Отправить уведомление или напоминание сотрудникам по документу.

    Используй когда:
    - Пользователь говорит «напомни», «уведоми», «предупреди»
    - Нужно напомнить о сроке исполнения

    Параметры:
    - document_id: UUID документа
    - recipient_ids: Список UUID получателей
    - message: Текст уведомления
    - notification_type: Тип (REMINDER, DEADLINE, CUSTOM)
    - deadline: Дедлайн в ISO 8601 (необязательно)
    - token: JWT токен
    """
    logger.info("send_notification: doc_id=%s... recipients=%d", document_id[:8], len(recipient_ids))
    if not recipient_ids:
        return json.dumps({"success": False, "error": "Необходимо указать получателей"}, ensure_ascii=False)
    if not message.strip():
        return json.dumps({"success": False, "error": "Текст уведомления не может быть пустым"}, ensure_ascii=False)
    try:
        body: dict[str, Any] = {
            "recipientIds": recipient_ids,
            "type": notification_type,
            "message": message.strip(),
        }
        if deadline:
            body["deadline"] = deadline
        await _post(f"api/document/{document_id}/notification", body=body, token=token)
        return json.dumps({
            "success": True,
            "message": f"Уведомление отправлено {len(recipient_ids)} сотруднику(-ам)",
        }, ensure_ascii=False)
    except Exception as exc:
        logger.error("send_notification error: %s", exc)
        return _format_error(exc, "send_notification")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def get_reference_data(
    reference_type: str,
    search_query: str = "",
    token: str = "",
) -> str:
    """
    Получить данные из справочников EDMS (типы документов, отделы, должности и т.д.).

    Используй когда:
    - Нужен список доступных значений для параметров
    - Пользователь спрашивает о типах, категориях, статусах

    Параметры:
    - reference_type: Тип справочника:
        document_types — виды документов
        departments — подразделения
        delivery_methods — способы доставки
        control_types — типы контроля
        citizen_types — виды обращений
    - search_query: Поиск по названию (необязательно)
    - token: JWT токен
    """
    logger.info("get_reference_data: type='%s' query='%s'", reference_type, search_query)
    endpoints = {
        "document_types": "api/document-type",
        "departments": "api/department",
        "delivery_methods": "api/delivery-method",
        "control_types": "api/control-type",
        "citizen_types": "api/citizen-type",
    }
    if reference_type not in endpoints:
        return json.dumps({
            "success": False,
            "error": f"Неизвестный справочник '{reference_type}'. "
                     f"Доступные: {', '.join(endpoints.keys())}",
        }, ensure_ascii=False)
    try:
        params = {}
        if search_query:
            params["fts"] = search_query
        data = await _get(endpoints[reference_type], params=params, token=token)
        items = data if isinstance(data, list) else data.get("content", [])
        return json.dumps({
            "success": True,
            "reference_type": reference_type,
            "items": items,
        }, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error("get_reference_data error: %s", exc)
        return _format_error(exc, "get_reference_data")


@mcp.tool()
async def health_check() -> str:
    """
    Проверить доступность EDMS API и MCP сервера.

    Используй когда:
    - Нужно проверить работоспособность системы
    - Диагностика проблем с подключением
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{EDMS_BASE_URL}/actuator/health")
            edms_ok = resp.status_code < 400
    except Exception:
        edms_ok = False

    return json.dumps({
        "success": True,
        "mcp_server": "healthy",
        "edms_api": "healthy" if edms_ok else "unavailable",
        "edms_url": EDMS_BASE_URL,
        "timestamp": datetime.now().isoformat(),
    }, ensure_ascii=False)


# ── Entry point ───────────────────────────────────────────────────────────────
import asyncio

if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
