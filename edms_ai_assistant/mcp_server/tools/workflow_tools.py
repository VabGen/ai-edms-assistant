# mcp-server/tools/workflow_tools.py
"""
Инструменты рабочих процессов: поручения, ознакомления, сотрудники, уведомления.
Перенесены из edms_ai_assistant/tools/task.py, introduction.py,
employee_search.py, doc_notification.py, doc_update_field.py.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastmcp import FastMCP

from ..clients.base_client import EdmsHttpClient
from ..clients.document_client import DocumentClient
from ..clients.employee_client import EmployeeClient
from ..models.task_models import TaskType
from ..services.introduction_service import IntroductionService
from ..services.task_service import TaskService
from ..utils.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)


def register_workflow_tools(mcp: FastMCP) -> None:
    """Регистрирует инструменты рабочих процессов."""

    @mcp.tool(
        description=(
            "Создать поручение по документу с поддержкой disambiguation. "
            "Поиск исполнителей по фамилии. Если фамилия неоднозначна — "
            "возвращает requires_disambiguation со списком сотрудников для выбора. "
            "После выбора вызвать повторно с selected_employee_ids."
        )
    )
    async def task_create_tool(
        token: str,
        document_id: str,
        task_text: str,
        executor_last_names: list[str] | None = None,
        selected_employee_ids: list[str] | None = None,
        responsible_last_name: str | None = None,
        planed_date_end: str | None = None,
        task_type: str = "GENERAL",
    ) -> dict[str, Any]:
        """
        Создать поручение по документу.

        Args:
            token: JWT-токен.
            document_id: UUID документа.
            task_text: Текст поручения.
            executor_last_names: Фамилии исполнителей ['Иванов', 'Петров'].
            selected_employee_ids: UUID выбранных сотрудников (после disambiguation).
            responsible_last_name: Фамилия ответственного исполнителя.
            planed_date_end: Дедлайн ISO 8601 (по умолчанию +7 дней).
            task_type: GENERAL | PROJECT | CONTROL.
        """
        if not any([executor_last_names, selected_employee_ids]):
            return {"status": "error", "message": "Укажите executor_last_names или selected_employee_ids."}
        if not task_text or not task_text.strip():
            return {"status": "error", "message": "Текст поручения не может быть пустым."}

        deadline: datetime | None = None
        if planed_date_end:
            try:
                deadline = datetime.fromisoformat(planed_date_end.replace("Z", "+00:00"))
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=UTC)
            except ValueError as e:
                return {"status": "error", "message": f"Неверный формат даты: {e}"}

        try:
            tt = TaskType(task_type.upper()) if task_type else TaskType.GENERAL
        except ValueError:
            tt = TaskType.GENERAL

        try:
            async with TaskService() as service:

                if selected_employee_ids:
                    employee_uuids = [UUID(eid) for eid in selected_employee_ids]
                    result = await service.create_task_by_employee_ids(
                        token=token, document_id=document_id, task_text=task_text,
                        employee_ids=employee_uuids, planed_date_end=deadline, task_type=tt,
                    )
                else:
                    result = await service.create_task(
                        token=token, document_id=document_id, task_text=task_text,
                        executor_last_names=executor_last_names or [],
                        planed_date_end=deadline, responsible_last_name=responsible_last_name,
                        task_type=tt,
                    )

                if result.status == "requires_disambiguation":
                    flat_candidates: list[dict[str, Any]] = []
                    for group in result.ambiguous_matches or []:
                        for match in group.get("matches", []):
                            flat_candidates.append({
                                "id": match.get("id", ""),
                                "full_name": match.get("full_name", ""),
                                "post": match.get("post", ""),
                                "department": match.get("department", ""),
                            })
                    return {
                        "status": "requires_disambiguation",
                        "message": "⚠️ Найдено несколько сотрудников. Выберите нужного:",
                        "ambiguous_matches": flat_candidates,
                        "instruction": "Выберите сотрудника и вызовите инструмент с selected_employee_ids.",
                    }

                if result.success:
                    response: dict[str, Any] = {
                        "status": "success",
                        "message": f"✅ Поручение создано. Исполнителей: {result.created_count}",
                        "created_count": result.created_count,
                        "requires_reload": True,
                    }
                    if result.not_found_employees:
                        response["not_found"] = result.not_found_employees
                        response["message"] += f" Не найдено: {', '.join(result.not_found_employees)}."
                    return response

                return {"status": "error", "message": result.error_message}

        except Exception as e:
            logger.error("task_create_tool failed: %s", e, exc_info=True)
            return {"status": "error", "message": f"Ошибка создания поручения: {e!s}"}

    @mcp.tool(
        description=(
            "Добавить сотрудников в список ознакомления с документом. "
            "Поиск по фамилии, отделу или группе. "
            "Если фамилия неоднозначна — возвращает disambiguation. "
            "После выбора вызвать повторно с selected_employee_ids."
        )
    )
    async def introduction_create_tool(
        token: str,
        document_id: str,
        last_names: list[str] | None = None,
        department_names: list[str] | None = None,
        group_names: list[str] | None = None,
        comment: str | None = None,
        selected_employee_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Создать список ознакомления с документом.

        Args:
            token: JWT-токен.
            document_id: UUID документа.
            last_names: Фамилии сотрудников ['Иванов', 'Петров'].
            department_names: Названия подразделений.
            group_names: Названия групп.
            comment: Комментарий к ознакомлению.
            selected_employee_ids: UUID выбранных сотрудников (после disambiguation).
        """
        try:
            async with IntroductionService() as service:

                if selected_employee_ids:
                    if not selected_employee_ids:
                        return {"status": "error", "message": "Не указаны ID сотрудников."}
                    result = await service.create_introduction(
                        token=token,
                        document_id=document_id,
                        employee_ids=[UUID(eid) for eid in selected_employee_ids],
                        comment=comment,
                    )
                    if result.success:
                        return {
                            "status": "success",
                            "message": f"✅ Добавлено {result.added_count} сотрудников в ознакомление.",
                            "added_count": result.added_count,
                            "requires_reload": True,
                        }
                    return {"status": "error", "message": result.error_message}

                resolution_result = await service.resolve_employees(
                    token=token,
                    last_names=last_names or [],
                    department_names=department_names or [],
                    group_names=group_names or [],
                )

                if resolution_result.ambiguous:
                    formatted_choices = []
                    for amb in resolution_result.ambiguous:
                        for match in amb.get("matches", []):
                            formatted_choices.append({
                                "id": match.get("id"),
                                "full_name": match.get("full_name", ""),
                                "post": match.get("post", ""),
                                "department": match.get("department", ""),
                                "search_term": amb.get("search_query", ""),
                            })
                    return {
                        "status": "requires_disambiguation",
                        "action_type": "select_employee",
                        "message": "Найдено несколько совпадений. Выберите нужных сотрудников:",
                        "ambiguous_matches": formatted_choices,
                        "instruction": "Выберите сотрудников и вызовите инструмент с selected_employee_ids.",
                    }

                if not resolution_result.employee_ids:
                    not_found_str = ", ".join(resolution_result.not_found) if resolution_result.not_found else "критерии не заданы"
                    return {
                        "status": "error",
                        "message": f"❌ Не найдено ни одного сотрудника. Не найдены: {not_found_str}",
                    }

                result = await service.create_introduction(
                    token=token,
                    document_id=document_id,
                    employee_ids=list(resolution_result.employee_ids),
                    comment=comment,
                )

                if result.success:
                    response: dict[str, Any] = {
                        "status": "success",
                        "message": f"✅ Добавлено {result.added_count} сотрудников в ознакомление.",
                        "added_count": result.added_count,
                        "requires_reload": True,
                    }
                    if resolution_result.not_found:
                        response["not_found"] = resolution_result.not_found
                        response["message"] += f" Не найдены: {', '.join(resolution_result.not_found)}."
                    return response

                return {"status": "error", "message": result.error_message}

        except Exception as e:
            logger.error("introduction_create_tool failed: %s", e, exc_info=True)
            return {"status": "error", "message": f"Ошибка: {e!s}"}

    @mcp.tool(
        description=(
            "Поиск сотрудников в реестре EDMS по ФИО, должности, отделу или UUID. "
            "При одном результате — полная карточка. При нескольких — список для уточнения."
        )
    )
    async def employee_search_tool(
        token: str,
        employee_id: str | None = None,
        last_name: str | None = None,
        first_name: str | None = None,
        middle_name: str | None = None,
        full_post_name: str | None = None,
        active_only: bool | None = None,
        fired_only: bool | None = None,
        department_names: list[str] | None = None,
        department_ids: list[str] | None = None,
        employee_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Поиск сотрудников по различным критериям.

        Args:
            token: JWT-токен.
            employee_id: UUID конкретного сотрудника.
            last_name: Фамилия (частичное совпадение).
            first_name: Имя.
            middle_name: Отчество.
            full_post_name: Название должности.
            active_only: True — только активные.
            fired_only: True — только уволенные.
            department_names: Названия отделов (UUID резолвится автоматически).
            department_ids: UUID отделов.
            employee_ids: Список UUID сотрудников для пакетного запроса.
        """
        if not any([employee_id, last_name, first_name, middle_name, full_post_name,
                    active_only, fired_only, department_names, department_ids, employee_ids]):
            return {"status": "error", "message": "Укажите хотя бы один параметр поиска."}

        # Прямой запрос по UUID
        if employee_id:
            try:
                async with EmployeeClient() as client:
                    raw = await client.get_employee(token, employee_id)
                if not raw:
                    return {"status": "not_found", "message": f"Сотрудник {employee_id} не найден."}
                return {"status": "found", "total": 1, "employee_card": _format_employee_full(raw)}
            except Exception as exc:
                return {"status": "error", "message": f"Ошибка: {exc}"}

        employee_filter: dict[str, Any] = {"includes": ["POST", "DEPARTMENT"]}
        if last_name:
            employee_filter["lastName"] = last_name.strip()
        if first_name:
            employee_filter["firstName"] = first_name.strip()
        if middle_name:
            employee_filter["middleName"] = middle_name.strip()
        if full_post_name:
            employee_filter["fullPostName"] = full_post_name.strip()
        if active_only is True:
            employee_filter["active"] = True
        if fired_only is True:
            employee_filter["fired"] = True
        if employee_ids:
            employee_filter["ids"] = employee_ids

        resolved_dept_ids: list[str] = list(department_ids or [])
        if department_names:
            from ..clients.department_client import DepartmentClient
            async with DepartmentClient() as dept_client:
                for name in department_names:
                    dept = await dept_client.find_by_name(token, name.strip())
                    if dept and dept.get("id"):
                        resolved_dept_ids.append(str(dept["id"]))
                    else:
                        logger.warning("Отдел не найден: '%s'", name)

        if resolved_dept_ids:
            employee_filter["departmentId"] = resolved_dept_ids

        try:
            async with EmployeeClient() as client:
                results = await client.search_employees_post(
                    token=token,
                    employee_filter=employee_filter,
                    pageable={"page": 0, "size": 20, "sort": "lastName,ASC"},
                )

            if not results:
                return {"status": "not_found", "message": "Сотрудники не найдены.", "employees": [], "total": 0}

            if len(results) == 1:
                return {"status": "found", "total": 1, "employee_card": _format_employee_full(results[0])}

            choices = [_format_employee_brief(r) for r in results[:20]]
            return {
                "status": "requires_action",
                "action_type": "select_employee",
                "message": f"Найдено {len(choices)} сотрудников. Уточните выбор:",
                "total": len(choices),
                "choices": choices,
            }

        except Exception as exc:
            logger.error("employee_search_tool failed: %s", exc, exc_info=True)
            return {"status": "error", "message": f"Ошибка поиска: {exc}"}

    @mcp.tool(
        description=(
            "Отправить уведомление или напоминание сотрудникам по документу. "
            "Используй когда нужно напомнить о сроке, уведомить исполнителей, "
            "предупредить о приближающемся дедлайне."
        )
    )
    async def doc_send_notification(
        token: str,
        document_id: str,
        recipient_ids: list[str],
        message: str,
        notification_type: str = "REMINDER",
        deadline: str | None = None,
    ) -> dict[str, Any]:
        """
        Отправить уведомление сотрудникам по документу.

        Args:
            token: JWT-токен.
            document_id: UUID документа.
            recipient_ids: Список UUID сотрудников-получателей.
            message: Текст уведомления.
            notification_type: REMINDER | DEADLINE | CUSTOM.
            deadline: Дедлайн ISO 8601 (опционально).
        """
        if not recipient_ids:
            return {"status": "error", "message": "Список получателей не может быть пустым."}
        if not message or not message.strip():
            return {"status": "error", "message": "Текст уведомления не может быть пустым."}

        payload: dict[str, Any] = {
            "recipientIds": [uid.strip() for uid in recipient_ids if uid.strip()],
            "type": notification_type.upper(),
            "message": message.strip(),
        }
        if deadline:
            payload["deadline"] = deadline

        try:
            async with EdmsHttpClient() as client:
                result = await client._make_request(
                    "POST", f"api/document/{document_id}/notification",
                    token=token, json=payload, is_json_response=False,
                )
            deadline_note = f" (дедлайн: {deadline[:10]})" if deadline else ""
            return {
                "status": "success",
                "message": f"✅ Уведомление отправлено {len(recipient_ids)} сотруднику(-ам){deadline_note}.",
                "notification_type": notification_type.upper(),
                "recipients_count": len(recipient_ids),
            }
        except Exception as exc:
            logger.error("doc_send_notification failed: %s", exc, exc_info=True)
            return {"status": "error", "message": f"Ошибка отправки: {exc}"}

    @mcp.tool(
        description=(
            "Обновить одно поле документа EDMS. "
            "Основные поля: shortSummary (заголовок ≤80 символов), note, pages. "
            "Поля обращения: fullAddress, phone, email, signed, fioApplicant, organizationName, reviewProgress."
        )
    )
    async def doc_update_field(
        document_id: str,
        token: str,
        field_name: str,
        field_value: str,
    ) -> dict[str, Any]:
        """
        Обновить одно поле документа.

        Args:
            document_id: UUID документа.
            token: JWT-токен.
            field_name: Имя поля (shortSummary, note, pages, fullAddress, phone, email, signed...).
            field_value: Новое значение.
        """
        _ALLOWED_FIELDS = {
            "shortSummary", "note", "pages", "additionalPages", "exemplarCount",
        }
        _ALLOWED_APPEAL_FIELDS = {
            "fullAddress", "phone", "email", "signed", "correspondentOrgNumber",
            "organizationName", "fioApplicant", "reviewProgress",
        }
        all_allowed = _ALLOWED_FIELDS | _ALLOWED_APPEAL_FIELDS

        if field_name not in all_allowed:
            return {
                "status": "error",
                "message": f"Поле '{field_name}' не поддерживается. Допустимые: {', '.join(sorted(all_allowed))}",
            }

        value: Any = field_value.strip()
        if field_name == "shortSummary" and len(value) > 80:
            value = value[:80]
            logger.warning("shortSummary обрезан до 80 символов")

        if field_name in ("pages", "additionalPages", "exemplarCount"):
            try:
                value = int(field_value.strip())
            except ValueError:
                return {"status": "error", "message": f"Поле '{field_name}' должно быть числом."}

        is_appeal_field = field_name in _ALLOWED_APPEAL_FIELDS
        operation_type = (
            "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE"
            if is_appeal_field else "DOCUMENT_MAIN_FIELDS_UPDATE"
        )

        body = {field_name: value}
        payload = [{"operationType": operation_type, "body": body}]
        json_payload = json.loads(json.dumps(payload, cls=CustomJSONEncoder))

        try:
            async with DocumentClient() as client:
                await client._make_request(
                    "POST", f"api/document/{document_id}/execute",
                    token=token, json=json_payload, is_json_response=False,
                )
            return {
                "status": "success",
                "message": f"✅ Поле «{field_name}» обновлено: «{value}».",
                "field_name": field_name,
                "new_value": value,
                "requires_reload": True,
            }
        except Exception as exc:
            logger.error("doc_update_field failed: %s", exc, exc_info=True)
            return {"status": "error", "message": f"Ошибка обновления поля: {exc}"}


def _format_employee_full(raw: dict[str, Any]) -> dict[str, Any]:
    """Полная карточка сотрудника."""
    post = raw.get("post") or {}
    dept = raw.get("department") or {}
    parts = [raw.get("lastName", ""), raw.get("firstName", ""), raw.get("middleName") or ""]
    full_name = " ".join(p for p in parts if p).strip()
    return {
        "основное": {
            "фио": full_name or "—",
            "должность": post.get("postName") if isinstance(post, dict) else "—",
            "департамент": dept.get("name") if isinstance(dept, dict) else "—",
            "статус": "Уволен" if raw.get("fired") else "Активен",
        },
        "контакты": {
            "email": raw.get("email"),
            "телефон": raw.get("phone"),
        },
        "id": str(raw.get("id", "")),
    }


def _format_employee_brief(raw: dict[str, Any]) -> dict[str, Any]:
    """Краткая карточка сотрудника для списка выбора."""
    post = raw.get("post") or {}
    dept = raw.get("department") or {}
    parts = [raw.get("lastName", ""), raw.get("firstName", ""), raw.get("middleName") or ""]
    full_name = " ".join(p for p in parts if p).strip()
    return {
        "id": str(raw.get("id", "")),
        "full_name": full_name or "—",
        "post": post.get("postName") if isinstance(post, dict) else "—",
        "department": dept.get("name") if isinstance(dept, dict) else "—",
        "active": raw.get("active"),
        "fired": raw.get("fired"),
    }