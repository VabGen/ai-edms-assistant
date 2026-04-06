# mcp-server/services/task_service.py
"""
Task Service — создание поручений с disambiguation.
Перенесён из edms_ai_assistant/services/task_service.py.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from ..clients.employee_client import EmployeeClient
from ..clients.task_client import TaskClient
from ..models.task_models import (
    CreateTaskRequest,
    CreateTaskRequestExecutor,
    TaskCreationResult,
    TaskType,
)
from ..utils.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)


def _to_uuid(value: Any) -> UUID:
    return UUID(value) if isinstance(value, str) else value


class TaskService:
    """
    Сервис создания поручений с поддержкой disambiguation.

    Резолвит исполнителей по фамилии, обрабатывает неоднозначности,
    создаёт поручения через TaskClient.
    """

    def __init__(self) -> None:
        self.employee_client = EmployeeClient()
        self.task_client = TaskClient()

    async def __aenter__(self) -> "TaskService":
        await self.employee_client.__aenter__()
        await self.task_client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.employee_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.task_client.__aexit__(exc_type, exc_val, exc_tb)

    async def collect_executors(
        self,
        token: str,
        last_names: list[str],
        responsible_last_name: str | None = None,
    ) -> tuple[list[CreateTaskRequestExecutor] | None, list[str], list[dict[str, Any]]]:
        """
        Резолвинг исполнителей по фамилии.

        Returns:
            (executors, not_found, ambiguous_results)
        """
        found_employees: list[dict[str, Any]] = []
        not_found: list[str] = []
        ambiguous_results: list[dict[str, Any]] = []

        for last_name in last_names:
            employees = await self.employee_client.search_employees(
                token, {"lastName": last_name, "includes": ["POST", "DEPARTMENT"]}
            )

            if not employees:
                not_found.append(last_name)
                continue

            if len(employees) > 1:
                ambiguous_results.append({
                    "search_query": last_name,
                    "matches": [self._format_employee_match(e) for e in employees],
                })
                continue

            found_employees.append(employees[0])

        if ambiguous_results:
            return None, not_found, ambiguous_results

        if not found_employees:
            return None, not_found, []

        responsible_employee: dict[str, Any] | None = None
        if responsible_last_name:
            responsible_employee = await self.employee_client.find_by_last_name_fts(
                token, responsible_last_name
            )

        executors: list[CreateTaskRequestExecutor] = []
        seen_ids: set[UUID] = set()

        if responsible_employee:
            resp_id = _to_uuid(responsible_employee["id"])
            executors.append(CreateTaskRequestExecutor(employeeId=resp_id, responsible=True))
            seen_ids.add(resp_id)

        for idx, emp in enumerate(found_employees):
            emp_id = _to_uuid(emp["id"])
            if emp_id in seen_ids:
                continue
            is_responsible = not responsible_employee and idx == 0
            executors.append(CreateTaskRequestExecutor(employeeId=emp_id, responsible=is_responsible))
            seen_ids.add(emp_id)

        return executors, not_found, []

    async def create_task(
        self,
        token: str,
        document_id: str,
        task_text: str,
        executor_last_names: list[str],
        planed_date_end: datetime | None = None,
        responsible_last_name: str | None = None,
        task_type: TaskType = TaskType.GENERAL,
    ) -> TaskCreationResult:
        """Создать поручение с резолвингом исполнителей по фамилии."""
        if not executor_last_names:
            return TaskCreationResult(
                success=False, status="error",
                error_message="Необходимо указать хотя бы одного исполнителя.",
            )
        if not task_text or not task_text.strip():
            return TaskCreationResult(
                success=False, status="error",
                error_message="Текст поручения не может быть пустым.",
            )

        executors, not_found, ambiguous_results = await self.collect_executors(
            token, executor_last_names, responsible_last_name
        )

        if ambiguous_results:
            return TaskCreationResult(
                success=False, status="requires_disambiguation",
                ambiguous_matches=ambiguous_results,
                not_found_employees=not_found,
            )

        if not executors:
            return TaskCreationResult(
                success=False, status="error",
                error_message=f"Не найдены сотрудники: {', '.join(not_found)}",
                not_found_employees=not_found,
            )

        return await self._submit_task(
            token=token, document_id=document_id, task_text=task_text,
            executors=executors, planed_date_end=planed_date_end,
            task_type=task_type, not_found=not_found,
        )

    async def create_task_by_employee_ids(
        self,
        token: str,
        document_id: str,
        task_text: str,
        employee_ids: list[UUID],
        planed_date_end: datetime | None = None,
        responsible_employee_id: UUID | None = None,
        task_type: TaskType = TaskType.GENERAL,
    ) -> TaskCreationResult:
        """Создать поручение для пре-выбранных сотрудников (после disambiguation)."""
        if not employee_ids:
            return TaskCreationResult(
                success=False, status="error",
                error_message="Необходимо указать хотя бы одного исполнителя.",
            )

        responsible_id = responsible_employee_id or employee_ids[0]
        executors: list[CreateTaskRequestExecutor] = []
        seen_ids: set[UUID] = set()

        for emp_id in employee_ids:
            if emp_id in seen_ids:
                continue
            executors.append(CreateTaskRequestExecutor(
                employeeId=emp_id, responsible=(emp_id == responsible_id),
            ))
            seen_ids.add(emp_id)

        return await self._submit_task(
            token=token, document_id=document_id, task_text=task_text,
            executors=executors, planed_date_end=planed_date_end, task_type=task_type,
        )

    async def _submit_task(
        self,
        token: str,
        document_id: str,
        task_text: str,
        executors: list[CreateTaskRequestExecutor],
        planed_date_end: datetime | None,
        task_type: TaskType,
        not_found: list[str] | None = None,
    ) -> TaskCreationResult:
        """Отправить подготовленный запрос на создание поручения."""
        if not planed_date_end:
            planed_date_end = datetime.now(UTC) + timedelta(days=7)
        elif planed_date_end.tzinfo is None:
            planed_date_end = planed_date_end.replace(tzinfo=UTC)

        formatted_text = (
            task_text[0].upper() + task_text[1:]
            if len(task_text) > 1 else task_text.upper()
        )

        task_request = CreateTaskRequest(
            taskText=formatted_text,
            planedDateEnd=planed_date_end,
            type=task_type,
            periodTask=False,
            endless=False,
            executors=executors,
        )

        try:
            payload = [json.loads(json.dumps(task_request.model_dump(mode="json"), cls=CustomJSONEncoder))]
            success = await self.task_client.create_tasks_batch(token, document_id, payload)

            if success:
                return TaskCreationResult(
                    success=True, status="success",
                    created_count=1,
                    not_found_employees=not_found or [],
                )

            return TaskCreationResult(
                success=False, status="error",
                error_message="Не удалось создать поручение. Проверьте права доступа.",
            )
        except Exception as exc:
            logger.error("Task creation failed: %s", exc, exc_info=True)
            return TaskCreationResult(
                success=False, status="error",
                error_message=f"Ошибка создания поручения: {exc}",
            )

    @staticmethod
    def _format_employee_match(employee: dict[str, Any]) -> dict[str, Any]:
        """Форматирует данные сотрудника для disambiguation."""
        ln = employee.get("lastName", "")
        fn = employee.get("firstName", "")
        mn = employee.get("middleName") or ""
        full_name = f"{ln} {fn} {mn}".strip()

        post_data = employee.get("post") or {}
        post_name = post_data.get("postName", "Не указана") if isinstance(post_data, dict) else "Не указана"

        dept_data = employee.get("department") or {}
        dept_name = dept_data.get("name", "Не указан") if isinstance(dept_data, dict) else "Не указан"

        return {
            "id": str(employee["id"]),
            "full_name": full_name,
            "post": post_name,
            "department": dept_name,
        }