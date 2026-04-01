# edms_ai_assistant/services/task_service.py
"""
EDMS AI Assistant — Task Service with Disambiguation Support.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.task_client import TaskClient
from edms_ai_assistant.models.task_models import (
    CreateTaskRequest,
    CreateTaskRequestExecutor,
    TaskCreationResult,
    TaskType,
)

logger = logging.getLogger(__name__)


class TaskService:
    """
    Service for managing document tasks (поручения) with Disambiguation.

    Responsibilities:
    - Резолвинг исполнителей по фамилии с обработкой неоднозначностей
    - Создание поручений через TaskClient
    - Управление lifecycle клиентов через async context manager
    """

    def __init__(self) -> None:
        """Initialises employee and task API clients."""
        self.employee_client = EmployeeClient()
        self.task_client = TaskClient()

    async def __aenter__(self) -> "TaskService":
        """Opens underlying HTTP clients."""
        await self.employee_client.__aenter__()
        await self.task_client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Closes underlying HTTP clients."""
        await self.employee_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.task_client.__aexit__(exc_type, exc_val, exc_tb)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    async def collect_executors(
        self,
        token: str,
        last_names: list[str],
        responsible_last_name: str | None = None,
    ) -> tuple[
        list[CreateTaskRequestExecutor] | None,
        list[str],
        list[dict[str, Any]],
    ]:
        """
        Resolves executor employees by last name with disambiguation support.

        Args:
            token: JWT bearer token.
            last_names: List of executor last names to search.
            responsible_last_name: Optional last name of the responsible executor.

        Returns:
            Tuple of:
            - executors: List of CreateTaskRequestExecutor (None if disambiguation needed).
            - not_found: Last names that yielded no results.
            - ambiguous_results: Disambiguation payloads (one per ambiguous name).
        """
        found_employees: list[dict[str, Any]] = []
        not_found: list[str] = []
        ambiguous_results: list[dict[str, Any]] = []

        for last_name in last_names:
            employees = await self.employee_client.search_employees(
                token,
                {"lastName": last_name, "includes": ["POST", "DEPARTMENT"]},
            )

            if not employees:
                logger.warning("[TASK-SERVICE] Employee not found: %s", last_name)
                not_found.append(last_name)
                continue

            if len(employees) > 1:
                logger.info(
                    "[TASK-SERVICE] Ambiguous match for '%s': %d results",
                    last_name,
                    len(employees),
                )
                ambiguous_results.append(
                    {
                        "search_query": last_name,
                        "matches": [self._format_employee_match(e) for e in employees],
                    }
                )
                continue

            # Только одно совпадение → OK
            logger.debug(
                "[TASK-SERVICE] Found single employee: %s -> %s",
                last_name,
                employees[0].get("id"),
            )
            found_employees.append(employees[0])

        # ── Неоднозначности → прерываем, возвращаем для выбора ────────────
        if ambiguous_results:
            return None, not_found, ambiguous_results

        if not found_employees:
            return None, not_found, []

        # ── Все найдены однозначно → формируем executors ───────────────────
        responsible_employee: dict[str, Any] | None = None
        if responsible_last_name:
            responsible_employee = await self.employee_client.find_by_last_name_fts(
                token, responsible_last_name
            )
            if not responsible_employee:
                logger.warning(
                    "[TASK-SERVICE] Responsible '%s' not found — first executor becomes responsible",
                    responsible_last_name,
                )

        executors: list[CreateTaskRequestExecutor] = []
        seen_ids: set[UUID] = set()

        if responsible_employee:
            resp_id = _to_uuid(responsible_employee["id"])
            executors.append(
                CreateTaskRequestExecutor(employeeId=resp_id, responsible=True)
            )
            seen_ids.add(resp_id)

        for idx, emp in enumerate(found_employees):
            emp_id = _to_uuid(emp["id"])
            if emp_id in seen_ids:
                continue
            is_responsible = not responsible_employee and idx == 0
            executors.append(
                CreateTaskRequestExecutor(employeeId=emp_id, responsible=is_responsible)
            )
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
        """
        Creates a task after resolving executors by last name.

        Returns requires_disambiguation if any name is ambiguous.

        Args:
            token: JWT bearer token.
            document_id: Target EDMS document UUID.
            task_text: Task body text.
            executor_last_names: List of executor last names.
            planed_date_end: Deadline (defaults to +7 days UTC).
            responsible_last_name: Optional responsible executor last name.
            task_type: Task type enum value.

        Returns:
            TaskCreationResult with status success | requires_disambiguation | error.
        """
        logger.info("[TASK-SERVICE] Creating task. Executors: %s", executor_last_names)

        if not executor_last_names:
            return TaskCreationResult(
                success=False,
                status="error",
                error_message="Необходимо указать хотя бы одного исполнителя.",
            )

        if not task_text or not task_text.strip():
            return TaskCreationResult(
                success=False,
                status="error",
                error_message="Текст поручения не может быть пустым.",
            )

        executors, not_found, ambiguous_results = await self.collect_executors(
            token, executor_last_names, responsible_last_name
        )

        if ambiguous_results:
            logger.info(
                "[TASK-SERVICE] Disambiguation required. Ambiguous: %d, Not found: %d",
                len(ambiguous_results),
                len(not_found),
            )
            return TaskCreationResult(
                success=False,
                status="requires_disambiguation",
                ambiguous_matches=ambiguous_results,
                not_found_employees=not_found,
            )

        if not executors:
            return TaskCreationResult(
                success=False,
                status="error",
                error_message=f"Не найдены сотрудники: {', '.join(not_found)}",
                not_found_employees=not_found,
            )

        return await self._submit_task(
            token=token,
            document_id=document_id,
            task_text=task_text,
            executors=executors,
            planed_date_end=planed_date_end,
            task_type=task_type,
            not_found=not_found,
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
        """
        Creates a task for pre-selected employees (post-disambiguation flow).

        Args:
            token: JWT bearer token.
            document_id: Target EDMS document UUID.
            task_text: Task body text.
            employee_ids: Pre-resolved executor UUIDs.
            planed_date_end: Deadline (defaults to +7 days UTC).
            responsible_employee_id: UUID of the responsible executor.
            task_type: Task type enum value.

        Returns:
            TaskCreationResult with status success | error.
        """
        logger.info(
            "[TASK-SERVICE] Creating task with pre-selected employees: %d",
            len(employee_ids),
        )

        if not employee_ids:
            return TaskCreationResult(
                success=False,
                status="error",
                error_message="Необходимо указать хотя бы одного исполнителя.",
            )

        responsible_id = responsible_employee_id or employee_ids[0]

        executors: list[CreateTaskRequestExecutor] = []
        seen_ids: set[UUID] = set()
        for emp_id in employee_ids:
            if emp_id in seen_ids:
                continue
            executors.append(
                CreateTaskRequestExecutor(
                    employeeId=emp_id,
                    responsible=(emp_id == responsible_id),
                )
            )
            seen_ids.add(emp_id)

        return await self._submit_task(
            token=token,
            document_id=document_id,
            task_text=task_text,
            executors=executors,
            planed_date_end=planed_date_end,
            task_type=task_type,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

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
        """
        Submits the prepared task request to the EDMS API.

        Args:
            token: JWT bearer token.
            document_id: Target document UUID.
            task_text: Task body text.
            executors: Resolved executor list.
            planed_date_end: Deadline datetime.
            task_type: Task type.
            not_found: Optional list of not-found names (for partial success).

        Returns:
            TaskCreationResult.
        """
        # Нормализация дедлайна
        if not planed_date_end:
            planed_date_end = datetime.now(UTC) + timedelta(days=7)
        elif planed_date_end.tzinfo is None:
            planed_date_end = planed_date_end.replace(tzinfo=UTC)

        # Капитализация первой буквы текста
        formatted_text = (
            task_text[0].upper() + task_text[1:]
            if len(task_text) > 1
            else task_text.upper()
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
            success = await self.task_client.create_tasks_batch(
                token, document_id, [task_request]
            )

            if success:
                logger.info(
                    "[TASK-SERVICE] Task created successfully. Executors: %d",
                    len(executors),
                )
                result = TaskCreationResult(
                    success=True,
                    status="success",
                    created_count=1,
                    not_found_employees=not_found or [],
                )
                return result

            return TaskCreationResult(
                success=False,
                status="error",
                error_message=(
                    "Не удалось создать поручение. "
                    "Проверьте права доступа или корректность данных."
                ),
            )

        except Exception as exc:
            logger.error("[TASK-SERVICE] Task creation failed: %s", exc, exc_info=True)
            return TaskCreationResult(
                success=False,
                status="error",
                error_message=f"Ошибка создания поручения: {exc}",
            )

    @staticmethod
    def _format_employee_match(employee: dict[str, Any]) -> dict[str, Any]:
        """
        Formats a raw employee dict for disambiguation UI payload.

        Args:
            employee: Raw API employee record.

        Returns:
            Dict with id, full_name, post, department keys.
        """
        last_name = employee.get("lastName", "")
        first_name = employee.get("firstName", "")
        middle_name = employee.get("middleName") or ""
        full_name = f"{last_name} {first_name} {middle_name}".strip()

        post_data = employee.get("post") or {}
        post_name = (
            post_data.get("postName", "Не указана")
            if isinstance(post_data, dict)
            else "Не указана"
        )

        dept_data = employee.get("department") or {}
        dept_name = (
            dept_data.get("name", "Не указан")
            if isinstance(dept_data, dict)
            else "Не указан"
        )

        return {
            "id": str(employee["id"]),
            "full_name": full_name,
            "post": post_name,
            "department": dept_name,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _to_uuid(value: Any) -> UUID:
    """Converts a string or UUID value to UUID.

    Args:
        value: Raw id value from API response.

    Returns:
        UUID instance.
    """
    return UUID(value) if isinstance(value, str) else value
