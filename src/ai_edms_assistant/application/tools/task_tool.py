# src/ai_edms_assistant/application/tools/task_tool.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.entities.task import TaskType
from ...domain.exceptions import TaskCreationError
from ...domain.repositories import (
    AbstractDocumentRepository,
    AbstractEmployeeRepository,
    AbstractTaskRepository,
)
from ...domain.services import TaskAssigner
from .base_tool import AbstractEdmsTool


class TaskCreateInput(BaseModel):
    """Input schema for task creation with disambiguation support.

    Attributes:
        token: JWT auth token (auto-injected by agent).
        document_id: UUID of parent document.
        task_text: Task instruction text (required).
        executor_last_names: List of executor surnames for fuzzy search.
        selected_employee_ids: UUID list after disambiguation (second call).
        responsible_last_name: Surname of responsible executor (optional).
        planed_date_end: Deadline in ISO 8601 (optional, defaults to +7 days).
        task_type: Workflow type (defaults to EXECUTION).
    """

    token: str = Field(..., description="JWT токен авторизации")
    document_id: UUID = Field(..., description="UUID документа")
    task_text: str = Field(..., min_length=1, description="Текст поручения")
    executor_last_names: list[str] | None = Field(
        default=None, description="Фамилии исполнителей для поиска"
    )
    selected_employee_ids: list[UUID] | None = Field(
        default=None, description="UUID выбранных сотрудников (после disambiguation)"
    )
    responsible_last_name: str | None = Field(
        default=None, description="Фамилия ответственного"
    )
    planed_date_end: str | None = Field(
        default=None, description="Срок в ISO 8601 (опционально)"
    )
    task_type: TaskType = Field(default=TaskType.EXECUTION, description="Тип поручения")


class TaskCreationTool(AbstractEdmsTool):
    """Tool for creating tasks on documents with executor disambiguation.

    Workflow:
        1. If executor_last_names provided → search employees by surname
        2. If ambiguous (multiple "Ivanov") → return "requires_disambiguation"
        3. User selects from list → agent calls again with selected_employee_ids
        4. Create task via repository

    Dependencies:
        - ``AbstractDocumentRepository``: Validate parent document exists.
        - ``AbstractEmployeeRepository``: Search executors by surname.
        - ``AbstractTaskRepository``: Create task via EDMS API.
        - ``TaskAssigner``: Validate executor assignments.
    """

    name: str = "task_create_tool"
    description: str = (
        "Создает поручение на документе. "
        "Поддерживает автоматический поиск исполнителей по фамилии. "
        "Если найдено несколько человек с одной фамилией, возвращает список для выбора."
    )
    args_schema: type[BaseModel] = TaskCreateInput

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        employee_repository: AbstractEmployeeRepository,
        task_repository: AbstractTaskRepository,
        task_assigner: TaskAssigner,
        **kwargs,
    ):
        """Initialize with injected dependencies.

        Args:
            document_repository: Repository for validating parent document.
            employee_repository: Repository for searching executors.
            task_repository: Repository for creating tasks.
            task_assigner: Domain service for validating assignments.
            **kwargs: Additional BaseTool arguments.
        """
        super().__init__(**kwargs)
        self._doc_repo = document_repository
        self._emp_repo = employee_repository
        self._task_repo = task_repository
        self._assigner = task_assigner

    async def _arun(
        self,
        token: str,
        document_id: UUID,
        task_text: str,
        executor_last_names: list[str] | None = None,
        selected_employee_ids: list[UUID] | None = None,
        responsible_last_name: str | None = None,
        planed_date_end: str | None = None,
        task_type: TaskType = TaskType.EXECUTION,
    ) -> dict[str, Any]:
        """Execute task creation with disambiguation support.

        Args:
            token: JWT token.
            document_id: Parent document UUID.
            task_text: Task instruction.
            executor_last_names: Surnames for fuzzy search (first call).
            selected_employee_ids: Resolved UUIDs (second call).
            responsible_last_name: Responsible executor surname.
            planed_date_end: Deadline ISO string.
            task_type: Task workflow type.

        Returns:
            Dict with status and message/data.
        """
        try:
            # Validate inputs
            if not any([executor_last_names, selected_employee_ids]):
                return self._handle_error(
                    ValueError(
                        "Необходимо указать executor_last_names или selected_employee_ids"
                    )
                )

            # Parse deadline
            deadline = self._parse_deadline(planed_date_end)

            # BRANCH 1: User already selected employees
            if selected_employee_ids:
                return await self._create_with_selected_ids(
                    token=token,
                    document_id=document_id,
                    task_text=task_text,
                    employee_ids=selected_employee_ids,
                    deadline=deadline,
                    task_type=task_type,
                )

            # BRANCH 2: Fuzzy search by surnames
            return await self._create_with_surnames(
                token=token,
                document_id=document_id,
                task_text=task_text,
                executor_last_names=executor_last_names,
                responsible_last_name=responsible_last_name,
                deadline=deadline,
                task_type=task_type,
            )

        except TaskCreationError as e:
            return self._handle_error(e)
        except Exception as e:
            return self._handle_error(e)

    async def _create_with_selected_ids(
        self,
        token: str,
        document_id: UUID,
        task_text: str,
        employee_ids: list[UUID],
        deadline: datetime,
        task_type: TaskType,
    ) -> dict[str, Any]:
        """Create task using resolved employee IDs (after disambiguation).

        Args:
            token: JWT token.
            document_id: Parent document UUID.
            task_text: Task text.
            employee_ids: Resolved executor UUIDs.
            deadline: Parsed deadline.
            task_type: Task type.

        Returns:
            Success response with created count.
        """
        # Fetch employees
        employees = await self._emp_repo.get_by_ids(
            entity_ids=employee_ids,
            token=token,
        )

        # Build candidates (first is responsible)
        candidates = [(emp, idx == 0) for idx, emp in enumerate(employees)]

        # Validate with TaskAssigner
        plan = self._assigner.build_plan(candidates=candidates, task_text=task_text)

        # Create via repository
        success = await self._task_repo.create(
            document_id=document_id,
            task_text=plan.task_text,
            deadline=deadline,
            executor_ids=plan.to_api_executor_list(),
            token=token,
            task_type=task_type,
        )

        if not success:
            return self._handle_error(
                TaskCreationError("EDMS API вернул ошибку при создании поручения")
            )

        return self._success_response(
            data={"created_count": len(employees)},
            message=f"✅ Поручение создано. Исполнителей: {len(employees)}",
        )

    async def _create_with_surnames(
        self,
        token: str,
        document_id: UUID,
        task_text: str,
        executor_last_names: list[str],
        responsible_last_name: str | None,
        deadline: datetime,
        task_type: TaskType,
    ) -> dict[str, Any]:
        """Create task by searching executors via surname (with disambiguation).

        Args:
            token: JWT token.
            document_id: Parent document UUID.
            task_text: Task text.
            executor_last_names: Surnames to search.
            responsible_last_name: Responsible surname (optional).
            deadline: Parsed deadline.
            task_type: Task type.

        Returns:
            Either disambiguation request or success response.
        """
        # Search all executors
        all_results, ambiguous = await self._search_executors(
            token=token, last_names=executor_last_names
        )

        # DISAMBIGUATION NEEDED
        if ambiguous:
            return {
                "status": "requires_disambiguation",
                "message": (
                    "⚠️ Найдено несколько сотрудников с одинаковыми фамилиями. "
                    "Выберите нужных из списка:"
                ),
                "ambiguous_matches": ambiguous,
                "instruction": (
                    "Выберите сотрудников, затем вызовите инструмент повторно "
                    "с параметром selected_employee_ids."
                ),
            }

        if not all_results:
            return self._handle_error(
                ValueError("Не найдено ни одного сотрудника по указанным фамилиям")
            )

        # Determine responsible
        candidates = self._build_candidates(
            employees=all_results,
            responsible_last_name=responsible_last_name,
        )

        # Validate and create
        plan = self._assigner.build_plan(candidates=candidates, task_text=task_text)

        success = await self._task_repo.create(
            document_id=document_id,
            task_text=plan.task_text,
            deadline=deadline,
            executor_ids=plan.to_api_executor_list(),
            token=token,
            task_type=task_type,
        )

        if not success:
            return self._handle_error(TaskCreationError("EDMS API вернул ошибку"))

        return self._success_response(
            data={"created_count": len(all_results)},
            message=f"✅ Поручение создано. Исполнителей: {len(all_results)}",
        )

    async def _search_executors(
        self, token: str, last_names: list[str]
    ) -> tuple[list, list]:
        """Search employees by surnames and detect ambiguity.

        Args:
            token: JWT token.
            last_names: List of surnames to search.

        Returns:
            Tuple of (all_found_employees, ambiguous_matches_list).
            ambiguous_matches_list is non-empty when disambiguation needed.
        """
        from ...domain.value_objects.filters import EmployeeFilter

        all_employees = []
        ambiguous = []

        for surname in last_names:
            filter_obj = EmployeeFilter(last_name=surname, active=True)
            results = await self._emp_repo.search(filters=filter_obj, token=token)

            if len(results) > 1:
                # Ambiguous - add to disambiguation list
                ambiguous.append(
                    {
                        "surname": surname,
                        "candidates": [
                            {
                                "id": str(emp.id),
                                "full_name": emp.short_name,
                                "post": emp.post.name if emp.post else "Не указана",
                                "department": (
                                    emp.department.name
                                    if emp.department
                                    else "Не указан"
                                ),
                            }
                            for emp in results
                        ],
                    }
                )
            elif len(results) == 1:
                all_employees.append(results[0])

        return all_employees, ambiguous

    def _build_candidates(
        self, employees: list, responsible_last_name: str | None
    ) -> list[tuple]:
        """Build candidate list with responsible flag.

        Args:
            employees: List of resolved Employee entities.
            responsible_last_name: Surname of responsible (optional).

        Returns:
            List of (Employee, is_responsible) tuples.
        """
        if not responsible_last_name:
            # First executor is responsible
            return [(emp, idx == 0) for idx, emp in enumerate(employees)]

        # Find responsible by surname
        responsible_idx = next(
            (
                i
                for i, emp in enumerate(employees)
                if emp.last_name.lower() == responsible_last_name.lower()
            ),
            0,
        )

        return [(emp, i == responsible_idx) for i, emp in enumerate(employees)]

    def _parse_deadline(self, deadline_str: str | None) -> datetime:
        """Parse deadline string or default to +7 days.

        Args:
            deadline_str: ISO 8601 string or None.

        Returns:
            Parsed datetime with UTC timezone.
        """
        if deadline_str:
            dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        # Default: +7 days
        return datetime.now(timezone.utc) + timedelta(days=7)

    def _run(self, *args, **kwargs) -> dict[str, Any]:
        """Sync execution not supported."""
        raise NotImplementedError("Use _arun for async execution")
