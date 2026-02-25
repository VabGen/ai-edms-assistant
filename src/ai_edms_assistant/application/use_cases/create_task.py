# src/ai_edms_assistant/application/use_cases/create_task.py
"""Use case for creating tasks on documents."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.entities.task import TaskType
from ...domain.exceptions import DocumentNotFoundError, TaskCreationError
from ...domain.repositories import (
    AbstractDocumentRepository,
    AbstractEmployeeRepository,
    AbstractTaskRepository,
)
from ...domain.services import TaskAssigner
from ..dto import TaskSummaryDto
from .base import AbstractUseCase


class CreateTaskRequest(BaseModel):
    """Request DTO for task creation.

    Attributes:
        document_id: UUID of the parent document.
        task_text: Task instruction text.
        deadline: Planned completion date.
        executor_ids: List of tuples ``(employee_uuid, is_responsible)``.
        token: JWT bearer token.
        task_type: Workflow type for the task. Defaults to EXECUTION.
        endless: Whether this is an endless task (no deadline).
    """

    document_id: UUID
    task_text: str = Field(..., min_length=1)
    deadline: datetime
    executor_ids: list[tuple[UUID, bool]] = Field(..., min_items=1)
    token: str
    task_type: TaskType = TaskType.EXECUTION
    endless: bool = False


class CreateTaskUseCase(AbstractUseCase[CreateTaskRequest, TaskSummaryDto]):
    """Use case: Create a new task on a document.

    Validates executor assignments, checks that the parent document exists,
    and delegates to the task repository for creation.

    Dependencies:
        - ``AbstractDocumentRepository``: Validate parent document exists.
        - ``AbstractEmployeeRepository``: Fetch and validate executors.
        - ``AbstractTaskRepository``: Create the task via EDMS API.
        - ``TaskAssigner``: Validate executor assignments.

    Business rules:
        - Parent document must exist.
        - At least one executor is required.
        - Exactly one executor must be designated as responsible.
        - Executors must be active (not fired).
    """

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        employee_repository: AbstractEmployeeRepository,
        task_repository: AbstractTaskRepository,
        task_assigner: TaskAssigner,
    ) -> None:
        """Initialize with injected dependencies.

        Args:
            document_repository: Repository for validating parent document.
            employee_repository: Repository for fetching executors.
            task_repository: Repository for creating tasks.
            task_assigner: Domain service for validating assignments.
        """
        self._doc_repo = document_repository
        self._emp_repo = employee_repository
        self._task_repo = task_repository
        self._assigner = task_assigner

    async def execute(self, request: CreateTaskRequest) -> TaskSummaryDto:
        """Execute the task creation use case.

        Args:
            request: Task creation request with document ID, text, executors.

        Returns:
            ``TaskSummaryDto`` representing the created task.

        Raises:
            DocumentNotFoundError: When the parent document does not exist.
            TaskCreationError: When executor validation fails.
        """
        document = await self._doc_repo.get_by_id(
            entity_id=request.document_id,
            token=request.token,
        )
        if document is None:
            raise DocumentNotFoundError(document_id=request.document_id)

        executor_uuids = [uuid for uuid, _ in request.executor_ids]
        employees = await self._emp_repo.get_by_ids(
            entity_ids=executor_uuids,
            token=request.token,
        )

        employee_map = {e.id: e for e in employees}

        missing = set(executor_uuids) - set(employee_map.keys())
        if missing:
            raise TaskCreationError(
                f"Сотрудники не найдены: {', '.join(str(m) for m in missing)}"
            )

        candidates = [
            (employee_map[uuid], is_resp) for uuid, is_resp in request.executor_ids
        ]

        plan = self._assigner.build_plan(
            candidates=candidates,
            task_text=request.task_text,
        )

        success = await self._task_repo.create(
            document_id=request.document_id,
            task_text=plan.task_text,
            deadline=request.deadline,
            executor_ids=plan.to_api_executor_list(),
            token=request.token,
            task_type=request.task_type,
            endless=request.endless,
        )

        if not success:
            raise TaskCreationError("EDMS API вернул ошибку при создании поручения")

        tasks = await self._task_repo.get_by_document_id(
            document_id=request.document_id,
            token=request.token,
        )

        if not tasks:
            raise TaskCreationError(
                "Поручение создано, но не удалось получить его из системы"
            )

        latest_task = max(
            tasks,
            key=lambda t: t.create_date if t.create_date else datetime.min,
        )

        return TaskSummaryDto.from_entity(latest_task)
