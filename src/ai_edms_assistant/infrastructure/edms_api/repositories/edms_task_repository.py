# src/ai_edms_assistant/infrastructure/edms_api/repositories/edms_task_repository.py
"""EDMS REST API implementation of AbstractTaskRepository.

All mapper logic uses TaskMapper from infrastructure/edms_api/mappers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from ....domain.entities.task import Task, TaskStatus, TaskType
from ....domain.repositories.base import Page, PageRequest
from ....domain.repositories.task_repository import AbstractTaskRepository
from ....domain.value_objects.filters import TaskFilter
from ...edms_api.http_client import EdmsHttpClient
from ...edms_api.mappers.task_mapper import TaskMapper

logger = logging.getLogger(__name__)


class EdmsTaskRepository(AbstractTaskRepository):
    """EDMS REST API implementation of AbstractTaskRepository.

    Uses EdmsHttpClient for HTTP requests and TaskMapper for DTO → domain mapping.
    All filters come from domain/value_objects/filters.py.

    Hierarchy methods (find_children / find_parents / find_task_tree) call
    dedicated EDMS endpoints — the backend executes recursive CTE queries.

    Attributes:
        http_client: Shared async EdmsHttpClient instance.
    """

    def __init__(self, http_client: EdmsHttpClient) -> None:
        """Initialize repository with HTTP client.

        Args:
            http_client: Configured EdmsHttpClient instance.
        """
        self.http_client = http_client

    # ------------------------------------------------------------------
    # BaseRepository implementation
    # ------------------------------------------------------------------

    async def get_by_id(
        self,
        entity_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> Task | None:
        """Fetch task by ID.

        Args:
            entity_id: Task UUID.
            token: JWT bearer token.
            organization_id: Optional org scope (unused for tasks).

        Returns:
            Task instance or None when not found.

        Raises:
            Exception: On HTTP errors after retries.
        """
        try:
            data = await self.http_client._make_request(
                method="GET",
                endpoint=f"api/task/{entity_id}",
                token=token,
            )
            return TaskMapper.from_dto(data) if data else None
        except Exception as exc:
            logger.error(
                f"Failed to fetch task {entity_id}",
                exc_info=True,
                extra={"task_id": str(entity_id)},
            )
            raise

    async def get_by_ids(
        self,
        entity_ids: list[UUID],
        token: str,
        organization_id: str | None = None,
    ) -> list[Task]:
        """Fetch multiple tasks by IDs (parallel individual GET requests).

        EDMS has no batch-by-ids endpoint for tasks.

        Args:
            entity_ids: List of task UUIDs.
            token: JWT bearer token.
            organization_id: Optional org scope (unused).

        Returns:
            List of found tasks (silently skips missing IDs).
        """
        if not entity_ids:
            return []

        import asyncio

        results = await asyncio.gather(
            *[self.get_by_id(eid, token) for eid in entity_ids],
            return_exceptions=True,
        )
        return [r for r in results if isinstance(r, Task)]

    async def find_page(
        self,
        token: str,
        organization_id: str | None = None,
        pagination: PageRequest | None = None,
    ) -> Page[Task]:
        """Fetch paginated task list.

        Delegates to search() with empty TaskFilter.

        Args:
            token: JWT bearer token.
            organization_id: Optional org scope (unused).
            pagination: Pagination parameters.

        Returns:
            Page[Task] with slice semantics.
        """
        return await self.search(TaskFilter(), token, pagination)

    # ------------------------------------------------------------------
    # TaskRepository-specific methods
    # ------------------------------------------------------------------

    async def get_by_document_id(
        self,
        document_id: UUID,
        token: str,
    ) -> list[Task]:
        """Fetch all tasks linked to a document.

        Args:
            document_id: Parent document UUID.
            token: JWT bearer token.

        Returns:
            List of tasks (empty list if none found).

        Raises:
            Exception: On HTTP errors after retries.
        """
        try:
            data = await self.http_client._make_request(
                method="GET",
                endpoint=f"api/document/{document_id}/task",
                token=token,
            )
            return TaskMapper.from_dto_list(data or [])
        except Exception as exc:
            logger.error(
                f"Failed to fetch tasks for document {document_id}",
                exc_info=True,
                extra={"document_id": str(document_id)},
            )
            raise

    async def get_by_executor(
        self,
        executor_id: UUID,
        token: str,
        active_only: bool = True,
        pagination: PageRequest | None = None,
    ) -> Page[Task]:
        """Fetch tasks assigned to an executor.

        Uses current_user_on_execution filter. For tasks of OTHER users,
        use search() with explicit executor filters.

        Args:
            executor_id: Employee UUID of the executor.
            token: JWT bearer token.
            active_only: When True, exclude completed/cancelled tasks.
            pagination: Pagination parameters.

        Returns:
            Page[Task] for the executor.
        """
        filters = TaskFilter(current_user_on_execution=True)
        page = await self.search(filters, token, pagination)

        if active_only:
            # Filter out completed/cancelled tasks
            filtered_items = [
                t
                for t in page.items
                if t.status not in (TaskStatus.COMPLETED, TaskStatus.OVERDUE)
            ]
            # Create new Page (frozen, cannot mutate items)
            page = Page(
                items=filtered_items,
                page=page.page,
                size=page.size,
                has_next=page.has_next,
                total=page.total,
            )

        return page

    async def search(
        self,
        filters: TaskFilter,
        token: str,
        pagination: PageRequest | None = None,
    ) -> Page[Task]:
        """Search tasks with filters and pagination.

        Args:
            filters: TaskFilter value object.
            token: JWT bearer token.
            pagination: Pagination parameters.

        Returns:
            Page[Task] with slice semantics.
        """
        pag = pagination or PageRequest()
        params = {**filters.as_api_params(), **pag.as_params()}

        data = await self.http_client._make_request(
            method="GET",
            endpoint="api/task",
            token=token,
            params=params,
        )
        data = data or {}

        # Handle both list response and paginated response
        if isinstance(data, list):
            return Page(
                items=TaskMapper.from_dto_list(data),
                page=0,
                size=len(data),
                has_next=False,
            )

        content = data.get("content", [])
        items = TaskMapper.from_dto_list(content)

        return Page(
            items=items,
            page=data.get("number", pag.page),
            size=data.get("size", pag.size),
            has_next=not data.get("last", True),
            total=data.get("totalElements"),
        )

    # ------------------------------------------------------------------
    # Hierarchy traversal (recursive CTE queries on backend)
    # ------------------------------------------------------------------

    async def find_children(
        self,
        task_id: UUID,
        token: str,
        organization_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[Task]:
        """Recursively fetch all child tasks.

        Backend executes WITH RECURSIVE CTE query.

        Args:
            task_id: Root task UUID.
            token: JWT bearer token.
            organization_id: Optional org scope.
            include_deleted: Include soft-deleted tasks.

        Returns:
            List of all descendant tasks including root.
        """
        params: dict = {}
        if organization_id:
            params["organizationId"] = organization_id
        if include_deleted:
            params["includeDeleted"] = "true"

        data = await self.http_client._make_request(
            method="GET",
            endpoint=f"api/task/{task_id}/children",
            token=token,
            params=params,
        )
        return TaskMapper.from_dto_list(data or [])

    async def find_parents(
        self,
        task_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[Task]:
        """Walk task parent chain upward to root.

        Backend executes upward recursive CTE query.

        Args:
            task_id: Starting task UUID.
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            List of tasks from given task up to root.
        """
        params: dict = {}
        if organization_id:
            params["organizationId"] = organization_id

        data = await self.http_client._make_request(
            method="GET",
            endpoint=f"api/task/{task_id}/parents",
            token=token,
            params=params,
        )
        return TaskMapper.from_dto_list(data or [])

    async def find_task_tree(
        self,
        task_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[Task]:
        """Fetch full task tree including siblings.

        Backend finds root ancestor, returns all descendants + siblings.

        Args:
            task_id: Any task UUID within the tree.
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            Flat list of all tasks in the tree.
        """
        params: dict = {}
        if organization_id:
            params["organizationId"] = organization_id

        data = await self.http_client._make_request(
            method="GET",
            endpoint=f"api/task/{task_id}/tree",
            token=token,
            params=params,
        )
        return TaskMapper.from_dto_list(data or [])

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def create(
        self,
        document_id: UUID,
        task_text: str,
        deadline: datetime | None,
        executor_ids: list[tuple[UUID, bool]],
        token: str,
        task_type: TaskType = TaskType.EXECUTION,
        endless: bool = False,
    ) -> bool:
        """Create new task via batch API.

        Args:
            document_id: Parent document UUID.
            task_text: Task instruction text.
            deadline: Planned completion date (ignored if endless=True).
            executor_ids: List of (employee_uuid, is_responsible) tuples.
            token: JWT bearer token.
            task_type: Workflow type (default EXECUTION).
            endless: Create task without deadline.

        Returns:
            True on success, False on API error.

        Raises:
            ValueError: When executor_ids is empty.
        """
        if not executor_ids:
            raise ValueError("At least one executor is required")

        payload = [
            {
                "taskText": task_text,
                "planedDateEnd": (
                    None if endless else (deadline.isoformat() if deadline else None)
                ),
                "type": task_type.value,
                "periodTask": False,
                "endless": endless,
                "executors": [
                    {"employeeId": str(eid), "responsible": is_resp}
                    for eid, is_resp in executor_ids
                ],
            }
        ]

        try:
            await self.http_client._make_request(
                method="POST",
                endpoint=f"api/document/{document_id}/task/batch",
                token=token,
                json=payload,
                is_json_response=False,
            )
            logger.info(
                f"Task created on document {document_id}",
                extra={
                    "document_id": str(document_id),
                    "executors": len(executor_ids),
                    "endless": endless,
                },
            )
            return True
        except Exception as exc:
            logger.error(
                f"Failed to create task on document {document_id}",
                exc_info=True,
                extra={"document_id": str(document_id)},
            )
            return False
