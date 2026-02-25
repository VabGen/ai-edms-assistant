# src/ai_edms_assistant/domain/repositories/task_repository.py
from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from uuid import UUID

from ..entities.task import Task, TaskType
from ..value_objects.filters import TaskFilter
from .base import AbstractRepository, Page, PageRequest


class AbstractTaskRepository(AbstractRepository[Task]):
    """Port (interface) for task / assignment (поручение) data access.

    Defines all task operations available to the application layer,
    including document-scoped lookup, executor-scoped queries, rich filtered
    search, recursive hierarchy traversal, and task creation.

    Implementation:
        ``infrastructure/edms_api/repositories/edms_task_repository.py``

    Consumers:
        - ``application/tools/task_tool.py`` — create, get by document
        - ``application/use_cases/create_task.py`` — task creation
        - ``application/use_cases/summarize_document.py`` — task context

    Note:
        ``TaskType`` is imported from ``domain/entities/task.py`` — it is a
        domain enum, not a filter type. ``TaskFilter`` comes from
        ``domain/value_objects/filters.py``.
    """

    # ------------------------------------------------------------------
    # Document-scoped lookup
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_by_document_id(
        self,
        document_id: UUID,
        token: str,
    ) -> list[Task]:
        """Fetch all tasks associated with a specific document.

        The primary method used when the agent answers
        "какие поручения по документу №01/123?".

        Calls ``GET /api/task?documentId={id}`` or
        ``GET /api/document/{id}/task-task-project``.

        Args:
            document_id: Parent document UUID.
            token: JWT bearer token.

        Returns:
            List of ``Task`` objects linked to the document.
            Returns ``[]`` when the document has no associated tasks.
        """

    # ------------------------------------------------------------------
    # Executor-scoped lookup
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_by_executor(
        self,
        executor_id: UUID,
        token: str,
        active_only: bool = True,
        pagination: PageRequest | None = None,
    ) -> Page[Task]:
        """Fetch all tasks assigned to a specific employee.

        Used when the agent answers "какие поручения у Иванова?".

        Calls ``POST /api/task/search`` with an executor UUID filter.
        When ``active_only=True``, excludes tasks with status
        ``COMPLETED``, ``OVERDUE`` (closed), and ``CANCEL``.

        Args:
            executor_id: Employee UUID of the executor (not the controller).
            token: JWT bearer token.
            active_only: When ``True``, exclude completed / cancelled tasks.
                Default is ``True``.
            pagination: Pagination parameters.

        Returns:
            ``Page[Task]`` for the given executor.
        """

    # ------------------------------------------------------------------
    # Filtered search
    # ------------------------------------------------------------------

    @abstractmethod
    async def search(
        self,
        filters: TaskFilter,
        token: str,
        pagination: PageRequest | None = None,
    ) -> Page[Task]:
        """Search tasks with rich filtering and pagination.

        Wraps ``POST /api/task/search`` with the full ``TaskFilter``
        parameter set. Used for complex agent queries like
        "покажи просроченные поручения текущего пользователя за март".

        Args:
            filters: ``TaskFilter`` value object from
                ``domain/value_objects/filters.py``. All fields are optional.
            token: JWT bearer token.
            pagination: Page/size/sort params.

        Returns:
            ``Page[Task]`` with Slice semantics.
        """

    # ------------------------------------------------------------------
    # Hierarchy traversal
    # ------------------------------------------------------------------

    @abstractmethod
    async def find_children(
        self,
        task_id: UUID,
        token: str,
        organization_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[Task]:
        """Recursively fetch all child (sub-)tasks of a given task.

        Mirrors Java ``findAllChild`` implemented as a WITH RECURSIVE CTE
        that traverses the ``parent_id`` tree downward.

        Used when the agent shows the full sub-task breakdown:
        "покажи все подпоручения поручения X".

        Args:
            task_id: Root task UUID.
            token: JWT bearer token.
            organization_id: Org scope.
            include_deleted: When ``True``, include soft-deleted tasks.
                Mirrors the Java CTE variant without ``deleted = false``.

        Returns:
            Flat list of all descendant ``Task`` objects including the root.
            Returns ``[root_task]`` when no children exist.
        """

    @abstractmethod
    async def find_parents(
        self,
        task_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[Task]:
        """Walk the task parent chain upward to the root.

        Mirrors Java ``findAllParent`` implemented as a WITH RECURSIVE CTE
        traversing ``parent_id`` upward.

        Used when the agent provides context:
        "в рамках какого поручения создано это подпоручение?".

        Args:
            task_id: Starting task UUID.
            token: JWT bearer token.
            organization_id: Org scope.

        Returns:
            List of ``Task`` objects from the given task up to the root,
            ordered from given task toward root.
            Returns ``[task_itself]`` for a root-level task.
        """

    @abstractmethod
    async def find_task_tree(
        self,
        task_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[Task]:
        """Fetch the full task tree including all siblings and descendants.

        Mirrors Java ``findAllParentByTaskId`` — first finds the root
        ancestor, then returns all descendants including sibling branches
        at every level.

        Used to render a complete task tree in the agent's response:
        "покажи всё дерево поручений документа X".

        Args:
            task_id: Any task UUID within the tree (not necessarily the root).
            token: JWT bearer token.
            organization_id: Org scope.

        Returns:
            Flat list of all ``Task`` objects in the tree, ordered by
            ``create_date DESC`` (mirrors Java query ORDER BY).
        """

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def create(
        self,
        document_id: UUID,
        task_text: str,
        deadline: datetime,
        executor_ids: list[tuple[UUID, bool]],
        token: str,
        task_type: TaskType = TaskType.EXECUTION,
        endless: bool = False,
    ) -> bool:
        """Create a new task on a document via the EDMS API.

        Wraps ``TaskClient.create_tasks_batch`` which calls
        ``POST /api/document/{docId}/task/batch``.

        Business rules enforced by the API:
        - At least one executor is required.
        - Exactly one executor must have ``is_responsible=True``.
        - ``deadline`` is required when ``endless=False``.

        Args:
            document_id: Parent document UUID.
            task_text: Text body of the task (поручение).
            deadline: Planned completion date.
            executor_ids: List of ``(employee_uuid, is_responsible)`` tuples.
                The executor with ``is_responsible=True`` is the primary
                responsible person (ответственный исполнитель).
            token: JWT bearer token.
            task_type: Workflow type for the task. Defaults to ``EXECUTION``.
            endless: When ``True``, creates a task without a deadline
                (бессрочное поручение). Overrides ``deadline`` param.

        Returns:
            ``True`` on HTTP 2xx success, ``False`` on any API error.
        """
