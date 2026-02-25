# src/ai_edms_assistant/infrastructure/edms_api/clients/task_client.py
"""EDMS Task HTTP Client — wraps /api/task/* and /api/document/{id}/task/* endpoints."""

from __future__ import annotations

import structlog
from datetime import datetime
from typing import Any
from uuid import UUID

from ..http_client import EdmsHttpClient
from ai_edms_assistant.domain.value_objects.filters import TaskFilter
from ai_edms_assistant.domain.entities import TaskType
from ....domain.repositories.base import PageRequest

logger = structlog.get_logger(__name__)


class EdmsTaskClient(EdmsHttpClient):
    """
    Low-level async client for EDMS /api/task/* endpoints.

    Task search uses GET (not POST) with query params from TaskFilter.
    Task creation uses POST /api/document/{id}/task/batch.
    Hierarchy traversal uses dedicated tree/children/parents endpoints.

    Returns raw dicts — domain mapping is in task_mapper.py.
    """

    async def get_by_id(
        self,
        task_id: UUID,
        token: str,
    ) -> dict[str, Any] | None:
        """
        GET /api/task/{id}

        Returns:
            Raw TaskDto dict or None on 404.
        """
        try:
            return await self._make_request("GET", f"api/task/{task_id}", token=token)
        except Exception as exc:
            logger.error("task_get_by_id_failed", task_id=str(task_id), error=str(exc))
            raise

    async def search(
        self,
        filters: TaskFilter,
        token: str,
        pagination: PageRequest | None = None,
    ) -> dict[str, Any]:
        """
        GET /api/task with TaskFilter as query params.

        EDMS uses GET (not POST) for task search.

        Returns:
            Spring Page or plain list depending on API version.
        """
        pag = pagination or PageRequest()
        params = {**filters.as_api_params(), **pag.as_params()}
        data = await self._make_request("GET", "api/task", token=token, params=params)
        return data or {}

    async def get_by_document(
        self,
        document_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """GET /api/document/{id}/task"""
        data = await self._make_request(
            "GET", f"api/document/{document_id}/task", token=token
        )
        return data if isinstance(data, list) else []

    async def get_children(
        self,
        task_id: UUID,
        token: str,
        organization_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """
        GET /api/task/{id}/children

        Backend runs recursive CTE (findAllChild).
        """
        params: dict[str, Any] = {}
        if organization_id:
            params["organizationId"] = organization_id
        if include_deleted:
            params["includeDeleted"] = True
        data = await self._make_request(
            "GET", f"api/task/{task_id}/children", token=token, params=params
        )
        return data if isinstance(data, list) else []

    async def get_parents(
        self,
        task_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        GET /api/task/{id}/parents

        Backend runs upward recursive CTE (findAllParent).
        """
        params: dict[str, Any] = {}
        if organization_id:
            params["organizationId"] = organization_id
        data = await self._make_request(
            "GET", f"api/task/{task_id}/parents", token=token, params=params
        )
        return data if isinstance(data, list) else []

    async def get_tree(
        self,
        task_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        GET /api/task/{id}/tree

        Returns root ancestor + all descendants + siblings.
        findAllParentByTaskId, ordered by createDate DESC.
        """
        params: dict[str, Any] = {}
        if organization_id:
            params["organizationId"] = organization_id
        data = await self._make_request(
            "GET", f"api/task/{task_id}/tree", token=token, params=params
        )
        return data if isinstance(data, list) else []

    async def create_batch(
        self,
        document_id: UUID,
        tasks: list[dict[str, Any]],
        token: str,
    ) -> bool:
        """
        POST /api/document/{id}/task/batch

        Args:
            tasks: List of task dicts. Each dict must have:
                   taskText, planedDateEnd, type, endless, periodTask, executors[].
                   Prepare with _build_task_payload() helper below.

        Returns:
            True on 200/201/204, False on error.
        """
        if not tasks:
            logger.warning("task_create_batch_empty")
            return False
        try:
            await self._make_request(
                "POST",
                f"api/document/{document_id}/task/batch",
                token=token,
                json=tasks,
                is_json_response=False,
            )
            logger.info("task_batch_created", doc_id=str(document_id), count=len(tasks))
            return True
        except Exception as exc:
            logger.error(
                "task_create_batch_failed", doc_id=str(document_id), error=str(exc)
            )
            return False

    @staticmethod
    def build_task_payload(
        task_text: str,
        executor_ids: list[tuple[UUID, bool]],
        deadline: datetime | None = None,
        task_type: TaskType = TaskType.EXECUTION,
        endless: bool = False,
    ) -> dict[str, Any]:
        """
        Build a single task dict for create_batch().

        Args:
            task_text:    Task description text.
            executor_ids: List of (employee_id, is_responsible) tuples.
            deadline:     Deadline datetime (ignored when endless=True).
            task_type:    TaskType enum value.
            endless:      Whether task has no deadline.

        Returns:
            Dict ready for inclusion in create_batch tasks list.

        Raises:
            ValueError: When executor_ids is empty.
        """
        if not executor_ids:
            raise ValueError("At least one executor is required")
        return {
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
