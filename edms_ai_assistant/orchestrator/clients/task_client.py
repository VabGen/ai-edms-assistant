# edms_ai_assistant/orchestrator/clients/task_client.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from edms_ai_assistant.orchestrator.models.task_models import CreateTaskRequest

from .base_client import EdmsBaseClient, EdmsHttpClient

logger = logging.getLogger(__name__)


class BaseTaskClient(EdmsBaseClient, ABC):
    """Abstract interface for task API clients."""

    @abstractmethod
    async def create_tasks_batch(
        self, token: str, document_id: str, tasks: list[CreateTaskRequest]
    ) -> bool:
        raise NotImplementedError


class TaskClient(BaseTaskClient, EdmsHttpClient):
    """Concrete async HTTP client for EDMS task API."""

    async def create_tasks_batch(
        self,
        token: str,
        document_id: str,
        tasks: list[CreateTaskRequest],
    ) -> bool:
        """Create a batch of tasks for a document.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses (NOT caught here —
                caller is responsible for handling permission errors etc.).

        Args:
            token: Bearer token.
            document_id: Target document UUID.
            tasks: List of task request objects.

        Returns:
            True on success.
        """
        if not tasks:
            logger.warning("[TASK-CLIENT] Empty tasks list — skipping API call")
            return False

        endpoint = f"api/document/{document_id}/task/batch"
        payload = [task.model_dump(mode="json") for task in tasks]

        await self._make_request(
            "POST",
            endpoint,
            token=token,
            json=payload,
            is_json_response=False,
        )

        logger.info(
            "[TASK-CLIENT] Successfully created %d task(s) for document %s",
            len(tasks),
            document_id,
        )
        return True
