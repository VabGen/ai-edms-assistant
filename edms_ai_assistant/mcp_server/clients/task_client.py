# mcp-server/clients/task_client.py
"""EDMS Task HTTP Client. Перенесён из edms_ai_assistant/clients/task_client.py."""
from __future__ import annotations

import logging

from .base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class TaskClient(EdmsHttpClient):
    """Async HTTP-клиент для EDMS Task API."""

    async def create_tasks_batch(
        self,
        token: str,
        document_id: str,
        tasks: list[dict],
    ) -> bool:
        """
        Создать пакет поручений: POST api/document/{documentId}/task/batch.

        Args:
            token: JWT-токен.
            document_id: UUID документа.
            tasks: Список поручений в формате dict (уже сериализованных).

        Returns:
            True при успехе.
        """
        if not tasks:
            logger.warning("Empty tasks list — skipping API call")
            return False

        endpoint = f"api/document/{document_id}/task/batch"
        await self._make_request(
            "POST", endpoint,
            token=token, json=tasks, is_json_response=False,
        )
        logger.info("Created %d task(s) for document %s", len(tasks), document_id)
        return True