# mcp-server/clients/attachment_client.py
"""EDMS Attachment HTTP Client. Перенесён из edms_ai_assistant/clients/attachment_client.py."""
from __future__ import annotations

import logging

from .base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class EdmsAttachmentClient(EdmsHttpClient):
    """Клиент для работы с вложениями документов."""

    async def get_attachment_content(
        self, token: str, document_id: str, attachment_id: str
    ) -> bytes:
        """
        Скачать содержимое вложения (сырые байты).

        GET api/document/{documentId}/attachment/{attachmentId}
        """
        logger.debug("Запрос вложения %s для документа %s", attachment_id, document_id)
        result = await self._make_request(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}",
            token=token,
            is_json_response=False,
            long_timeout=True,
        )
        return result or b""