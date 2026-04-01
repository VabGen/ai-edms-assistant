# edms_ai_assistant/clients/attachment_client.py
import logging

from .base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class EdmsAttachmentClient(EdmsHttpClient):
    """Клиент для работы с контентом и файлами вложений в СЭД."""

    async def get_attachment_content(
        self, token: str, document_id: str, attachment_id: str
    ) -> bytes:
        """
        Скачать содержимое вложения (сырые байты).
        GET api/document/{documentId}/attachment/{attachmentId}
        """
        logger.debug(
            f"Запрос контента вложения {attachment_id} для документа {document_id}"
        )

        return await self._make_request(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}",
            token=token,
            is_json_response=False,
            long_timeout=True,
        )
