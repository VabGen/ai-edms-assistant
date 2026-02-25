# src/ai_edms_assistant/infrastructure/edms_api/clients/attachment_client.py
"""EDMS Attachment HTTP Client — binary file download."""

from __future__ import annotations

import structlog
from uuid import UUID

from ..http_client import EdmsHttpClient

logger = structlog.get_logger(__name__)


class EdmsAttachmentClient(EdmsHttpClient):
    """
    Low-level async client for EDMS attachment content endpoints.

    Downloads raw file bytes. No JSON response — is_json_response=False always.
    Uses long_timeout=True for large files.
    """

    async def get_content(
        self,
        document_id: UUID,
        attachment_id: UUID,
        token: str,
    ) -> bytes:
        """
        GET /api/document/{documentId}/attachment/{attachmentId}

        Args:
            document_id:   UUID of the parent document.
            attachment_id: UUID of the attachment file.
            token:         JWT bearer token.

        Returns:
            Raw file bytes.
        """
        logger.debug(
            "attachment_download_start",
            doc_id=str(document_id),
            att_id=str(attachment_id),
        )
        return await self._make_request(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}",
            token=token,
            is_json_response=False,
            long_timeout=True,
        )

    async def get_metadata(
        self,
        document_id: UUID,
        attachment_id: UUID,
        token: str,
    ) -> dict:
        """
        GET /api/document/{documentId}/attachment/{attachmentId}/metadata

        Returns:
            AttachmentDto dict (id, fileName, fileSize, mimeType, versionNumber).
        """
        data = await self._make_request(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}/metadata",
            token=token,
        )
        return data or {}
