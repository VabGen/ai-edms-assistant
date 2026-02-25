# src/ai_edms_assistant/infrastructure/adapters/introduction_adapter.py
"""Adapter for creating introduction lists via EDMS API.

This demonstrates how infrastructure layer provides concrete implementations
that tools depend on. The tool calls this adapter, which calls EDMS API clients.
"""

from __future__ import annotations

from uuid import UUID

from ai_edms_assistant.infrastructure.edms_api.clients.document_client import (
    EdmsDocumentClient,
)


class IntroductionAdapter:
    """Adapter for EDMS introduction list operations.

    NOTE: In full architecture, this would be injected into IntroductionTool
    or wrapped in a repository implementation.
    """

    def __init__(self, base_url: str):
        self._base_url = base_url
        self._client: EdmsDocumentClient | None = None

    async def __aenter__(self):
        self._client = EdmsDocumentClient(base_url=self._base_url)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.__aexit__(*args)

    async def create_introduction(
        self,
        document_id: UUID,
        employee_ids: list[UUID],
        token: str,
        comment: str | None = None,
    ) -> bool:
        """Create introduction list via EDMS API.

        Args:
            document_id: Parent document UUID.
            employee_ids: List of employee UUIDs to add.
            token: JWT bearer token.
            comment: Optional comment.

        Returns:
            True on success, False on failure.

        NOTE: Real implementation would call:
        POST /api/document/{document_id}/introduction
        Body: {"employeeIds": [...], "comment": "..."}
        """
        if not self._client:
            raise RuntimeError("Client not initialized")

        # Placeholder - real implementation would:
        # await self._client.create_introduction(document_id, employee_ids, comment, token)

        # For now, simulate success
        return True
