# src/ai_edms_assistant/infrastructure/edms_api/clients/document_client.py
"""EDMS Document HTTP Client — wraps /api/document/* endpoints."""

from __future__ import annotations

import structlog
from typing import Any
from uuid import UUID

from ..http_client import EdmsHttpClient
from ai_edms_assistant.domain.value_objects.filters import (
    DocumentFilter,
    DocumentLinkFilter,
)
from ....domain.repositories.base import PageRequest

logger = structlog.get_logger(__name__)


class EdmsDocumentClient(EdmsHttpClient):
    """
    Low-level async client for EDMS /api/document/* endpoints.

    Returns raw dict / list[dict] (API DTOs).
    Domain mapping is in infrastructure/mappers/document_mapper.py.

    Inherits EdmsHttpClient for transport, retry logic, and auth headers.
    """

    async def get_by_id(
        self,
        document_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        GET /api/document/{id}

        Returns:
            Raw DocumentDto dict or None on 404.
        """
        params: dict[str, Any] = {}
        if organization_id:
            params["organizationId"] = organization_id
        try:
            return await self._make_request(
                "GET", f"api/document/{document_id}", token=token, params=params
            )
        except Exception as exc:
            logger.error(
                "document_get_by_id_failed", doc_id=str(document_id), error=str(exc)
            )
            raise

    async def search(
        self,
        filters: DocumentFilter,
        token: str,
        pagination: PageRequest | None = None,
    ) -> dict[str, Any]:
        """
        GET /api/document with DocumentFilter serialized as query params.

        Runs filters.validate() (domain rules) before the request.

        Returns:
            Spring Page: {content, number, size, last, totalElements}.
        """
        filters.validate()
        pag = pagination or PageRequest()
        params = {**filters.as_api_params(), **pag.as_params()}
        data = await self._make_request(
            "GET", "api/document", token=token, params=params
        )
        return data or {}

    async def get_versions(
        self,
        document_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """GET /api/document/{id}/version — all document versions."""
        data = await self._make_request(
            "GET", f"api/document/{document_id}/version", token=token
        )
        return data if isinstance(data, list) else []

    async def get_links(
        self,
        filters: DocumentLinkFilter,
        token: str,
    ) -> list[dict[str, Any]]:
        """GET /api/document/{docId}/nomenclature-affair-document-link"""
        if not filters.doc_id:
            raise ValueError("DocumentLinkFilter.doc_id is required")
        params = filters.as_api_params()
        params.pop("docId", None)
        data = await self._make_request(
            "GET",
            f"api/document/{filters.doc_id}/nomenclature-affair-document-link",
            token=token,
            params=params,
        )
        return data if isinstance(data, list) else []

    async def execute_operation(
        self,
        document_id: UUID,
        operation: str,
        payload: dict[str, Any],
        token: str,
    ) -> None:
        """
        POST /api/document/{id}/execute

        Args:
            operation: One of DocumentOperationType enum values
                       (DOCUMENT_MAIN_FIELDS_UPDATE, DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE, …).
            payload:   Operation-specific dict (see resources_openapi.py for schemas).
        """
        await self._make_request(
            "POST",
            f"api/document/{document_id}/execute",
            token=token,
            json=[{"type": operation, **payload}],
            is_json_response=False,
        )

    async def get_responsible(
        self,
        document_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """GET /api/document/{id}/responsible — contract responsible executors."""
        data = await self._make_request(
            "GET", f"api/document/{document_id}/responsible", token=token
        )
        return data if isinstance(data, list) else []

    async def get_tasks(
        self,
        document_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """GET /api/document/{id}/task — tasks linked to document."""
        data = await self._make_request(
            "GET", f"api/document/{document_id}/task", token=token
        )
        return data if isinstance(data, list) else []
