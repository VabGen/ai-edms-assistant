# src/ai_edms_assistant/infrastructure/edms_api/repositories/edms_document_repository.py
from __future__ import annotations

import structlog
from uuid import UUID
from typing import Any

from ai_edms_assistant.infrastructure.edms_api.http_client import EdmsHttpClient
from ....domain.entities.document import Document, DocumentStatus
from ....domain.entities.employee import UserInfo
from ....domain.entities.attachment import Attachment
from ....domain.entities.appeal import DocumentAppeal
from ai_edms_assistant.domain.value_objects.filters import (
    DocumentFilter,
    DocumentLinkFilter,
)
from ....domain.repositories.document_repository import AbstractDocumentRepository
from ....domain.repositories.base import Page, PageRequest

logger = structlog.get_logger(__name__)


class EdmsDocumentRepository(AbstractDocumentRepository):
    """
    EDMS REST API implementation of AbstractDocumentRepository.

    Uses EdmsHttpClient._make_request() exclusively — no .get()/.post() shortcuts.
    All filters come from domain/filters.py; resources_openapi.py is NOT used.

    Attributes:
        http_client: Shared async EdmsHttpClient instance.
    """

    def __init__(self, http_client: EdmsHttpClient) -> None:
        self.http_client = http_client

    # ------------------------------------------------------------------
    # Mapper: raw API dict → domain Document
    # ------------------------------------------------------------------

    def _map_to_entity(self, data: dict[str, Any]) -> Document:
        """
        Maps a raw EDMS API DocumentDto dict to a domain Document.

        Handles nested author/executor/initiator UserInfo objects and the
        optional attachments + documentAppeal nested structures.
        """

        def _user(u: dict | None) -> UserInfo | None:
            if not u:
                return None
            return UserInfo(
                id=UUID(u["id"]) if u.get("id") else None,
                name=u.get("name") or u.get("fullName", "Не указано"),
                organization_id=u.get("organizationId"),
                department_name=u.get("departmentName"),
                post_name=u.get("postName"),
                email=u.get("email"),
            )

        attachments: list[Attachment] = [
            Attachment(
                id=UUID(a["id"]),
                file_name=a.get("fileName", "unnamed"),
                file_size=a.get("fileSize", 0),
                mime_type=a.get("mimeType"),
                version_number=a.get("versionNumber", 1),
            )
            for a in data.get("attachments", [])
        ]

        appeal_raw = data.get("documentAppeal")
        appeal = (
            DocumentAppeal(
                id=UUID(appeal_raw["id"]),
                appeal_number=appeal_raw.get("appealNumber"),
                applicant_name=appeal_raw.get("applicantName"),
                description=appeal_raw.get("description"),
            )
            if appeal_raw
            else None
        )

        status: DocumentStatus | None = None
        raw_state = data.get("state") or data.get("status")
        if raw_state:
            try:
                status = DocumentStatus(raw_state)
            except ValueError:
                logger.warning(
                    "unknown_document_status", raw=raw_state, doc_id=data.get("id")
                )

        return Document(
            id=UUID(data["id"]),
            organization_id=data.get("organizationId"),
            reg_number=data.get("regNumber"),
            reg_date=data.get("regDate"),
            create_date=data.get("createDate"),
            short_summary=data.get("shortSummary"),
            summary=data.get("summary"),
            status=status,
            document_type_name=data.get("documentTypeName"),
            author=_user(data.get("author")),
            responsible_executor=_user(data.get("responsibleExecutor")),
            initiator=_user(data.get("initiator")),
            attachments=attachments,
            appeal=appeal,
            custom_fields=data.get("customFields", {}),
        )

    # ------------------------------------------------------------------
    # BaseRepository
    # ------------------------------------------------------------------

    async def get_by_id(
        self,
        entity_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> Document | None:
        """GET /api/document/{id}"""
        params: dict = {}
        if organization_id:
            params["organizationId"] = organization_id

        try:
            data = await self.http_client._make_request(
                "GET", f"api/document/{entity_id}", token=token, params=params
            )
            return self._map_to_entity(data) if data else None
        except Exception as exc:
            logger.error(
                "edms_document_get_by_id_failed", doc_id=str(entity_id), error=str(exc)
            )
            raise

    async def get_by_ids(
        self,
        entity_ids: list[UUID],
        token: str,
        organization_id: str | None = None,
    ) -> list[Document]:
        """
        Batch-fetch documents via parallel GET /api/document/{id} calls.

        EDMS API has no dedicated batch-by-ids endpoint for documents.
        Uses asyncio.gather for concurrent requests.
        """
        if not entity_ids:
            return []
        import asyncio

        results = await asyncio.gather(
            *[self.get_by_id(eid, token, organization_id) for eid in entity_ids],
            return_exceptions=True,
        )
        return [r for r in results if isinstance(r, Document)]

    async def find_page(
        self,
        token: str,
        organization_id: str | None = None,
        pagination: PageRequest | None = None,
    ) -> Page[Document]:
        """Delegates to search() with an empty DocumentFilter."""
        f = DocumentFilter()
        return await self.search(f, token, pagination)

    # ------------------------------------------------------------------
    # AbstractDocumentRepository-specific
    # ------------------------------------------------------------------

    async def get_by_reg_number(
        self,
        reg_number: str,
        token: str,
        organization_id: str | None = None,
    ) -> Document | None:
        """Delegates to search() with regNumber filter, size=1."""
        page = await self.search(
            DocumentFilter(reg_number=reg_number),
            token,
            PageRequest(size=1),
        )
        return page.items[0] if page.items else None

    async def get_with_attachments(
        self,
        document_id: UUID,
        token: str,
    ) -> Document | None:
        """
        GET /api/document/{id}

        _map_to_entity already parses attachments[] from the standard response.
        No separate API call needed — EDMS returns attachments inline.
        """
        return await self.get_by_id(document_id, token)

    async def get_versions(
        self,
        document_id: UUID,
        token: str,
    ) -> list[Document]:
        """GET /api/document/{id}/version"""
        try:
            data = await self.http_client._make_request(
                "GET", f"api/document/{document_id}/version", token=token
            )
            return [self._map_to_entity(v) for v in (data or [])]
        except Exception as exc:
            logger.error(
                "edms_document_get_versions_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            raise

    async def search(
        self,
        filters: DocumentFilter,
        token: str,
        pagination: PageRequest | None = None,
    ) -> Page[Document]:
        """
        GET /api/document with DocumentFilter serialized as query params.

        Calls filters.validate() before the HTTP request to enforce
        domain rules (DocumentFilterValidator.validate()).
        """
        filters.validate()
        pag = pagination or PageRequest()
        params = {**filters.as_api_params(), **pag.as_params()}

        data = await self.http_client._make_request(
            "GET", "api/document", token=token, params=params
        )
        data = data or {}

        items = [self._map_to_entity(row) for row in data.get("content", [])]
        return Page(
            items=items,
            page=data.get("number", pag.page),
            size=data.get("size", pag.size),
            has_next=not data.get("last", True),
            total=data.get("totalElements"),
        )

    async def find_by_organization(
        self,
        organization_id: str,
        token: str,
        pagination: PageRequest | None = None,
    ) -> Page[Document]:
        """Delegates to search() with onlyUserOrganization=True."""
        return await self.search(
            DocumentFilter(only_user_organization=True),
            token,
            pagination,
        )

    async def get_links(
        self,
        filters: DocumentLinkFilter,
        token: str,
    ) -> list[dict]:
        """GET /api/document/{docId}/nomenclature-affair-document-link"""
        if not filters.doc_id:
            raise ValueError("DocumentLinkFilter.doc_id is required for get_links()")

        params = filters.as_api_params()
        params.pop("docId", None)  # already in URL path

        data = await self.http_client._make_request(
            "GET",
            f"api/document/{filters.doc_id}/nomenclature-affair-document-link",
            token=token,
            params=params,
        )
        return data if isinstance(data, list) else []

    async def update_fields(
        self,
        document_id: UUID,
        operation: str,
        payload: dict,
        token: str,
    ) -> bool:
        """
        POST /api/document/{id}/execute

        Wraps the DocumentOperationExecutor pattern used in appeal_autofill.py.
        Operations: DOCUMENT_MAIN_FIELDS_UPDATE, DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE, etc.
        """
        try:
            await self.http_client._make_request(
                "POST",
                f"api/document/{document_id}/execute",
                token=token,
                json=[{"type": operation, **payload}],
                is_json_response=False,
            )
            return True
        except Exception as exc:
            logger.error(
                "edms_document_update_fields_failed",
                doc_id=str(document_id),
                operation=operation,
                error=str(exc),
            )
            return False
