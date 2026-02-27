# src/ai_edms_assistant/infrastructure/edms_api/repositories/edms_document_repository.py
"""EDMS Document Repository — REST API implementation."""

from __future__ import annotations

import asyncio
import structlog
from typing import Any
from uuid import UUID

from ai_edms_assistant.infrastructure.edms_api.http_client import EdmsHttpClient
from ....domain.entities.appeal import DocumentAppeal
from ....domain.entities.attachment import Attachment
from ....domain.entities.document import Document, DocumentStatus
from ....domain.entities.employee import UserInfo
from ai_edms_assistant.domain.value_objects.filters import (
    DocumentFilter,
    DocumentLinkFilter,
)
from ....domain.repositories.base import Page, PageRequest
from ....domain.repositories.document_repository import AbstractDocumentRepository

logger = structlog.get_logger(__name__)


class EdmsDocumentRepository(AbstractDocumentRepository):
    """EDMS REST API implementation of AbstractDocumentRepository.

    Uses EdmsHttpClient._make_request() exclusively.
    All filters come from domain/filters.py.

    Attributes:
        http_client: Shared async EdmsHttpClient instance.
    """

    def __init__(self, http_client: EdmsHttpClient) -> None:
        """Initialize repository with shared HTTP client.

        Args:
            http_client: Configured EdmsHttpClient instance.
        """
        self.http_client = http_client

    # ------------------------------------------------------------------
    # Null-safe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_uuid(raw: Any) -> UUID | None:
        """Parse UUID with None fallback — never raises.

        Args:
            raw: Any value — UUID string, UUID object, or None/null.

        Returns:
            UUID instance or None.
        """
        if not raw:
            return None
        try:
            return UUID(str(raw))
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _safe_str(raw: Any, fallback: str = "") -> str:
        """Return string value with fallback — never raises.

        Args:
            raw: Any value.
            fallback: Default string if raw is None/empty.

        Returns:
            String value or fallback.
        """
        if raw is None:
            return fallback
        return str(raw)

    # ------------------------------------------------------------------
    # Internal mapper: raw API dict → domain entities
    # ------------------------------------------------------------------

    def _map_to_entity(self, data: dict[str, Any]) -> Document | None:
        """Map raw EDMS DocumentDto dict to domain Document.

        Args:
            data: Raw dict from EDMS API response.

        Returns:
            Populated Document entity, or None if ``id`` is absent.
        """
        doc_id = self._safe_uuid(data.get("id"))
        if doc_id is None:
            logger.warning(
                "document_mapper_missing_id",
                extra={"keys": list(data.keys())[:10]},
            )
            return None

        # ── Status ───────────────────────────────────────────────────────────
        status: DocumentStatus | None = None
        raw_state = data.get("state") or data.get("status")
        if raw_state:
            try:
                status = DocumentStatus(raw_state)
            except ValueError:
                logger.warning(
                    "unknown_document_status",
                    extra={"raw": raw_state, "doc_id": str(doc_id)},
                )

        # ── Participants ──────────────────────────────────────────────────────
        author = self._map_user(data.get("author"))
        responsible_executor = self._map_user(data.get("responsibleExecutor"))
        initiator = self._map_user(data.get("initiator"))

        # ── Attachments ───────────────────────────────────────────────────────
        raw_attachments = (
            data.get("attachmentDocument") or data.get("attachments") or []
        )
        attachments: list[Attachment] = []
        for a in raw_attachments:
            if not isinstance(a, dict):
                continue
            att_id = self._safe_uuid(a.get("id"))
            if att_id is None:
                logger.debug(
                    "attachment_missing_id_skipped",
                    extra={"doc_id": str(doc_id)},
                )
                continue
            try:
                attachments.append(
                    Attachment(
                        id=att_id,
                        file_name=a.get("fileName") or a.get("name") or "unnamed",
                        file_size=a.get("fileSize") or a.get("size") or 0,
                        mime_type=a.get("mimeType"),
                        version_number=a.get("versionNumber") or 1,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "attachment_map_failed",
                    extra={"doc_id": str(doc_id), "error": str(exc)},
                )

        # ── Appeal ────────────────────────────────────────────────────────────
        appeal: DocumentAppeal | None = None
        appeal_raw = data.get("documentAppeal")
        if appeal_raw and isinstance(appeal_raw, dict):
            try:
                appeal = DocumentAppeal(
                    id=self._safe_uuid(appeal_raw.get("id")),
                    appeal_number=appeal_raw.get("appealNumber"),
                    applicant_name=(
                        appeal_raw.get("fioApplicant")
                        or appeal_raw.get("applicantName")
                    ),
                    description=appeal_raw.get("description"),
                )
            except Exception as exc:
                logger.warning(
                    "appeal_map_failed",
                    extra={
                        "doc_id": str(doc_id),
                        "appeal_id": appeal_raw.get("id"),
                        "error": str(exc),
                    },
                )

        # ── Build Document ────────────────────────────────────────────────────
        try:
            return Document(
                id=doc_id,
                organization_id=data.get("organizationId"),
                reg_number=data.get("regNumber"),
                reg_date=data.get("regDate"),
                create_date=data.get("createDate"),
                short_summary=data.get("shortSummary"),
                summary=data.get("summary"),
                status=status,
                document_type_name=data.get("documentTypeName"),
                author=author,
                responsible_executor=responsible_executor,
                initiator=initiator,
                attachments=attachments,
                appeal=appeal,
                custom_fields=data.get("customFields") or {},
            )
        except Exception as exc:
            logger.error(
                "document_entity_build_failed",
                extra={"doc_id": str(doc_id), "error": str(exc)},
            )
            return None

    def _map_user(self, u: dict | None) -> UserInfo | None:
        """Map raw user dict to UserInfo value object.

        Null-safe: returns None for None input or missing ``id``.

        Args:
            u: Raw user dict from API, or None.

        Returns:
            UserInfo or None.
        """
        if not u or not isinstance(u, dict):
            return None
        try:
            return UserInfo(
                id=self._safe_uuid(u.get("id")),
                name=u.get("name") or u.get("fullName") or "Не указано",
                organization_id=u.get("organizationId"),
                department_name=u.get("departmentName"),
                post_name=u.get("postName"),
                email=u.get("email"),
            )
        except Exception as exc:
            logger.debug(
                "user_map_failed",
                extra={"error": str(exc)},
            )
            return None

    # ------------------------------------------------------------------
    # AbstractRepository base
    # ------------------------------------------------------------------

    async def get_by_id(
        self,
        entity_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> Document | None:
        """Fetch document by UUID.

        GET /api/document/{id}

        Args:
            entity_id: Document UUID.
            token: JWT bearer token.
            organization_id: Optional org scope for multi-tenant.

        Returns:
            Domain Document or None if not found / id=null in response.
        """
        params: dict[str, Any] = {}
        if organization_id:
            params["organizationId"] = organization_id

        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{entity_id}",
                token=token,
                params=params,
            )
            if not data:
                return None
            return self._map_to_entity(data)

        except Exception as exc:
            logger.error(
                "edms_document_get_by_id_failed",
                doc_id=str(entity_id),
                error=str(exc),
            )
            raise

    async def get_by_ids(
        self,
        entity_ids: list[UUID],
        token: str,
        organization_id: str | None = None,
    ) -> list[Document]:
        """Batch-fetch documents via parallel GET calls.

        EDMS API has no dedicated batch-by-ids endpoint.
        Uses asyncio.gather for concurrent requests.

        Args:
            entity_ids: List of document UUIDs.
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            List of successfully mapped documents (failures silently skipped).
        """
        if not entity_ids:
            return []

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
        """Fetch paginated document list with empty filter.

        Args:
            token: JWT bearer token.
            organization_id: Optional org scope.
            pagination: Page/size/sort params.

        Returns:
            Page[Document].
        """
        return await self.search(DocumentFilter(), token, pagination)

    # ------------------------------------------------------------------
    # AbstractDocumentRepository
    # ------------------------------------------------------------------

    async def get_by_reg_number(
        self,
        reg_number: str,
        token: str,
        organization_id: str | None = None,
    ) -> Document | None:
        """Find document by registration number.

        Args:
            reg_number: Exact or partial registration number.
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            Best-matching Document or None.
        """
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
        """Fetch document with attachments pre-populated.

        EDMS returns attachments inline in the standard GET /api/document/{id}
        response — no separate API call needed.

        Args:
            document_id: Document UUID.
            token: JWT bearer token.

        Returns:
            Document with attachments list, or None.
        """
        return await self.get_by_id(document_id, token)

    async def get_versions(
        self,
        document_id: UUID,
        token: str,
    ) -> list[Document]:
        """Fetch all historical versions of a document.

        GET /api/document/{id}/version

        Args:
            document_id: Document UUID.
            token: JWT bearer token.

        Returns:
            List of Document version snapshots. Empty list on error.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/version",
                token=token,
            )
            result: list[Document] = []
            for v in data or []:
                doc = self._map_to_entity(v)
                if doc is not None:
                    result.append(doc)
            return result
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
        """Search documents with rich filtering.

        GET /api/document with DocumentFilter serialized as query params.
        Calls filters.validate() before the HTTP request.

        Args:
            filters: DocumentFilter value object.
            token: JWT bearer token.
            pagination: Page/size/sort params. Defaults to page 0, size 20.

        Returns:
            Page[Document] with Spring Page semantics.
        """
        filters.validate()
        pag = pagination or PageRequest()
        params = {**filters.as_api_params(), **pag.as_params()}

        data = await self.http_client._make_request(
            "GET", "api/document", token=token, params=params
        )
        data = data or {}

        items: list[Document] = []
        for row in data.get("content", []):
            doc = self._map_to_entity(row)
            if doc is not None:
                items.append(doc)

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
        """Fetch documents for a specific organization.

        Args:
            organization_id: EDMS organization identifier.
            token: JWT bearer token.
            pagination: Pagination parameters.

        Returns:
            Page[Document] scoped to the organization.
        """
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
        """Fetch document cross-references.

        GET /api/document/{docId}/nomenclature-affair-document-link

        Args:
            filters: DocumentLinkFilter with required doc_id.
            token: JWT bearer token.

        Returns:
            List of link dicts.

        Raises:
            ValueError: When filters.doc_id is not set.
        """
        if not filters.doc_id:
            raise ValueError("DocumentLinkFilter.doc_id is required for get_links()")

        params = filters.as_api_params()
        params.pop("docId", None)

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
        """Execute named field-update operation.

        POST /api/document/{id}/execute

        Args:
            document_id: Target document UUID.
            operation: EDMS operation constant (e.g. DOCUMENT_MAIN_FIELDS_UPDATE).
            payload: Operation-specific field dict.
            token: JWT bearer token.

        Returns:
            True on HTTP 2xx success, False on any API error.
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
