# src/ai_edms_assistant/domain/repositories/document_repository.py
from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from ..entities.document import Document
from ..value_objects.filters import DocumentFilter, DocumentLinkFilter
from .base import AbstractRepository, Page, PageRequest


class AbstractDocumentRepository(AbstractRepository[Document]):
    """Port (interface) for document data access.

    Defines the complete set of read and write operations the application
    layer can perform on ``Document`` aggregates. All filtering uses
    hand-crafted domain value objects from ``domain/value_objects/filters.py``
    — never raw dicts or ``resources_openapi.py`` types.

    Implementation:
        ``infrastructure/edms_api/repositories/edms_document_repository.py``

    Consumers:
        - ``application/use_cases/summarize_document.py``
        - ``application/use_cases/compare_documents.py``
        - ``application/use_cases/extract_appeal_data.py``
        - ``application/tools/document_tool.py``
        - ``application/tools/autofill_tool.py``
        - ``application/tools/comparison_tool.py``
    """

    @abstractmethod
    async def get_with_attachments(
        self,
        document_id: UUID,
        token: str,
    ) -> Document | None:
        """Fetch a document with its attachment list pre-populated.

        Calls ``GET /api/document/{id}`` with an ``includes=ATTACHMENTS``
        query parameter. Used by attachment processing tools to avoid a
        separate attachments request.

        Args:
            document_id: Document UUID.
            token: JWT bearer token.

        Returns:
            ``Document`` with ``attachments`` list populated, or ``None``
            when the document does not exist.
        """

    @abstractmethod
    async def get_by_reg_number(
        self,
        reg_number: str,
        token: str,
        organization_id: str | None = None,
    ) -> Document | None:
        """Find a document by its registration number.

        Calls ``GET /api/document?regNumber=...``. Used by the AI agent to
        resolve user references like "документ №01/123-п".

        Args:
            reg_number: Exact or partial registration number string.
            token: JWT bearer token.
            organization_id: Org scope for multi-tenant requests.

        Returns:
            Best-matching ``Document``, or ``None`` when not found.
        """

    @abstractmethod
    async def get_versions(
        self,
        document_id: UUID,
        token: str,
    ) -> list[Document]:
        """Fetch all historical versions of a document.

        Calls ``GET /api/document/{id}/version``. Used by the comparison
        tool to diff document versions.

        Args:
            document_id: UUID of the document whose versions to list.
            token: JWT bearer token.

        Returns:
            List of ``Document`` version snapshots ordered oldest-first.
            Returns ``[]`` when the document has no prior versions.
        """

    @abstractmethod
    async def search(
        self,
        filters: DocumentFilter,
        token: str,
        pagination: PageRequest | None = None,
    ) -> Page[Document]:
        """Search documents with rich filtering and pagination.

        Wraps ``POST /api/document/search`` with the full ``DocumentFilter``
        parameter set. Filters are validated at the domain level before the
        API request is dispatched.

        Args:
            filters: ``DocumentFilter`` value object from
                ``domain/value_objects/filters.py``.
            token: JWT bearer token.
            pagination: Page/size/sort params. Defaults to page 0, size 20.

        Returns:
            ``Page[Document]`` with Slice semantics (``has_next`` flag).
        """

    @abstractmethod
    async def find_by_organization(
        self,
        organization_id: str,
        token: str,
        pagination: PageRequest | None = None,
    ) -> Page[Document]:
        """Fetch all documents belonging to a given organization.

        Mirrors Java ``findAllByOrganizationId``. Used in multi-tenant
        contexts where documents must be scoped to a specific organization.

        Args:
            organization_id: EDMS organization identifier string.
            token: JWT bearer token.
            pagination: Pagination parameters.

        Returns:
            ``Page[Document]`` scoped to the organization.
        """

    @abstractmethod
    async def get_links(
        self,
        filters: DocumentLinkFilter,
        token: str,
    ) -> list[dict]:
        """Fetch document cross-references (связи между документами).

        Calls ``GET /api/document/{id}/nomenclature-affair-document-link``.
        Returns raw link data — the infrastructure mapper converts these to
        value objects before returning to the application layer.

        Args:
            filters: ``DocumentLinkFilter`` specifying the source document
                and optional link type filter.
            token: JWT bearer token.

        Returns:
            List of link dicts. Mapped to value objects by the infrastructure
            layer before reaching use cases.
        """

    @abstractmethod
    async def update_fields(
        self,
        document_id: UUID,
        operation: str,
        payload: dict,
        token: str,
    ) -> bool:
        """Execute a named field-update operation on a document.

        Calls ``POST /api/document/{id}/execute`` with an operation constant
        and field payload. Used by the autofill tool for operations such as
        ``DOCUMENT_MAIN_FIELDS_UPDATE`` and ``APPEAL_FIELDS_UPDATE``.

        Args:
            document_id: Target document UUID.
            operation: EDMS operation constant string (e.g.
                ``"DOCUMENT_MAIN_FIELDS_UPDATE"``).
            payload: Dict of fields to update. Structure depends on
                the operation type — validated by the EDMS API.
            token: JWT bearer token.

        Returns:
            ``True`` on HTTP 2xx success, ``False`` on any API error.
        """
