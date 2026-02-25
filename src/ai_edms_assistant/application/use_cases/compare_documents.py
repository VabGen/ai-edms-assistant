# src/ai_edms_assistant/application/use_cases/compare_documents.py
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from ...domain.exceptions import DocumentNotFoundError
from ...domain.repositories import AbstractDocumentRepository
from ...domain.services import DocumentComparer
from ..dto import ComparisonResultDto
from .base import AbstractUseCase


class CompareDocumentsRequest(BaseModel):
    """Request DTO for document comparison.

    Attributes:
        base_document_id: UUID of the older (base) document.
        new_document_id: UUID of the newer document to compare against.
        token: JWT bearer token.
    """

    base_document_id: UUID
    new_document_id: UUID
    token: str


class CompareDocumentsUseCase(
    AbstractUseCase[CompareDocumentsRequest, ComparisonResultDto]
):
    """Use case: Compare two document versions and return a structured diff.

    Fetches both documents from the repository and uses the
    ``DocumentComparer`` domain service to generate a diff of changed fields.

    Dependencies:
        - ``AbstractDocumentRepository``: Fetch both documents.
        - ``DocumentComparer``: Generate the field-level diff.

    Business rules:
        - Both documents must exist.
        - Documents are compared as-is (no special handling for versions —
          that's determined by the caller).
    """

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        document_comparer: DocumentComparer,
    ) -> None:
        """Initialize with injected dependencies.

        Args:
            document_repository: Repository for fetching documents.
            document_comparer: Domain service for comparing documents.
        """
        self._doc_repo = document_repository
        self._comparer = document_comparer

    async def execute(self, request: CompareDocumentsRequest) -> ComparisonResultDto:
        """Execute the comparison use case.

        Args:
            request: Comparison request with base and new document IDs.

        Returns:
            ``ComparisonResultDto`` with summary and changed fields.

        Raises:
            DocumentNotFoundError: When either document does not exist.
        """
        # 1. Fetch base document
        base_doc = await self._doc_repo.get_by_id(
            entity_id=request.base_document_id,
            token=request.token,
        )
        if base_doc is None:
            raise DocumentNotFoundError(document_id=request.base_document_id)

        # 2. Fetch new document
        new_doc = await self._doc_repo.get_by_id(
            entity_id=request.new_document_id,
            token=request.token,
        )
        if new_doc is None:
            raise DocumentNotFoundError(document_id=request.new_document_id)

        # 3. Compare using domain service
        comparison_result = self._comparer.compare(base=base_doc, new=new_doc)

        # 4. Convert to DTO and return
        return ComparisonResultDto.from_comparison_result(comparison_result)
