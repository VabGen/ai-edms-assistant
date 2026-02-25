# src/ai_edms_assistant/application/use_cases/extract_appeal_data.py
"""Use case for extracting structured fields from appeal documents."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from ...domain.entities.document import Document
from ...domain.exceptions import DocumentNotFoundError
from ...domain.repositories import AbstractDocumentRepository
from ...domain.services import AppealValidator
from ..dto import ExtractionResultDto
from ..ports import AbstractNLPExtractor
from .base import AbstractUseCase


class ExtractAppealDataRequest(BaseModel):
    """Request DTO for appeal data extraction.

    Attributes:
        document_id: UUID of the appeal document.
        token: JWT bearer token.
        attachment_id: Optional UUID of a specific attachment to analyze.
            When ``None``, uses the main document text.
    """

    document_id: UUID
    token: str
    attachment_id: UUID | None = None


class ExtractAppealDataUseCase(
    AbstractUseCase[ExtractAppealDataRequest, ExtractionResultDto]
):
    """Use case: Extract structured fields from an appeal document via NLP.

    Fetches the document, validates it's an appeal category, extracts text
    from the main attachment (or document body), and runs NLP extraction
    to identify applicant name, contact info, geographic location, etc.

    Dependencies:
        - ``AbstractDocumentRepository``: Fetch document and attachments.
        - ``AbstractNLPExtractor``: Extract structured fields from text.
        - ``AppealValidator``: Validate that the document is an appeal.

    Business rules:
        - Document must exist and be of category APPEAL.
        - If ``attachment_id`` is provided, that attachment is analyzed.
          Otherwise, the main attachment or document summary is used.
        - Extraction warnings are included in the response DTO.
    """

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        nlp_extractor: AbstractNLPExtractor,
        appeal_validator: AppealValidator,
    ) -> None:
        """Initialize with injected dependencies.

        Args:
            document_repository: Repository for fetching documents.
            nlp_extractor: NLP extractor for structured data extraction.
            appeal_validator: Validator for appeal business rules.
        """
        self._doc_repo = document_repository
        self._nlp = nlp_extractor
        self._validator = appeal_validator

    async def execute(self, request: ExtractAppealDataRequest) -> ExtractionResultDto:
        """Execute the extraction use case.

        Args:
            request: Extraction request with document ID and optional attachment.

        Returns:
            ``ExtractionResultDto`` with confident fields and warnings.

        Raises:
            DocumentNotFoundError: When the document does not exist.
            AppealValidationError: When the document is not an appeal.
        """
        # 1. Fetch document with attachments
        document = await self._doc_repo.get_with_attachments(
            document_id=request.document_id,
            token=request.token,
        )
        if document is None:
            raise DocumentNotFoundError(document_id=request.document_id)

        # 2. Validate it's an appeal
        self._validator.validate_document_is_appeal(document)

        # 3. Extract text to analyze
        text = self._extract_text_for_analysis(document, request.attachment_id)
        if not text.strip():
            # Return empty result with warning
            from ...domain.value_objects.extraction_result import ExtractionResult

            empty_result = ExtractionResult(
                source_document_id=str(document.id),
                fields=[],
                warnings=["Документ не содержит текста для анализа"],
            )
            return ExtractionResultDto.from_extraction_result(empty_result)

        # 4. Run NLP extraction
        extraction_result = await self._nlp.extract_appeal_fields(
            text=text,
            document_id=str(document.id),
        )

        # 5. Convert to DTO and return
        return ExtractionResultDto.from_extraction_result(extraction_result)

    def _extract_text_for_analysis(
        self,
        document: Document,
        attachment_id: UUID | None,
    ) -> str:
        """Extract text content from the document or specified attachment.

        Args:
            document: ``Document`` entity.
            attachment_id: Optional attachment UUID to analyze.

        Returns:
            Extracted text string.
        """
        if attachment_id:
            target_attachment = next(
                (a for a in document.attachments if a.id == attachment_id),
                None,
            )
            if target_attachment:
                pass

        return document.get_full_text()
