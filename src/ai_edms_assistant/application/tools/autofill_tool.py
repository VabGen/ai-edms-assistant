# src/ai_edms_assistant/application/tools/autofill_tool.py
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.entities.document import DocumentCategory
from ...domain.exceptions import AppealValidationError, DocumentNotFoundError
from ...domain.repositories import AbstractDocumentRepository
from ...domain.services import AppealValidator
from ..dto import ExtractionResultDto
from ..ports import AbstractNLPExtractor, AbstractStorage
from .base_tool import AbstractEdmsTool


class AppealAutofillInput(BaseModel):
    """Input schema for appeal autofill.

    Attributes:
        token: JWT auth token.
        document_id: UUID of appeal document.
        attachment_id: Specific attachment UUID (optional).
            If None, tool selects main attachment automatically.
    """

    token: str = Field(..., description="JWT токен")
    document_id: UUID = Field(..., description="UUID документа категории APPEAL")
    attachment_id: UUID | None = Field(
        default=None, description="UUID конкретного вложения"
    )


class AppealAutofillTool(AbstractEdmsTool):
    """Tool for auto-filling appeal document fields via NLP extraction.

    Workflow:
        1. Validate document is APPEAL category
        2. Select target attachment (main or specified)
        3. Extract text from attachment
        4. Run NLP extraction to identify fields
        5. Update document via EDMS API (NOTE: requires infrastructure layer)

    Dependencies:
        - ``AbstractDocumentRepository``: Fetch document and attachments.
        - ``AbstractNLPExtractor``: Extract structured fields from text.
        - ``AbstractStorage``: Download attachment file bytes.
        - ``AppealValidator``: Validate document is appeal category.
    """

    name: str = "autofill_appeal_document"
    description: str = (
        "Автоматически заполняет карточку обращения (APPEAL) через LLM-анализ. "
        "Извлекает текст из вложения, анализирует его с помощью NLP и заполняет "
        "поля документа (заявитель, контакты, адрес, тип заявителя и т.д.)."
    )
    args_schema: type[BaseModel] = AppealAutofillInput

    _MIN_TEXT_LENGTH: int = 50
    _SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".doc", ".rtf")

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        nlp_extractor: AbstractNLPExtractor,
        storage: AbstractStorage,
        appeal_validator: AppealValidator,
        **kwargs,
    ):
        """Initialize with injected dependencies.

        Args:
            document_repository: Repository for fetching documents.
            nlp_extractor: NLP extractor for field extraction.
            storage: Storage backend for downloading files.
            appeal_validator: Validator for appeal business rules.
            **kwargs: Additional BaseTool arguments.
        """
        super().__init__(**kwargs)
        self._doc_repo = document_repository
        self._nlp = nlp_extractor
        self._storage = storage
        self._validator = appeal_validator

    async def _arun(
        self,
        token: str,
        document_id: UUID,
        attachment_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Execute appeal autofill.

        Args:
            token: JWT token.
            document_id: Appeal document UUID.
            attachment_id: Specific attachment UUID (optional).

        Returns:
            Dict with extracted fields and warnings.
        """
        try:
            # 1. Load document with attachments
            document = await self._load_document(token, document_id)

            # 2. Validate it's an appeal
            self._validate_appeal_category(document)

            # 3. Select target attachment
            target_attachment, warnings = self._select_attachment(
                document, attachment_id
            )

            # 4. Extract text from attachment
            extracted_text = await self._extract_text_from_attachment(
                token=token,
                document_id=document_id,
                attachment=target_attachment,
            )

            # 5. Validate text length
            if len(extracted_text) < self._MIN_TEXT_LENGTH:
                return self._handle_error(
                    ValueError("Текст не извлечён или слишком короткий")
                )

            # 6. Run NLP extraction
            extraction_result = await self._nlp.extract_appeal_fields(
                text=extracted_text,
                document_id=str(document_id),
            )

            # 7. Convert to DTO
            dto = ExtractionResultDto.from_extraction_result(extraction_result)

            # 8. NOTE: In real implementation, we would update EDMS API here:
            # - Build payload from confident_fields
            # - Call POST /api/document/{id}/execute with APPEAL_FIELDS_UPDATE
            # - Handle API response
            #
            # For now, return extraction results without API call

            result = self._success_response(
                data={
                    "confident_fields": dto.confident_fields,
                    "overall_confidence": dto.overall_confidence,
                    "warnings": warnings + dto.warnings,
                    "attachment_used": target_attachment.file_name,
                },
                message="Документ успешно проанализирован",
            )

            # Add partial success note if low confidence
            if dto.overall_confidence < 0.7:
                result["partial_success"] = True
                result[
                    "message"
                ] += f" (низкая общая уверенность: {dto.overall_confidence:.1%})"

            return result

        except DocumentNotFoundError as e:
            return self._handle_error(e)
        except AppealValidationError as e:
            return self._handle_error(e)
        except Exception as e:
            return self._handle_error(e)

    async def _load_document(self, token: str, document_id: UUID):
        """Load document with attachments.

        Args:
            token: JWT token.
            document_id: Document UUID.

        Returns:
            Document entity.

        Raises:
            DocumentNotFoundError: When document doesn't exist.
        """
        document = await self._doc_repo.get_with_attachments(
            document_id=document_id,
            token=token,
        )
        if not document:
            raise DocumentNotFoundError(document_id=document_id)
        return document

    def _validate_appeal_category(self, document) -> None:
        """Validate document is APPEAL category.

        Args:
            document: Document entity.

        Raises:
            AppealValidationError: When not an appeal.
        """
        if document.document_category != DocumentCategory.APPEAL:
            raise AppealValidationError(
                f"Документ должен быть категории APPEAL, "
                f"а не {document.document_category}"
            )

    def _select_attachment(
        self, document, attachment_id: UUID | None
    ) -> tuple[Any, list[str]]:
        """Select target attachment for text extraction.

        Args:
            document: Document entity with attachments.
            attachment_id: Specific attachment UUID (optional).

        Returns:
            Tuple of (selected_attachment, warnings_list).

        Raises:
            ValueError: When no attachments available.
        """
        if not document.attachments:
            raise ValueError("В документе отсутствуют вложения")

        warnings = []

        # If specific ID requested
        if attachment_id:
            target = next(
                (a for a in document.attachments if a.id == attachment_id),
                None,
            )
            if not target:
                warnings.append(
                    f"Вложение ID={attachment_id} не найдено, используется автоподбор"
                )
            else:
                return target, warnings

        # Auto-select: prefer supported formats
        target = next(
            (
                a
                for a in document.attachments
                if a.file_name.lower().endswith(self._SUPPORTED_EXTENSIONS)
            ),
            document.attachments[0],  # Fallback to first
        )

        return target, warnings

    async def _extract_text_from_attachment(
        self, token: str, document_id: UUID, attachment: Any
    ) -> str:
        """Extract text content from attachment.

        Args:
            token: JWT token.
            document_id: Parent document UUID.
            attachment: Attachment entity.

        Returns:
            Extracted text string.

        Raises:
            ValueError: When extraction fails.
        """
        # Download file
        storage_key = f"documents/{document_id}/attachments/{attachment.id}"

        try:
            file_bytes = await self._storage.download(storage_key)
        except Exception as download_err:
            raise ValueError(f"Не удалось скачать файл: {download_err}")

        # Extract text (simplified - real impl would use FileProcessor)
        text = self._extract_text_simple(file_bytes, attachment.file_name)

        if not text or len(text) < self._MIN_TEXT_LENGTH:
            raise ValueError(
                f"Не удалось извлечь достаточно текста из файла {attachment.file_name}"
            )

        return text

    def _extract_text_simple(self, file_bytes: bytes, filename: str) -> str:
        """Simplified text extraction (placeholder for FileProcessor).

        Args:
            file_bytes: Raw file bytes.
            filename: Original filename.

        Returns:
            Extracted text string.
        """
        # NOTE: Real implementation would use FileProcessor from
        # application/processors/ with proper PDF/DOCX/TXT parsing

        try:
            # Try UTF-8 decode for text files
            if filename.lower().endswith(".txt"):
                return file_bytes.decode("utf-8", errors="ignore")

            # For binary formats, return placeholder
            # (Real implementation would use pypdf2, python-docx, etc.)
            return f"[Извлечение текста из {filename} требует FileProcessor]"

        except Exception:
            return ""

    def _run(self, *args, **kwargs):
        """Sync execution not supported."""
        raise NotImplementedError("Use _arun for async execution")
