# src/ai_edms_assistant/domain/exceptions/document_exceptions.py
from __future__ import annotations

from uuid import UUID

from .base import DomainError


class DocumentNotFoundError(DomainError):
    """Raised when a requested document does not exist in the EDMS.

    Corresponds to HTTP 404 from ``GET /api/document/{id}``.

    Attributes:
        document_id: UUID of the missing document.

    Example:
        >>> raise DocumentNotFoundError(document_id=some_uuid)
        DocumentNotFoundError: Документ 'abc-123' не найден
    """

    def __init__(self, document_id: UUID | str) -> None:
        super().__init__(
            message=f"Документ '{document_id}' не найден",
            code="DOCUMENT_NOT_FOUND",
        )
        self.document_id = document_id


class DocumentAccessDeniedError(DomainError):
    """Raised when the current user lacks permission to access a document.

    Corresponds to HTTP 403 from the EDMS API.

    Attributes:
        document_id: UUID of the document.
        user_hint: Optional description of the required permission.

    Example:
        >>> raise DocumentAccessDeniedError(document_id=uuid, user_hint="ДСП документ")
    """

    def __init__(
        self,
        document_id: UUID | str,
        user_hint: str | None = None,
    ) -> None:
        hint = f" ({user_hint})" if user_hint else ""
        super().__init__(
            message=f"Нет доступа к документу '{document_id}'{hint}",
            code="DOCUMENT_ACCESS_DENIED",
        )
        self.document_id = document_id
        self.user_hint = user_hint


class DocumentUpdateError(DomainError):
    """Raised when a document field update operation fails.

    Wraps EDMS API errors from ``POST /api/document/{id}/execute``.
    Used by the autofill tool when the EDMS rejects a payload.

    Attributes:
        document_id: UUID of the target document.
        operation: The operation constant that failed.
        reason: API error message or reason string.

    Example:
        >>> raise DocumentUpdateError(
        ...     document_id=uuid,
        ...     operation="APPEAL_FIELDS_UPDATE",
        ...     reason="Поле citizenTypeId обязательно",
        ... )
    """

    def __init__(
        self,
        document_id: UUID | str,
        operation: str,
        reason: str,
    ) -> None:
        super().__init__(
            message=(
                f"Ошибка обновления документа '{document_id}' "
                f"(операция: {operation}): {reason}"
            ),
            code="DOCUMENT_UPDATE_FAILED",
        )
        self.document_id = document_id
        self.operation = operation
        self.reason = reason


class AttachmentNotFoundError(DomainError):
    """Raised when a requested attachment does not exist.

    Corresponds to HTTP 404 from ``GET /api/attachment/{id}``.

    Attributes:
        attachment_id: UUID of the missing attachment.

    Example:
        >>> raise AttachmentNotFoundError(attachment_id=uuid)
    """

    def __init__(self, attachment_id: UUID | str) -> None:
        super().__init__(
            message=f"Вложение '{attachment_id}' не найдено",
            code="ATTACHMENT_NOT_FOUND",
        )
        self.attachment_id = attachment_id


class AttachmentTextExtractionError(DomainError):
    """Raised when text cannot be extracted from an attachment.

    Indicates that the file format is unsupported, the file is corrupted,
    or the OCR pipeline failed.

    Attributes:
        attachment_id: UUID of the attachment.
        file_name: Original file name for user-facing messages.
        reason: Technical reason for extraction failure.

    Example:
        >>> raise AttachmentTextExtractionError(
        ...     attachment_id=uuid,
        ...     file_name="contract.pdf",
        ...     reason="PDF is password-protected",
        ... )
    """

    def __init__(
        self,
        attachment_id: UUID | str,
        file_name: str,
        reason: str,
    ) -> None:
        super().__init__(
            message=f"Не удалось извлечь текст из '{file_name}': {reason}",
            code="ATTACHMENT_EXTRACTION_FAILED",
        )
        self.attachment_id = attachment_id
        self.file_name = file_name
        self.reason = reason


class TaskNotFoundError(DomainError):
    """Raised when a requested task does not exist in the EDMS.

    Attributes:
        task_id: UUID of the missing task.

    Example:
        >>> raise TaskNotFoundError(task_id=uuid)
    """

    def __init__(self, task_id: UUID | str) -> None:
        super().__init__(
            message=f"Поручение '{task_id}' не найдено",
            code="TASK_NOT_FOUND",
        )
        self.task_id = task_id


class TaskCreationError(DomainError):
    """Raised when task creation fails due to a business rule violation.

    Distinct from infrastructure errors — this is raised when the domain
    validates the create request before sending it to the API.

    Attributes:
        reason: Description of why the task cannot be created.

    Example:
        >>> raise TaskCreationError("Не указан ответственный исполнитель")
    """

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Невозможно создать поручение: {reason}",
            code="TASK_CREATION_FAILED",
        )
        self.reason = reason
