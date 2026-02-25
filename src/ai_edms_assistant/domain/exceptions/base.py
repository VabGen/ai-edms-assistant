# src/ai_edms_assistant/domain/exceptions/base.py
from __future__ import annotations


class AppError(Exception):
    """Root exception for all application-level errors.

    All custom exceptions in this project inherit from ``AppError``.
    This enables a single ``except AppError`` catch at the interface layer
    (FastAPI exception handler) to intercept any known error type and map
    it to an appropriate HTTP response.

    Attributes:
        message: Human-readable error description.
        code: Optional machine-readable error code for API responses.

    Example:
        >>> try:
        ...     raise DocumentNotFoundError(document_id=uuid4())
        ... except AppError as e:
        ...     print(e.message)
        'Document abc123 not found'
    """

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return self.message


class DomainError(AppError):
    """Base class for domain-layer business rule violations.

    Raised when an operation violates a business invariant — e.g. attempting
    to create a task on an archived document, or assigning an executor who
    does not exist.

    Domain errors are always the result of a business rule check, not a
    technical failure. They should be communicated back to the user with
    a descriptive message.

    Example:
        >>> raise DomainError("Невозможно создать поручение для архивного документа")
    """


class InfrastructureError(AppError):
    """Base class for infrastructure-layer technical errors.

    Raised when an external system (EDMS API, LLM, vector store) returns
    an unexpected response or becomes unavailable. Infrastructure errors
    indicate a technical problem, not a business rule violation.

    Example:
        >>> raise InfrastructureError("EDMS API вернул 503", code="EDMS_UNAVAILABLE")
    """
