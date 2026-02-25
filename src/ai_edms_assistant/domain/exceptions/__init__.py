# src/ai_edms_assistant/domain/exceptions/__init__.py
"""Domain exception hierarchy for the AI EDMS Assistant.

All exceptions inherit from ``AppError`` to enable a single catch-all
handler at the interface layer.

Hierarchy:
    AppError
    ├── DomainError                    (business rule violations)
    │   ├── DocumentNotFoundError
    │   ├── DocumentAccessDeniedError
    │   ├── DocumentUpdateError
    │   ├── AttachmentNotFoundError
    │   ├── AttachmentTextExtractionError
    │   ├── TaskNotFoundError
    │   ├── TaskCreationError
    │   ├── FilterValidationError
    │   ├── AppealValidationError
    │   ├── ExtractionValidationError
    │   └── EmployeeResolutionError
    └── InfrastructureError            (technical failures — defined in base)
"""

from .base import AppError, DomainError, InfrastructureError
from .document_exceptions import (
    AttachmentNotFoundError,
    AttachmentTextExtractionError,
    DocumentAccessDeniedError,
    DocumentNotFoundError,
    DocumentUpdateError,
    TaskCreationError,
    TaskNotFoundError,
)
from .validation_exceptions import (
    AppealValidationError,
    EmployeeResolutionError,
    ExtractionValidationError,
    FilterValidationError,
)

__all__ = [
    "AppError",
    "DomainError",
    "InfrastructureError",
    "DocumentNotFoundError",
    "DocumentAccessDeniedError",
    "DocumentUpdateError",
    "AttachmentNotFoundError",
    "AttachmentTextExtractionError",
    "TaskNotFoundError",
    "TaskCreationError",
    "FilterValidationError",
    "AppealValidationError",
    "ExtractionValidationError",
    "EmployeeResolutionError",
]
