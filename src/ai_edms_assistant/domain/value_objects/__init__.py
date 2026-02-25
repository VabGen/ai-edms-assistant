# src/ai_edms_assistant/domain/value_objects/__init__.py
"""Domain value objects.

Value objects are immutable data containers that represent concepts
without identity. Unlike entities (which have a lifecycle and UUID),
value objects are compared by their field values.

All value objects inherit from ``DomainModel`` with ``frozen=True``.
"""

from .document_id import DocumentId
from .employee_id import EmployeeId
from .extraction_result import ExtractionResult, FieldExtractionResult
from .file_metadata import FileMetadata
from .filters import (
    DocumentFilter,
    DocumentFilterInclude,
    DocumentIoOption,
    DocumentLinkFilter,
    DocumentLinkType,
    EmployeeFilter,
    EmployeeFilterInclude,
    TaskFilter,
    TaskFilterInclude,
    UserActionFilter,
    UserActionType,
)
from .introduction import Introduction
from .meeting import DocumentQuestion, Speaker, SpeakerType
from .responsible_executor import DocumentResponsibleExecutor

__all__ = [
    # Type-safe ID wrappers
    "DocumentId",
    "EmployeeId",
    # NLP extraction
    "ExtractionResult",
    "FieldExtractionResult",
    # File processing
    "FileMetadata",
    # Filters
    "DocumentFilter",
    "DocumentFilterInclude",
    "DocumentIoOption",
    "DocumentLinkFilter",
    "DocumentLinkType",
    "EmployeeFilter",
    "EmployeeFilterInclude",
    "TaskFilter",
    "TaskFilterInclude",
    "UserActionFilter",
    "UserActionType",
    # Document-related
    "Introduction",
    "DocumentQuestion",
    "Speaker",
    "SpeakerType",
    "DocumentResponsibleExecutor",
]
