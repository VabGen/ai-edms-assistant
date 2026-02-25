# src/ai_edms_assistant/application/use_cases/__init__.py
"""Application use cases (business scenarios).

Use cases orchestrate domain entities, services, and repositories to
implement user-facing operations. They represent the "what" of the
application — what the user wants to accomplish.

Architecture:
    - Each use case implements ``AbstractUseCase[Request, Response]``.
    - Dependencies (repositories, services, ports) are injected via constructor.
    - Use cases are transport-agnostic — no HTTP, no JSON, no FastAPI.
    - Interface layer (FastAPI endpoints) calls use cases and serializes results.

Available use cases:
    SummarizeDocumentUseCase: Generate a summary of a document's content.
    ExtractAppealDataUseCase: Extract structured fields from an appeal.
    CompareDocumentsUseCase: Compare two document versions.
    CreateTaskUseCase: Create a new task on a document.
"""

from .base import AbstractUseCase
from .compare_documents import CompareDocumentsRequest, CompareDocumentsUseCase
from .create_task import CreateTaskRequest, CreateTaskUseCase
from .extract_appeal_data import ExtractAppealDataRequest, ExtractAppealDataUseCase
from .summarize_document import SummarizeDocumentRequest, SummarizeDocumentUseCase

__all__ = [
    "AbstractUseCase",
    "SummarizeDocumentUseCase",
    "SummarizeDocumentRequest",
    "ExtractAppealDataUseCase",
    "ExtractAppealDataRequest",
    "CompareDocumentsUseCase",
    "CompareDocumentsRequest",
    "CreateTaskUseCase",
    "CreateTaskRequest",
]
