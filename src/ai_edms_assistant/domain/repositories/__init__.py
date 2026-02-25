# src/ai_edms_assistant/domain/repositories/__init__.py
"""Domain repository ports (interfaces) for the AI EDMS Assistant.

This module exposes the public ABC interfaces that the application layer
uses to interact with data. Implementations live in the infrastructure layer
and are bound via the DI container (``container.py``).

Pagination primitives:
    PageRequest: Immutable pagination parameters (page, size, sort).
    Page: Generic paginated result set with Slice semantics.

Repository ABCs:
    AbstractRepository: Generic base interface for all repositories.
    AbstractDocumentRepository: Document aggregate data access.
    AbstractEmployeeRepository: Employee / org-structure data access.
    AbstractTaskRepository: Task / assignment data access.

Architecture rule:
    Use cases and tools MUST depend only on these ABCs, never on concrete
    infrastructure implementations. This enables test doubles (fakes/mocks)
    without touching infrastructure code.

Example:
    >>> # In a use case — depend on the ABC, not the implementation
    >>> from ai_edms_assistant.domain.repositories import AbstractDocumentRepository
    >>>
    >>> class SummarizeDocumentUseCase:
    ...     def __init__(self, repo: AbstractDocumentRepository) -> None:
    ...         self._repo = repo
"""

from .base import AbstractRepository, Page, PageRequest
from .document_repository import AbstractDocumentRepository
from .employee_repository import AbstractEmployeeRepository
from .task_repository import AbstractTaskRepository

__all__ = [
    "PageRequest",
    "Page",
    "AbstractRepository",
    "AbstractDocumentRepository",
    "AbstractEmployeeRepository",
    "AbstractTaskRepository",
]
