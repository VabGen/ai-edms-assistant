# src/ai_edms_assistant/application/use_cases/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class AbstractUseCase(ABC, Generic[RequestT, ResponseT]):
    """Base class for all application use cases.

    A use case represents a single user-facing operation — e.g.
    "summarize a document", "create a task", "compare two versions".
    Each use case has a single ``execute`` method that takes a request DTO
    and returns a response DTO.

    Design principles:
        - **Single Responsibility**: Each use case does one thing.
        - **Dependency Injection**: Receive all dependencies (repositories,
          services, ports) via constructor.
        - **No HTTP logic**: Use cases are transport-agnostic. They know
          nothing about FastAPI, HTTP status codes, or JSON serialization.
        - **Domain-driven**: Orchestrate domain entities, services, and
          repositories. Do NOT contain business logic — delegate to domain.

    Type parameters:
        RequestT: Input DTO type.
        ResponseT: Output DTO type.
    """

    @abstractmethod
    async def execute(self, request: RequestT) -> ResponseT:
        """Execute the use case with the given request.

        Args:
            request: Input DTO specific to this use case.

        Returns:
            Output DTO or result value.

        Raises:
            DomainError: When a business rule is violated.
            InfrastructureError: When an external system fails.
        """
