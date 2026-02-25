# src/ai_edms_assistant/domain/value_objects/responsible_executor.py
"""Document responsible executor value object."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from ..entities.base import DomainModel
from ..entities.employee import UserInfo


class DocumentResponsibleExecutor(DomainModel):
    """Employee responsible for preparing materials for a document.

    Maps to DocumentResponsibleExecutorDto from Java. Used in meeting
    documents to track who is responsible for preparing materials for
    each agenda question.

    Immutable because responsibility assignments are set during meeting
    planning and represent a point-in-time commitment.

    Attributes:
        id: Responsible executor record UUID.
        employee: UserInfo reference to the responsible employee.
        is_responsible: Whether this is the primary responsible person.
            When multiple executors exist, one should be marked as primary.
        order: Display order in the list of responsible executors.
        create_date: When this responsibility assignment was created.
        comment: Optional note about the responsibility scope.

    Example:
        >>> from uuid import uuid4
        >>> executor = DocumentResponsibleExecutor(
        ...     id=uuid4(),
        ...     employee=UserInfo(...),
        ...     is_responsible=True,
        ... )
        >>> executor.is_primary
        True
    """

    id: UUID
    employee: UserInfo

    is_responsible: bool = Field(default=False, alias="responsible")
    order: int = 0
    create_date: datetime | None = Field(default=None, alias="createDate")
    comment: str | None = None

    @property
    def is_primary(self) -> bool:
        """Returns True for the primary responsible executor.

        Returns:
            ``True`` when ``is_responsible=True``.
        """
        return self.is_responsible

    def __str__(self) -> str:
        """Returns human-readable summary."""
        role = "Ответственный" if self.is_primary else "Исполнитель"
        return f"{role}: {self.employee.name}"
