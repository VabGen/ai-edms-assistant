# src/ai_edms_assistant/domain/value_objects/introduction.py
"""Introduction list value object for document familiarization tracking."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from ..entities.base import DomainModel
from ..entities.employee import UserInfo


class Introduction(DomainModel):
    """Single entry in a document introduction (familiarization) list.

    Tracks whether an employee has reviewed/acknowledged a document.
    Immutable because introduction records are audit trail artifacts —
    once created, they represent a point-in-time state.

    Used in Document.introduction list to track who needs to review
    the document and who has already completed the review.

    Attributes:
        id: Introduction record UUID.
        employee: UserInfo reference to the employee who must review.
        date_view: Timestamp when the employee viewed the document.
            ``None`` when not yet viewed.
        is_viewed: Whether the employee has acknowledged the document.
        comment: Optional employee comment on the review.
        order: Display order in the introduction list.
        create_date: When this introduction record was created.

    Example:
        >>> from uuid import uuid4
        >>> intro = Introduction(
        ...     id=uuid4(),
        ...     employee=UserInfo(...),
        ...     date_view=None,
        ...     is_viewed=False,
        ... )
        >>> intro.is_pending
        True
    """

    id: UUID
    employee: UserInfo

    date_view: datetime | None = Field(default=None, alias="dateView")
    is_viewed: bool = Field(default=False, alias="viewed")
    comment: str | None = None
    order: int = 0
    create_date: datetime | None = Field(default=None, alias="createDate")

    @property
    def is_pending(self) -> bool:
        """Returns True when the introduction is pending review.

        Returns:
            ``True`` when ``is_viewed=False`` and ``date_view`` is ``None``.
        """
        return not self.is_viewed and self.date_view is None

    @property
    def is_completed(self) -> bool:
        """Returns True when the employee has completed the review.

        Returns:
            ``True`` when ``is_viewed=True`` and ``date_view`` is set.
        """
        return self.is_viewed and self.date_view is not None

    def __str__(self) -> str:
        """Returns human-readable summary for logging."""
        status = "✓" if self.is_completed else "⧗"
        return f"{status} {self.employee.name}"
