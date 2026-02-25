# src/ai_edms_assistant/domain/value_objects/employee_id.py
from __future__ import annotations

from uuid import UUID

from ..entities.base import DomainModel


class EmployeeId(DomainModel):
    """Type-safe wrapper for an employee UUID.

    Prevents accidental passing of a document or task UUID to methods
    expecting an employee identifier.

    Immutable by inheritance from ``DomainModel`` (``frozen=True``).

    Attributes:
        value: The underlying UUID value.

    Example:
        >>> emp_id = EmployeeId(value=UUID("123e4567-e89b-12d3-a456-426614174000"))
        >>> emp_id == EmployeeId.from_str("123e4567-e89b-12d3-a456-426614174000")
        True
    """

    value: UUID

    @classmethod
    def from_str(cls, raw: str) -> "EmployeeId":
        """Parse a UUID string into an ``EmployeeId``.

        Args:
            raw: UUID string in standard hyphenated format.

        Returns:
            A new ``EmployeeId`` instance.

        Raises:
            ValueError: When the string is not a valid UUID.
        """
        return cls(value=UUID(raw))

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(self.value)
