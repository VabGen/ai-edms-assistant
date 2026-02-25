# src/ai_edms_assistant/domain/value_objects/document_id.py
from __future__ import annotations

from uuid import UUID

from ..entities.base import DomainModel


class DocumentId(DomainModel):
    """Type-safe wrapper for a document UUID.

    Using a dedicated ``DocumentId`` type instead of raw ``UUID`` prevents
    accidentally passing an employee UUID where a document UUID is expected —
    a common source of silent bugs in large codebases.

    Immutable by inheritance from ``DomainModel`` (``frozen=True``).

    Attributes:
        value: The underlying UUID value.

    Example:
        >>> doc_id = DocumentId(value=UUID("123e4567-e89b-12d3-a456-426614174000"))
        >>> str(doc_id)
        '123e4567-e89b-12d3-a456-426614174000'
    """

    value: UUID

    @classmethod
    def from_str(cls, raw: str) -> "DocumentId":
        """Parse a UUID string into a ``DocumentId``.

        Args:
            raw: UUID string in standard hyphenated format.

        Returns:
            A new ``DocumentId`` instance.

        Raises:
            ValueError: When the string is not a valid UUID.

        Example:
            >>> DocumentId.from_str("123e4567-e89b-12d3-a456-426614174000")
        """
        return cls(value=UUID(raw))

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(self.value)
