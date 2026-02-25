# src/ai_edms_assistant/application/dto/document_dto.py
"""Document-related Data Transfer Objects.

This module contains DTOs for document, task, comparison, and extraction
operations. All DTOs provide from_entity() / from_result() classmethods
to convert domain objects to transport-agnostic representations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentSummaryDto(BaseModel):
    """Lightweight document summary for use case outputs.

    Used when a use case returns document references without needing
    the full ``Document`` entity. Reduces coupling — use cases can return
    this DTO instead of exposing the entire domain model to the interface.

    Attributes:
        id: Document UUID.
        reg_number: Registration number (for user-facing display).
        reg_date: Registration date.
        short_summary: Document title / brief content.
        status: Current workflow status (as string, not enum).
        category: Document category (as string).

    Example:
        >>> from uuid import uuid4
        >>> dto = DocumentSummaryDto(
        ...     id=uuid4(),
        ...     reg_number="01/123-п",
        ...     short_summary="О проведении совещания",
        ...     status="REGISTERED",
        ...     category="ORDER",
        ... )
    """

    id: UUID
    reg_number: str | None = None
    reg_date: datetime | None = None
    short_summary: str | None = None
    status: str | None = None
    category: str | None = None

    @classmethod
    def from_entity(cls, doc: Any) -> DocumentSummaryDto:
        """Convert a ``Document`` entity to a summary DTO.

        Args:
            doc: Domain ``Document`` entity.

        Returns:
            Lightweight ``DocumentSummaryDto``.
        """
        return cls(
            id=doc.id,
            reg_number=doc.reg_number,
            reg_date=doc.reg_date,
            short_summary=doc.short_summary,
            status=doc.status.value if doc.status else None,
            category=doc.document_category.value if doc.document_category else None,
        )


class TaskSummaryDto(BaseModel):
    """Lightweight task summary for use case outputs.

    Used when returning task references without the full ``Task`` entity.

    Attributes:
        id: Task UUID.
        task_number: Human-readable task number.
        text: Task instruction text.
        status: Current task status (as string).
        deadline: Planned completion date.
        responsible_executor_name: Name of the responsible executor.

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime
        >>> dto = TaskSummaryDto(
        ...     id=uuid4(),
        ...     task_number="2025-001",
        ...     text="Подготовить отчёт к совещанию",
        ...     status="IN_PROGRESS",
        ...     deadline=datetime(2025, 12, 1),
        ...     responsible_executor_name="Иванов И.И.",
        ... )
    """

    id: UUID
    task_number: str | None = None
    text: str
    status: str
    deadline: datetime | None = None
    responsible_executor_name: str | None = None

    @classmethod
    def from_entity(cls, task: Any) -> TaskSummaryDto:
        """Convert a ``Task`` entity to a summary DTO.

        Args:
            task: Domain ``Task`` entity.

        Returns:
            Lightweight ``TaskSummaryDto``.

        Example:
            >>> task_dto = TaskSummaryDto.from_entity(task)
        """
        responsible = task.responsible_executor
        return cls(
            id=task.id,
            task_number=task.task_number,
            text=task.text,
            status=task.status.value,
            deadline=task.deadline,
            responsible_executor_name=responsible.name if responsible else None,
        )


class ComparisonResultDto(BaseModel):
    """DTO for document comparison results.

    Carries the diff output from ``DocumentComparer`` service without
    exposing the full ``ComparisonResult`` value object to the interface.

    Attributes:
        base_doc_id: UUID string of the older document.
        new_doc_id: UUID string of the newer document.
        summary: Human-readable summary of changes.
        changed_fields: List of field names that changed.
        total_changes: Count of detected changes.

    Example:
        >>> dto = ComparisonResultDto(
        ...     base_doc_id="abc-123",
        ...     new_doc_id="abc-124",
        ...     summary="Изменено 3 поля: status, reg_number, summary",
        ...     changed_fields=["status", "reg_number", "summary"],
        ...     total_changes=3,
        ... )
    """

    base_doc_id: str
    new_doc_id: str
    summary: str
    changed_fields: list[str] = Field(default_factory=list)
    total_changes: int = 0

    @classmethod
    def from_comparison_result(cls, result: Any) -> ComparisonResultDto:
        """Convert a ``ComparisonResult`` value object to a DTO.

        Args:
            result: ``ComparisonResult`` from ``DocumentComparer`` domain service.

        Returns:
            ``ComparisonResultDto`` for API serialization.
        """
        return cls(
            base_doc_id=result.base_doc_id,
            new_doc_id=result.new_doc_id,
            summary=result.summary,
            changed_fields=[d.field_name for d in result.changed_fields],
            total_changes=len(result.changed_fields),
        )


class ExtractionResultDto(BaseModel):
    """DTO for NLP extraction results.

    Simplifies the ``ExtractionResult`` value object for API responses.

    Attributes:
        confident_fields: Dict of field_name → extracted_value for all
            fields that passed the confidence threshold.
        warnings: List of warning messages (low confidence, missing fields).
        overall_confidence: Average confidence across all fields (0.0–1.0).

    Example:
        >>> dto = ExtractionResultDto(
        ...     confident_fields={
        ...         "applicant_name": "Иванов Иван Иванович",
        ...         "phone": "+375291234567",
        ...         "email": "ivanov@example.com",
        ...     },
        ...     warnings=["Поле 'full_address' не извлечено", "Низкая уверенность для 'email'"],
        ...     overall_confidence=0.82,
        ... )
    """

    confident_fields: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    overall_confidence: float = 0.0

    @classmethod
    def from_extraction_result(cls, result: Any) -> ExtractionResultDto:
        """Convert an ``ExtractionResult`` value object to a DTO.

        Args:
            result: ``ExtractionResult`` from NLP extractor port.

        Returns:
            ``ExtractionResultDto`` for API serialization.
        """
        return cls(
            confident_fields=result.confident_fields,
            warnings=result.warnings,
            overall_confidence=result.overall_confidence,
        )
