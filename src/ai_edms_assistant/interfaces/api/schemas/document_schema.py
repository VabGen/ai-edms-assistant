# src/ai_edms_assistant/interfaces/api/schemas/document_schema.py
"""Pydantic response schemas for document and task endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Attachment
# ─────────────────────────────────────────────────────────────────────────────


class AttachmentBriefResponse(BaseModel):
    """Brief attachment representation for document lists.

    Maps to Java ``AttachmentDocumentDto``.

    Attributes:
        id: Attachment UUID.
        name: Original file name.
        size: File size in bytes.
        has_signature: True when the attachment has at least one ЭЦП signature.
        attachment_type: Attachment role (ATTACHMENT / PRINT_DOCUMENT / etc.).
    """

    id: UUID
    name: Optional[str] = None
    size: Optional[int] = None
    has_signature: bool = False
    attachment_type: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Document
# ─────────────────────────────────────────────────────────────────────────────


class DocumentBriefResponse(BaseModel):
    """Brief document representation for list endpoints.

    Maps to ``DocumentSummaryDto`` / ``DocumentDto``.

    Audit fixes:
        - Added ``reg_date`` (was absent, present in ``DocumentSummaryDto``).
        - Added ``category`` (was absent, present in ``DocumentSummaryDto``).
        - ``author_name`` computed from ``Document.author.name`` by the mapper.
          Not a raw entity field — populated by infrastructure mapper layer.

    Attributes:
        id: Document UUID.
        reg_number: Registration number (e.g. ``"01/123-п"``).
        reg_date: Registration date.
        short_summary: Brief document title / content.
        status: Current workflow status string.
        category: Document category constant (INTERN / INCOMING / etc.).
        create_date: Document creation date.
        author_name: Full name of the document author (computed).
    """

    id: UUID
    reg_number: Optional[str] = None
    reg_date: Optional[datetime] = None
    short_summary: Optional[str] = None
    status: Optional[str] = None
    category: Optional[str] = None
    create_date: Optional[datetime] = None
    author_name: Optional[str] = None

    @classmethod
    def from_entity(cls, doc: Any) -> "DocumentBriefResponse":
        """Create from a domain ``Document`` entity.

        Args:
            doc: Domain ``Document`` entity.

        Returns:
            Populated ``DocumentBriefResponse``.
        """
        author_name: str | None = None
        if doc.author:
            parts = [
                getattr(doc.author, "last_name", "") or "",
                getattr(doc.author, "first_name", "") or "",
                getattr(doc.author, "middle_name", "") or "",
            ]
            author_name = " ".join(p for p in parts if p).strip() or None

        return cls(
            id=doc.id,
            reg_number=doc.reg_number,
            reg_date=doc.reg_date,
            short_summary=doc.short_summary,
            status=doc.status.value if doc.status else None,
            category=(doc.document_category.value if doc.document_category else None),
            create_date=doc.create_date,
            author_name=author_name,
        )


class DocumentDetailResponse(BaseModel):
    """Detailed document response for agent tools and API endpoints.

    Provides a richer view than ``DocumentBriefResponse`` for contexts where
    the full document context is needed (e.g., LLM prompt assembly).

    Attributes:
        id: Document UUID.
        reg_number: Registration number.
        reg_date: Registration date.
        short_summary: Brief title / content.
        summary: Full document text body.
        note: Annotation / note.
        status: Current workflow status.
        category: Document category constant.
        document_type_name: Human-readable document type name.
        profile_name: Document profile name.
        author_name: Full name of the author.
        responsible_executor_name: Full name of the responsible executor.
        create_date: Creation date.
        days_execution: Days allocated for execution.
        control_flag: True when document is under control.
        dsp_flag: True when document has ДСП classification.
        attachments: List of attached file summaries.
        tasks_count: Total number of tasks on the document.
    """

    id: UUID
    reg_number: Optional[str] = None
    reg_date: Optional[datetime] = None
    short_summary: Optional[str] = None
    summary: Optional[str] = None
    note: Optional[str] = None
    status: Optional[str] = None
    category: Optional[str] = None
    document_type_name: Optional[str] = None
    profile_name: Optional[str] = None
    author_name: Optional[str] = None
    responsible_executor_name: Optional[str] = None
    create_date: Optional[datetime] = None
    days_execution: Optional[int] = None
    control_flag: bool = False
    dsp_flag: bool = False
    attachments: list[AttachmentBriefResponse] = Field(default_factory=list)
    tasks_count: int = 0

    @classmethod
    def from_entity(cls, doc: Any) -> "DocumentDetailResponse":
        """Create from a domain ``Document`` entity.

        Args:
            doc: Domain ``Document`` entity.

        Returns:
            Populated ``DocumentDetailResponse``.
        """

        def _name(user: Any) -> str | None:
            if not user:
                return None
            parts = [
                getattr(user, "last_name", "") or "",
                getattr(user, "first_name", "") or "",
                getattr(user, "middle_name", "") or "",
            ]
            return " ".join(p for p in parts if p).strip() or None

        attachments = [
            AttachmentBriefResponse(
                id=att.id,
                name=getattr(att, "name", None) or getattr(att, "file_name", None),
                size=getattr(att, "size", None),
                has_signature=bool(getattr(att, "signs", None)),
                attachment_type=(
                    att.attachment_type.value
                    if hasattr(att.attachment_type, "value")
                    else str(att.attachment_type) if att.attachment_type else None
                ),
            )
            for att in (doc.attachments or [])
        ]

        return cls(
            id=doc.id,
            reg_number=doc.reg_number,
            reg_date=doc.reg_date,
            short_summary=doc.short_summary,
            summary=doc.summary,
            note=doc.note,
            status=doc.status.value if doc.status else None,
            category=(doc.document_category.value if doc.document_category else None),
            document_type_name=doc.document_type_name,
            profile_name=doc.profile_name,
            author_name=_name(doc.author),
            responsible_executor_name=_name(doc.responsible_executor),
            create_date=doc.create_date,
            days_execution=doc.days_execution,
            control_flag=bool(doc.control_flag),
            dsp_flag=bool(doc.dsp_flag),
            attachments=attachments,
            tasks_count=doc.count_task or len(doc.tasks or []),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────


class TaskBriefResponse(BaseModel):
    """Brief task (поручение) representation for list endpoints.

    Maps to ``TaskSummaryDto`` / ``TaskDto``.

    Audit fixes:
        - Added ``task_number`` (was absent, maps to ``TaskDto.taskNumber``).
        - Renamed ``responsible_name`` → ``responsible_executor_name`` to match
          Java semantics and ``TaskSummaryDto``.
        - Added ``is_endless`` (maps to ``TaskDto.endless``).
        - Added ``executors_count`` (maps to ``TaskDto.countExec``).

    Attributes:
        id: Task UUID.
        task_number: Human-readable task number (e.g. ``"2025-001"``).
        text: Task instruction text (maps to Java ``taskText``).
        status: Current task status (``ON_EXECUTION`` / ``EXECUTED``).
        deadline: Planned completion date (maps to Java ``planedDateEnd``).
        responsible_executor_name: Full name of the responsible executor.
            Mapped from the executor with ``responsible=True`` in
            ``TaskDto.taskExecutors`` list.
        is_endless: True when the task has no deadline.
        executors_count: Total number of executors on the task.
    """

    id: UUID
    task_number: Optional[str] = None
    text: str
    status: str
    deadline: Optional[datetime] = None
    responsible_executor_name: Optional[str] = Field(
        default=None,
        alias="responsible_name",  # backward compat: old clients may send responsible_name
        serialization_alias="responsible_executor_name",
    )
    is_endless: bool = False
    executors_count: int = 0

    model_config = {"populate_by_name": True}

    @classmethod
    def from_entity(cls, task: Any) -> "TaskBriefResponse":
        """Create from a domain ``Task`` entity.

        Resolves the responsible executor from ``Task.responsible_executor``
        (domain entity pre-computes this from the executors list).

        Args:
            task: Domain ``Task`` entity.

        Returns:
            Populated ``TaskBriefResponse``.
        """
        responsible = getattr(task, "responsible_executor", None)
        responsible_name: str | None = None
        if responsible:
            parts = [
                getattr(responsible, "last_name", "") or "",
                getattr(responsible, "first_name", "") or "",
                getattr(responsible, "middle_name", "") or "",
            ]
            responsible_name = " ".join(p for p in parts if p).strip() or None
            if not responsible_name:
                responsible_name = getattr(responsible, "name", None)

        return cls(
            id=task.id,
            task_number=getattr(task, "task_number", None),
            text=task.text,
            status=(
                task.status.value if hasattr(task.status, "value") else str(task.status)
            ),
            deadline=getattr(task, "deadline", None),
            responsible_executor_name=responsible_name,
            is_endless=bool(getattr(task, "is_endless", False)),
            executors_count=getattr(task, "executors_count", 0),
        )
