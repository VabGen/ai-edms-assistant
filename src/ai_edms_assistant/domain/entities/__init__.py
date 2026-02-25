# src/ai_edms_assistant/domain/entities/__init__.py
"""Domain entities for the AI EDMS Assistant.

This module exposes the public API of the domain entities layer.
Import from this module, not from individual entity files, to
ensure stable public interfaces and enable future refactoring.

Entities (mutable aggregates):
    Document: Central EDMS document aggregate.
    Task: Task / assignment entity (Поручение).
    Employee: Full employee entity for org-structure operations.
    DocumentAppeal: Appeal entity (embedded in Document).
    Attachment: File attachment entity.

Value objects (immutable):
    UserInfo: Lightweight user reference (embedded in Document, Task).
    GeoLocation: Geographic hierarchy for appeal applicants.
    AttachmentSignature: ЭЦП metadata for an attachment.
    DocumentVersion: Immutable document version snapshot.
    ControlInfo: Document control metadata.
    AutoControlSettings: Automatic control configuration.
    DocumentRecipient: Addressee / correspondent reference.
    TaskExecutor: Executor assignment record within a Task.

Enums:
    DocumentStatus: Document lifecycle statuses.
    DocumentCategory: Document type categories.
    DocumentCreateType: How the document was created.
    MeetingFormType: Meeting format type.
    AttachmentType: Semantic role of an attachment.
    ContentType: File format for parser selection.
    DeclarantType: Appeal applicant type (physical / legal entity).
    AppealChannel: Appeal intake channel.
    TaskStatus: Task execution statuses.
    TaskType: Task workflow type.

Example:
    >>> from ai_edms_assistant.domain.entities import Document, DocumentStatus
    >>> doc = Document(id=uuid4(), organization_id="org-1")
    >>> doc.is_registered
    False
"""

from uuid import uuid4

from .appeal import AppealChannel, DeclarantType, DocumentAppeal, GeoLocation
from .attachment import Attachment, AttachmentSignature, AttachmentType, ContentType
from .base import DomainModel, MutableDomainModel
from .document import (
    AutoControlSettings,
    ControlInfo,
    Document,
    DocumentCategory,
    DocumentCreateType,
    DocumentRecipient,
    DocumentStatus,
    DocumentVersion,
    MeetingFormType,
)
from .employee import Employee, UserInfo
from .task import Task, TaskExecutor, TaskStatus, TaskType

__all__ = [
    "DomainModel",
    "MutableDomainModel",
    "Document",
    "DocumentStatus",
    "DocumentCategory",
    "DocumentCreateType",
    "MeetingFormType",
    "DocumentVersion",
    "ControlInfo",
    "AutoControlSettings",
    "DocumentRecipient",
    "Task",
    "TaskStatus",
    "TaskType",
    "TaskExecutor",
    "Employee",
    "UserInfo",
    "DocumentAppeal",
    "GeoLocation",
    "DeclarantType",
    "AppealChannel",
    "Attachment",
    "AttachmentSignature",
    "AttachmentType",
    "ContentType",
]
