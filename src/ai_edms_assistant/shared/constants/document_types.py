# src/ai_edms_assistant/shared/constants/document_types.py
"""
Global EDMS document type constants.

All document types in EDMS are defined here,
and are used across domain, service and tool layers to avoid magic strings.

Import::

    from ai_edms_assistant.shared.constants.document_types import DocumentCategory
    if doc.category == DocumentCategory.APPEAL: ...
"""

from __future__ import annotations

from enum import StrEnum


class DocumentCategory(StrEnum):
    """
    Top-level document categories in EDMS.
    """

    INTERN = "INTERN"  # Внутренние документы
    INCOMING = "INCOMING"  # Входящие документы
    OUTGOING = "OUTGOING"  # Исходящие документы
    APPEAL = "APPEAL"  # Обращения граждан


class DocumentOperation(StrEnum):
    """
    Document operation types for POST /api/document/{id}/execute.
    """

    MAIN_FIELDS_UPDATE = "DOCUMENT_MAIN_FIELDS_UPDATE"
    APPEAL_FIELDS_UPDATE = "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE"
    CORRESPONDENT_UPDATE = "DOCUMENT_CORRESPONDENT_UPDATE"
    LINK_UPDATE = "DOCUMENT_LINK_UPDATE"
    RECIPIENT_LIST_UPDATE = "DOCUMENT_RECIPIENT_LIST_UPDATE"
    CONTRACT_FIELDS_UPDATE = "DOCUMENT_CONTRACT_FIELDS_UPDATE"
    NOMENCLATURE_AFFAIR_UPDATE = "DOCUMENT_NOMENCLATURE_AFFAIR_UPDATE"
    UNDRAFT = "DOCUMENT_UNDRAFT"
    UPDATE_CUSTOM_FIELDS = "DOCUMENT_UPDATE_CUSTOM_FIELDS"
    ACCESS_GRIEF_UPDATE = "DOCUMENT_ACCESS_GRIEF_UPDATE"
    ARCHIVE = "DOCUMENT_ARCHIVE"


class TaskOperationType(StrEnum):
    """Task-related operation types."""

    CREATE = "TASK_CREATE"
    EDIT = "TASK_EDIT"
    TO_CONTROL = "TASK_TO_CONTROL"
    REMOVE_CONTROL = "TASK_REMOVE_CONTROL"
    AFTER_EXECUTION = "TASK_AFTER_EXECUTION"
    FOR_REVISION = "TASK_FOR_REVISION"


class SupportedFileExtension(StrEnum):
    """File extensions supported for text extraction."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


# Максимальный размер извлекаемого текста (символов) для LLM контекста
MAX_EXTRACTED_TEXT_CHARS: int = 15_000

# Минимальная длина текста для анализа (защита от пустых вложений)
MIN_APPEAL_TEXT_CHARS: int = 50
