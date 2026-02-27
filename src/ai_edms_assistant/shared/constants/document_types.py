# src/ai_edms_assistant/shared/constants/document_types.py
"""Global EDMS document type constants.

Import examples::

    from ai_edms_assistant.shared.constants.document_types import (
        DocumentCategory, DocumentOperation, TaskStatus, SupportedFileExtension,
    )
    if doc.document_category == DocumentCategory.APPEAL: ...
    if status == TaskStatus.ON_EXECUTION: ...

Design principles:
    - Every constant maps 1-to-1 with the Java backend enum/constant.
    - ``StrEnum`` allows direct string comparison without ``.value``.
    - New categories / operations MUST be added here, not scattered in code.

Audit against Java source (2025-06):
    - ``DocumentCategoryConstants``: all 9 variants covered.
    - ``DocumentStatus``: all 26 variants covered.
    - ``TaskStatus``: both variants covered.
    - ``ContentType`` (AttachmentDocumentDto): all 11 variants mapped.
    - ``DocumentOperation``: 11 EDMS execute operations covered.
    - ``TaskOperationType``: 6 task workflow operations covered.
"""

from __future__ import annotations

from enum import StrEnum

# ─────────────────────────────────────────────────────────────────────────────
# Document categories
# ─────────────────────────────────────────────────────────────────────────────


class DocumentCategory(StrEnum):
    """Top-level document categories in EDMS.

    Maps to ``DocumentCategoryConstants`` Java enum. All 9 variants included.
    Previously only INTERN / INCOMING / OUTGOING / APPEAL were present —
    MEETING / QUESTION / MEETING_QUESTION / CONTRACT / CUSTOM were missing.

    Attributes:
        INTERN: Внутренний документ.
        INCOMING: Входящий документ.
        OUTGOING: Исходящий документ.
        MEETING: Совещание.
        QUESTION: Вопрос повестки.
        MEETING_QUESTION: Повестка заседания.
        APPEAL: Обращение гражданина.
        CONTRACT: Договор.
        CUSTOM: Пользовательский документ.
    """

    INTERN = "INTERN"
    INCOMING = "INCOMING"
    OUTGOING = "OUTGOING"
    MEETING = "MEETING"
    QUESTION = "QUESTION"
    MEETING_QUESTION = "MEETING_QUESTION"
    APPEAL = "APPEAL"
    CONTRACT = "CONTRACT"
    CUSTOM = "CUSTOM"

    # ── Human-readable labels (Russian) ───────────────────────────────────────

    @property
    def label(self) -> str:
        """Return Russian display name matching Java enum constructor arg.

        Returns:
            Russian category name used in EDMS UI.
        """
        _labels: dict[str, str] = {
            "INTERN": "Внутренний документ",
            "INCOMING": "Входящий документ",
            "OUTGOING": "Исходящий документ",
            "MEETING": "Совещание",
            "QUESTION": "Вопрос повестки",
            "MEETING_QUESTION": "Повестка заседания",
            "APPEAL": "Обращение",
            "CONTRACT": "Договор",
            "CUSTOM": "Пользовательский документ",
        }
        return _labels.get(self.value, self.value)

    @property
    def is_meeting_type(self) -> bool:
        """True for MEETING, QUESTION, MEETING_QUESTION.

        Returns:
            Whether this category belongs to the meeting family.
        """
        return self in (
            DocumentCategory.MEETING,
            DocumentCategory.QUESTION,
            DocumentCategory.MEETING_QUESTION,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Document lifecycle statuses (for use in constants layer — mirrors domain enum)
# ─────────────────────────────────────────────────────────────────────────────


class DocumentStatus(StrEnum):
    """Document lifecycle status constants.

    Attributes:
        DRAFT: Черновик.
        NEW: Новый.
        REGISTERED: Зарегистрирован.
        EXECUTION: На исполнении.
        EXECUTED: Исполнен.
        ARCHIVE: Архив.
        (see full list below)
    """

    DRAFT = "DRAFT"
    NEW = "NEW"
    STATEMENT = "STATEMENT"
    APPROVED = "APPROVED"
    SIGNING = "SIGNING"
    SIGNED = "SIGNED"
    AGREEMENT = "AGREEMENT"
    AGREED = "AGREED"
    REVIEW = "REVIEW"
    REVIEWED = "REVIEWED"
    REGISTRATION = "REGISTRATION"
    REGISTERED = "REGISTERED"
    EXECUTION = "EXECUTION"
    EXECUTED = "EXECUTED"
    DISPATCH = "DISPATCH"
    SENT = "SENT"
    REJECT = "REJECT"
    CANCEL = "CANCEL"
    PREPARATION = "PREPARATION"
    PAPERWORK = "PAPERWORK"
    FORMALIZED = "FORMALIZED"
    ACCEPTANCE = "ACCEPTANCE"
    ACCEPTED = "ACCEPTED"
    CONTRACT_EXECUTION = "CONTRACT_EXECUTION"
    CONTRACT_CLOSED = "CONTRACT_CLOSED"
    ARCHIVE = "ARCHIVE"
    DELETED = "DELETED"
    ALL = "ALL"

    @property
    def label(self) -> str:
        """Return Russian display label matching Java enum constructor arg.

        Returns:
            Russian status name for EDMS UI rendering.
        """
        _labels: dict[str, str] = {
            "DRAFT": "Черновик",
            "NEW": "Новый",
            "STATEMENT": "На утверждении",
            "APPROVED": "Утвержден",
            "SIGNING": "На подписании",
            "SIGNED": "Подписан",
            "AGREEMENT": "На согласовании",
            "AGREED": "Согласован",
            "REVIEW": "На рассмотрении",
            "REVIEWED": "Рассмотрен",
            "REGISTRATION": "На регистрации",
            "REGISTERED": "Зарегистрирован",
            "EXECUTION": "На исполнении",
            "EXECUTED": "Исполнен",
            "DISPATCH": "На отправке",
            "SENT": "Отправлен",
            "REJECT": "Отклонен",
            "CANCEL": "Аннулирован",
            "PREPARATION": "Подготовка",
            "PAPERWORK": "На оформлении",
            "FORMALIZED": "Оформлен",
            "ACCEPTANCE": "На одобрении",
            "ACCEPTED": "Одобрен",
            "CONTRACT_EXECUTION": "Исполнение договора",
            "CONTRACT_CLOSED": "Закрыт",
            "ARCHIVE": "Архив",
            "DELETED": "Удален",
            "ALL": "Все",
        }
        return _labels.get(self.value, self.value)


# ─────────────────────────────────────────────────────────────────────────────
# Task statuses
# ─────────────────────────────────────────────────────────────────────────────


class TaskStatus(StrEnum):
    """Task lifecycle status constants.

    Maps to Java ``TaskStatus`` enum (ON_EXECUTION / EXECUTED).
    Previously absent from the constants layer — tools and NLP service used
    raw strings, risking silent mismatches.

    Attributes:
        ON_EXECUTION: Поручение на исполнении.
        EXECUTED: Поручение исполнено.
    """

    ON_EXECUTION = "ON_EXECUTION"
    EXECUTED = "EXECUTED"

    @property
    def label(self) -> str:
        """Return Russian display label.

        Returns:
            Russian task status name.
        """
        return {
            "ON_EXECUTION": "На исполнении",
            "EXECUTED": "Исполнено",
        }.get(self.value, self.value)


# ─────────────────────────────────────────────────────────────────────────────
# Document execute operations
# ─────────────────────────────────────────────────────────────────────────────


class DocumentOperation(StrEnum):
    """Document operation types for ``POST /api/document/{id}/execute``.

    Each constant is the ``operationType`` payload value sent to the EDMS
    backend. Maps to EDMS internal operation codes.

    Attributes:
        MAIN_FIELDS_UPDATE: Update core document fields.
        APPEAL_FIELDS_UPDATE: Update appeal-specific fields.
        CORRESPONDENT_UPDATE: Update correspondent.
        LINK_UPDATE: Add/remove document links.
        RECIPIENT_LIST_UPDATE: Update addressee list.
        CONTRACT_FIELDS_UPDATE: Update contract fields (sum, dates, etc.).
        NOMENCLATURE_AFFAIR_UPDATE: Assign to nomenclature affair.
        UNDRAFT: Move from DRAFT → NEW.
        UPDATE_CUSTOM_FIELDS: Update ``customFields`` map.
        ACCESS_GRIEF_UPDATE: Change access classification.
        ARCHIVE: Move document to archive.
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


# ─────────────────────────────────────────────────────────────────────────────
# Task workflow operations
# ─────────────────────────────────────────────────────────────────────────────


class TaskOperationType(StrEnum):
    """Task-related operation types for EDMS task workflow endpoints.

    Attributes:
        CREATE: Create a new task.
        EDIT: Edit task text / deadline.
        TO_CONTROL: Place task under control.
        REMOVE_CONTROL: Remove task from control.
        AFTER_EXECUTION: Mark task as executed.
        FOR_REVISION: Send task back for revision.
    """

    CREATE = "TASK_CREATE"
    EDIT = "TASK_EDIT"
    TO_CONTROL = "TASK_TO_CONTROL"
    REMOVE_CONTROL = "TASK_REMOVE_CONTROL"
    AFTER_EXECUTION = "TASK_AFTER_EXECUTION"
    FOR_REVISION = "TASK_FOR_REVISION"


# ─────────────────────────────────────────────────────────────────────────────
# File extensions for text extraction
# ─────────────────────────────────────────────────────────────────────────────


class SupportedFileExtension(StrEnum):
    """File extensions supported for full-text extraction by the AI pipeline.

    These are extensions that the text-extraction layer can process.
    Maps loosely to ``ContentType`` Java enum in ``AttachmentDocumentDto``.

    Text-extractable formats:
        PDF, DOCX, DOC, TXT

    Image formats (OCR pipeline, future):
        PNG, JPG, JPEG, TIFF, TIF, BMP, GIF

    Attributes:
        PDF: PDF documents.
        DOCX: Microsoft Word (Open XML).
        DOC: Microsoft Word (legacy binary).
        TXT: Plain text.
    """

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"

    @classmethod
    def from_filename(cls, filename: str) -> "SupportedFileExtension | None":
        """Extract and validate file extension from a filename.

        Args:
            filename: Original file name (e.g. ``"contract.docx"``).

        Returns:
            Matching ``SupportedFileExtension`` member, or ``None`` if
            the extension is not supported for text extraction.

        Example:
            >>> SupportedFileExtension.from_filename("report.pdf")
            <SupportedFileExtension.PDF: 'pdf'>
            >>> SupportedFileExtension.from_filename("image.jpg")
            None
        """
        if not filename:
            return None
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        for member in cls:
            if member.value == ext:
                return member
        return None


class ImageFileExtension(StrEnum):
    """Image file extensions from Java ``ContentType`` enum.

    Tracked separately from text-extractable formats. Future OCR pipeline
    will use these when processing scanned documents.

    Attributes:
        PNG, JPG, JPEG, TIFF, TIF, BMP, GIF: Standard image formats.
    """

    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    TIFF = "tiff"
    TIF = "tif"
    BMP = "bmp"
    GIF = "gif"

    @classmethod
    def is_image(cls, filename: str) -> bool:
        """Check whether a filename has an image extension.

        Args:
            filename: File name to check.

        Returns:
            ``True`` when the extension matches any image format.
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return any(m.value == ext for m in cls)


# ─────────────────────────────────────────────────────────────────────────────
# Numeric limits
# ─────────────────────────────────────────────────────────────────────────────

# Максимальный размер извлекаемого текста (символов) для LLM контекста.
# Ограничение по размеру контекстного окна модели.
MAX_EXTRACTED_TEXT_CHARS: int = 15_000

# Минимальная длина текста для анализа (защита от пустых вложений).
MIN_APPEAL_TEXT_CHARS: int = 50

# Максимальный размер файла для загрузки (байты): 50 МБ.
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024
