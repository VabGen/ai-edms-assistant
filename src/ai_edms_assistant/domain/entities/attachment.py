# src/ai_edms_assistant/domain/entities/attachment.py
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from .base import DomainModel, MutableDomainModel

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AttachmentType(StrEnum):
    """Semantic role of a file within an EDMS document.

    Determines how the file participates in document workflows:
    - ``ATTACHMENT`` — основной файл документа.
    - ``PRINT_DOCUMENT`` — печатная форма (PDF-рендер).
    - ``PROJECT_SOLUTION`` — проект решения.
    - ``RATIONALE`` — обоснование.
    - ``DOCUMENTS_QUESTION`` — материалы к вопросу повестки.
    - ``INTRODUCTION_LIST`` — лист ознакомления.
    - ``AGREEMENT_LIST`` — лист согласования.
    - ``DECISION`` — решение.
    - ``RKK`` — регистрационно-контрольная карточка.
    """

    ATTACHMENT = "ATTACHMENT"
    PRINT_DOCUMENT = "PRINT_DOCUMENT"
    PROJECT_SOLUTION = "PROJECT_SOLUTION"
    RATIONALE = "RATIONALE"
    DOCUMENTS_QUESTION = "DOCUMENTS_QUESTION"
    INTRODUCTION_LIST = "INTRODUCTION_LIST"
    AGREEMENT_LIST = "AGREEMENT_LIST"
    DECISION = "DECISION"
    RKK = "RKK"


class AttachmentDocumentType(StrEnum):
    DOCUMENT = "DOCUMENT"
    MINI_DOCUMENT = "MINI_DOCUMENT"
    ADDITIONAL_DOCUMENT = "ADDITIONAL_DOCUMENT"


class ContentType(StrEnum):
    """File format of an attachment.

    Used by ``FileProcessor`` and ``NLPExtractor`` to select the appropriate
    parser or OCR pipeline.

    Note:
        ``StrEnum`` allows direct string comparison without ``.value`` access:
        ``content_type == "PDF"`` works correctly.
    """

    DOC = "DOC"
    DOCX = "DOCX"
    TIF = "TIF"
    TIFF = "TIFF"
    JPG = "JPG"
    JPEG = "JPEG"
    BMP = "BMP"
    PDF = "PDF"
    GIF = "GIF"
    PNG = "PNG"
    OTHER = "OTHER"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class Signature(DomainModel):
    """Full signature data (вложен в AttachmentSignature)."""

    data: str
    key_id: str | None = Field(default=None, alias="keyId")
    signer: str | None = None
    sign_time: datetime | None = Field(default=None, alias="signtime")
    signer_date: datetime | None = Field(default=None, alias="signerDate")
    start: datetime | None = None
    end: datetime | None = None
    cert_serial: str | None = Field(default=None, alias="certSerial")
    issuer: str | None = None
    signer_fio: str | None = Field(default=None, alias="signerFio")
    signer_post: str | None = Field(default=None, alias="signerPost")
    signer_org: str | None = Field(default=None, alias="signerOrg")
    personal_number: str | None = Field(default=None, alias="personalNumber")
    operation_type: str | None = Field(default=None, alias="operationType")
    orig_signature: str | None = Field(default=None, alias="origSignature")
    sign_count: int | None = Field(default=None, alias="signCount")
    # TODO ... + attr_cert_* fields (еще ~10 полей)


_TEXT_EXTRACTABLE_FORMATS: frozenset[ContentType] = frozenset(
    {
        ContentType.DOC,
        ContentType.DOCX,
        ContentType.PDF,
        ContentType.TIF,
        ContentType.TIFF,
    }
)
"""Formats supported by text extraction / OCR pipeline.

Defined at module level as a constant to avoid reconstruction on every
``is_text_extractable`` property access.
"""


class AttachmentSignature(DomainModel):
    """Electronic digital signature (ЭЦП) metadata for an attachment.

    Immutable because signatures are write-once artifacts — once applied
    they must not be changed.

    Attributes:
        id: Signature record UUID.
        date: Timestamp when the signature was applied.
        is_verified: Whether the signature has been cryptographically verified.
    """

    id: UUID
    date: datetime | None = None
    is_verified: bool | None = Field(default=None, alias="check")
    signature: Signature | None = None


# ---------------------------------------------------------------------------
# Main entity
# ---------------------------------------------------------------------------


class Attachment(MutableDomainModel):
    """Domain entity for a file attached to an EDMS document.

    Represents a single attachment record as returned by the Java backend
    ``AttachmentDocumentDto``. The entity tracks both the semantic role of
    the file (``attachment_type``) and its storage location in MinIO/S3.

    Storage-specific fields (``storage_path``, ``bucket_name``, ``minio_name``)
    are populated by the infrastructure layer (``AttachmentMapper``) and
    must not be set in domain logic.

    Attributes:
        id: Attachment UUID.
        file_name: Original file name shown to users.
        file_size: File size in bytes.
        attachment_type: Semantic role within the document workflow.
        content_type: File format used to select parser/OCR pipeline.
        mime_type: MIME type string (optional, from HTTP headers).
        storage_path: Full path in MinIO / S3 (set by infrastructure layer).
        bucket_name: MinIO bucket name (set by infrastructure layer).
        minio_name: Internal MinIO object name (set by infrastructure layer).
        content_hash: File integrity hash (``hashED`` in Java DTO).
        author_id: UUID of the employee who uploaded the file.
        upload_date: Timestamp of the upload.
        version_number: Sequential version counter for this attachment.
        is_deleted: Soft-delete flag.
        signs: List of applied electronic signatures.
    """

    id: UUID
    document_id: UUID | None = Field(default=None, alias="documentId")
    file_name: str = Field(alias="name")
    file_size: int = Field(default=0, alias="size")

    attachment_type: AttachmentType | None = Field(default=None, alias="type")
    attachment_document_type: AttachmentDocumentType | None = Field(
        default=None, alias="attachmentDocumentType"
    )
    content_type: ContentType | None = Field(default=None, alias="contentType")
    mime_type: str | None = Field(default=None, alias="mimeType")

    storage_path: str | None = Field(default=None, alias="storagePath")
    bucket_name: str | None = Field(default=None, alias="bucketName")
    minio_name: str | None = Field(default=None, alias="minioName")
    source_original_name: str | None = Field(default=None, alias="sourceOriginalName")
    content_hash: str | None = Field(default=None, alias="hashED")

    author_id: UUID | None = Field(default=None, alias="authorId")
    upload_date: datetime | None = Field(default=None, alias="uploadDate")
    modify_date: datetime | None = Field(default=None, alias="modifyDate")
    last_modify_user_id: UUID | None = Field(default=None, alias="lastModifyUserId")
    version_number: int = Field(default=1, alias="versionNumber")

    is_deleted: bool = Field(default=False, alias="deleted")
    signs: list[AttachmentSignature] = Field(default_factory=list)

    @property
    def is_signed(self) -> bool:
        """Returns True when the attachment has at least one digital signature."""
        return bool(self.signs)

    @property
    def is_text_extractable(self) -> bool:
        """Returns True when the file format supports text extraction or OCR.

        Used by ``FileProcessor`` to decide whether to invoke the NLP pipeline.
        When ``content_type`` is ``None``, returns ``True`` as a safe default
        (attempt extraction and fail gracefully downstream).

        Returns:
            ``True`` for DOC, DOCX, PDF, TIF, TIFF formats and unknown types.
        """
        if self.content_type is None:
            return True
        return self.content_type in _TEXT_EXTRACTABLE_FORMATS

    @property
    def size_kb(self) -> float:
        """Returns file size in kilobytes (rounded to 1 decimal).

        Returns:
            File size in KB.
        """
        return round(self.file_size / 1024, 1)

    def __str__(self) -> str:
        return f"{self.file_name} ({self.size_kb} KB)"
