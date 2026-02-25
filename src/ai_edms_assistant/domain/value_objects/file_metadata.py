# src/ai_edms_assistant/domain/value_objects/file_metadata.py
"""FileMetadata value object for file processing operations."""

from __future__ import annotations

from uuid import UUID

from ..entities.attachment import ContentType
from ..entities.base import DomainModel


class FileMetadata(DomainModel):
    """Immutable metadata snapshot for a file passed to the NLP pipeline.

    Carries the minimal information needed by ``FileProcessor`` and
    ``NLPExtractor`` to fetch, parse, and route a file — without loading
    the full ``Attachment`` entity.

    Used as the input contract between ``AttachmentTool`` (application layer)
    and ``FileProcessor`` (application processor), keeping the pipeline
    decoupled from the full attachment aggregate.

    Attributes:
        attachment_id: UUID of the source attachment.
        file_name: Original file name (used for logging and user messages).
        content_type: File format — determines which parser to invoke.
        size_bytes: File size in bytes — used for chunking strategy selection.
        storage_path: Full path in MinIO / S3 for downloading the file.
            ``None`` when the file is provided as inline bytes.
        document_id: UUID of the parent document (for context injection).

    Example:
        >>> from uuid import uuid4
        >>> meta = FileMetadata(
        ...     attachment_id=uuid4(),
        ...     file_name="contract.pdf",
        ...     content_type=ContentType.PDF,
        ...     size_bytes=204800,
        ...     storage_path="bucket/documents/contract.pdf",
        ... )
        >>> meta.is_large
        False
        >>> meta.size_kb
        200.0
        >>> meta.extension
        'pdf'
    """

    attachment_id: UUID
    file_name: str
    content_type: ContentType
    size_bytes: int
    storage_path: str | None = None
    document_id: UUID | None = None

    _LARGE_FILE_THRESHOLD_BYTES: int = 10 * 1024 * 1024

    @property
    def size_kb(self) -> float:
        """Returns file size in kilobytes.

        Returns:
            Size in KB rounded to 1 decimal place.

        Example:
            200.0
        """
        return round(self.size_bytes / 1024, 1)

    @property
    def size_mb(self) -> float:
        """Returns file size in megabytes.

        Returns:
            Size in MB rounded to 2 decimal places.

        Example:
            0.20
        """
        return round(self.size_bytes / (1024 * 1024), 2)

    @property
    def is_large(self) -> bool:
        """Returns True when the file exceeds the large-file threshold (10 MB).

        Large files require chunked processing instead of single-pass parsing.

        Returns:
            ``True`` when ``size_bytes > 10 MB``.

        Example:
            False
        """
        return self.size_bytes > self._LARGE_FILE_THRESHOLD_BYTES

    @property
    def extension(self) -> str:
        """Returns the lowercase file extension without the leading dot.

        Returns:
            Extension string like ``'pdf'``, ``'docx'``, ``'txt'``.
            Falls back to the ``content_type`` value when no extension
            is present in ``file_name``.
        """
        if "." in self.file_name:
            return self.file_name.rsplit(".", 1)[-1].lower()
        return self.content_type.value.lower()

    @property
    def can_extract_text(self) -> bool:
        """Check if text extraction is supported for this file type.

        Returns:
            ``True`` for PDF, DOCX, TXT, HTML, RTF files.

        Example:
            True
        """
        return self.extension in ("pdf", "docx", "txt", "html", "htm", "rtf", "doc")

    @property
    def is_text_file(self) -> bool:
        """Check if file is plain text.

        Returns:
            ``True`` for .txt files.
        """
        return self.extension == "txt"

    @property
    def is_pdf(self) -> bool:
        """Check if file is PDF.

        Returns:
            ``True`` for .pdf files.
        """
        return self.extension == "pdf"

    @property
    def is_docx(self) -> bool:
        """Check if file is DOCX.

        Returns:
            ``True`` for .docx files.
        """
        return self.extension == "docx"

    @property
    def is_doc(self) -> bool:
        """Check if file is legacy DOC.

        Returns:
            ``True`` for .doc files.
        """
        return self.extension == "doc"

    @property
    def is_html(self) -> bool:
        """Check if file is HTML.

        Returns:
            ``True`` for .html or .htm files.
        """
        return self.extension in ("html", "htm")

    @property
    def is_rtf(self) -> bool:
        """Check if file is RTF.

        Returns:
            ``True`` for .rtf files.
        """
        return self.extension == "rtf"

    def __str__(self) -> str:
        """String representation for logging.

        Returns:
            Human-readable string with filename, size, and content type.

        Example:
            'contract.pdf (200.0 KB, PDF)'
        """
        return f"{self.file_name} ({self.size_kb} KB, {self.content_type.value})"
