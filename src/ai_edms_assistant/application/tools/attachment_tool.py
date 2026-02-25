# src/ai_edms_assistant/application/tools/attachment_tool.py
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.repositories import AbstractDocumentRepository
from ...domain.value_objects.file_metadata import FileMetadata
from ..ports import AbstractStorage
from .base_tool import AbstractEdmsTool


class AttachmentFetchInput(BaseModel):
    """Input schema for attachment text extraction.

    Attributes:
        token: JWT auth token.
        document_id: Parent document UUID.
        attachment_id: Specific attachment UUID (optional).
            If None, returns list of available attachments.
    """

    token: str = Field(..., description="JWT токен")
    document_id: UUID = Field(..., description="UUID документа")
    attachment_id: UUID | None = Field(
        default=None, description="UUID вложения (опционально)"
    )


class AttachmentTool(AbstractEdmsTool):
    """Tool for extracting text content from document attachments.

    Workflow:
        1. If attachment_id=None → return list of attachments
        2. If attachment_id provided → download, extract text, return preview

    Dependencies:
        - ``AbstractDocumentRepository``: Fetch document with attachments.
        - ``AbstractStorage``: Download attachment file bytes.
    """

    name: str = "doc_get_file_content"
    description: str = (
        "Извлекает содержимое файла для анализа. "
        "Если ID не указан, возвращает список доступных файлов."
    )
    args_schema: type[BaseModel] = AttachmentFetchInput

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        storage: AbstractStorage,
        **kwargs,
    ):
        """Initialize with injected dependencies.

        Args:
            document_repository: Repository for fetching documents.
            storage: Storage backend for downloading files.
            **kwargs: Additional BaseTool arguments.
        """
        super().__init__(**kwargs)
        self._doc_repo = document_repository
        self._storage = storage

    async def _arun(
        self,
        token: str,
        document_id: UUID,
        attachment_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Execute attachment text extraction.

        Args:
            token: JWT token.
            document_id: Parent document UUID.
            attachment_id: Specific attachment UUID (optional).

        Returns:
            Dict with file list or extracted text.
        """
        try:
            # Fetch document with attachments
            document = await self._doc_repo.get_with_attachments(
                document_id=document_id, token=token
            )
            if not document:
                return self._handle_error(
                    ValueError(f"Документ {document_id} не найден")
                )

            if not document.attachments:
                return self._success_response(
                    data={"files": []},
                    message="В документе отсутствуют вложения",
                )

            # MODE 1: List attachments
            if not attachment_id:
                files_info = [
                    {
                        "id": str(att.id),
                        "name": att.file_name,
                        "size_kb": round(att.file_size / 1024, 1),
                        "content_type": (
                            att.content_type.value if att.content_type else "unknown"
                        ),
                        "can_extract_text": att.can_extract_text,
                    }
                    for att in document.attachments
                ]
                return {
                    "status": "need_selection",
                    "message": (
                        "Укажите attachment_id для анализа текста. "
                        "Список доступных файлов:"
                    ),
                    "data": {"files": files_info},
                }

            # MODE 2: Extract text from specific attachment
            target = next(
                (a for a in document.attachments if a.id == attachment_id),
                None,
            )
            if not target:
                return self._handle_error(
                    ValueError(f"Вложение {attachment_id} не найдено")
                )

            # Download file
            # NOTE: In real implementation, we'd construct storage_key
            # from attachment metadata. For now, placeholder.
            storage_key = f"documents/{document_id}/attachments/{attachment_id}"

            try:
                file_bytes = await self._storage.download(storage_key)
            except Exception as download_err:
                return self._handle_error(
                    ValueError(f"Не удалось скачать файл: {download_err}")
                )

            # Extract text (simplified - real impl would use FileProcessor)
            text_content = self._extract_text_simple(file_bytes, target.file_name)

            # Truncate for preview
            preview = text_content[:15000]
            is_truncated = len(text_content) > 15000

            return self._success_response(
                data={
                    "meta": {
                        "name": target.file_name,
                        "size_kb": round(target.file_size / 1024, 1),
                        "content_type": (
                            target.content_type.value if target.content_type else None
                        ),
                    },
                    "text_preview": preview,
                    "is_truncated": is_truncated,
                    "total_chars": len(text_content),
                },
                message="Текст извлечён",
            )

        except Exception as e:
            return self._handle_error(e)

    def _extract_text_simple(self, file_bytes: bytes, filename: str) -> str:
        """Simplified text extraction (placeholder for FileProcessor).

        Args:
            file_bytes: Raw file bytes.
            filename: Original filename.

        Returns:
            Extracted text string.
        """
        # NOTE: Real implementation would use FileProcessor from
        # application/processors/ with proper PDF/DOCX/TXT parsing
        try:
            # Try UTF-8 decode for text files
            if filename.lower().endswith(".txt"):
                return file_bytes.decode("utf-8", errors="ignore")

            # For binary formats, return placeholder
            return (
                f"[Файл {filename} требует парсинга через FileProcessor. "
                f"Размер: {len(file_bytes)} байт]"
            )
        except Exception:
            return "[Ошибка извлечения текста]"

    def _run(self, *args, **kwargs) -> dict[str, Any]:
        """Sync execution not supported."""
        raise NotImplementedError("Use _arun for async execution")
