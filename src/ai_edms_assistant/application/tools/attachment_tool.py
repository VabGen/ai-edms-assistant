# src/ai_edms_assistant/application/tools/attachment_tool.py
"""
Tool: doc_get_file_content — скачивание и извлечение текста из вложений EDMS.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from ...domain.entities.attachment import Attachment, ContentType
from ...domain.entities.document import Document
from .base_tool import AbstractEdmsTool

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Input schema
# ─────────────────────────────────────────────────────────────────────────────


class AttachmentFetchInput(BaseModel):
    """Input schema for doc_get_file_content tool.

    Attributes:
        token: JWT bearer token (injected automatically by orchestrator).
        document_id: Parent document UUID.
        attachment_id: Specific attachment UUID. When None — returns file list.
        analysis_mode: Extraction mode (text | metadata | full).
    """

    token: str = Field(..., description="JWT токен авторизации")
    document_id: UUID = Field(..., description="UUID документа в EDMS")
    attachment_id: UUID | None = Field(
        default=None,
        description="UUID конкретного файла. Если не указан — вернёт список всех файлов.",
    )
    analysis_mode: str = Field(
        default="text",
        description="Режим: 'text' (текст), 'metadata' (только мета), 'full' (всё)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction helpers (inline — без зависимости от ContentType enum)
# ─────────────────────────────────────────────────────────────────────────────


def _ext(filename: str) -> str:
    """Return lowercase extension without dot.

    Args:
        filename: File name string.

    Returns:
        Lowercase extension like 'pdf', 'docx', 'txt'.
    """
    return os.path.splitext(filename)[1].lstrip(".").lower()


def _extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from raw file bytes.

    Попытка использовать shared/utils/file_utils.py::extract_text_from_bytes
    с inline-fallback для каждого формата.

    Supported formats (определяется по расширению файла):
        .docx  → python-docx (io.BytesIO)
        .pdf   → PyPDF2 (io.BytesIO)
        .txt   → UTF-8 decode
        .doc   → Сообщение о необходимости конвертации
        .rtf   → striprtf (если установлен)
        .html  → BeautifulSoup (если установлен) или regex
        другие → Информационное сообщение

    НЕ использует ContentType enum — только строковые расширения.

    Args:
        file_bytes: Raw file content bytes.
        filename: Original filename including extension.

    Returns:
        Extracted text string. Never raises — returns error description on failure.
    """
    try:
        from ...shared.utils.file_utils import (
            extract_text_from_bytes as _shared_extract,
        )

        result = _shared_extract(file_bytes, filename)
        if result:
            log.debug(
                "text_extracted_via_shared_utils", filename=filename, chars=len(result)
            )
            return result
    except (ImportError, Exception):
        pass

    ext = _ext(filename)

    # ── DOCX ──────────────────────────────────────────────────────────────────
    if ext == "docx":
        try:
            import docx

            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)
            log.debug("docx_extracted", filename=filename, chars=len(text))
            return text or f"[DOCX файл '{filename}' не содержит текстовых параграфов]"
        except ImportError:
            return (
                f"[Для извлечения текста из DOCX установите: pip install python-docx]\n"
                f"Файл: {filename}, размер: {len(file_bytes)} байт"
            )
        except Exception as exc:
            log.warning("docx_extraction_failed", filename=filename, error=str(exc))
            return f"[Ошибка извлечения текста из DOCX '{filename}': {exc}]"

    # ── PDF ───────────────────────────────────────────────────────────────────
    if ext == "pdf":
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)
            text = "\n".join(pages_text)
            log.debug(
                "pdf_extracted",
                filename=filename,
                pages=len(reader.pages),
                chars=len(text),
            )
            return (
                text
                or f"[PDF '{filename}' не содержит извлекаемого текста (возможно, скан)]"
            )
        except ImportError:
            return (
                f"[Для извлечения текста из PDF установите: pip install PyPDF2]\n"
                f"Файл: {filename}, размер: {len(file_bytes)} байт"
            )
        except Exception as exc:
            log.warning("pdf_extraction_failed", filename=filename, error=str(exc))
            return f"[Ошибка извлечения текста из PDF '{filename}': {exc}]"

    # ── TXT ───────────────────────────────────────────────────────────────────
    if ext == "txt":
        try:
            text = file_bytes.decode("utf-8", errors="ignore").strip()
            log.debug("txt_extracted", filename=filename, chars=len(text))
            return text or f"[TXT файл '{filename}' пуст]"
        except Exception as exc:
            return f"[Ошибка чтения TXT '{filename}': {exc}]"

    # ── RTF ───────────────────────────────────────────────────────────────────
    if ext == "rtf":
        try:
            from striprtf.striprtf import rtf_to_text

            rtf_str = file_bytes.decode("utf-8", errors="ignore")
            text = rtf_to_text(rtf_str).strip()
            log.debug("rtf_extracted", filename=filename, chars=len(text))
            return text or f"[RTF '{filename}' не содержит текста]"
        except ImportError:
            return (
                f"[Для RTF установите: pip install striprtf]\n"
                f"Файл: {filename}, размер: {len(file_bytes)} байт"
            )
        except Exception as exc:
            return f"[Ошибка извлечения RTF '{filename}': {exc}]"

    # ── HTML ──────────────────────────────────────────────────────────────────
    if ext in ("html", "htm"):
        try:
            html_str = file_bytes.decode("utf-8", errors="ignore")
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html_str, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                import re

                text = re.sub(r"<[^>]+>", " ", html_str)
                text = re.sub(r"\s+", " ", text).strip()
            log.debug("html_extracted", filename=filename, chars=len(text))
            return text or f"[HTML '{filename}' не содержит текста]"
        except Exception as exc:
            return f"[Ошибка извлечения HTML '{filename}': {exc}]"

    # ── DOC (legacy binary) ───────────────────────────────────────────────────
    if ext == "doc":
        return (
            f"[Формат .doc (legacy binary) требует специализированных библиотек. "
            f"Рекомендуется конвертировать в .docx или .pdf через LibreOffice.]\n"
            f"Файл: {filename}, размер: {len(file_bytes)} байт"
        )

    # ── TIFF/изображения — нет текста ─────────────────────────────────────────
    if ext in ("tif", "tiff", "jpg", "jpeg", "png", "gif", "bmp"):
        return (
            f"[Файл '{filename}' является изображением. "
            f"Для извлечения текста из изображений используйте OCR.]\n"
            f"Размер: {len(file_bytes)} байт"
        )

    # ── Неизвестный формат ────────────────────────────────────────────────────
    return (
        f"[Формат '.{ext}' не поддерживается для извлечения текста. "
        f"Поддерживаются: .docx, .pdf, .txt, .rtf, .html]\n"
        f"Файл: {filename}, размер: {len(file_bytes)} байт"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Tool
# ─────────────────────────────────────────────────────────────────────────────


class AttachmentTool(AbstractEdmsTool):
    """Tool for extracting text from EDMS document attachments.

    Downloads files via REST API (EdmsAttachmentClient) — no MinIO/Storage required.

    Workflow:
        MODE 1 (attachment_id=None):
            → GET document with attachments
            → Return list of files with metadata

        MODE 2 (attachment_id provided):
            → GET document with attachments (find target)
            → Download bytes via EdmsAttachmentClient.get_content()
            → Extract text via _extract_text_from_bytes() (no ContentType enum usage)
            → Return text_preview + metadata

        ContentType enum НЕ используется для маппинга расширений — только строки.
    """

    name: str = "doc_get_file_content"
    description: str = (
        "Скачивает и извлекает текст из файла-вложения документа EDMS. "
        "Без attachment_id возвращает список доступных файлов с метаданными."
    )
    args_schema: type[BaseModel] = AttachmentFetchInput

    _MAX_PREVIEW_CHARS: int = 15_000

    def __init__(
        self,
        document_repository: Any,
        attachment_client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AttachmentTool.

        Args:
            document_repository: Repository implementing get_with_attachments().
                Used to fetch document metadata + attachment list.
            attachment_client: EdmsAttachmentClient instance for file download.
                If None — creates EdmsAttachmentClient() lazily on first use.
            **kwargs: Additional AbstractEdmsTool arguments.
        """
        super().__init__(**kwargs)
        self._doc_repo = document_repository
        self._attachment_client = attachment_client

    def _get_attachment_client(self) -> Any:
        """Get or lazily create EdmsAttachmentClient.

        Returns:
            EdmsAttachmentClient instance.
        """
        if self._attachment_client is None:
            from ...infrastructure.edms_api.clients.attachment_client import (
                EdmsAttachmentClient,
            )

            self._attachment_client = EdmsAttachmentClient()
        return self._attachment_client

    async def _arun(
        self,
        token: str,
        document_id: UUID,
        attachment_id: UUID | None = None,
        analysis_mode: str = "text",
    ) -> dict[str, Any]:
        """Execute file content extraction.

        Args:
            token: JWT bearer token.
            document_id: Parent document UUID.
            attachment_id: Target attachment UUID (None = list mode).
            analysis_mode: 'text' | 'metadata' | 'full'.

        Returns:
            Dict with status, data (file list or extracted text), message.
        """
        try:
            # ── Шаг 1: Получаем документ с вложениями ─────────────────────────
            document: Document | None = await self._doc_repo.get_with_attachments(
                document_id=document_id, token=token
            )

            if document is None:
                return self._handle_error(
                    ValueError(f"Документ {document_id} не найден или недоступен")
                )

            attachments: list[Attachment] = document.attachments or []

            if not attachments:
                return self._success_response(
                    data={"files": []},
                    message="В документе отсутствуют вложения.",
                )

            log.info(
                "downloading_attachment_via_api",
                document_id=str(document_id),
                attachment_id=str(attachment_id) if attachment_id else "none",
                file_name=next(
                    (a.file_name for a in attachments if a.id == attachment_id),
                    "listing",
                ),
            )

            # ── Шаг 2a: Режим списка файлов (attachment_id не указан) ─────────
            if attachment_id is None:
                files_info = [
                    {
                        "id": str(att.id),
                        "name": att.file_name,
                        "size_kb": round((att.file_size or 0) / 1024, 1),
                        "content_type": (
                            att.content_type.value
                            if att.content_type
                            else _ext(att.file_name).upper() or "unknown"
                        ),
                        "can_extract_text": _ext(att.file_name)
                        in ("pdf", "docx", "txt", "rtf", "html", "htm", "doc"),
                    }
                    for att in attachments
                ]
                return {
                    "status": "need_selection",
                    "message": (
                        f"Найдено {len(files_info)} вложений. "
                        "Укажите attachment_id для извлечения текста."
                    ),
                    "data": {"files": files_info},
                }

            # ── Шаг 2b: Режим извлечения текста конкретного файла ────────────
            target: Attachment | None = next(
                (a for a in attachments if a.id == attachment_id), None
            )
            if target is None:
                return self._handle_error(
                    ValueError(
                        f"Вложение {attachment_id} не найдено в документе {document_id}"
                    )
                )

            # ── Шаг 3: Только метаданные ──────────────────────────────────────
            if analysis_mode == "metadata":
                return self._success_response(
                    data={
                        "id": str(target.id),
                        "name": target.file_name,
                        "size_kb": round((target.file_size or 0) / 1024, 1),
                        "content_type": (
                            target.content_type.value
                            if target.content_type
                            else _ext(target.file_name).upper() or "OTHER"
                        ),
                        "upload_date": (
                            str(target.upload_date) if target.upload_date else None
                        ),
                        "version": target.version_number,
                    },
                    message=f"Метаданные файла '{target.file_name}'",
                )

            # ── Шаг 4: Скачиваем байты через EdmsAttachmentClient ─────────────
            try:
                attachment_client = self._get_attachment_client()
                async with attachment_client as client:
                    file_bytes: bytes = await client.get_content(
                        document_id=document_id,
                        attachment_id=attachment_id,
                        token=token,
                    )
                log.info(
                    "attachment_downloaded",
                    file_name=target.file_name,
                    size_bytes=len(file_bytes),
                )
            except Exception as download_exc:
                log.error(
                    "attachment_download_failed",
                    attachment_id=str(attachment_id),
                    document_id=str(document_id),
                    error=str(download_exc),
                )
                return self._handle_error(
                    ValueError(
                        f"Не удалось скачать файл '{target.file_name}': {download_exc}"
                    )
                )

            # ── Шаг 5: Извлекаем текст (БЕЗ ContentType enum!) ───────────────
            try:
                text_content = _extract_text_from_bytes(file_bytes, target.file_name)
                log.debug(
                    "text_extracted",
                    filename=target.file_name,
                    chars=len(text_content),
                )
            except Exception as extract_exc:
                log.error(
                    "attachment_extraction_failed",
                    attachment_id=str(attachment_id),
                    document_id=str(document_id),
                    error=str(extract_exc),
                )
                text_content = (
                    f"[Ошибка извлечения текста из '{target.file_name}': {extract_exc}]"
                )

            # ── Шаг 6: Формируем ответ ────────────────────────────────────────
            preview = text_content[: self._MAX_PREVIEW_CHARS]
            is_truncated = len(text_content) > self._MAX_PREVIEW_CHARS

            response_data: dict[str, Any] = {
                "meta": {
                    "id": str(target.id),
                    "name": target.file_name,
                    "size_kb": round((target.file_size or 0) / 1024, 1),
                    "content_type": (
                        target.content_type.value
                        if target.content_type
                        else _ext(target.file_name).upper() or "OTHER"
                    ),
                },
                "content": preview,
                "text_preview": preview,
                "is_truncated": is_truncated,
                "total_chars": len(text_content),
            }

            if analysis_mode == "full":
                response_data["all_files"] = [
                    {"id": str(a.id), "name": a.file_name} for a in attachments
                ]

            return self._success_response(
                data=response_data,
                message=f"Текст извлечён из '{target.file_name}' ({len(text_content)} симв.)",
            )

        except Exception as exc:
            log.error(
                "attachment_tool_error",
                document_id=str(document_id),
                attachment_id=str(attachment_id) if attachment_id else None,
                error=str(exc),
                exc_info=True,
            )
            return self._handle_error(exc)

    def _run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Sync execution not supported — use async."""
        raise NotImplementedError("AttachmentTool requires async execution via _arun()")
