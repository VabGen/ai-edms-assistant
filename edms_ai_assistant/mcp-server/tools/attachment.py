# edms_ai_assistant/tools/attachment.py
"""
EDMS AI Assistant — Attachment Fetch Tool.

Извлекает содержимое вложений из документов EDMS.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS: int = 15_000
_SUPPORTED_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".txt",
        ".rtf",
        ".odt",
        ".md",
    }
)
_SUPPORTED_TABLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".xlsx",
        ".xls",
        ".csv",
    }
)
_ALL_SUPPORTED: frozenset[str] = (
    _SUPPORTED_TEXT_EXTENSIONS | _SUPPORTED_TABLE_EXTENSIONS
)


# ─── Input schema ─────────────────────────────────────────────────────────────


class AttachmentFetchInput(BaseModel):
    """Validated input for the doc_get_file_content tool.

    Attributes:
        document_id: EDMS document UUID (injected by orchestrator).
        token: User JWT bearer token (injected by orchestrator).
        attachment_id: Attachment UUID or filename hint. When a filename is
            provided (e.g. "Шаблон обложки.docx"), the tool performs a
            case-insensitive name/stem lookup. If omitted, the first
            attachment is used with an informational log entry.
        analysis_mode: Extraction strategy — text | tables | metadata | full.
        summary_type: Summarisation format to forward to doc_summarize_text.
    """

    document_id: str = Field(..., description="UUID документа в СЭД")
    token: str = Field(..., description="Токен авторизации пользователя (JWT)")
    attachment_id: str | None = Field(
        None,
        description=(
            "UUID вложения или имя файла (например: «Шаблон обложки.docx»). "
            "Если не указан — берётся первое вложение документа."
        ),
    )
    analysis_mode: str = Field(
        "text",
        description=(
            "Режим анализа: "
            "'text' — текст (PDF/DOCX/TXT/RTF), "
            "'tables' — таблицы (Excel/CSV), "
            "'metadata' — только метаданные, "
            "'full' — текст + таблицы + метаданные."
        ),
    )
    summary_type: str | None = Field(
        None,
        description=(
            "Тип суммаризации для передачи следующему инструменту: "
            "extractive | abstractive | thesis."
        ),
    )

    @field_validator("analysis_mode")
    @classmethod
    def validate_analysis_mode(cls, v: str) -> str:
        """Normalise and validate analysis_mode value.

        Args:
            v: Raw mode string.

        Returns:
            Lowercase validated mode.

        Raises:
            ValueError: If mode is not one of the supported values.
        """
        allowed = {"text", "tables", "metadata", "full"}
        normalised = v.strip().lower()
        if normalised not in allowed:
            raise ValueError(
                f"Неизвестный режим анализа '{v}'. "
                f"Допустимые значения: {', '.join(sorted(allowed))}."
            )
        return normalised


# ─── Private helpers ──────────────────────────────────────────────────────────


def _get_attachment_name(attachment: Any) -> str:
    """Extract display name from an attachment domain object.

    Args:
        attachment: Attachment object from DocumentDto.attachmentDocument.

    Returns:
        Name string or empty string.
    """
    return (
        getattr(attachment, "name", None) or getattr(attachment, "fileName", None) or ""
    )


def _get_attachment_id(attachment: Any) -> str:
    """Extract UUID string from an attachment domain object.

    Args:
        attachment: Attachment object.

    Returns:
        UUID string or empty string.
    """
    return str(getattr(attachment, "id", "") or "")


def _resolve_attachment(attachments: list[Any], hint: str) -> Any | None:
    """Resolve an attachment from a list by UUID or filename hint.

    Resolution order:
    1. Exact UUID match.
    2. Exact case-insensitive filename match (including extension).
    3. Exact stem match (filename without extension).
    4. Partial stem containment (handles "(1)", "(2)" copy variants).

    Args:
        attachments: All attachment objects for the document.
        hint: A UUID string or filename string from the caller.

    Returns:
        First matching attachment, or None.
    """
    hint_stripped = hint.strip()

    if UUID_RE.match(hint_stripped):
        found = next(
            (a for a in attachments if _get_attachment_id(a) == hint_stripped),
            None,
        )
        if found is not None:
            logger.debug("Resolved attachment by UUID: %s...", hint_stripped[:8])
            return found

    hint_lower = hint_stripped.lower()
    hint_stem = Path(hint_lower).stem

    # Exact filename match
    for att in attachments:
        if _get_attachment_name(att).lower() == hint_lower:
            logger.info("Resolved attachment by exact name: '%s'", hint_stripped)
            return att

    # Exact stem match
    for att in attachments:
        att_stem = Path(_get_attachment_name(att).lower()).stem
        if att_stem == hint_stem and hint_stem:
            logger.info("Resolved attachment by stem: '%s'", hint_stripped)
            return att

    # Partial stem containment
    for att in attachments:
        att_stem = Path(_get_attachment_name(att).lower()).stem
        if hint_stem and att_stem and (hint_stem in att_stem or att_stem in hint_stem):
            logger.info(
                "Resolved attachment by partial stem: '%s' ~ '%s'",
                hint_stem,
                att_stem,
            )
            return att

    return None


def _build_attachment_meta(attachment: Any) -> dict[str, Any]:
    """Build a human-readable metadata dict from an attachment domain object.

    Args:
        attachment: Attachment object from DocumentDto.

    Returns:
        Dict with display-ready metadata fields in Russian.
    """
    name = _get_attachment_name(attachment) or "unknown"
    size_bytes: int = getattr(attachment, "size", None) or 0
    upload_date = getattr(attachment, "uploadDate", None)
    signs = getattr(attachment, "signs", None) or []

    att_type_obj = getattr(attachment, "attachmentDocumentType", None)
    att_type: str | None = None
    if att_type_obj:
        att_type = getattr(att_type_obj, "name", None) or str(att_type_obj)

    formatted_date: str | None = None
    if upload_date:
        try:
            formatted_date = (
                upload_date.strftime("%d.%m.%Y %H:%M")
                if hasattr(upload_date, "strftime")
                else str(upload_date)[:16]
            )
        except Exception:
            formatted_date = str(upload_date)

    return {
        "название": name,
        "тип_вложения": att_type,
        "размер_кб": round(size_bytes / 1024, 2) if size_bytes else 0,
        "дата_загрузки": formatted_date,
        "есть_эцп": bool(signs),
        "id": _get_attachment_id(attachment) or None,
    }


# ─── Tool ─────────────────────────────────────────────────────────────────────


@tool("doc_get_file_content", args_schema=AttachmentFetchInput)
async def doc_get_file_content(
    document_id: str,
    token: str,
    attachment_id: str | None = None,
    analysis_mode: str = "text",
    summary_type: str | None = None,
) -> dict[str, Any]:
    """Extract and analyse the content of a document attachment from EDMS.

    Accepts ``attachment_id`` as either a UUID or a filename string.
    Performs multi-strategy name resolution before falling back to
    the first attachment with an info-level log entry.

    Analysis modes:
    - ``text``     : Plain text extraction (PDF, DOCX, TXT, RTF, ODT, MD).
    - ``tables``   : Structured table extraction (XLSX, XLS, CSV).
    - ``metadata`` : Attachment metadata only — no file download.
    - ``full``     : Text + tables + metadata in a single response.

    Args:
        document_id: EDMS document UUID (injected by orchestrator).
        token: User JWT bearer token (injected by orchestrator).
        attachment_id: Attachment UUID or filename to read.
        analysis_mode: One of text | tables | metadata | full.
        summary_type: Summarisation format to forward to doc_summarize_text.

    Returns:
        Dict containing:
        - ``status``        : 'success' | 'info' | 'warning' | 'error'
        - ``content``       : Extracted text (for text/full modes).
        - ``tables``        : Extracted table data (for tables/full modes).
        - ``file_info``     : Attachment metadata dict.
        - ``summary_type``  : Forwarded from input for pipeline continuity.
        - ``is_truncated``  : bool — whether text was trimmed to 15 000 chars.
        - ``total_chars``   : int — full character count before truncation.
    """
    logger.info(
        "doc_get_file_content called",
        extra={
            "document_id": document_id[:8] + "...",
            "attachment_id": attachment_id,
            "analysis_mode": analysis_mode,
            "summary_type": summary_type,
        },
    )

    # ── Получение метаданных документа ───────────────────────────────────────
    try:
        async with DocumentClient() as doc_client:
            raw_data = await doc_client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw_data)
            attachments: list[Any] = doc.attachmentDocument or []
    except Exception as exc:
        logger.error("Failed to fetch document metadata: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"Ошибка получения метаданных документа: {exc}",
            "summary_type": summary_type,
        }

    if not attachments:
        return {
            "status": "info",
            "message": "В документе нет вложений.",
            "summary_type": summary_type,
        }

    # ── Разрешение целевого вложения ──────────────────────────────────────────
    if attachment_id:
        target = _resolve_attachment(attachments, attachment_id)
        if target is None:
            available = [
                {"id": _get_attachment_id(a), "name": _get_attachment_name(a)}
                for a in attachments
            ]
            names_str = ", ".join(f"«{item['name']}»" for item in available)
            return {
                "status": "requires_disambiguation",
                "message": (
                    f"Вложение «{attachment_id}» не найдено в документе. "
                    f"Выберите вложение для анализа:"
                ),
                "available_attachments": available,
            }
        resolved_id = _get_attachment_id(target)
        logger.info("Attachment resolved: '%s' → %s...", attachment_id, resolved_id[:8])
    else:
        if len(attachments) == 1:
            target = attachments[0]
            resolved_id = _get_attachment_id(target)
            logger.info("Single attachment auto-selected: %s...", resolved_id[:8])
        else:
            available = [
                {
                    "id": _get_attachment_id(a),
                    "name": _get_attachment_name(a) or "без имени",
                }
                for a in attachments
                if _get_attachment_id(a)
            ]
            names_str = ", ".join(f"«{item['name']}»" for item in available)
            logger.info(
                "No attachment_id and multiple attachments — returning disambiguation"
            )
            return {
                "status": "requires_disambiguation",
                "message": (
                    f"В документе несколько вложений. "
                    f"Укажите, какое вложение открыть для анализа: {names_str}."
                ),
                "available_attachments": available,
            }

    file_info = _build_attachment_meta(target)
    file_name: str = file_info["название"]
    suffix: str = Path(file_name).suffix.lower() if file_name else ".tmp"

    # ── Режим metadata: без скачивания ────────────────────────────────────────
    if analysis_mode == "metadata":
        return {
            "status": "success",
            "mode": "metadata",
            "file_info": file_info,
            "summary_type": summary_type,
        }

    # ── Скачивание вложения ───────────────────────────────────────────────────
    att_doc_id: str = str(getattr(target, "documentId", None) or document_id)
    try:
        async with EdmsAttachmentClient() as attach_client:
            content_bytes = await attach_client.get_attachment_content(
                token, att_doc_id, resolved_id
            )
    except Exception as exc:
        logger.error(
            "Failed to download attachment '%s': %s", file_name, exc, exc_info=True
        )
        return {
            "status": "error",
            "message": f"Ошибка скачивания «{file_name}»: {exc}",
            "file_info": file_info,
            "summary_type": summary_type,
        }

    if not content_bytes:
        return {
            "status": "error",
            "message": f"Файл «{file_name}» пустой или недоступен для скачивания.",
            "file_info": file_info,
            "summary_type": summary_type,
        }

    # ── Обработка через FileProcessorService ─────────────────────────────────
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name

        # ── full: текст + таблицы + метаданные ───────────────────────────────
        if analysis_mode == "full":
            structured = await FileProcessorService.extract_structured_data(tmp_path)
            text_content: str = structured.get("text", "")
            return {
                "status": "success",
                "mode": "full",
                "file_info": file_info,
                "content": text_content[:_MAX_TEXT_CHARS],
                "is_truncated": len(text_content) > _MAX_TEXT_CHARS,
                "total_chars": len(text_content),
                "metadata": structured.get("metadata"),
                "stats": structured.get("stats"),
                "tables": structured.get("tables"),
                "summary_type": summary_type,
            }

        # ── tables: только таблицы (Excel/CSV) ───────────────────────────────
        if analysis_mode == "tables":
            if suffix not in _SUPPORTED_TABLE_EXTENSIONS:
                return {
                    "status": "error",
                    "message": (
                        f"Режим 'tables' поддерживается только для Excel/CSV. "
                        f"Текущий формат: '{suffix}'. "
                        "Используй режим 'text' для текстовых документов."
                    ),
                    "file_info": file_info,
                    "summary_type": summary_type,
                }
            structured = await FileProcessorService.extract_structured_data(tmp_path)
            tables = structured.get("tables", [])
            return {
                "status": "success",
                "mode": "tables",
                "file_info": file_info,
                "tables": tables,
                "tables_count": len(tables) if tables else 0,
                "summary_type": summary_type,
            }

        # ── text: извлечение текста (default) ────────────────────────────────
        if suffix not in _ALL_SUPPORTED:
            return {
                "status": "warning",
                "message": (
                    f"Формат '{suffix}' не поддерживается для извлечения текста. "
                    f"Поддерживаемые: {', '.join(sorted(_ALL_SUPPORTED))}. "
                    "Метаданные файла возвращены."
                ),
                "file_info": file_info,
                "summary_type": summary_type,
            }

        text_content = await FileProcessorService.extract_text_async(tmp_path)

        if not text_content or text_content.startswith(("Ошибка:", "Формат файла")):
            return {
                "status": "error",
                "message": (
                    f"Не удалось извлечь текст из «{file_name}». "
                    "Возможно, файл является сканом или защищён паролем. "
                    f"Подробности: {text_content}"
                ),
                "file_info": file_info,
                "summary_type": summary_type,
            }

        return {
            "status": "success",
            "mode": "text",
            "file_info": file_info,
            "content": text_content[:_MAX_TEXT_CHARS],
            "is_truncated": len(text_content) > _MAX_TEXT_CHARS,
            "total_chars": len(text_content),
            "summary_type": summary_type,
        }

    except Exception as exc:
        logger.error(
            "File processing error for '%s': %s", file_name, exc, exc_info=True
        )
        return {
            "status": "error",
            "message": (
                f"Не удалось обработать «{file_name}»: {exc}. "
                "Возможно, файл повреждён или защищён паролем."
            ),
            "file_info": file_info,
            "summary_type": summary_type,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as cleanup_err:
                logger.warning(
                    "Failed to remove temp file '%s': %s", tmp_path, cleanup_err
                )
