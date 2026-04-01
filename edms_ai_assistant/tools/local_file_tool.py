from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.services.file_processor import FileProcessorService

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 15000
_ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".rtf",
    ".odt",
    ".md",
    ".xlsx",
    ".xls",
    ".csv",
}


class LocalFileInput(BaseModel):
    """Validated input for read_local_file_content.

    Attributes:
        file_path: Absolute path to the local file to read.
    """

    file_path: str = Field(
        ...,
        description=(
            "ПОЛНЫЙ абсолютный путь к локальному файлу. "
            "Берётся из блока <local_file_path> в system prompt."
        ),
        min_length=1,
        max_length=500,
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Reject placeholder strings; require a real path.

        Args:
            v: Raw path value.

        Returns:
            Stripped path string.

        Raises:
            ValueError: If path looks like a placeholder or is blank.
        """
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Путь к файлу не может быть пустым")
        lower = cleaned.lower()
        if "file_path" in lower or "<" in cleaned or ">" in cleaned:
            raise ValueError(
                "Передан плейсхолдер вместо реального пути. "
                "Используй значение из <local_file_path> в system prompt."
            )
        return cleaned


@tool("read_local_file_content", args_schema=LocalFileInput)
async def read_local_file_content(file_path: str) -> dict[str, Any]:
    """Extract text and metadata from a local file on disk.

    Supported formats: PDF, DOCX, DOC, TXT, RTF, ODT, MD, XLSX, XLS, CSV.

    This is an async tool — it delegates CPU work to a thread-pool executor
    so the event loop is never blocked.

    Args:
        file_path: Absolute path to the file (from <local_file_path> in context).

    Returns:
        Dict with keys:
        - status     : «success» | «error»
        - meta       : File metadata dict.
        - content    : Extracted text (up to 15 000 chars).
        - is_truncated: bool — whether text was trimmed.
        - total_chars: int — full character count before truncation.
        - message    : Error description (only for status=error).
    """
    logger.info("read_local_file_content called", extra={"file_path": file_path})

    path = Path(file_path)

    if not path.exists():
        logger.error("File not found: %s", file_path)
        return {
            "status": "error",
            "message": (
                f"Файл не найден: '{file_path}'. "
                "Проверьте, что файл был загружен и путь указан верно."
            ),
        }

    if not path.is_file():
        return {
            "status": "error",
            "message": f"Указанный путь не является файлом: '{file_path}'.",
        }

    suffix = path.suffix.lower()
    size_bytes = path.stat().st_size
    size_mb = round(size_bytes / (1024 * 1024), 2)

    meta = {
        "имя_файла": path.name,
        "расширение": suffix,
        "размер_мб": size_mb,
        "путь": str(path),
    }

    if suffix not in _ALLOWED_EXTENSIONS:
        return {
            "status": "error",
            "message": (
                f"Формат '{suffix}' не поддерживается. "
                f"Поддерживаемые форматы: {', '.join(sorted(_ALLOWED_EXTENSIONS))}."
            ),
            "meta": meta,
        }

    text_content: str = await FileProcessorService.extract_text_async(str(path))

    if (
        not text_content
        or text_content.startswith("Ошибка:")
        or text_content.startswith("Формат файла")
    ):
        logger.error("Text extraction failed for %s: %s", file_path, text_content[:200])
        return {
            "status": "error",
            "message": (
                f"Не удалось извлечь текст из '{path.name}'. "
                f"Возможно, файл зашифрован, повреждён или является отсканированным изображением. "
                f"Подробности: {text_content[:300]}"
            ),
            "meta": meta,
        }

    total_chars = len(text_content)
    is_truncated = total_chars > _MAX_CONTENT_CHARS
    content = text_content[:_MAX_CONTENT_CHARS] if is_truncated else text_content

    logger.info(
        "local_file_read_success",
        extra={
            "file": path.name,
            "total_chars": total_chars,
            "is_truncated": is_truncated,
            "size_mb": size_mb,
        },
    )

    return {
        "status": "success",
        "meta": meta,
        "content": content,
        "is_truncated": is_truncated,
        "total_chars": total_chars,
    }
