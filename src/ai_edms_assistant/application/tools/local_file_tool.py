# src/ai_edms_assistant/application/tools/local_file_tool.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ...domain.value_objects.file_metadata import FileMetadata
from ..processors import AbstractProcessor
from .base_tool import AbstractEdmsTool


class LocalFileInput(BaseModel):
    """Input schema for local file reading.

    Attributes:
        file_path: Full path to local file. Must be absolute path.
    """

    file_path: str = Field(
        ...,
        description="ПОЛНЫЙ путь к локальному файлу. "
        "Возьми его ИЗ ТЕКСТА ЗАПРОСА в блоке [ДОСТУПЕН ЛОКАЛЬНЫЙ ФАЙЛ].",
    )


class LocalFileTool(AbstractEdmsTool):
    """Tool for extracting text from locally uploaded files.

    Supports: PDF, DOCX, TXT, DOC, RTF via FileProcessor.
    Used when user uploads file directly to the interface (not via EDMS).

    Dependencies:
        - ``AbstractProcessor[FileMetadata, str]``: File processor for text extraction.
    """

    name: str = "read_local_file_content"
    description: str = (
        "Извлекает текст из локального файла (PDF, DOCX, TXT). "
        "Используй для анализа документов, загруженных пользователем напрямую."
    )
    args_schema: type[BaseModel] = LocalFileInput

    _SUPPORTED_EXTENSIONS: tuple[str, ...] = (
        ".pdf",
        ".docx",
        ".txt",
        ".doc",
        ".rtf",
        ".html",
        ".htm",
    )

    def __init__(
        self,
        file_processor: AbstractProcessor[FileMetadata, str] | None = None,
        **kwargs,
    ):
        """Initialize with optional file processor.

        Args:
            file_processor: Processor for extracting text from files.
                If None, uses simplified fallback logic.
            **kwargs: Additional BaseTool arguments.
        """
        super().__init__(**kwargs)
        self._processor = file_processor

    async def _arun(self, file_path: str) -> dict[str, Any]:
        """Execute local file text extraction.

        Args:
            file_path: Absolute path to local file.

        Returns:
            Dict with extracted text and metadata.
        """
        try:
            # Validate path
            if "file_path" in file_path or not file_path.strip():
                return self._handle_error(
                    ValueError(
                        "Передан некорректный путь. "
                        "Найдите реальный путь в блоке [ДОСТУПЕН ЛОКАЛЬНЫЙ ФАЙЛ]."
                    )
                )

            path = Path(file_path)

            if not path.exists():
                return self._handle_error(
                    ValueError("Файл не найден по указанному пути")
                )

            # Build file metadata
            file_meta = self._build_file_metadata(path)

            # Extract text
            if self._processor:
                # Use injected processor
                text_content = await self._processor.process(file_meta)
            else:
                # Fallback: simplified extraction
                text_content = self._extract_text_fallback(path)

            # Truncate for preview
            preview = text_content[:15000]
            is_truncated = len(text_content) > 15000

            return self._success_response(
                data={
                    "meta": {
                        "name": path.name,
                        "size_kb": file_meta.size_kb,
                        "extension": file_meta.extension,
                    },
                    "content": preview,
                    "is_truncated": is_truncated,
                    "total_chars": len(text_content),
                },
                message="Текст извлечён",
            )

        except Exception as e:
            return self._handle_error(e)

    def _build_file_metadata(self, path: Path) -> FileMetadata:
        """Build FileMetadata value object from path.

        Args:
            path: Path to file.

        Returns:
            FileMetadata value object.
        """
        from ...domain.entities.attachment import ContentType

        # Infer content type from extension
        ext = path.suffix.lower()
        content_type_map = {
            ".pdf": ContentType.PDF,
            ".docx": ContentType.DOCX,
            ".doc": ContentType.DOC,
            ".txt": ContentType.TXT,
            ".rtf": ContentType.RTF,
            ".html": ContentType.HTML,
            ".htm": ContentType.HTML,
        }
        content_type = content_type_map.get(ext, ContentType.UNKNOWN)

        from uuid import uuid4

        return FileMetadata(
            attachment_id=uuid4(),  # Generate temporary UUID
            file_name=path.name,
            content_type=content_type,
            size_bytes=path.stat().st_size,
            storage_path=str(path.absolute()),
        )

    def _extract_text_fallback(self, path: Path) -> str:
        """Fallback text extraction when no processor available.

        Args:
            path: Path to file.

        Returns:
            Extracted text string.
        """
        ext = path.suffix.lower()

        # TXT files - direct read
        if ext == ".txt":
            try:
                return path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                return f"[Ошибка чтения .txt файла: {e}]"

        # HTML files - read as text (simplified, no HTML parsing)
        if ext in (".html", ".htm"):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                # Very basic HTML stripping (real impl would use BeautifulSoup)
                import re

                content = re.sub(r"<[^>]+>", " ", content)
                content = re.sub(r"\s+", " ", content)
                return content.strip()
            except Exception as e:
                return f"[Ошибка чтения .html файла: {e}]"

        # PDF files - requires pypdf2 or similar
        if ext == ".pdf":
            try:
                # Attempt to use pypdf2 if available
                import PyPDF2

                text_parts = []
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text_parts.append(page.extract_text())
                return "\n".join(text_parts)
            except ImportError:
                return (
                    f"[PDF парсинг требует PyPDF2. Установите: pip install pypdf2]\n"
                    f"Файл: {path.name}, размер: {path.stat().st_size} байт"
                )
            except Exception as e:
                return f"[Ошибка парсинга PDF: {e}]"

        # DOCX files - requires python-docx
        if ext == ".docx":
            try:
                import docx

                doc = docx.Document(str(path))
                text_parts = [p.text for p in doc.paragraphs]
                return "\n".join(text_parts)
            except ImportError:
                return (
                    f"[DOCX парсинг требует python-docx. "
                    f"Установите: pip install python-docx]\n"
                    f"Файл: {path.name}, размер: {path.stat().st_size} байт"
                )
            except Exception as e:
                return f"[Ошибка парсинга DOCX: {e}]"

        # DOC/RTF - requires more complex libraries
        if ext in (".doc", ".rtf"):
            return (
                f"[Формат {ext} требует специализированных библиотек. "
                f"Рекомендуется конвертировать в .docx или .pdf]\n"
                f"Файл: {path.name}, размер: {path.stat().st_size} байт"
            )

        # Unknown format
        return (
            f"[Неподдерживаемый формат: {ext}]\n"
            f"Поддерживаются: {', '.join(self._SUPPORTED_EXTENSIONS)}"
        )

    def _run(self, *args, **kwargs):
        """Sync execution not supported."""
        raise NotImplementedError("Use _arun for async execution")
