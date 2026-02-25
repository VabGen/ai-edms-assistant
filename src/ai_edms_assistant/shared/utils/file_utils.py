# src/ai_edms_assistant/shared/utils/file_utils.py
"""
File content extraction utilities.

Migrated from edms_ai_assistant/utils/file_utils.py.
Supports .pdf, .docx, .txt with graceful ImportError handling.
"""

from __future__ import annotations

import io
import structlog
from typing import Optional

logger = structlog.get_logger(__name__)

try:
    import docx2txt
except ImportError:
    docx2txt = None
    logger.warning("docx2txt not installed — .docx extraction unavailable")

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None
    logger.warning("PyPDF2 not installed — .pdf extraction unavailable")


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> Optional[str]:
    """
    Extract plain text from file bytes based on filename extension.

    Supported formats: .pdf, .docx, .txt

    Args:
        file_bytes: Raw file content.
        filename:   Original filename including extension.

    Returns:
        Extracted text string or None on failure / unsupported format.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    try:
        if ext == "pdf" and PdfReader is not None:
            reader = PdfReader(io.BytesIO(file_bytes))
            return (
                "\n".join((page.extract_text() or "") for page in reader.pages).strip()
                or None
            )

        if ext == "docx" and docx2txt is not None:
            return docx2txt.process(io.BytesIO(file_bytes))

        if ext == "txt":
            return file_bytes.decode("utf-8", errors="ignore").strip() or None

        logger.warning("unsupported_file_format", ext=ext, filename=filename)
        return None

    except Exception as exc:
        logger.error("file_text_extraction_failed", filename=filename, error=str(exc))
        return None


def truncate_text(text: str, max_chars: int = 15_000) -> tuple[str, bool]:
    """
    Truncate text to max_chars, returning (truncated, was_truncated).

    Args:
        text:      Input string.
        max_chars: Maximum character count (default 15 000).

    Returns:
        Tuple of (possibly truncated text, truncation flag).
    """
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True
