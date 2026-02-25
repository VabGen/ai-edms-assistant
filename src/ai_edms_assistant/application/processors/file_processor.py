# src/ai_edms_assistant/application/processors/file_processor.py
"""File processor for extracting text from various file formats."""

from __future__ import annotations

import logging
from pathlib import Path

from ...domain.value_objects.file_metadata import FileMetadata
from .base_processor import AbstractProcessor

logger = logging.getLogger(__name__)


class FileProcessor(AbstractProcessor[FileMetadata, str]):
    """Production-ready file processor with support for PDF, DOCX, TXT, HTML, RTF.

    Uses appropriate libraries for each format:
    - PDF: pypdf2
    - DOCX: python-docx
    - TXT/HTML: native Python
    - RTF: striprtf
    """

    async def process(self, metadata: FileMetadata) -> str:
        """Extract text from file based on extension.

        Args:
            metadata: FileMetadata with storage_path.

        Returns:
            Extracted text string.

        Raises:
            ValueError: When file format is unsupported or extraction fails.
        """
        if not metadata.can_extract_text:
            raise ValueError(f"Unsupported file format: {metadata.extension}")

        path = Path(metadata.storage_path)
        if not path.exists():
            raise ValueError(f"File not found: {path}")

        try:
            if metadata.extension == ".pdf":
                return await self._process_pdf(path)
            elif metadata.extension == ".docx":
                return await self._process_docx(path)
            elif metadata.extension == ".txt":
                return await self._process_txt(path)
            elif metadata.extension in (".html", ".htm"):
                return await self._process_html(path)
            elif metadata.extension == ".rtf":
                return await self._process_rtf(path)
            else:
                raise ValueError(f"Unsupported extension: {metadata.extension}")

        except Exception as e:
            logger.error(f"Failed to extract text from {metadata.file_name}: {e}")
            raise ValueError(f"Extraction failed: {e}")

    async def _process_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            import PyPDF2

            text_parts = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text())
            return "\n".join(text_parts)
        except ImportError:
            raise ValueError("PDF support requires: pip install pypdf2")

    async def _process_docx(self, path: Path) -> str:
        """Extract text from DOCX."""
        try:
            import docx

            doc = docx.Document(str(path))
            text_parts = [p.text for p in doc.paragraphs]
            return "\n".join(text_parts)
        except ImportError:
            raise ValueError("DOCX support requires: pip install python-docx")

    async def _process_txt(self, path: Path) -> str:
        """Extract text from TXT."""
        return path.read_text(encoding="utf-8", errors="ignore")

    async def _process_html(self, path: Path) -> str:
        """Extract text from HTML."""
        try:
            from bs4 import BeautifulSoup

            html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Fallback: basic regex
            import re

            content = path.read_text(encoding="utf-8", errors="ignore")
            content = re.sub(r"<[^>]+>", " ", content)
            return re.sub(r"\s+", " ", content).strip()

    async def _process_rtf(self, path: Path) -> str:
        """Extract text from RTF."""
        try:
            from striprtf.striprtf import rtf_to_text

            rtf = path.read_text(encoding="utf-8", errors="ignore")
            return rtf_to_text(rtf)
        except ImportError:
            raise ValueError("RTF support requires: pip install striprtf")
