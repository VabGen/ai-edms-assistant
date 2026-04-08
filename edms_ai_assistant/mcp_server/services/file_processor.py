# edms_ai_assistant/services/file_processor.py
"""
EDMS AI Assistant - Enhanced File Processor Service.

Поддерживает: PDF, DOCX, DOC (legacy Word 97-2003), TXT, XLSX, XLS.
"""

import asyncio
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


def _extract_doc_via_fitz(file_path: str) -> str:
    """Extract text from .doc or .docx using PyMuPDF (fitz).

    PyMuPDF supports legacy Word 97-2003 (.doc) files via OLE2 parsing
    and modern .docx without external system dependencies.

    Args:
        file_path: Absolute path to the Word document.

    Returns:
        Extracted plain text, or empty string on failure.

    Raises:
        ImportError: If pymupdf is not installed.
        Exception: If fitz fails to open or parse the file.
    """
    import fitz  # type: ignore[import]

    doc = fitz.open(file_path)
    pages_text: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text and text.strip():
            pages_text.append(text.strip())
    doc.close()
    return "\n\n".join(pages_text)


def _extract_doc_via_mammoth(file_path: str) -> str:
    """Extract text from .doc/.docx using mammoth library as fallback.

    Converts to HTML internally and strips tags to get plain text.
    More tolerant of malformed Word documents than docx2txt.

    Args:
        file_path: Absolute path to the Word document.

    Returns:
        Extracted plain text, or empty string on failure.

    Raises:
        ImportError: If mammoth is not installed.
    """

    import mammoth  # type: ignore[import]

    with open(file_path, "rb") as f:
        result = mammoth.extract_raw_text(f)

    return result.value.strip()


def _extract_docx_via_docx2txt(file_path: str) -> str:
    """Extract text from .docx using docx2txt (ZIP-based format only).

    Args:
        file_path: Absolute path to a .docx file.

    Returns:
        Extracted plain text.

    Raises:
        KeyError: If the file is not a valid .docx ZIP archive.
    """
    import docx2txt  # type: ignore[import]

    return docx2txt.process(file_path) or ""


# ── Utility: DOC to DOCX conversion ─────────────────────────────────────

def _convert_doc_to_docx(doc_path: str) -> str:
    """Convert legacy .doc (Word 97-2003) to modern .docx using LibreOffice.

    Requires LibreOffice installed and 'soffice' in PATH.
    Windows: https://www.libreoffice.org/download/download/
    Linux: sudo apt install libreoffice

    Args:
        doc_path: Absolute path to .doc file.

    Returns:
        Absolute path to converted .docx file.

    Raises:
        RuntimeError: If conversion fails or LibreOffice not found.
    """
    import subprocess
    from pathlib import Path

    doc_path_obj = Path(doc_path)
    out_dir = doc_path_obj.parent

    try:
        soffice_cmd = "soffice"
        if not shutil.which(soffice_cmd):
            possible_paths = [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            ]
            for p in possible_paths:
                if Path(p).exists():
                    soffice_cmd = p
                    break

        result = subprocess.run(
            [
                soffice_cmd,
                "--headless",
                "--convert-to", "docx:MS Word 2007 XML",
                "--outdir", str(out_dir),
                str(doc_path_obj),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

        converted_path = out_dir / f"{doc_path_obj.stem}.docx"
        if converted_path.exists():
            logger.info(
                "DOC converted to DOCX via LibreOffice: %s → %s",
                doc_path, converted_path,
            )
            return str(converted_path)
        else:
            raise RuntimeError(f"Converted file not found: {converted_path}")

    except FileNotFoundError:
        raise RuntimeError(
            "LibreOffice not found. Install from https://www.libreoffice.org/ "
            "and ensure 'soffice' is in PATH or specify full path."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"LibreOffice conversion timed out for: {doc_path}")
    except subprocess.CalledProcessError as e:
        logger.error(
            "LibreOffice conversion failed: %s (stderr: %s)",
            e, e.stderr,
        )
        raise RuntimeError(f"Conversion failed: {e.stderr or e}")


class FileProcessorService:
    """Service for text extraction from multiple file formats.

    Extraction strategies by extension:
        .pdf    → PyPDFLoader (LangChain)
        .docx   → docx2txt → mammoth → fitz (fallback chain)
        .doc    → fitz (PyMuPDF) → mammoth (fallback chain)
        .txt    → TextLoader (LangChain)
        .xlsx   → openpyxl
        .xls    → xlrd
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".txt",
        ".xlsx",
        ".xls",
    }

    @classmethod
    async def extract_text_async(cls, file_path: str) -> str:
        """Async text extraction delegating CPU work to thread pool.

        Args:
            file_path: Absolute path to file.

        Returns:
            Extracted text string.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, cls.extract_text, file_path)

    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Synchronous text extraction with format-specific strategy.

        Args:
            file_path: Absolute path to file.

        Returns:
            Extracted text or human-readable error message.
        """
        path = Path(file_path)

        if not path.exists():
            error_msg = f"Файл не найден по пути: {file_path}"
            logger.error(error_msg, extra={"file_path": file_path})
            return f"Ошибка: {error_msg}"

        ext = path.suffix.lower()

        if ext not in cls.SUPPORTED_EXTENSIONS:
            warning_msg = f"Формат файла {ext} пока не поддерживается для анализа."
            logger.warning(
                "Unsupported file format: %s",
                ext,
                extra={"file_path": file_path, "extension": ext},
            )
            return warning_msg

        try:
            if ext in (".xlsx", ".xls"):
                return cls._extract_from_excel(file_path, ext)

            if ext == ".doc":
                return cls._extract_doc(file_path)

            if ext == ".docx":
                return cls._extract_docx(file_path)

            if ext == ".pdf":
                return cls._extract_pdf(file_path)

            if ext == ".txt":
                return cls._extract_txt(file_path)

            return f"Формат {ext} не поддерживается."

        except Exception as e:
            error_msg = f"Произошла техническая ошибка при чтении файла {ext}: {e!s}"
            logger.error(
                "File parsing error: %s",
                e,
                exc_info=True,
                extra={"file_path": file_path, "extension": ext},
            )
            return error_msg

    # ── Format-specific extractors ────────────────────────────────────────────

    @classmethod
    def _extract_doc(cls, file_path: str) -> str:
        """Extract text from legacy .doc (Word 97-2003) files.

        Uses a fallback chain:
            1. PyMuPDF (fitz) — handles OLE2 binary format natively
            2. mammoth — HTML-based extraction, more tolerant of corruption

        Args:
            file_path: Absolute path to .doc file.

        Returns:
            Extracted text or error message.
        """
        try:
            text = _extract_doc_via_fitz(file_path)
            if text and len(text.strip()) > 10:
                logger.info(
                    "DOC extracted via fitz",
                    extra={"file_path": file_path, "chars": len(text)},
                )
                return text
            logger.debug("fitz returned empty text for .doc, trying mammoth")
        except Exception as fitz_err:
            logger.warning(
                "fitz failed for .doc '%s': %s — trying mammoth",
                file_path,
                fitz_err,
            )

        try:
            text = _extract_doc_via_mammoth(file_path)
            if text and len(text.strip()) > 10:
                logger.info(
                    "DOC extracted via mammoth",
                    extra={"file_path": file_path, "chars": len(text)},
                )
                return text
        except ImportError:
            logger.warning("mammoth not installed — install with: uv add mammoth")
        except Exception as mammoth_err:
            logger.warning("mammoth failed for .doc '%s': %s", file_path, mammoth_err)

        try:
            logger.info("Attempting LibreOffice conversion for .doc: %s", file_path)
            docx_path = _convert_doc_to_docx(file_path)

            # Extract from converted .docx using existing pipeline
            text = cls._extract_docx(docx_path)

            # Clean up converted file (optional: keep for debugging)
            try:
                Path(docx_path).unlink()
                logger.debug("Removed temporary converted file: %s", docx_path)
            except Exception as cleanup_err:
                logger.warning("Failed to remove temp file %s: %s", docx_path, cleanup_err)

            if text and len(text.strip()) > 10:
                logger.info(
                    "DOC extracted via LibreOffice conversion → docx pipeline",
                    extra={"original": file_path, "converted": docx_path, "chars": len(text)},
                )
                return text

        except RuntimeError as conv_err:
            logger.warning(
                "LibreOffice conversion failed for '%s': %s",
                file_path, conv_err,
            )
        except Exception as e:
            logger.error(
                "Unexpected error during DOC extraction for '%s': %s",
                file_path, e, exc_info=True,
            )

        return (
            "Не удалось извлечь текст из файла .doc.\n"
            "Возможные причины:\n"
            "• Файл повреждён или в неподдерживаемом бинарном формате.\n"
            "• LibreOffice не установлен (требуется для конвертации).\n"
            "Рекомендация: пересохраните документ в формате .docx через Microsoft Word или LibreOffice."
        )

    @classmethod
    def _extract_docx(cls, file_path: str) -> str:
        """Extract text from .docx (Office Open XML ZIP) files.

        Uses a fallback chain:
            1. docx2txt — fast and lightweight
            2. mammoth — more robust for malformed files
            3. PyMuPDF (fitz) — last resort

        Args:
            file_path: Absolute path to .docx file.

        Returns:
            Extracted text or error message.
        """
        try:
            text = _extract_docx_via_docx2txt(file_path)
            if text and len(text.strip()) > 10:
                logger.info(
                    "DOCX extracted via docx2txt",
                    extra={"file_path": file_path, "chars": len(text)},
                )
                return text
        except KeyError as ke:
            logger.warning(
                "docx2txt failed (invalid ZIP structure) for '%s': %s — trying mammoth",
                file_path,
                ke,
            )
        except Exception as e:
            logger.warning(
                "docx2txt failed for '%s': %s — trying mammoth", file_path, e
            )

        try:
            text = _extract_doc_via_mammoth(file_path)
            if text and len(text.strip()) > 10:
                logger.info(
                    "DOCX extracted via mammoth (docx2txt fallback)",
                    extra={"file_path": file_path, "chars": len(text)},
                )
                return text
        except ImportError:
            logger.warning("mammoth not installed")
        except Exception as e:
            logger.warning("mammoth failed for '%s': %s — trying fitz", file_path, e)

        try:
            text = _extract_doc_via_fitz(file_path)
            if text and len(text.strip()) > 10:
                logger.info(
                    "DOCX extracted via fitz (final fallback)",
                    extra={"file_path": file_path, "chars": len(text)},
                )
                return text
        except Exception as e:
            logger.error("fitz also failed for '%s': %s", file_path, e)

        return (
            "Не удалось извлечь текст из файла .docx. "
            "Файл может быть повреждён или иметь нестандартную структуру."
        )

    @classmethod
    def _extract_pdf(cls, file_path: str) -> str:
        """Extract text from PDF using LangChain PyPDFLoader.

        Args:
            file_path: Absolute path to .pdf file.

        Returns:
            Extracted text or error message.
        """
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            return (
                "Файл успешно прочитан, но в нем не обнаружено текстового содержимого."
            )
        full_text = "\n\n".join(doc.page_content for doc in docs).strip()
        if not full_text:
            return (
                "Содержимое PDF пусто или состоит из изображений без текстового слоя."
            )
        logger.info(
            "PDF extracted successfully",
            extra={"file_path": file_path, "chars": len(full_text), "pages": len(docs)},
        )
        return full_text

    @classmethod
    def _extract_txt(cls, file_path: str) -> str:
        """Extract text from plain .txt files.

        Args:
            file_path: Absolute path to .txt file.

        Returns:
            Extracted text or error message.
        """
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        if not docs:
            return "Текстовый файл пуст."
        full_text = "\n\n".join(doc.page_content for doc in docs).strip()
        logger.info(
            "TXT extracted successfully",
            extra={"file_path": file_path, "chars": len(full_text)},
        )
        return full_text

    @classmethod
    def _extract_from_excel(cls, file_path: str, ext: str) -> str:
        """Extract text from Excel files preserving table structure.

        Args:
            file_path: Absolute path to Excel file.
            ext: File extension (.xlsx or .xls).

        Returns:
            Formatted text with table rows or error message.
        """
        try:
            if ext == ".xlsx":
                import openpyxl

                wb = openpyxl.load_workbook(file_path, data_only=True)
            else:
                import xlrd

                wb = xlrd.open_workbook(file_path)

            extracted_text = []

            if ext == ".xlsx":
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    extracted_text.append(f"\n{'=' * 50}\nЛИСТ: {sheet_name}\n{'=' * 50}\n")
                    for row in sheet.iter_rows(values_only=True):
                        if any(cell is not None for cell in row):
                            row_text = " | ".join(
                                str(cell) if cell is not None else "" for cell in row
                            )
                            extracted_text.append(row_text)
            else:
                for sheet_idx in range(wb.nsheets):
                    sheet = wb.sheet_by_index(sheet_idx)
                    extracted_text.append(f"\n{'=' * 50}\nЛИСТ: {sheet.name}\n{'=' * 50}\n")
                    for row_idx in range(sheet.nrows):
                        row = sheet.row_values(row_idx)
                        if any(cell for cell in row):
                            row_text = " | ".join(
                                str(cell) if cell else "" for cell in row
                            )
                            extracted_text.append(row_text)

            result = "\n".join(extracted_text).strip()
            logger.info(
                "Excel extracted successfully",
                extra={"file_path": file_path, "extension": ext, "chars": len(result)},
            )
            return result

        except ImportError as e:
            logger.error("Missing dependency for Excel: %s", e)
            return (
                "Ошибка: Для обработки Excel файлов требуется установить библиотеки "
                "'openpyxl' (для .xlsx) или 'xlrd' (для .xls)."
            )
        except Exception as e:
            logger.error("Excel extraction error: %s", e, exc_info=True)
            return f"Ошибка при чтении Excel файла: {e!s}"

    # ── Utility methods (unchanged) ───────────────────────────────────────────

    @classmethod
    async def extract_structured_data(cls, file_path: str) -> dict[str, Any]:
        """Extract structured data including text, metadata, and tables.

        Args:
            file_path: Absolute path to file.

        Returns:
            Dict with text, metadata, stats, and tables keys.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        text = await cls.extract_text_async(file_path)
        stats = {
            "chars": len(text),
            "words": len(text.split()),
            "lines": text.count("\n"),
            "digits": len([c for c in text if c.isdigit()]),
        }
        file_stat = path.stat()
        metadata = {
            "filename": path.name,
            "extension": ext,
            "size_bytes": file_stat.st_size,
            "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
        }
        result: dict[str, Any] = {
            "text": text,
            "metadata": metadata,
            "stats": stats,
            "tables": None,
        }
        if ext in (".xlsx", ".xls"):
            result["tables"] = await cls._extract_excel_tables(file_path, ext)
        return result

    @classmethod
    async def _extract_excel_tables(
            cls, file_path: str, ext: str
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, cls._extract_tables_sync, file_path, ext
        )

    @classmethod
    def _extract_tables_sync(cls, file_path: str, ext: str) -> list[dict[str, Any]]:
        try:
            if ext == ".xlsx":
                import openpyxl

                wb = openpyxl.load_workbook(file_path, data_only=True)
            else:
                import xlrd

                wb = xlrd.open_workbook(file_path)
            tables = []
            if ext == ".xlsx":
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    data = [
                        list(row)
                        for row in sheet.iter_rows(values_only=True)
                        if any(c is not None for c in row)
                    ]
                    if data:
                        tables.append(
                            {
                                "sheet_name": sheet_name,
                                "headers": data[0],
                                "data": data[1:],
                                "rows_count": len(data) - 1,
                            }
                        )
            else:
                for i in range(wb.nsheets):
                    sheet = wb.sheet_by_index(i)
                    data = [
                        sheet.row_values(r)
                        for r in range(sheet.nrows)
                        if any(sheet.row_values(r))
                    ]
                    if data:
                        tables.append(
                            {
                                "sheet_name": sheet.name,
                                "headers": data[0],
                                "data": data[1:],
                                "rows_count": len(data) - 1,
                            }
                        )
            return tables
        except Exception as e:
            logger.error("Table extraction error: %s", e, exc_info=True)
            return []

    @classmethod
    def validate_file_path(cls, file_path: str) -> str | None:
        if not file_path or not file_path.strip():
            return "Путь к файлу не может быть пустым"
        path = Path(file_path)
        if not path.exists():
            return f"Файл не найден: {file_path}"
        if not path.is_file():
            return f"Указанный путь не является файлом: {file_path}"
        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(cls.SUPPORTED_EXTENSIONS))
            return f"Неподдерживаемый формат {ext}. Поддерживаются: {supported}"
        return None

    @classmethod
    def get_file_info(cls, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            return {"exists": False, "error": f"Файл не найден: {file_path}"}
        try:
            stat = path.stat()
            return {
                "exists": True,
                "name": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "is_supported": path.suffix.lower() in cls.SUPPORTED_EXTENSIONS,
                "absolute_path": str(path.absolute()),
            }
        except Exception as e:
            return {"exists": True, "error": f"Не удалось получить информацию: {e!s}"}
