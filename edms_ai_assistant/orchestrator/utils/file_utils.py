# utils\file_utils.py
import io
import logging

logger = logging.getLogger(__name__)

try:
    import docx2txt
    from pypdf import PdfReader
except ImportError:
    logger.warning("Не удалось импортировать модули docx2txt и PyPDF")
    docx2txt = None
    PdfReader = None


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str | None:
    """
    Извлекает текст из байтов файла (поддержка .docx, .pdf, .txt).
    """
    try:
        file_stream = io.BytesIO(file_bytes)
        ext = filename.lower().split(".")[-1] if "." in filename else ""

        if ext == "pdf" and PdfReader:
            reader = PdfReader(file_stream)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()

        elif ext == "docx" and docx2txt:
            return docx2txt.process(file_stream)

        elif ext == "txt":
            return file_bytes.decode("utf-8", errors="ignore").strip()

        else:
            logger.warning(f"Неподдерживаемый формат файла: {ext}")
            return None

    except Exception as e:
        logger.error(f"Ошибка извлечения текста из {filename}: {e}")
        return None
