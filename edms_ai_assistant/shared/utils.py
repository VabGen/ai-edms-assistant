# edms_ai_assistant/shared/utils.py
"""
Общие утилиты.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

# ── Regex ─────────────────────────────────────────────────────────────────

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

DOC_NUMBER_RE = re.compile(r"^DOC-\d{1,10}$", re.IGNORECASE)
JWT_RE = re.compile(r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$")


# ── JSON Encoder ──────────────────────────────────────────────────────────


class CustomJSONEncoder(json.JSONEncoder):
    """
    Кастомный JSON encoder для сериализации специальных типов данных.

    Поддерживает:
    - UUID → str
    - datetime → ISO 8601 с timezone
    - Enum → value
    - Pydantic models → dict
    """

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "dict"):
            return obj.dict()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            if obj.tzinfo is not None:
                return obj.isoformat()
            return obj.isoformat() + "Z"
        if isinstance(obj, Enum):
            return obj.value
        return json.JSONEncoder.default(self, obj)


# ── Hash utils ────────────────────────────────────────────────────────────


def get_file_hash(file_path: str) -> str:
    """Генерирует SHA-256 хэш содержимого файла."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 хэш строки."""
    return hashlib.sha256(text.encode()).hexdigest()


async def sha256_file_async(file_path: str, chunk_size: int = 65536) -> str:
    """Асинхронно вычисляет SHA-256 файла по чанкам."""
    import aiofiles

    h = hashlib.sha256()
    async with aiofiles.open(file_path, "rb") as f:
        while chunk := await f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ── API utils ─────────────────────────────────────────────────────────────


def prepare_auth_headers(token: str) -> dict[str, str]:
    """Создает стандартные заголовки для EDMS API."""
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def handle_api_error(response: httpx.Response, request_info: str) -> None:
    """
    Проверяет статус ответа и вызывает исключение при ошибке (>= 400).

    Args:
        response: httpx.Response объект.
        request_info: Описание запроса для логирования.
    """
    if response.is_error:
        try:
            error_details = response.json()
        except (json.JSONDecodeError, AttributeError):
            error_details = {"text": response.text[:200]}

        logger.error(
            "API Error [%d] for %s. Details: %s",
            response.status_code,
            request_info,
            error_details,
        )
        response.raise_for_status()


# ── File utils ────────────────────────────────────────────────────────────


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str | None:
    """
    Извлекает текст из байтов файла.

    Поддерживаемые форматы: .docx, .pdf, .txt
    """
    try:
        file_stream = io.BytesIO(file_bytes)
        ext = filename.lower().split(".")[-1] if "." in filename else ""

        if ext == "pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(file_stream)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text.strip()
            except ImportError:
                logger.warning("pypdf не установлен")
                return None

        elif ext == "docx":
            try:
                import docx2txt

                return docx2txt.process(file_stream)
            except ImportError:
                logger.warning("docx2txt не установлен")
                return None

        elif ext == "txt":
            return file_bytes.decode("utf-8", errors="ignore").strip()

        else:
            logger.warning("Неподдерживаемый формат файла: %s", ext)
            return None

    except Exception as e:
        logger.error("Ошибка извлечения текста из %s: %s", filename, e)
        return None


# ── Format utils ──────────────────────────────────────────────────────────


def format_datetime_ru(dt: "datetime | str | None") -> str:
    """Форматирует datetime в русском формате: 01 января 2025, 14:30."""
    from datetime import timezone

    if dt is None:
        return "—"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    _MONTHS = [
        "",
        "января",
        "февраля",
        "марта",
        "апреля",
        "мая",
        "июня",
        "июля",
        "августа",
        "сентября",
        "октября",
        "ноября",
        "декабря",
    ]
    local = dt.astimezone(timezone.utc)
    return f"{local.day:02d} {_MONTHS[local.month]} {local.year}, {local.hour:02d}:{local.minute:02d}"


def truncate(text: str, max_len: int = 200, suffix: str = "...") -> str:
    """Обрезает текст до max_len символов."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def pluralize_ru(n: int, one: str, few: str, many: str) -> str:
    """Склоняет существительное по числу (1 документ, 2 документа, 5 документов)."""
    n_abs = abs(n) % 100
    if 11 <= n_abs <= 19:
        return f"{n} {many}"
    r = n_abs % 10
    if r == 1:
        return f"{n} {one}"
    if 2 <= r <= 4:
        return f"{n} {few}"
    return f"{n} {many}"
