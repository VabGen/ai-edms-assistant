# edms_ai_assistant/mcp_server/tools/content_tools.py
"""
Инструменты работы с содержимым файлов и вложений.

Переписано под монорепо: FastMCP вместо langchain, llm_client вместо get_chat_model,
сырые dict вместо generated DTO.

Содержит:
  - doc_get_file_content      (из attachment.py)
  - read_local_file_content   (из local_file_tool.py)
  - doc_compare_attachment_with_local (из file_compare_tool.py)
  - doc_summarize_text        (из summarization.py)
"""
from __future__ import annotations

import difflib
import json
import logging
import os
import re
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from edms_ai_assistant.mcp_server.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.mcp_server.clients.document_client import DocumentClient
from edms_ai_assistant.mcp_server.services.file_processor import FileProcessorService
from edms_ai_assistant.shared.utils.utils import UUID_RE

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS: int = 15_000
_MAX_DIFF_LINES: int = 60
_MAX_SUMMARY_CHARS: int = 12_000
_HEAD_FRACTION: float = 0.67

_SUPPORTED_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt", ".md"}
)
_SUPPORTED_TABLE_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls", ".csv"})
_ALL_SUPPORTED: frozenset[str] = _SUPPORTED_TEXT_EXTENSIONS | _SUPPORTED_TABLE_EXTENSIONS

_PATH_PLACEHOLDERS: frozenset[str] = frozenset(
    {"", "local_file", "local_file_path", "/path/to/file",
     "none", "null", "<local_file_path>", "<path>"}
)

_SUMMARY_TYPE_ALIASES: dict[str, str] = {
    "факты": "extractive", "ключевые факты": "extractive",
    "extractive": "extractive", "1": "extractive",
    "пересказ": "abstractive", "краткий пересказ": "abstractive",
    "abstractive": "abstractive", "2": "abstractive",
    "тезисы": "thesis", "тезисный план": "thesis",
    "thesis": "thesis", "3": "thesis",
}


class SummarizeType(StrEnum):
    """Форматы суммаризации."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    THESIS = "thesis"


# ── Вспомогательные функции ───────────────────────────────────────────────────


def _get_att_name(attachment: Any) -> str:
    if isinstance(attachment, dict):
        return (
            attachment.get("name") or attachment.get("fileName")
            or attachment.get("originalName") or ""
        )
    return (
        getattr(attachment, "name", None) or getattr(attachment, "fileName", None)
        or getattr(attachment, "originalName", None) or ""
    )


def _get_att_id(attachment: Any) -> str:
    if isinstance(attachment, dict):
        return str(attachment.get("id", "") or "")
    return str(getattr(attachment, "id", "") or "")


def _resolve_attachment(attachments: list[Any], hint: str) -> Any | None:
    """Резолвит вложение по UUID или имени файла (4 уровня fallback)."""
    hint_stripped = hint.strip()
    if UUID_RE.match(hint_stripped):
        found = next(
            (a for a in attachments if _get_att_id(a) == hint_stripped), None
        )
        if found is not None:
            return found

    hint_lower = hint_stripped.lower()
    hint_stem = Path(hint_lower).stem

    for att in attachments:
        if _get_att_name(att).lower() == hint_lower:
            return att
    for att in attachments:
        att_stem = Path(_get_att_name(att).lower()).stem
        if att_stem == hint_stem and hint_stem:
            return att
    for att in attachments:
        att_stem = Path(_get_att_name(att).lower()).stem
        if hint_stem and att_stem and (hint_stem in att_stem or att_stem in hint_stem):
            return att
    return None


def _build_attachment_meta(attachment: Any) -> dict[str, Any]:
    """Строит читаемый dict метаданных вложения."""
    name = _get_att_name(attachment) or "unknown"
    size_bytes: int = (
        (attachment.get("size") if isinstance(attachment, dict) else getattr(attachment, "size", None)) or 0
    )
    return {
        "название": name,
        "размер_кб": round(size_bytes / 1024, 2) if size_bytes else 0,
        "id": _get_att_id(attachment) or None,
    }


def _normalise_summary_type(value: Any) -> SummarizeType:
    """Резолвит алиасы к канонической SummarizeType."""
    if isinstance(value, SummarizeType):
        return value
    raw = str(value).strip().lower() if value else ""
    canonical = _SUMMARY_TYPE_ALIASES.get(raw, raw)
    try:
        return SummarizeType(canonical)
    except ValueError:
        logger.warning("Unknown summary_type '%s' — falling back to extractive", value)
        return SummarizeType.EXTRACTIVE


def _unwrap_json_envelope(text: str) -> str:
    """Разворачивает JSON-обёртку от doc_get_file_content."""
    clean = text.strip()
    if not (clean.startswith("{") and clean.endswith("}")):
        return clean
    try:
        data: dict[str, Any] = json.loads(clean)
        for key in ("content", "text", "document_info"):
            extracted = data.get(key)
            if extracted and isinstance(extracted, str) and len(extracted) > 10:
                return extracted.strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return clean


def _heuristic_recommendation(text: str) -> dict[str, str]:
    """Эвристически рекомендует формат суммаризации."""
    if not text:
        return {"recommended": "abstractive", "reason": "Текст пуст или очень короткий."}

    chars = len(text)
    lines = text.count("\n")
    numeric_groups = len(re.findall(r"\d+", text))

    if chars > 5_000 or numeric_groups > 20:
        return {
            "recommended": "thesis",
            "reason": (
                f"Объёмный документ ({chars} симв.) или много числовых данных "
                f"({numeric_groups} чисел) — тезисный план удобнее."
            ),
        }
    if lines < 5:
        return {
            "recommended": "abstractive",
            "reason": f"Компактный текст ({lines} строк) — краткого пересказа достаточно.",
        }
    return {
        "recommended": "extractive",
        "reason": "Структурированный текст с конкретными данными — список фактов будет полезнее.",
    }


def _truncate_for_llm(text: str, max_length: int = _MAX_SUMMARY_CHARS) -> str:
    """Обрезает текст сохраняя начало и конец."""
    if len(text) <= max_length:
        return text
    head = int(max_length * _HEAD_FRACTION)
    tail = max_length - head
    return (
        text[:head]
        + "\n\n[... часть содержимого пропущена ...]\n\n"
        + text[-tail:]
    )


async def _execute_summarization(text: str, summary_type: SummarizeType) -> dict[str, Any]:
    """Выполняет суммаризацию через LLM."""
    from edms_ai_assistant.llm_client import get_llm_response

    if len(text) < 50:
        return {
            "status": "success",
            "content": "Текст слишком короткий для глубокого анализа.",
            "meta": {"format_used": summary_type.value, "text_length": len(text), "was_truncated": False},
        }

    was_truncated = len(text) > _MAX_SUMMARY_CHARS
    processing_text = _truncate_for_llm(text)

    base_constraints = (
        "Опирайся ИСКЛЮЧИТЕЛЬНО на текст документа. Не выдумывай факты. "
        "Отвечай строго на русском языке. "
        "НЕ используй вводные фразы («В данном документе», «Вот ваш анализ»). "
        "Начинай ответ сразу с сути."
    )

    instructions = {
        SummarizeType.EXTRACTIVE: (
            "Выполни экстрактивную суммаризацию. "
            "Найди и выпиши: конкретные даты, суммы, имена лиц, сроки, метрики. "
            "Оформи СТРОГО нумерованным списком. Каждый пункт — одна конкретная сущность."
        ),
        SummarizeType.ABSTRACTIVE: (
            "Сформируй краткий пересказ сути документа своими словами (1–2 абзаца). "
            "Пиши так, чтобы руководитель за 10 секунд понял всю суть."
        ),
        SummarizeType.THESIS: (
            "Создай иерархический тезисный план. "
            "Используй нумерацию (1., 1.1., 1.2., 2.). "
            "Каждый тезис — одно ёмкое законченное предложение."
        ),
    }

    system = f"{base_constraints}\n\n{instructions[summary_type]}"
    prompt = f"<document>\n{processing_text}\n</document>\n\nВыдай результат:"

    try:
        summary = await get_llm_response(prompt, system=system)
        return {
            "status": "success",
            "content": summary.strip(),
            "meta": {
                "format_used": summary_type.value,
                "text_length": len(text),
                "was_truncated": was_truncated,
            },
        }
    except Exception as exc:
        logger.error("LLM summarization failed: %s", exc, exc_info=True)
        return {"status": "error", "message": f"Не удалось проанализировать документ: {exc}"}


# ── FastMCP tool регистрация ──────────────────────────────────────────────────


def register_content_tools(mcp: FastMCP) -> None:
    """Регистрирует инструменты работы с содержимым файлов."""

    @mcp.tool(
        description=(
            "Извлечь и проанализировать содержимое вложения документа EDMS. "
            "Поддерживает: PDF, DOCX, DOC, TXT, XLSX, XLS. "
            "Режимы: text (по умолчанию), tables (Excel/CSV), metadata, full. "
            "Если attachment_id не указан и вложений несколько — возвращает список для выбора."
        )
    )
    async def doc_get_file_content(
        document_id: str,
        token: str,
        attachment_id: str | None = None,
        analysis_mode: str = "text",
        summary_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Извлечь содержимое вложения из документа EDMS.

        Args:
            document_id: UUID документа.
            token: JWT-токен.
            attachment_id: UUID вложения или имя файла.
            analysis_mode: text | tables | metadata | full.
            summary_type: Тип суммаризации для передачи дальше.
        """
        mode = analysis_mode.strip().lower()
        if mode not in ("text", "tables", "metadata", "full"):
            mode = "text"

        try:
            async with DocumentClient() as doc_client:
                raw_data = await doc_client.get_document_metadata(token, document_id)

            if not raw_data:
                return {"status": "error", "message": "Документ не найден.", "summary_type": summary_type}

            attachments = raw_data.get("attachmentDocument") or []

            if not attachments:
                return {"status": "info", "message": "В документе нет вложений.", "summary_type": summary_type}

            # Резолвинг вложения
            if attachment_id:
                target = _resolve_attachment(attachments, attachment_id)
                if target is None:
                    available = [
                        {"id": _get_att_id(a), "name": _get_att_name(a)}
                        for a in attachments
                    ]
                    return {
                        "status": "requires_disambiguation",
                        "message": f"Вложение «{attachment_id}» не найдено. Выберите из списка:",
                        "available_attachments": available,
                    }
            elif len(attachments) == 1:
                target = attachments[0]
            else:
                available = [
                    {"id": _get_att_id(a), "name": _get_att_name(a) or "без имени"}
                    for a in attachments if _get_att_id(a)
                ]
                return {
                    "status": "requires_disambiguation",
                    "message": "В документе несколько вложений. Укажите какое открыть:",
                    "available_attachments": available,
                }

            resolved_id = _get_att_id(target)
            file_info = _build_attachment_meta(target)
            file_name = file_info["название"]
            suffix = Path(file_name).suffix.lower() if file_name else ".tmp"

            att_doc_id = str(
                (target.get("documentId") if isinstance(target, dict) else getattr(target, "documentId", None))
                or document_id
            )

            # Только метаданные
            if mode == "metadata":
                return {"status": "success", "mode": "metadata", "file_info": file_info, "summary_type": summary_type}

            # Скачивание
            async with EdmsAttachmentClient() as att_client:
                content_bytes = await att_client.get_attachment_content(token, att_doc_id, resolved_id)

            if not content_bytes:
                return {
                    "status": "error",
                    "message": f"Файл «{file_name}» пустой или недоступен.",
                    "file_info": file_info,
                    "summary_type": summary_type,
                }

            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content_bytes)
                    tmp_path = tmp.name

                if mode == "full":
                    structured = await FileProcessorService.extract_structured_data(tmp_path)
                    text_content = structured.get("text", "")
                    return {
                        "status": "success", "mode": "full", "file_info": file_info,
                        "content": text_content[:_MAX_TEXT_CHARS],
                        "is_truncated": len(text_content) > _MAX_TEXT_CHARS,
                        "total_chars": len(text_content),
                        "tables": structured.get("tables"),
                        "stats": structured.get("stats"),
                        "summary_type": summary_type,
                    }

                if mode == "tables":
                    if suffix not in _SUPPORTED_TABLE_EXTENSIONS:
                        return {
                            "status": "error",
                            "message": f"Режим 'tables' только для Excel/CSV. Текущий: '{suffix}'.",
                            "file_info": file_info, "summary_type": summary_type,
                        }
                    structured = await FileProcessorService.extract_structured_data(tmp_path)
                    tables = structured.get("tables", [])
                    return {
                        "status": "success", "mode": "tables", "file_info": file_info,
                        "tables": tables, "tables_count": len(tables) if tables else 0,
                        "summary_type": summary_type,
                    }

                # text mode
                if suffix not in _ALL_SUPPORTED:
                    return {
                        "status": "warning",
                        "message": f"Формат '{suffix}' не поддерживается. Поддерживаемые: {', '.join(sorted(_ALL_SUPPORTED))}",
                        "file_info": file_info, "summary_type": summary_type,
                    }

                text_content = await FileProcessorService.extract_text_async(tmp_path)
                if not text_content or text_content.startswith(("Ошибка:", "Формат файла")):
                    return {
                        "status": "error",
                        "message": f"Не удалось извлечь текст из «{file_name}»: {text_content}",
                        "file_info": file_info, "summary_type": summary_type,
                    }

                return {
                    "status": "success", "mode": "text", "file_info": file_info,
                    "content": text_content[:_MAX_TEXT_CHARS],
                    "is_truncated": len(text_content) > _MAX_TEXT_CHARS,
                    "total_chars": len(text_content),
                    "summary_type": summary_type,
                }

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        except Exception as exc:
            logger.error("doc_get_file_content failed: %s", exc, exc_info=True)
            return {"status": "error", "message": f"Ошибка: {exc}"}

    @mcp.tool(
        description=(
            "Прочитать локальный файл, загруженный пользователем. "
            "file_path берётся из контекста агента. "
            "Поддерживает: PDF, DOCX, DOC, TXT, RTF, ODT, MD, XLSX, XLS, CSV."
        )
    )
    async def read_local_file_content(
        file_path: str,
    ) -> dict[str, Any]:
        """
        Извлечь текст из локального файла.

        Args:
            file_path: Абсолютный путь к файлу (из контекста агента).
        """
        cleaned = file_path.strip()
        if not cleaned or cleaned.lower() in _PATH_PLACEHOLDERS or "<" in cleaned:
            return {
                "status": "error",
                "message": "Передан плейсхолдер вместо реального пути. Используй значение из контекста агента.",
            }

        path = Path(cleaned)
        if not path.exists():
            return {"status": "error", "message": f"Файл не найден: '{cleaned}'."}
        if not path.is_file():
            return {"status": "error", "message": f"Указанный путь не является файлом: '{cleaned}'."}

        suffix = path.suffix.lower()
        size_bytes = path.stat().st_size
        meta = {
            "имя_файла": path.name,
            "расширение": suffix,
            "размер_мб": round(size_bytes / (1024 * 1024), 2),
            "путь": str(path),
        }

        if suffix not in _ALL_SUPPORTED:
            return {
                "status": "error",
                "message": f"Формат '{suffix}' не поддерживается. Поддерживаемые: {', '.join(sorted(_ALL_SUPPORTED))}",
                "meta": meta,
            }

        text_content = await FileProcessorService.extract_text_async(str(path))
        if not text_content or text_content.startswith(("Ошибка:", "Формат файла")):
            return {
                "status": "error",
                "message": f"Не удалось извлечь текст из '{path.name}': {text_content[:300]}",
                "meta": meta,
            }

        total_chars = len(text_content)
        is_truncated = total_chars > _MAX_TEXT_CHARS
        return {
            "status": "success",
            "meta": meta,
            "content": text_content[:_MAX_TEXT_CHARS],
            "is_truncated": is_truncated,
            "total_chars": total_chars,
        }

    @mcp.tool(
        description=(
            "Сравнить локальный файл с вложением документа EDMS. "
            "Используй когда пользователь загрузил файл и просит сравнить его с вложением. "
            "НЕ используй для сравнения двух документов EDMS — для этого doc_compare_documents."
        )
    )
    async def doc_compare_attachment_with_local(
        token: str,
        document_id: str,
        local_file_path: str,
        attachment_id: str | None = None,
        original_filename: str | None = None,
    ) -> dict[str, Any]:
        """
        Сравнить локальный файл с вложением документа EDMS.

        Args:
            token: JWT-токен.
            document_id: UUID документа.
            local_file_path: Путь к локальному файлу.
            attachment_id: UUID вложения или его имя.
            original_filename: Оригинальное имя загруженного файла.
        """
        local_path = Path(local_file_path)
        display_name = (
            original_filename.strip()
            if original_filename and original_filename.strip()
            else local_path.name
        )

        if not local_path.exists():
            return {
                "status": "error",
                "message": f"Загруженный файл «{display_name}» не найден. Загрузите файл заново.",
            }

        try:
            async with DocumentClient() as doc_client:
                raw_data = await doc_client.get_document_metadata(token, document_id)

            if not raw_data:
                return {"status": "error", "message": "Документ не найден."}

            attachments = raw_data.get("attachmentDocument") or []
            if not attachments:
                return {"status": "error", "message": "В документе нет вложений для сравнения."}

        except Exception as exc:
            return {"status": "error", "message": f"Ошибка получения метаданных: {exc}"}

        # Резолвинг вложения
        if attachment_id:
            target = _resolve_attachment(attachments, attachment_id)
            if target is None:
                available = [
                    {"id": _get_att_id(a), "name": _get_att_name(a) or "без имени"}
                    for a in attachments if _get_att_id(a)
                ]
                return {
                    "status": "requires_disambiguation",
                    "message": f"Вложение «{attachment_id}» не найдено. Выберите из списка:",
                    "available_attachments": available,
                }
        else:
            target = _resolve_attachment(attachments, display_name)
            if target is None and display_name != local_path.name:
                target = _resolve_attachment(attachments, local_path.name)
            if target is None:
                available = [
                    {"id": _get_att_id(a), "name": _get_att_name(a) or "без имени"}
                    for a in attachments if _get_att_id(a)
                ]
                return {
                    "status": "requires_disambiguation",
                    "message": f"Не удалось определить вложение для «{display_name}». Выберите:",
                    "available_attachments": available,
                }

        resolved_id = _get_att_id(target)
        resolved_name = _get_att_name(target) or "attachment"
        resolved_suffix = Path(resolved_name).suffix.lower() or ".tmp"
        att_doc_id = str(
            (target.get("documentId") if isinstance(target, dict) else getattr(target, "documentId", None))
            or document_id
        )

        # Скачивание вложения
        try:
            async with EdmsAttachmentClient() as att_client:
                att_bytes = await att_client.get_attachment_content(token, att_doc_id, resolved_id)
        except Exception as exc:
            return {"status": "error", "message": f"Ошибка скачивания «{resolved_name}»: {exc}"}

        if not att_bytes:
            return {"status": "error", "message": f"Вложение «{resolved_name}» пустое."}

        # Извлечение текста
        local_text_raw = await FileProcessorService.extract_text_async(str(local_path))
        if not local_text_raw or local_text_raw.startswith(("Ошибка:", "Формат файла")):
            return {
                "status": "error",
                "message": f"Не удалось извлечь текст из «{display_name}»: {local_text_raw}",
            }

        att_text_raw = ""
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=resolved_suffix) as tmp:
                tmp.write(att_bytes)
                tmp_path = tmp.name
            att_text_raw = await FileProcessorService.extract_text_async(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # Нормализация и сравнение
        def _norm(text: str) -> str:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()

        local_text = _norm(local_text_raw[:_MAX_TEXT_CHARS])
        att_text = _norm(att_text_raw[:_MAX_TEXT_CHARS])
        are_identical = local_text == att_text
        similarity = round(
            difflib.SequenceMatcher(None, local_text, att_text, autojunk=False).ratio() * 100, 1
        )

        diff_result: list[dict[str, str]] = []
        if not are_identical:
            raw_diff = difflib.unified_diff(
                local_text.splitlines(keepends=True),
                att_text.splitlines(keepends=True),
                fromfile=f"Загруженный: {display_name}",
                tofile=f"Вложение EDMS: {resolved_name}",
                lineterm="",
                n=2,
            )
            for line in raw_diff:
                if line.startswith(("---", "+++", "@@")):
                    continue
                stripped = line[1:].strip()
                if not stripped:
                    continue
                if line.startswith("+"):
                    diff_result.append({"type": "added_in_attachment", "content": stripped})
                elif line.startswith("-"):
                    diff_result.append({"type": "removed_from_local", "content": stripped})
                if len(diff_result) >= _MAX_DIFF_LINES:
                    break

        return {
            "status": "success",
            "are_identical": are_identical,
            "similarity_percent": similarity,
            "local_file": display_name,
            "attachment_name": resolved_name,
            "local_stats": {"chars": len(local_text), "lines": local_text.count("\n") + 1},
            "attachment_stats": {"chars": len(att_text), "lines": att_text.count("\n") + 1},
            "differences": diff_result,
            "summary": (
                f"Файлы идентичны (схожесть: {similarity}%)."
                if are_identical
                else f"Файлы различаются (схожесть: {similarity}%). Различий: {len(diff_result)}."
            ),
        }

    @mcp.tool(
        description=(
            "Суммаризация текста документа. "
            "Если summary_type не указан — возвращает requires_choice с вариантами для выбора пользователем. "
            "Human-in-the-Loop: агент ОБЯЗАН показать выбор пользователю, не угадывать. "
            "Форматы: extractive (ключевые факты), abstractive (краткий пересказ), thesis (тезисный план)."
        )
    )
    async def doc_summarize_text(
        text: str,
        summary_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Суммаризация текста документа через LLM.

        Args:
            text: Текст документа (до 50000 символов или JSON-обёртка от doc_get_file_content).
            summary_type: extractive | abstractive | thesis | None (предложит выбор).
        """
        if not text or not text.strip():
            return {"status": "error", "message": "Текст не может быть пустым."}

        clean_text = _unwrap_json_envelope(text.strip())

        # Нет типа — предлагаем выбор пользователю
        if not summary_type:
            hint = _heuristic_recommendation(clean_text)
            return {
                "status": "requires_choice",
                "message": "Выберите формат анализа документа:",
                "options": [
                    {
                        "key": "extractive",
                        "label": "Ключевые факты",
                        "description": "Конкретные данные, даты, суммы нумерованным списком.",
                    },
                    {
                        "key": "abstractive",
                        "label": "Краткий пересказ",
                        "description": "Суть документа своими словами в 1–2 абзацах.",
                    },
                    {
                        "key": "thesis",
                        "label": "Тезисный план",
                        "description": "Структурированный план с разделами и подпунктами.",
                    },
                ],
                "hint": hint["recommended"],
                "hint_reason": hint["reason"],
            }

        try:
            normalised_type = _normalise_summary_type(summary_type)
            return await _execute_summarization(clean_text, normalised_type)
        except Exception as exc:
            logger.error("doc_summarize_text failed: %s", exc, exc_info=True)
            return {"status": "error", "message": f"Ошибка суммаризации: {exc}"}