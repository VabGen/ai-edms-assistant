# mcp-server/tools/content_tools.py
"""
Инструменты работы с содержимым файлов и вложений.
Перенесены из edms_ai_assistant/tools/attachment.py, local_file_tool.py,
file_compare_tool.py, summarization.py.
"""
from __future__ import annotations

import difflib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..clients.attachment_client import EdmsAttachmentClient
from ..clients.document_client import DocumentClient
from ..llm import get_llm_response
from ..services.file_processor import FileProcessorService
from ..utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 15_000
_MAX_DIFF_LINES = 60
_MAX_SUMMARY_CHARS = 12_000

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt", ".md",
    ".xlsx", ".xls", ".csv",
})

_PATH_PLACEHOLDERS: frozenset[str] = frozenset({
    "", "local_file", "local_file_path", "/path/to/file",
    "none", "null", "<local_file_path>", "<path>",
})


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
        found = next((a for a in attachments if _get_att_id(a) == hint_stripped), None)
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
    ) -> dict[str, Any]:
        """
        Извлечь содержимое вложения из документа EDMS.

        Args:
            document_id: UUID документа.
            token: JWT-токен.
            attachment_id: UUID вложения или имя файла.
            analysis_mode: text | tables | metadata | full.
        """
        mode = analysis_mode.strip().lower()
        if mode not in ("text", "tables", "metadata", "full"):
            mode = "text"

        try:
            async with DocumentClient() as doc_client:
                raw_data = await doc_client.get_document_metadata(token, document_id)

            if not raw_data:
                return {"status": "error", "message": "Документ не найден."}

            attachments = raw_data.get("attachmentDocument") or []

            if not attachments:
                return {"status": "info", "message": "В документе нет вложений."}

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
            file_name = _get_att_name(target) or "attachment"
            suffix = Path(file_name).suffix.lower() or ".tmp"

            att_doc_id = str(
                (target.get("documentId") if isinstance(target, dict)
                 else getattr(target, "documentId", None)) or document_id
            )

            # Только метаданные
            if mode == "metadata":
                size_bytes = (
                    target.get("size") if isinstance(target, dict)
                    else getattr(target, "size", 0)
                ) or 0
                return {
                    "status": "success",
                    "mode": "metadata",
                    "file_info": {
                        "название": file_name,
                        "размер_кб": round(size_bytes / 1024, 2),
                        "id": resolved_id,
                    },
                }

            # Скачивание
            async with EdmsAttachmentClient() as att_client:
                content_bytes = await att_client.get_attachment_content(
                    token, att_doc_id, resolved_id
                )

            if not content_bytes:
                return {"status": "error", "message": f"Файл «{file_name}» пустой или недоступен."}

            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content_bytes)
                    tmp_path = tmp.name

                if mode == "full":
                    structured = await FileProcessorService.extract_structured_data(tmp_path)
                    text_content = structured.get("text", "")
                    return {
                        "status": "success",
                        "mode": "full",
                        "file_name": file_name,
                        "content": text_content[:_MAX_TEXT_CHARS],
                        "is_truncated": len(text_content) > _MAX_TEXT_CHARS,
                        "total_chars": len(text_content),
                        "tables": structured.get("tables"),
                        "stats": structured.get("stats"),
                    }

                if mode == "tables":
                    if suffix not in {".xlsx", ".xls", ".csv"}:
                        return {
                            "status": "error",
                            "message": f"Режим 'tables' только для Excel/CSV. Текущий: '{suffix}'.",
                        }
                    structured = await FileProcessorService.extract_structured_data(tmp_path)
                    tables = structured.get("tables", [])
                    return {
                        "status": "success",
                        "mode": "tables",
                        "file_name": file_name,
                        "tables": tables,
                        "tables_count": len(tables) if tables else 0,
                    }

                # text mode
                if suffix not in _SUPPORTED_EXTENSIONS:
                    return {
                        "status": "warning",
                        "message": f"Формат '{suffix}' не поддерживается. Поддерживаемые: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}",
                        "file_name": file_name,
                    }

                text_content = await FileProcessorService.extract_text_async(tmp_path)
                if not text_content or text_content.startswith(("Ошибка:", "Формат файла")):
                    return {
                        "status": "error",
                        "message": f"Не удалось извлечь текст из «{file_name}»: {text_content}",
                    }

                return {
                    "status": "success",
                    "mode": "text",
                    "file_name": file_name,
                    "content": text_content[:_MAX_TEXT_CHARS],
                    "is_truncated": len(text_content) > _MAX_TEXT_CHARS,
                    "total_chars": len(text_content),
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
            "file_path берётся из контекста агента автоматически. "
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

        if suffix not in _SUPPORTED_EXTENSIONS:
            return {
                "status": "error",
                "message": f"Формат '{suffix}' не поддерживается. Поддерживаемые: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}",
                "meta": meta,
            }

        text_content = await FileProcessorService.extract_text_async(str(path))
        if not text_content or text_content.startswith(("Ошибка:", "Формат файла")):
            return {
                "status": "error",
                "message": f"Не удалось извлечь текст: {text_content[:300]}",
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
            local_file_path: Путь к локальному файлу (из контекста агента).
            attachment_id: UUID вложения или его имя для сравнения.
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
                    "message": f"Не удалось автоматически определить вложение для «{display_name}». Выберите:",
                    "available_attachments": available,
                }

        resolved_id = _get_att_id(target)
        resolved_name = _get_att_name(target) or "attachment"
        resolved_suffix = Path(resolved_name).suffix.lower() or ".tmp"
        att_doc_id = str(
            (target.get("documentId") if isinstance(target, dict)
             else getattr(target, "documentId", None)) or document_id
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
            return {"status": "error", "message": f"Не удалось извлечь текст из «{display_name}»: {local_text_raw}"}

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
            text: Текст документа (до 50000 символов).
            summary_type: extractive | abstractive | thesis | None (предложит выбор).
        """
        if not text or not text.strip():
            return {"status": "error", "message": "Текст не может быть пустым."}

        clean_text = text.strip()

        # Разворачиваем JSON-обёртку если пришла от doc_get_file_content
        if clean_text.startswith("{") and clean_text.endswith("}"):
            try:
                import json
                data = json.loads(clean_text)
                for key in ("content", "text", "document_info"):
                    extracted = data.get(key)
                    if extracted and isinstance(extracted, str) and len(extracted) > 10:
                        clean_text = extracted.strip()
                        break
            except Exception:
                pass

        # Нет типа — предлагаем выбор
        if not summary_type:
            chars = len(clean_text)
            numeric_groups = len(re.findall(r"\d+", clean_text))
            if chars > 5000 or numeric_groups > 20:
                hint = "thesis"
                reason = f"Объёмный документ ({chars} симв.) — тезисный план удобнее."
            elif clean_text.count("\n") < 5:
                hint = "abstractive"
                reason = "Компактный текст — краткого пересказа достаточно."
            else:
                hint = "extractive"
                reason = "Структурированный текст — список фактов будет полезнее."

            return {
                "status": "requires_choice",
                "message": "Выберите формат анализа документа:",
                "options": [
                    {"key": "extractive", "label": "Ключевые факты", "description": "Конкретные данные, даты, суммы нумерованным списком."},
                    {"key": "abstractive", "label": "Краткий пересказ", "description": "Суть документа своими словами в 1–2 абзацах."},
                    {"key": "thesis", "label": "Тезисный план", "description": "Структурированный план с разделами и подпунктами."},
                ],
                "hint": hint,
                "hint_reason": reason,
            }

        # Нормализация типа
        type_aliases = {
            "факты": "extractive", "ключевые факты": "extractive",
            "пересказ": "abstractive", "краткий пересказ": "abstractive",
            "тезисы": "thesis", "тезисный план": "thesis",
        }
        st = type_aliases.get(summary_type.lower().strip(), summary_type.lower().strip())
        if st not in ("extractive", "abstractive", "thesis"):
            st = "extractive"

        if len(clean_text) < 50:
            return {
                "status": "success",
                "content": "Текст слишком короткий для анализа.",
                "meta": {"format_used": st, "text_length": len(clean_text), "was_truncated": False},
            }

        # Усечение
        max_len = _MAX_SUMMARY_CHARS
        was_truncated = len(clean_text) > max_len
        head = int(max_len * 0.67)
        tail = max_len - head
        if was_truncated:
            processing_text = (
                clean_text[:head] + "\n\n[... часть содержимого пропущена ...]\n\n" + clean_text[-tail:]
            )
        else:
            processing_text = clean_text

        instructions = {
            "extractive": (
                "Выдели ключевые факты: конкретные даты, суммы, имена, сроки и обязательства. "
                "Оформи СТРОГО нумерованным списком. Каждый пункт — одна конкретная мысль."
            ),
            "abstractive": (
                "Напиши связный краткий пересказ сути документа своими словами (1–2 абзаца). "
                "Пиши как для руководителя, который видит документ впервые."
            ),
            "thesis": (
                "Сформируй структурированный тезисный план. "
                "Используй нумерацию разделов (1., 1.1., 1.2.). Каждый тезис — одно ёмкое предложение."
            ),
        }

        system = (
            "Ты — аналитик системы электронного документооборота (СЭД). "
            f"Задача: {instructions[st]} "
            "Отвечай строго на русском языке. "
            "НЕ начинай со слов «В данном документе», «Данный текст», «Документ» — "
            "сразу переходи к содержанию."
        )

        try:
            summary = await get_llm_response(
                f"ТЕКСТ ДОКУМЕНТА:\n{processing_text}\n\nРЕЗУЛЬТАТ АНАЛИЗА:",
                system=system,
            )
            return {
                "status": "success",
                "content": summary.strip(),
                "meta": {
                    "format_used": st,
                    "text_length": len(clean_text),
                    "was_truncated": was_truncated,
                },
            }
        except Exception as exc:
            logger.error("doc_summarize_text LLM failed: %s", exc, exc_info=True)
            return {"status": "error", "message": f"Ошибка суммаризации: {exc}"}