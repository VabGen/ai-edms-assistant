# edms_ai_assistant/tools/document_versions.py
"""
EDMS AI Assistant — Document Versions Tool.

Получает все версии документа и АВТОМАТИЧЕСКИ сравнивает каждую пару
соседних версий (v1↔v2, v2↔v3, ...) без вопросов к пользователю.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)

# ── Поля метаданных для сравнения версий ─────────────────────────────────────
_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("regNumber", "Рег. номер"),
    ("regDate", "Дата регистрации"),
    ("status", "Статус"),
    ("shortSummary", "Краткое содержание"),
    ("correspondentName", "Корреспондент"),
    ("outRegNumber", "Исходящий номер"),
    ("outRegDate", "Исходящая дата"),
    ("author", "Автор"),
)


def _att_name(attachment: Any) -> str:
    """Извлекает имя вложения с fallback на originalName."""
    if isinstance(attachment, dict):
        return (
            attachment.get("name")
            or attachment.get("originalName")
            or attachment.get("fileName")
            or ""
        )
    return (
        getattr(attachment, "name", None)
        or getattr(attachment, "originalName", None)
        or ""
    )


def _compare_metadata(doc1: dict[str, Any], doc2: dict[str, Any]) -> dict[str, Any]:
    """
    Compares metadata fields of two document versions.

    Args:
        doc1: First document metadata dict.
        doc2: Second document metadata dict.

    Returns:
        Dict with changed fields: {field_label: {from: val1, to: val2}}.
    """
    changes: dict[str, Any] = {}
    for field_key, field_label in _METADATA_FIELDS:
        v1 = doc1.get(field_key)
        v2 = doc2.get(field_key)
        if v1 != v2:
            changes[field_label] = {
                "было": str(v1) if v1 is not None else "—",
                "стало": str(v2) if v2 is not None else "—",
            }
    return changes


def _compare_attachments(doc1: dict[str, Any], doc2: dict[str, Any]) -> dict[str, Any]:
    """
    Compares attachment lists of two document versions.

    Uses 'name' field (AttachmentDocumentDto) with fallback to 'originalName'.

    Args:
        doc1: First document metadata dict.
        doc2: Second document metadata dict.

    Returns:
        Dict with added, removed, common attachment names and counts.
    """
    att1: list[Any] = doc1.get("attachmentDocument") or []
    att2: list[Any] = doc2.get("attachmentDocument") or []

    names1: set[str] = {_att_name(a) for a in att1 if _att_name(a)}
    names2: set[str] = {_att_name(a) for a in att2 if _att_name(a)}

    return {
        "добавлены_в_новой": sorted(names2 - names1) or [],
        "удалены_из_старой": sorted(names1 - names2) or [],
        "присутствуют_в_обеих": sorted(names1 & names2) or [],
        "кол-во_вложений_старая": len(att1),
        "кол-во_вложений_новая": len(att2),
    }


class DocumentVersionsInput(BaseModel):
    """Схема входных данных для получения и сравнения всех версий документа."""

    document_id: str = Field(..., description="UUID документа")
    token: str = Field(..., description="Токен авторизации пользователя (JWT)")


@tool("doc_get_versions", args_schema=DocumentVersionsInput)
async def doc_get_versions(document_id: str, token: str) -> dict[str, Any]:
    """Retrieve all document versions and compare each consecutive pair automatically.

    BEHAVIOUR:
    - Fetches ALL versions of the document (N versions).
    - Compares every consecutive pair: v1↔v2, v2↔v3, ..., v(N-1)↔vN.
    - Returns full aggregated comparison results for ALL pairs.
    - Agent MUST NOT ask the user which versions to compare — all pairs are compared.
    - If only 1 version exists — informs user, no comparison possible.
    - If exactly 2 versions — compares them (1 pair).
    - If N > 2 versions — compares all N-1 consecutive pairs and presents full history.

    Args:
        document_id: EDMS document UUID.
        token: JWT bearer token.

    Returns:
        Dict with all versions metadata and full pair-wise comparison results.
    """
    try:
        async with DocumentClient() as client:
            versions = await client.get_document_versions(token, document_id)

            if not versions:
                return {
                    "status": "success",
                    "total_versions": 0,
                    "message": "У документа только одна версия — сравнивать не с чем.",
                }

            sorted_versions = sorted(versions, key=lambda v: v.get("version", 0))
            total = len(sorted_versions)

            # ── Сбор метаданных всех версий ─────────────────────────────────
            versions_info: list[dict[str, Any]] = []
            version_ids: dict[str, str] = {}  # "1" → doc_uuid

            for v in sorted_versions:
                vnum = v.get("version") or (len(versions_info) + 1)
                doc_id = str(v.get("documentId") or "")
                if doc_id:
                    version_ids[str(vnum)] = doc_id
                versions_info.append(
                    {
                        "version_number": vnum,
                        "created_date": str(v.get("createDate") or ""),
                    }
                )

            if total == 1:
                return {
                    "status": "success",
                    "total_versions": 1,
                    "versions": versions_info,
                    "message": "Документ имеет только одну версию — история изменений недоступна.",
                }

            # ── Последовательное сравнение всех соседних пар ────────────────
            # Для N версий выполняем N-1 сравнений: (1,2), (2,3), ..., (N-1,N).
            # Каждое сравнение — отдельный API-запрос за метаданными двух версий.
            comparisons: list[dict[str, Any]] = []
            errors: list[str] = []

            version_nums = sorted(version_ids.keys(), key=lambda x: int(x))

            for i in range(len(version_nums) - 1):
                from_vnum = version_nums[i]
                to_vnum = version_nums[i + 1]
                from_id = version_ids[from_vnum]
                to_id = version_ids[to_vnum]

                try:
                    doc_from = await client.get_document_metadata(token, from_id)
                    doc_to = await client.get_document_metadata(token, to_id)

                    if not doc_from or not doc_to:
                        errors.append(
                            f"Версия {from_vnum} или {to_vnum}: метаданные недоступны"
                        )
                        continue

                    meta_diff = _compare_metadata(doc_from, doc_to)
                    att_diff = _compare_attachments(doc_from, doc_to)

                    comparisons.append(
                        {
                            "pair": f"Версия {from_vnum} → Версия {to_vnum}",
                            "from_version": int(from_vnum),
                            "to_version": int(to_vnum),
                            "metadata_changes": meta_diff,
                            "metadata_changed": bool(meta_diff),
                            "attachment_changes": att_diff,
                            "attachments_changed": (
                                bool(att_diff.get("добавлены_в_новой"))
                                or bool(att_diff.get("удалены_из_старой"))
                            ),
                        }
                    )

                    logger.debug(
                        "Compared v%s↔v%s: meta_changes=%d att_added=%d att_removed=%d",
                        from_vnum,
                        to_vnum,
                        len(meta_diff),
                        len(att_diff.get("добавлены_в_новой", [])),
                        len(att_diff.get("удалены_из_старой", [])),
                    )

                except Exception as pair_exc:
                    err = f"Ошибка сравнения версий {from_vnum}↔{to_vnum}: {pair_exc}"
                    logger.error(err)
                    errors.append(err)

            # ── Агрегированный вывод ─────────────────────────────────────────
            has_any_changes = any(
                c.get("metadata_changed") or c.get("attachments_changed")
                for c in comparisons
            )

            logger.info(
                "doc_get_versions completed: total=%d pairs=%d changes=%s errors=%d",
                total,
                len(comparisons),
                has_any_changes,
                len(errors),
            )

            return {
                "status": "success",
                "total_versions": total,
                "versions": versions_info,
                "version_ids": version_ids,
                "comparisons": comparisons,
                "has_any_changes": has_any_changes,
                "errors": errors if errors else None,
                "message": (
                    f"Документ имеет {total} версии. "
                    f"Выполнено {len(comparisons)} сравнений: "
                    + ", ".join(c["pair"] for c in comparisons)
                    + (
                        ". Изменения обнаружены."
                        if has_any_changes
                        else ". Версии идентичны."
                    )
                ),
                "comparison_complete": True,
                "instruction": (
                    "Все сравнения уже выполнены. Сформулируй полный ответ пользователю "
                    "на основе поля 'comparisons'. НЕ вызывай doc_compare — данные уже есть."
                ),
            }

    except Exception as e:
        logger.error("[DOC-VERSIONS-TOOL] Error: %s", e, exc_info=True)
        return {"status": "error", "message": f"Ошибка получения версий: {e!s}"}
