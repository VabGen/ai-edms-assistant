# edms_ai_assistant/mcp_server/tools/document_tools.py
"""Инструменты работы с документами для FastMCP."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from edms_ai_assistant.config import settings
from edms_ai_assistant.mcp_server.clients.document_client import (
    FULL_DOC_INCLUDES,
    DocumentClient,
)
from edms_ai_assistant.mcp_server.services.document_enricher import DocumentEnricher

logger = logging.getLogger(__name__)

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

_MAX_DOC_CONTENT_CHARS = 8000


def _clean(d: Any) -> Any:
    """Рекурсивно удаляет None, пустые списки и словари."""
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items() if v not in (None, [], {}, "")}
    if isinstance(d, list):
        return [_clean(i) for i in d if i not in (None, [], {}, "")]
    return d


def _truncate_doc_for_llm(doc: dict[str, Any]) -> dict[str, Any]:
    """Обрезает документ до разумного размера для передачи в LLM.

    Удаляет потенциально огромные поля (вложения с base64, длинные тексты),
    оставляет только ключевые метаданные.
    """
    import json

    if len(json.dumps(doc, ensure_ascii=False, default=str)) <= _MAX_DOC_CONTENT_CHARS:
        return doc

    key_fields = [
        "id",
        "regNumber",
        "reservedRegNumber",
        "regDate",
        "status",
        "prevStatus",
        "shortSummary",
        "summary",
        "note",
        "docCategoryConstant",
        "correspondentName",
        "outRegNumber",
        "outRegDate",
        "author",
        "initiator",
        "responsibleExecutor",
        "whoSigned",
        "controlFlag",
        "daysExecution",
        "createDate",
        "profileName",
        "documentType",
        "registrationJournal",
        "version",
        "pages",
        "countTask",
        "completedTaskCount",
        "introductionCount",
        "contractNumber",
        "contractDate",
        "contractSum",
    ]
    slim = {k: doc[k] for k in key_fields if k in doc and doc[k] not in (None, [], {})}

    attachments = doc.get("attachmentDocument") or []
    if attachments:
        slim["attachmentDocument"] = [
            {
                "id": str(a.get("id", "")),
                "name": a.get("name", ""),
                "size": a.get("size"),
            }
            for a in attachments[:10]
        ]

    # Добавляем краткий список поручений
    tasks = doc.get("taskList") or []
    if tasks:
        slim["taskList"] = [
            {
                "taskNumber": t.get("taskNumber"),
                "taskText": (t.get("taskText") or "")[:200],
                "taskStatus": t.get("taskStatus"),
                "planedDateEnd": t.get("planedDateEnd"),
            }
            for t in tasks[:5]
        ]

    logger.debug(
        "Document truncated for LLM: %d → %d chars",
        len(json.dumps(doc, ensure_ascii=False, default=str)),
        len(json.dumps(slim, ensure_ascii=False, default=str)),
    )
    return slim


def _att_name(attachment: Any) -> str:
    if isinstance(attachment, dict):
        return (
            attachment.get("name")
            or attachment.get("originalName")
            or attachment.get("fileName")
            or ""
        )
    return ""


def _compare_metadata(doc1: dict, doc2: dict) -> dict[str, Any]:
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


def _compare_attachments(doc1: dict, doc2: dict) -> dict[str, Any]:
    att1: list = doc1.get("attachmentDocument") or []
    att2: list = doc2.get("attachmentDocument") or []
    names1 = {_att_name(a) for a in att1 if _att_name(a)}
    names2 = {_att_name(a) for a in att2 if _att_name(a)}
    return {
        "добавлены_в_новой": sorted(names2 - names1) or [],
        "удалены_из_старой": sorted(names1 - names2) or [],
        "присутствуют_в_обеих": sorted(names1 & names2) or [],
        "кол-во_вложений_старая": len(att1),
        "кол-во_вложений_новая": len(att2),
    }


def _format_author(author: dict[str, Any] | None) -> str:
    if not author:
        return "—"
    parts = [
        author.get("lastName", ""),
        author.get("firstName", ""),
        author.get("middleName", ""),
    ]
    return " ".join(p for p in parts if p).strip() or "—"


def register_document_tools(mcp: FastMCP) -> None:
    """Регистрирует все инструменты работы с документами."""

    @mcp.tool(
        description=(
            "Получить полный анализ документа СЭД: метаданные, участников, "
            "вложения, поручения, контроль. Используй для детальной информации "
            "о конкретном документе по его UUID."
        )
    )
    async def doc_get_details(
        document_id: str,
        token: str,
    ) -> dict[str, Any]:
        """
        Анализирует документ СЭД и все его вложенные сущности.

        Args:
            document_id: UUID документа.
            token: JWT-токен авторизации.
        """
        try:
            # base_url берётся из settings — не нужен как аргумент
            enricher = DocumentEnricher(base_url=settings.EDMS_BASE_URL)
            async with DocumentClient() as client:
                raw = await client.get_document_metadata(
                    token=token,
                    document_id=document_id,
                    includes=FULL_DOC_INCLUDES,
                )

            if not raw:
                return {
                    "status": "error",
                    "error": f"Документ {document_id} не найден.",
                }

            enriched = await enricher.enrich(raw, token=token)
            cleaned = _clean(enriched)

            # Обрезаем для LLM чтобы не получить 500 от Ollama
            trimmed = _truncate_doc_for_llm(cleaned)

            return {"status": "success", "document": trimmed}

        except Exception as exc:
            logger.error(
                "doc_get_details failed for %s: %s",
                document_id,
                exc,
                exc_info=True,
            )
            return {
                "status": "error",
                "error": f"Ошибка обработки документа: {exc}",
            }

    @mcp.tool(
        description=(
            "Получить все версии документа и автоматически сравнить соседние версии. "
            "Возвращает полную историю изменений метаданных и вложений."
        )
    )
    async def doc_get_versions(
        document_id: str,
        token: str,
    ) -> dict[str, Any]:
        """
        Получить версии документа и сравнить все соседние пары.

        Args:
            document_id: UUID документа.
            token: JWT-токен авторизации.
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
            versions_info: list[dict[str, Any]] = []
            version_ids: dict[str, str] = {}

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
                    "message": "Документ имеет только одну версию.",
                }

            comparisons: list[dict[str, Any]] = []
            errors: list[str] = []
            version_nums = sorted(version_ids.keys(), key=lambda x: int(x))

            async with DocumentClient() as client:
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
                    except Exception as pair_exc:
                        errors.append(
                            f"Ошибка сравнения {from_vnum}↔{to_vnum}: {pair_exc}"
                        )

            has_changes = any(
                c.get("metadata_changed") or c.get("attachments_changed")
                for c in comparisons
            )
            return {
                "status": "success",
                "total_versions": total,
                "versions": versions_info,
                "comparisons": comparisons,
                "has_any_changes": has_changes,
                "errors": errors if errors else None,
                "message": (
                    f"Документ имеет {total} версии. "
                    f"Выполнено {len(comparisons)} сравнений. "
                    + ("Изменения обнаружены." if has_changes else "Версии идентичны.")
                ),
            }

        except Exception as e:
            logger.error("doc_get_versions failed: %s", e, exc_info=True)
            return {"status": "error", "message": f"Ошибка получения версий: {e!s}"}

    @mcp.tool(
        description=(
            "Сравнить два документа или версии документа по метаданным и вложениям."
        )
    )
    async def doc_compare_documents(
        document_id_1: str,
        document_id_2: str,
        token: str,
        comparison_focus: str = "all",
    ) -> dict[str, Any]:
        """
        Сравнить два документа.

        Args:
            document_id_1: UUID первого документа.
            document_id_2: UUID второго документа.
            token: JWT-токен.
            comparison_focus: metadata | attachments | all.
        """
        try:
            async with DocumentClient() as client:
                doc1 = await client.get_document_metadata(token, document_id_1)
                doc2 = await client.get_document_metadata(token, document_id_2)

            if not doc1 or not doc2:
                return {
                    "status": "error",
                    "message": "Один или оба документа не найдены.",
                }

            differences: dict[str, Any] = {}
            if comparison_focus in ("metadata", "all"):
                differences["metadata"] = _compare_metadata(doc1, doc2)
            if comparison_focus in ("attachments", "all"):
                differences["attachments"] = _compare_attachments(doc1, doc2)

            return {
                "status": "success",
                "document_1_id": document_id_1,
                "document_2_id": document_id_2,
                "differences": differences,
                "has_differences": any(bool(v) for v in differences.values()),
            }

        except Exception as e:
            logger.error("doc_compare_documents failed: %s", e, exc_info=True)
            return {"status": "error", "message": f"Ошибка сравнения: {e!s}"}

    @mcp.tool(
        description=(
            "Поиск документов в EDMS по фильтрам: краткое содержание, "
            "рег. номер, категория, статус, даты, автор. Возвращает до 10 документов."
        )
    )
    async def doc_search_tool(
        token: str,
        short_summary: str | None = None,
        reg_number: str | None = None,
        doc_category: str | None = None,
        status: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        author_last_name: str | None = None,
        author_current_user: bool | None = None,
        process_executor_current_user: bool | None = None,
        task_executor_current_user: bool | None = None,
        introduction_current_user: bool | None = None,
    ) -> dict[str, Any]:
        """
        Поиск документов по широкому набору фильтров.

        Args:
            token: JWT-токен.
            short_summary: Поиск по краткому содержанию.
            reg_number: Регистрационный номер.
            doc_category: INCOMING|OUTGOING|INTERN|APPEAL|CONTRACT|MEETING.
            status: Список статусов.
            date_from: Дата регистрации от (YYYY-MM-DD).
            date_to: Дата регистрации до (YYYY-MM-DD).
            author_last_name: Фамилия автора.
            author_current_user: True — только мои документы.
        """
        from datetime import datetime as dt

        def to_iso_start(date_str: str) -> str:
            d = dt.fromisoformat(date_str)
            return d.replace(hour=0, minute=0, second=0).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"
            )

        def to_iso_end(date_str: str) -> str:
            d = dt.fromisoformat(date_str)
            return d.replace(hour=23, minute=59, second=59).strftime(
                "%Y-%m-%dT%H:%M:%S.999Z"
            )

        doc_filter: dict[str, Any] = {}
        if short_summary:
            doc_filter["shortSummary"] = short_summary
        if reg_number:
            doc_filter["regNumber"] = reg_number
        if doc_category:
            doc_filter["categoryConstants"] = [doc_category.strip().upper()]
        if status:
            doc_filter["status"] = status
        if date_from:
            doc_filter["dateRegStart"] = to_iso_start(date_from)
        if date_to:
            doc_filter["dateRegEnd"] = to_iso_end(date_to)
        if author_last_name:
            doc_filter["authorLastName"] = author_last_name
        if author_current_user is True:
            doc_filter["authorCurrentUser"] = True
        if process_executor_current_user is True:
            doc_filter["processExecutorCurrentUser"] = True
        if task_executor_current_user is True:
            doc_filter["taskExecutorCurrentUser"] = True
        if introduction_current_user is True:
            doc_filter["introductionCurrentUser"] = True

        if not doc_filter:
            return {
                "status": "error",
                "message": "Укажите хотя бы один параметр поиска.",
            }

        try:
            async with DocumentClient() as client:
                raw_docs = await client.search_documents(
                    token=token,
                    doc_filter=doc_filter,
                    pageable={"page": 0, "size": 10},
                )

            if not raw_docs:
                return {
                    "status": "success",
                    "message": "Документы не найдены.",
                    "documents": [],
                    "total": 0,
                }

            documents = [
                {
                    "id": str(d.get("id", "")),
                    "reg_number": (
                        d.get("regNumber") or d.get("reservedRegNumber") or "—"
                    ),
                    "reg_date": str(d.get("regDate", ""))[:10],
                    "category": str(d.get("docCategoryConstant", "—")),
                    "short_summary": (d.get("shortSummary") or "")[:200],
                    "author": _format_author(d.get("author")),
                    "status": str(d.get("status", "—")),
                }
                for d in raw_docs[:10]
            ]
            return {
                "status": "success",
                "total": len(documents),
                "documents": documents,
            }

        except Exception as exc:
            logger.error("doc_search_tool failed: %s", exc, exc_info=True)
            return {"status": "error", "message": f"Ошибка поиска: {exc}"}
