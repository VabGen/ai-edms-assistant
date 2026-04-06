# mcp-server/services/document_enricher.py
"""
Document Enricher — дозапросы вложенных объектов по UUID.
Перенесён из edms_ai_assistant/services/document_enricher.py.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..clients.base_client import EdmsHttpClient

logger = logging.getLogger(__name__)

RawDoc = dict[str, Any]


class _RefClient(EdmsHttpClient):
    """Лёгкий HTTP-клиент для одиночных справочных запросов."""

    async def get_correspondent(self, token: str, cid: str) -> dict[str, Any] | None:
        r = await self._make_request("GET", f"api/correspondent/{cid}", token=token)
        return r if isinstance(r, dict) and r else None

    async def get_reg_journal(self, token: str, jid: str) -> dict[str, Any] | None:
        r = await self._make_request("GET", f"api/reg-journal/{jid}", token=token)
        return r if isinstance(r, dict) and r else None

    async def get_document_type(self, token: str, dtid: str) -> dict[str, Any] | None:
        r = await self._make_request("GET", f"api/document-type/{dtid}", token=token)
        return r if isinstance(r, dict) and r else None

    async def get_currency(self, token: str, cid: str) -> dict[str, Any] | None:
        r = await self._make_request("GET", f"api/currency/{cid}", token=token)
        return r if isinstance(r, dict) and r else None

    async def get_control_type(self, token: str, ctid: str) -> dict[str, Any] | None:
        r = await self._make_request("GET", f"api/control-type/{ctid}", token=token)
        return r if isinstance(r, dict) and r else None

    async def get_introduction_list(self, token: str, doc_id: str) -> list[dict[str, Any]]:
        r = await self._make_request("GET", f"api/introduction/document/{doc_id}", token=token)
        if isinstance(r, list):
            return r
        if isinstance(r, dict):
            c = r.get("content") or r.get("items")
            if isinstance(c, list):
                return c
        return []

    async def get_document_appeal(self, token: str, doc_id: str) -> dict[str, Any] | None:
        r = await self._make_request("GET", f"api/document-appeal/document/{doc_id}", token=token)
        return r if isinstance(r, dict) and r else None

    async def get_document_recipients(self, token: str, doc_id: str) -> list[dict[str, Any]]:
        r = await self._make_request("GET", f"api/document/{doc_id}/recipient", token=token)
        if isinstance(r, list):
            return r
        if isinstance(r, dict):
            c = r.get("content") or r.get("items")
            if isinstance(c, list):
                return c
        return []

    async def get_document_tasks(self, token: str, doc_id: str) -> list[dict[str, Any]]:
        r = await self._make_request(
            "GET", f"api/document/{doc_id}/task",
            token=token, params={"fetchExecutors": "true"},
        )
        if isinstance(r, list):
            return r
        if isinstance(r, dict):
            c = r.get("content") or r.get("items") or r.get("data")
            if isinstance(c, list):
                return c
        return []

    async def get_contract_responsible(self, token: str, doc_id: str) -> list[dict[str, Any]]:
        r = await self._make_request("GET", f"api/document/{doc_id}/responsible", token=token)
        if isinstance(r, list):
            return r
        if isinstance(r, dict):
            c = r.get("content") or r.get("items") or r.get("data")
            if isinstance(c, list):
                return c
        return []


def _has_nested(obj: dict[str, Any], key: str) -> bool:
    """Проверяет наличие непустого вложенного объекта."""
    val = obj.get(key)
    if isinstance(val, dict):
        return bool(val)
    if isinstance(val, list):
        return bool(val)
    return False


class DocumentEnricher:
    """
    Обогащает сырой DocumentDto dict данными из дополнительных API-эндпоинтов.

    Все запросы выполняются параллельно через asyncio.gather.
    При ошибке любого запроса поле остаётся как есть (UUID-строка).
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    async def enrich(self, doc: RawDoc, token: str) -> RawDoc:
        """
        Обогатить сырой dict документа вложенными объектами.

        Args:
            doc: Сырой DocumentDto dict.
            token: JWT-токен.

        Returns:
            Тот же dict с заполненными вложенными объектами.
        """
        if not doc:
            return doc

        doc_id: str | None = str(doc.get("id", "")) or None
        category: str = str(doc.get("docCategoryConstant", "") or "")

        tasks: dict[str, Any] = {}

        async with _RefClient(base_url=self._base_url) as client:

            if doc.get("correspondentId") and not _has_nested(doc, "correspondent"):
                tasks["correspondent"] = client.get_correspondent(token, str(doc["correspondentId"]))

            if doc.get("journalId") and not _has_nested(doc, "registrationJournal"):
                tasks["registrationJournal"] = client.get_reg_journal(token, str(doc["journalId"]))

            if doc.get("documentTypeId") and not _has_nested(doc, "documentType"):
                tasks["documentType"] = client.get_document_type(token, str(doc["documentTypeId"]))

            if doc.get("currencyId") and not _has_nested(doc, "currency"):
                tasks["currency"] = client.get_currency(token, str(doc["currencyId"]))

            control = doc.get("control") or {}
            if (
                isinstance(control, dict)
                and control.get("controlTypeId")
                and not _has_nested(control, "controlType")
            ):
                tasks["_controlType"] = client.get_control_type(token, str(control["controlTypeId"]))

            if doc_id and not doc.get("introduction"):
                tasks["introduction"] = client.get_introduction_list(token, doc_id)

            if doc_id and category == "APPEAL" and not doc.get("documentAppeal"):
                tasks["documentAppeal"] = client.get_document_appeal(token, doc_id)

            if doc_id and not doc.get("recipientList"):
                tasks["recipientList"] = client.get_document_recipients(token, doc_id)

            if doc_id and not doc.get("taskList"):
                tasks["taskList"] = client.get_document_tasks(token, doc_id)

            if doc_id and category == "CONTRACT" and not doc.get("contractResponsible"):
                tasks["contractResponsible"] = client.get_contract_responsible(token, doc_id)

            # Корреспонденты адресатов (batch)
            recipient_list: list[dict[str, Any]] = doc.get("recipientList") or []
            recipient_corr_tasks: list[tuple[int, Any]] = []
            for idx, recipient in enumerate(recipient_list):
                if (
                    isinstance(recipient, dict)
                    and recipient.get("correspondentId")
                    and not _has_nested(recipient, "correspondent")
                ):
                    recipient_corr_tasks.append(
                        (idx, client.get_correspondent(token, str(recipient["correspondentId"])))
                    )

            if tasks:
                task_keys = list(tasks.keys())
                results = await asyncio.gather(
                    *[tasks[k] for k in task_keys], return_exceptions=True
                )
                for key, result in zip(task_keys, results):
                    if isinstance(result, Exception):
                        logger.warning("Enrichment failed for '%s': %s", key, result)
                        continue
                    if key == "_controlType":
                        if isinstance(doc.get("control"), dict) and result:
                            doc["control"]["controlType"] = result
                    elif key == "introduction":
                        if result:
                            doc["introduction"] = result
                    else:
                        if result is not None:
                            doc[key] = result

            if not recipient_list and doc.get("recipientList"):
                recipient_list = doc["recipientList"]

            if recipient_corr_tasks:
                indices = [idx for idx, _ in recipient_corr_tasks]
                corr_results = await asyncio.gather(
                    *[coro for _, coro in recipient_corr_tasks], return_exceptions=True
                )
                for idx, result in zip(indices, corr_results):
                    if isinstance(result, Exception):
                        continue
                    if result and isinstance(recipient_list[idx], dict):
                        recipient_list[idx]["correspondent"] = result

        return doc