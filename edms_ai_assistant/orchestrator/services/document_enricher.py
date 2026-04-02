# edms_ai_assistant/services/document_enricher.py
"""
EDMS AI Assistant — Document Enricher Service.

Покрываемые связи:
    correspondentId               → GET /api/correspondent/{id}
    journalId                     → GET /api/reg-journal/{id}
    documentTypeId                → GET /api/document-type/{id}
    currencyId                    → GET /api/currency/{id}
    control.controlTypeId         → GET /api/control-type/{id}
    doc.id (introduction)         → GET /api/introduction/document/{id}
    doc.id (appeal)               → GET /api/document-appeal/document/{id}
    doc.id (recipientList)        → GET /api/document/{id}/recipient
    doc.id (taskList+executors)   → GET /api/document/{id}/task?fetchExecutors=true
    doc.id (responsible)          → GET /api/document/{id}/responsible  (CONTRACT only)
    recipientList[N].correspondentId → GET /api/correspondent/{id}  (batch)

Использование:
    enricher = DocumentEnricher(base_url=settings.EDMS_API_URL)
    enriched_doc = await enricher.enrich(raw_doc_dict, token=jwt_token)
    analysis = nlp_service.process_document(enriched_doc)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from edms_ai_assistant.clients.base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


RawDoc = dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
# Thin reference clients  (один метод — один GET-запрос)
# ══════════════════════════════════════════════════════════════════════════════


class _RefClient(EdmsHttpClient):
    """Lightweight HTTP client for single-resource reference lookups.

    Используется только внутри DocumentEnricher.
    Все методы возвращают Optional[Dict] — None при любой ошибке / 404.
    """

    async def get_correspondent(
        self, token: str, correspondent_id: str
    ) -> dict[str, Any] | None:
        """Fetches CorrespondentDto by UUID.

        Args:
            token: JWT bearer token.
            correspondent_id: CorrespondentDto UUID.

        Returns:
            CorrespondentDto as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", f"api/correspondent/{correspondent_id}", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_reg_journal(
        self, token: str, journal_id: str
    ) -> dict[str, Any] | None:
        """Fetches RegistrationJournalDto by UUID.

        Args:
            token: JWT bearer token.
            journal_id: RegistrationJournalDto UUID.

        Returns:
            RegistrationJournalDto as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", f"api/reg-journal/{journal_id}", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_document_type(
        self, token: str, document_type_id: str
    ) -> dict[str, Any] | None:
        """Fetches DocumentTypeDto by ID.

        Args:
            token: JWT bearer token.
            document_type_id: DocumentTypeDto Long ID (as string).

        Returns:
            DocumentTypeDto as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", f"api/document-type/{document_type_id}", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_currency(self, token: str, currency_id: str) -> dict[str, Any] | None:
        """Fetches CurrencyDto by UUID.

        Args:
            token: JWT bearer token.
            currency_id: CurrencyDto UUID.

        Returns:
            CurrencyDto as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", f"api/currency/{currency_id}", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_control_type(
        self, token: str, control_type_id: str
    ) -> dict[str, Any] | None:
        """Fetches ControlTypeDto by UUID.

        Args:
            token: JWT bearer token.
            control_type_id: ControlTypeDto UUID.

        Returns:
            ControlTypeDto as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", f"api/control-type/{control_type_id}", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_introduction_list(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches IntroductionDto list for a document.

        Args:
            token: JWT bearer token.
            document_id: Document UUID.

        Returns:
            List of IntroductionDto dicts, empty list on failure.
        """
        result = await self._make_request(
            "GET", f"api/introduction/document/{document_id}", token=token
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            content = result.get("content") or result.get("items")
            if isinstance(content, list):
                return content
        return []

    async def get_document_appeal(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches DocumentAppealDto for a document (APPEAL category).

        Args:
            token: JWT bearer token.
            document_id: Document UUID.

        Returns:
            DocumentAppealDto as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", f"api/document-appeal/document/{document_id}", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_document_recipients(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches recipient/contractor list via GET api/document/{id}/recipient.

        includes=RECIPIENT не работает в Java — всегда возвращает пустой список.
        Единственный рабочий способ — отдельный GET-запрос к этому эндпоинту.

        Args:
            token: JWT bearer token.
            document_id: Document UUID.

        Returns:
            List of recipient dicts (id, name, unp, status, dateSend, toPeople).
            Empty list on failure or 404.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/recipient", token=token
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            content = result.get("content") or result.get("items")
            if isinstance(content, list):
                return content
        return []

    async def get_document_tasks(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches task list with executors via GET api/document/{id}/task.

        Вызывается с параметром fetchExecutors=true (TaskFilter.fetchExecutors)
        чтобы Java вложила TaskExecutorsDto[] в каждый TaskDto за один запрос.

        taskList через FULL_DOC_INCLUDES всегда возвращает пустой список —
        это подтверждено диагностикой. Данный эндпоинт — единственный рабочий.

        Args:
            token: JWT bearer token.
            document_id: Document UUID.

        Returns:
            List of TaskDto dicts with taskExecutors populated.
            Empty list on failure or 404.
        """
        result = await self._make_request(
            "GET",
            f"api/document/{document_id}/task",
            token=token,
            params={"fetchExecutors": "true"},
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Поддержка Page<TaskDto> и wrapper-объектов
            content = result.get("content") or result.get("items") or result.get("data")
            if isinstance(content, list):
                return content
        return []

    async def get_contract_responsible(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches contract responsible persons via GET api/document/{id}/responsible.

        Возвращает список ContractResponsibleDto — сотрудников, ответственных
        за исполнение договора со стороны организации.

        Структура ответа (подтверждена диагностикой):
            [{ id, documentId, document, user: UserInfoDto, createDate }]

        Args:
            token: JWT bearer token.
            document_id: Document UUID.

        Returns:
            List of ContractResponsibleDto dicts. Empty list on failure or 404.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/responsible", token=token
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            content = result.get("content") or result.get("items") or result.get("data")
            if isinstance(content, list):
                return content
        return []


# ══════════════════════════════════════════════════════════════════════════════
# DocumentEnricher
# ══════════════════════════════════════════════════════════════════════════════


class DocumentEnricher:
    """Enriches a raw DocumentDto dict with data from secondary API endpoints.

    Берёт сырой dict из DocumentClient.get_document_metadata() и параллельно
    дозапрашивает связанные объекты по UUID-полям.

    Все запросы выполняются через asyncio.gather — максимальный параллелизм,
    минимальная задержка.

    Принцип «не ломать»: если любой доп. запрос упал — поле остаётся как есть
    (UUID-строка), enrich() не бросает исключений наружу.

    Args:
        base_url: Base URL of the EDMS Java API.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    async def enrich(self, doc: RawDoc, token: str) -> RawDoc:
        """Enrich a raw DocumentDto dict with all secondary API data.

        Выполняет параллельные запросы для всех UUID-полей документа и
        встраивает полученные объекты обратно в dict.

        Args:
            doc: Raw DocumentDto dict from DocumentClient.
            token: JWT bearer token for all sub-requests.

        Returns:
            Same dict mutated in-place with enriched nested objects.
            Returns original doc unchanged if any critical error occurs.
        """
        if not doc:
            return doc

        doc_id: str | None = str(doc.get("id", "")) or None
        category: str = str(doc.get("docCategoryConstant", "") or "")

        # Планируем задачи — только те поля, у которых есть UUID
        tasks: dict[str, Any] = {}

        async with _RefClient(base_url=self._base_url) as client:

            # ── Корреспондент ─────────────────────────────────────────────────
            # correspondentId есть, но correspondent (объект) не пришёл через includes
            if doc.get("correspondentId") and not _has_nested(doc, "correspondent"):
                tasks["correspondent"] = client.get_correspondent(
                    token, str(doc["correspondentId"])
                )

            # ── Журнал регистрации ────────────────────────────────────────────
            if doc.get("journalId") and not _has_nested(doc, "registrationJournal"):
                tasks["registrationJournal"] = client.get_reg_journal(
                    token, str(doc["journalId"])
                )

            # ── Вид документа ─────────────────────────────────────────────────
            if doc.get("documentTypeId") and not _has_nested(doc, "documentType"):
                tasks["documentType"] = client.get_document_type(
                    token, str(doc["documentTypeId"])
                )

            # ── Валюта (договоры) ─────────────────────────────────────────────
            if doc.get("currencyId") and not _has_nested(doc, "currency"):
                tasks["currency"] = client.get_currency(token, str(doc["currencyId"]))

            # ── Тип контроля ──────────────────────────────────────────────────
            control = doc.get("control") or {}
            if (
                isinstance(control, dict)
                and control.get("controlTypeId")
                and not _has_nested(control, "controlType")
            ):
                tasks["_controlType"] = client.get_control_type(
                    token, str(control["controlTypeId"])
                )

            # ── Лист ознакомления ─────────────────────────────────────────────
            # introduction[] отсутствует в includes — всегда дозапрашиваем
            if doc_id and not doc.get("introduction"):
                tasks["introduction"] = client.get_introduction_list(token, doc_id)

            # ── Обращение граждан ─────────────────────────────────────────────
            if doc_id and category == "APPEAL" and not doc.get("documentAppeal"):
                tasks["documentAppeal"] = client.get_document_appeal(token, doc_id)

            # ── Адресаты/контрагенты ──────────────────────────────────────────
            if doc_id and not doc.get("recipientList"):
                tasks["recipientList"] = client.get_document_recipients(token, doc_id)

            # ── Корреспонденты адресатов (batch) ─────────────────────────────
            if doc_id and not doc.get("taskList"):
                tasks["taskList"] = client.get_document_tasks(token, doc_id)

            # ── Ответственные по договору ─────────────────────────────────────
            if doc_id and category == "CONTRACT" and not doc.get("contractResponsible"):
                tasks["contractResponsible"] = client.get_contract_responsible(
                    token, doc_id
                )

            recipient_list: list[dict[str, Any]] = doc.get("recipientList") or []
            recipient_corr_tasks: list[tuple[int, Any]] = []
            for idx, recipient in enumerate(recipient_list):
                if (
                    isinstance(recipient, dict)
                    and recipient.get("correspondentId")
                    and not _has_nested(recipient, "correspondent")
                ):
                    cid = str(recipient["correspondentId"])
                    recipient_corr_tasks.append(
                        (idx, client.get_correspondent(token, cid))
                    )

            # ── Выполняем все задачи параллельно ─────────────────────────────
            if tasks:
                task_keys = list(tasks.keys())
                task_coros = [tasks[k] for k in task_keys]
                logger.debug(
                    "Enriching document with secondary API calls",
                    extra={
                        "document_id": doc_id,
                        "tasks": task_keys,
                        "recipient_corr_count": len(recipient_corr_tasks),
                    },
                )
                results = await asyncio.gather(*task_coros, return_exceptions=True)
                for key, result in zip(task_keys, results):
                    if isinstance(result, Exception):
                        logger.warning(
                            "Enrichment sub-request failed",
                            extra={"task": key, "error": str(result)},
                        )
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

            # Корреспонденты внутри адресатов
            if recipient_corr_tasks:
                indices = [idx for idx, _ in recipient_corr_tasks]
                corr_coros = [coro for _, coro in recipient_corr_tasks]
                corr_results = await asyncio.gather(*corr_coros, return_exceptions=True)
                for idx, result in zip(indices, corr_results):
                    if isinstance(result, Exception):
                        logger.warning(
                            "Recipient correspondent enrichment failed",
                            extra={"recipient_index": idx, "error": str(result)},
                        )
                        continue
                    if result and isinstance(recipient_list[idx], dict):
                        recipient_list[idx]["correspondent"] = result

        logger.debug(
            "Document enrichment complete",
            extra={"document_id": doc_id, "category": category},
        )
        return doc


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _has_nested(obj: dict[str, Any], key: str) -> bool:
    """Return True if *obj* already has a non-empty nested object at *key*.

    Prevents redundant API calls when includes already populated the field.

    Args:
        obj: Dict to inspect.
        key: Field name to check.

    Returns:
        True if the field exists and is a non-empty dict or non-empty list.
    """
    val = obj.get(key)
    if isinstance(val, dict):
        return bool(val)
    if isinstance(val, list):
        return bool(val)
    return False
