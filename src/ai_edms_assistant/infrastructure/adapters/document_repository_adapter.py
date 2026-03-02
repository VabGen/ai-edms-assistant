# src/ai_edms_assistant/infrastructure/adapters/document_repository_adapter.py
"""Default adapter: IDocumentRepository → EdmsDocumentClient.

Единственная точка подключения Application Layer к Document HTTP API.
Application Layer знает только об IDocumentRepository (порт/протокол).
Эта реализация — конкретный адаптер Infrastructure Layer.

Архитектурные решения:
    1. EdmsDocumentClient (наследник EdmsHttpClient) используется для всех запросов.
       Он имеет _make_request() который нужен EdmsDocumentRepository.
    2. EdmsDocumentRepository получает EdmsDocumentClient (НЕ httpx.AsyncClient!).
       Это ключевое — EdmsDocumentRepository.get_recipients() вызывает
       self.http_client._make_request(), которого нет у httpx.AsyncClient.
    3. Shared client в get_related_data(): все корутины repo.get_*() выполняются
       в одном async with EdmsDocumentClient() — один connection pool, один auth.

Graceful degradation:
    Все ошибки поглощаются, возвращают None или {}.
    Агент продолжает с частичным <document_context>.

Changelog:
    2026-02-d: BUGFIX — заменён httpx.AsyncClient на EdmsDocumentClient.
               Ранее get_related_data() создавал EdmsDocumentRepository(httpx.AsyncClient()),
               но repository вызывает self.http_client._make_request() → AttributeError.
"""

from __future__ import annotations

import structlog

from ...domain.entities.document import Document

log = structlog.get_logger(__name__)


class DocumentRepositoryAdapter:
    """Concrete IDocumentRepository implementation via EdmsDocumentClient.

    Default adapter используется когда нет явного DI в EdmsDocumentAgent.
    Stateless — безопасно создавать на каждый запрос.
    """

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        """Fetch domain Document entity by UUID.

        Pipeline:
            1. Open EdmsDocumentClient as async context manager.
            2. GET /api/document/{doc_id} → raw dict.
            3. DocumentMapper.from_dto(raw) → domain Document.

        Args:
            token: JWT bearer token for EDMS API auth.
            doc_id: Document UUID string.

        Returns:
            Populated ``Document`` entity, or ``None`` on any error.
        """
        try:
            from uuid import UUID as _UUID
            from ..edms_api.clients.document_client import EdmsDocumentClient
            from ..edms_api.mappers.document_mapper import DocumentMapper

            async with EdmsDocumentClient() as client:
                raw = await client.get_by_id(
                    document_id=_UUID(str(doc_id)),
                    token=token,
                )
                if not raw:
                    log.warning("document_not_found", doc_id=doc_id)
                    return None

                doc = DocumentMapper.from_dto(raw)
                log.debug(
                    "document_fetched",
                    doc_id=doc_id,
                    reg_number=getattr(doc, "reg_number", None),
                    attachments_count=len(getattr(doc, "attachments", []) or []),
                    has_recipients=len(getattr(doc, "recipient_list", []) or []) > 0,
                    has_correspondent=bool(getattr(doc, "correspondent_name", None)),
                )
                return doc

        except Exception as exc:
            log.warning("document_fetch_failed", doc_id=doc_id, error=str(exc))
            return None

    async def get_related_data(
        self,
        token: str,
        doc_id: str,
        *,
        include_recipients: bool = True,
        include_responsible: bool = True,
        include_control: bool = True,
        include_tasks: bool = True,
        include_nomenclature: bool = True,
        include_repeat_identical: bool = False,
        include_history: bool = False,
        include_bpmn: bool = True,
        include_contract_versions: bool = False,
    ) -> dict:
        """Fetch all related data for document context enrichment.

        BUGFIX 2026-02-d:
            Ранее использовался httpx.AsyncClient(timeout=30.0).
            EdmsDocumentRepository вызывает self.http_client._make_request() —
            метод EdmsHttpClient, которого нет у httpx.AsyncClient → AttributeError.
            Теперь используется EdmsDocumentClient (наследник EdmsHttpClient).

        Все запросы выполняются параллельно через asyncio.gather()
        в одном EdmsDocumentClient context (shared connection pool).

        Args:
            token: JWT bearer token.
            doc_id: Document UUID string.
            include_recipients: GET /{id}/recipient.
            include_responsible: GET /{id}/responsible.
            include_control: GET /{id}/control.
            include_tasks: GET /{id}/task-task-project.
            include_nomenclature: GET /{id}/nomenclature-affair.
            include_repeat_identical: GET /{id}/repeat-identical.
            include_history: GET /{id}/history/v2 (тяжёлый, по умолчанию выкл).
            include_bpmn: GET /{id}/bpmn.
            include_contract_versions: GET /{id}/contract-version-info (тяжёлый).

        Returns:
            Dict: recipients, responsible, control, tasks, task_projects,
            nomenclature, repeat_identical, history, bpmn, contract_versions.
            Ошибочные ключи → [] или None, никогда не raises.
        """
        try:
            import asyncio as _asyncio
            from uuid import UUID as _UUID
            from ..edms_api.clients.document_client import EdmsDocumentClient
            from ..edms_api.repositories.edms_document_repository import (
                EdmsDocumentRepository,
            )

            doc_uuid = _UUID(str(doc_id))

            # ── КЛЮЧЕВОЕ: EdmsDocumentClient, а НЕ httpx.AsyncClient ──────────
            # EdmsDocumentRepository._make_request() требует EdmsHttpClient.
            # EdmsDocumentClient наследует EdmsHttpClient → имеет _make_request().
            async with EdmsDocumentClient() as client:
                repo = EdmsDocumentRepository(client)

                # ── Строим словарь корутин по boolean флагам ──────────────────
                coro_map: dict[str, object] = {}

                if include_recipients:
                    coro_map["recipients"] = repo.get_recipients(doc_uuid, token)
                if include_responsible:
                    coro_map["responsible"] = repo.get_contract_responsible(
                        doc_uuid, token
                    )
                if include_control:
                    coro_map["control"] = repo.get_control(doc_uuid, token)
                if include_tasks:
                    # get_tasks_and_projects → {"tasks": [...], "taskProjects": [...]}
                    coro_map["_tasks_raw"] = repo.get_tasks_and_projects(
                        doc_uuid, token
                    )
                if include_nomenclature:
                    coro_map["nomenclature"] = repo.get_nomenclature_affairs(
                        doc_uuid, token
                    )
                if include_repeat_identical:
                    coro_map["repeat_identical"] = repo.get_repeat_identical(
                        doc_uuid, token
                    )
                if include_history:
                    coro_map["history"] = repo.get_history(doc_uuid, token)
                if include_bpmn:
                    coro_map["bpmn"] = repo.get_bpmn_activity(doc_uuid, token)
                if include_contract_versions:
                    coro_map["contract_versions"] = repo.get_contract_version_info(
                        doc_uuid, token
                    )

                if not coro_map:
                    return {}

                # ── Параллельный запуск всех корутин ─────────────────────────
                keys = list(coro_map.keys())
                results = await _asyncio.gather(
                    *[coro_map[k] for k in keys],
                    return_exceptions=True,
                )

                # ── Graceful degradation: ошибки не пробрасываем ──────────────
                output: dict = {}
                for key, result in zip(keys, results):
                    if isinstance(result, Exception):
                        log.warning(
                            "related_data_partial_failure",
                            key=key,
                            doc_id=doc_id,
                            error=str(result),
                        )
                        # control/bpmn/_tasks_raw → None; остальные → []
                        output[key] = (
                            None
                            if key in ("control", "bpmn", "_tasks_raw")
                            else []
                        )
                    else:
                        output[key] = result

                # ── Распаковываем tasks_and_projects ──────────────────────────
                if "_tasks_raw" in output:
                    raw_tasks = output.pop("_tasks_raw") or {}
                    output["tasks"] = (
                        raw_tasks.get("tasks", [])
                        if isinstance(raw_tasks, dict)
                        else []
                    )
                    output["task_projects"] = (
                        raw_tasks.get("taskProjects", [])
                        if isinstance(raw_tasks, dict)
                        else []
                    )

                log.debug(
                    "related_data_fetched",
                    doc_id=doc_id,
                    keys=list(output.keys()),
                    recipients_count=len(output.get("recipients") or []),
                    tasks_count=len(output.get("tasks") or []),
                    has_control=output.get("control") is not None,
                    has_bpmn=output.get("bpmn") is not None,
                )
                return output

        except Exception as exc:
            log.warning(
                "related_data_fetch_failed",
                doc_id=doc_id,
                error=str(exc),
            )
            return {}