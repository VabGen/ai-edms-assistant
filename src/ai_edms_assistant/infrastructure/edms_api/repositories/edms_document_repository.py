# src/ai_edms_assistant/infrastructure/edms_api/repositories/edms_document_repository.py
"""EDMS Document Repository — REST API implementation.

Changelog 2026-02-c (ПОЛНОЕ РАСШИРЕНИЕ):
  + get_recipients(): GET /{id}/recipient — все адресаты документа
  + get_contract_responsible(): GET /{id}/responsible — ответственные по договору
  + get_contract_version_info(): GET /{id}/contract-version-info — версии договора
  + get_control(): GET /{id}/control — контроль документа (ControlDto)
  + get_history(): GET /{id}/history/v2 — история изменений
  + get_repeat_identical(): GET /{id}/repeat-identical — повторные/идентичные обращения
  + get_nomenclature_affairs(): GET /{id}/nomenclature-affair — дела (номенклатура)
  + get_bpmn_activity(): GET /{id}/bpmn — текущий BPMN-процесс
  + get_tasks_and_projects(): GET /{id}/task-task-project — задачи и проектные задачи
  + get_document_full(): Агрегирующий метод — параллельная загрузка всех связанных данных

Changelog 2026-02-b:
  + _map_version_to_entity(): корректный маппинг DocumentVersionDto
  + get_versions(): использует _map_version_to_entity
  + _map_to_entity(): подключён DocumentMapper.from_dto() для полного маппинга
  + _map_user(): исправлен (UserInfo не имеет поля "name" напрямую)

Архитектурная заметка:
  Все новые методы возвращают raw dict / list[dict] (не domain entities),
  так как эти данные используются только в DocumentContextBuilder для LLM-контекста.
  Domain-level маппинг добавлять по необходимости, когда появятся domain entities.
"""

from __future__ import annotations

import asyncio
import structlog
from typing import Any
from uuid import UUID

from ai_edms_assistant.infrastructure.edms_api.http_client import EdmsHttpClient
from ....domain.entities.appeal import DocumentAppeal
from ....domain.entities.attachment import Attachment
from ....domain.entities.document import Document, DocumentStatus
from ....domain.entities.employee import UserInfo
from ai_edms_assistant.domain.value_objects.filters import (
    DocumentFilter,
    DocumentLinkFilter,
)
from ....domain.repositories.base import Page, PageRequest
from ....domain.repositories.document_repository import AbstractDocumentRepository

logger = structlog.get_logger(__name__)


class EdmsDocumentRepository(AbstractDocumentRepository):
    """EDMS REST API implementation of AbstractDocumentRepository.

    Attributes:
        http_client: Shared async EdmsHttpClient instance.
    """

    def __init__(self, http_client: EdmsHttpClient) -> None:
        """Initialize repository with shared HTTP client.

        Args:
            http_client: Configured EdmsHttpClient instance.
        """
        self.http_client = http_client

    # ------------------------------------------------------------------
    # Null-safe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_uuid(raw: Any) -> UUID | None:
        """Parse UUID with None fallback — never raises.

        Args:
            raw: Any value — UUID string, UUID object, or None/null.

        Returns:
            UUID instance or None.
        """
        if not raw:
            return None
        try:
            return UUID(str(raw))
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _safe_str(raw: Any, fallback: str = "") -> str:
        """Return string value with fallback — never raises.

        Args:
            raw: Any value.
            fallback: Default string if raw is None/empty.

        Returns:
            String value or fallback.
        """
        if raw is None:
            return fallback
        return str(raw)

    # ------------------------------------------------------------------
    # Internal mapper: DocumentVersionDto → domain Document
    # ------------------------------------------------------------------

    def _map_version_to_entity(
        self,
        version_dto: dict[str, Any],
    ) -> Document | None:
        """Map Java DocumentVersionDto to domain Document.

        CRITICAL: Java GET /api/document/{id}/version returns
        List<DocumentVersionDto>, NOT List<DocumentDto>.

        DocumentVersionDto structure:
            {
                "id": <UUID of version record>,
                "version": <int — version number>,
                "documentId": <UUID of parent document>,
                "document": { ... full DocumentDto ... },
                "deleted": false
            }

        Strategy:
          1. Extract the nested "document" dict (full DocumentDto).
          2. Map it via _map_to_entity() to get a full Document entity.
          3. Patch version_number onto the entity from v["version"].
          4. If "document" is absent — fallback to mapping v itself
             (handles older API versions that return flat DocumentDto).

        Args:
            version_dto: Raw DocumentVersionDto dict from EDMS API.

        Returns:
            Fully populated domain Document entity, or None on failure.
        """
        if not isinstance(version_dto, dict):
            return None

        # ── Шаг 1: извлекаем вложенный DocumentDto ───────────────────
        doc_data = version_dto.get("document")

        if doc_data and isinstance(doc_data, dict) and doc_data.get("id"):
            # Нормальный случай: DocumentVersionDto содержит document
            doc = self._map_to_entity(doc_data)
        else:
            # Fallback: API вернул плоскую структуру (совместимость)
            logger.warning(
                "document_version_missing_nested_document",
                extra={
                    "version_id": version_dto.get("id"),
                    "document_id": version_dto.get("documentId"),
                },
            )
            doc = self._map_to_entity(version_dto)

        if doc is None:
            return None

        # ── Шаг 2: патчим version_number из поля "version" (int) ─────
        version_number = version_dto.get("version")
        if version_number is not None:
            try:
                doc = doc.model_copy(
                    update={"version_number_snapshot": int(version_number)}
                )
            except Exception:
                logger.debug(
                    "document_version_number_patch_skipped",
                    extra={
                        "version_number": version_number,
                        "doc_id": str(doc.id),
                    },
                )

        return doc

    # ------------------------------------------------------------------
    # Internal mapper: raw DocumentDto dict → domain Document
    # ------------------------------------------------------------------

    def _map_to_entity(self, data: dict[str, Any]) -> Document | None:
        """Map raw EDMS DocumentDto dict to domain Document.

        Delegates to DocumentMapper.from_dto() for full field coverage.
        Falls back to manual mapping if mapper is unavailable.

        Args:
            data: Raw dict from EDMS API response.

        Returns:
            Populated Document entity, or None if ``id`` is absent.
        """
        doc_id = self._safe_uuid(data.get("id"))
        if doc_id is None:
            logger.warning(
                "document_mapper_missing_id",
                extra={"keys": list(data.keys())[:10]},
            )
            return None

        # ── Используем DocumentMapper для полного покрытия полей ──────
        try:
            from ...edms_api.mappers.document_mapper import DocumentMapper
            return DocumentMapper.from_dto(data)
        except Exception as exc:
            logger.warning(
                "document_mapper_fallback",
                extra={"error": str(exc), "doc_id": str(doc_id)},
            )

        # ── Fallback: минимальный маппинг ─────────────────────────────
        status: DocumentStatus | None = None
        raw_state = data.get("state") or data.get("status")
        if raw_state:
            try:
                status = DocumentStatus(raw_state)
            except ValueError:
                pass

        doc_type_raw = data.get("documentType") or {}
        document_type_name: str | None = (
            doc_type_raw.get("typeName") if isinstance(doc_type_raw, dict) else None
        )

        delivery_raw = data.get("deliveryMethod") or {}
        delivery_method_name: str | None = (
            delivery_raw.get("deliveryName") if isinstance(delivery_raw, dict) else None
        )

        author = self._map_user(data.get("author"))
        responsible_executor = self._map_user(data.get("responsibleExecutor"))
        initiator = self._map_user(data.get("initiator"))
        who_signed = self._map_user(data.get("whoSigned"))

        raw_attachments = data.get("attachmentDocument") or data.get("attachments") or []
        attachments: list[Attachment] = []

        for a in raw_attachments:
            if not isinstance(a, dict):
                continue
            att_id = self._safe_uuid(a.get("id"))
            if att_id is None:
                continue
            try:
                attachments.append(
                    Attachment(
                        id=att_id,
                        file_name=a.get("fileName") or a.get("name") or "unnamed",
                        file_size=a.get("fileSize") or a.get("size") or 0,
                        mime_type=a.get("mimeType"),
                        version_number=a.get("versionNumber") or 1,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "attachment_map_failed",
                    extra={"doc_id": str(doc_id), "error": str(exc)},
                )

        try:
            return Document(
                id=doc_id,
                organization_id=data.get("organizationId"),
                reg_number=data.get("regNumber"),
                reg_date=data.get("regDate"),
                out_reg_number=data.get("outRegNumber"),
                out_reg_date=data.get("outRegDate"),
                create_date=data.get("createDate"),
                short_summary=data.get("shortSummary"),
                summary=data.get("summary"),
                note=data.get("note"),
                status=status,
                document_type_name=document_type_name,
                delivery_method_name=delivery_method_name,
                correspondent_name=data.get("correspondentName"),
                author=author,
                responsible_executor=responsible_executor,
                initiator=initiator,
                who_signed=who_signed,
                control_flag=data.get("controlFlag") or False,
                version_flag=data.get("versionFlag") or False,
                pages=data.get("pages"),
                exemplar_count=data.get("exemplarCount"),
                attachments=attachments,
                custom_fields=data.get("customFields") or {},
            )
        except Exception as exc:
            logger.error(
                "document_entity_build_failed",
                extra={"doc_id": str(doc_id), "error": str(exc)},
            )
            return None

    def _map_user(self, u: dict | None) -> UserInfo | None:
        """Map raw user dict to UserInfo value object.

        Args:
            u: Raw UserInfoDto dict from API, or None.

        Returns:
            UserInfo or None.
        """
        if not u or not isinstance(u, dict):
            return None
        try:
            return UserInfo(
                id=self._safe_uuid(u.get("id")),
                firstName=u.get("firstName"),
                lastName=u.get("lastName"),
                middleName=u.get("middleName"),
                organizationId=u.get("organizationId"),
                departmentId=self._safe_uuid(u.get("departmentId")),
                departmentName=u.get("departmentName"),
                postName=u.get("postName"),
                email=u.get("email"),
            )
        except Exception as exc:
            logger.debug("user_map_failed", extra={"error": str(exc)})
            return None

    # ------------------------------------------------------------------
    # AbstractRepository base
    # ------------------------------------------------------------------

    async def get_by_id(
        self,
        entity_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> Document | None:
        """Fetch document by UUID.

        Mirrors Java: GET /api/document/{id}
        Java findById() includes:
          ATTACHMENT, CURRENCY, ACCESS_GRIEF, DOC_VERSION, DOCUMENT_FORM_DEFINITION,
          DELIVERY_METHOD, DOCUMENT_TYPE, CITIZEN_TYPE, PARENT_SUBJECT, SOLUTION_RESULT

        Args:
            entity_id: Document UUID.
            token: JWT bearer token.
            organization_id: Optional org scope for multi-tenant.

        Returns:
            Domain Document or None.
        """
        params: dict[str, Any] = {}
        if organization_id:
            params["organizationId"] = organization_id

        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{entity_id}",
                token=token,
                params=params,
            )
            if not data:
                return None
            return self._map_to_entity(data)
        except Exception as exc:
            logger.error(
                "edms_document_get_by_id_failed",
                doc_id=str(entity_id),
                error=str(exc),
            )
            raise

    async def get_by_ids(
        self,
        entity_ids: list[UUID],
        token: str,
        organization_id: str | None = None,
    ) -> list[Document]:
        """Batch-fetch documents via parallel GET calls.

        Args:
            entity_ids: List of document UUIDs.
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            List of successfully mapped documents.
        """
        if not entity_ids:
            return []
        results = await asyncio.gather(
            *[self.get_by_id(eid, token, organization_id) for eid in entity_ids],
            return_exceptions=True,
        )
        return [r for r in results if isinstance(r, Document)]

    async def find_page(
        self,
        token: str,
        organization_id: str | None = None,
        pagination: PageRequest | None = None,
    ) -> Page[Document]:
        """Fetch paginated document list with empty filter."""
        return await self.search(DocumentFilter(), token, pagination)

    # ------------------------------------------------------------------
    # ── НОВЫЕ МЕТОДЫ: Связанные данные документа (Java endpoints) ─────
    # ------------------------------------------------------------------

    async def get_recipients(
        self,
        document_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """Fetch all recipients (адресаты) of a document.

        Mirrors Java:
            GET /api/document/{id}/recipient
            DocumentController.getDocumentRecipientList()
            → documentRecipientService.findByDocumentId(documentId, user)

        Returns List<DocumentRecipientDto> with fields:
            id, documentId, name, type (NORMAL/AISMV/GTB_ORG),
            status (SENDING/SENDED/RECEIVED/AISMV_SENDED/FAILED/CANCEL),
            delivered (bool), system (bool), lock (bool),
            deliveryMethodId, correspondentId, toPeople,
            signDate (Instant), unp (string),
            deliveryMethod.deliveryName

        Args:
            document_id: UUID of the document.
            token: JWT bearer token.

        Returns:
            List of DocumentRecipientDto dicts.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/recipient",
                token=token,
            )
            result = data if isinstance(data, list) else []
            logger.info(
                "document_recipients_fetched",
                doc_id=str(document_id),
                count=len(result),
            )
            return result
        except Exception as exc:
            logger.error(
                "edms_document_get_recipients_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            return []

    async def get_contract_responsible(
        self,
        document_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """Fetch contract responsible employees.

        Mirrors Java:
            GET /api/document/{documentId}/responsible
            DocumentController.getContractResponsible()
            → contractResponsibleService.findByDocumentId(documentId, user)

        Returns List<ContractResponsibleDto> with fields:
            id, documentId, documentOrgId,
            user: UserInfoDto {id, firstName, lastName, middleName, postName, departmentName}

        Used for CONTRACT category documents to know who is responsible
        for execution, tracking, and signing the contract.

        Args:
            document_id: UUID of the document.
            token: JWT bearer token.

        Returns:
            List of ContractResponsibleDto dicts.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/responsible",
                token=token,
            )
            result = data if isinstance(data, list) else []
            logger.info(
                "contract_responsible_fetched",
                doc_id=str(document_id),
                count=len(result),
            )
            return result
        except Exception as exc:
            logger.error(
                "edms_contract_responsible_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            return []

    async def get_contract_version_info(
        self,
        document_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """Fetch contract version history with attachments.

        Mirrors Java:
            GET /api/document/{documentId}/contract-version-info
            DocumentController.getVersionInfo()
            → contractVersionInfoService.findAll(documentId, user)

        Returns List<ContractVersionInfoDto> with fields:
            id, documentId, documentOrgId,
            createDate, version (int), versionDescription,
            attachments: List<ContractVersionAttachmentDto> {
                id, originalName, size, contentType, uploadDate
            }

        Used to show the full amendment history of a contract
        (e.g., version 1 → version 2 after additional agreement).

        Args:
            document_id: UUID of the document.
            token: JWT bearer token.

        Returns:
            List of ContractVersionInfoDto dicts.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/contract-version-info",
                token=token,
            )
            result = data if isinstance(data, list) else []
            logger.info(
                "contract_version_info_fetched",
                doc_id=str(document_id),
                count=len(result),
            )
            return result
        except Exception as exc:
            logger.error(
                "edms_contract_version_info_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            return []

    async def get_control(
        self,
        document_id: UUID,
        token: str,
    ) -> dict[str, Any] | None:
        """Fetch control (контроль) settings of a document.

        Mirrors Java:
            GET /api/document/{documentId}/control
            DocumentController.getControl()
            → controlService.findDocumentControl(documentId, user)

        Returns ControlDto with fields:
            id, documentId, controlTypeId, controlType.name,
            deadline (Instant), executionDate (Instant),
            controlUser: UserInfoDto, note (str),
            status (ACTIVE/REMOVED/EXECUTED)

        Control means the document is being monitored for on-time execution.
        Returns empty ControlDto {} if no control is set.

        Args:
            document_id: UUID of the document.
            token: JWT bearer token.

        Returns:
            ControlDto dict or None.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/control",
                token=token,
            )
            if not data or not isinstance(data, dict):
                return None
            # Java returns empty ControlDto {} when no control — treat as None
            if not data.get("id"):
                return None
            logger.info(
                "document_control_fetched",
                doc_id=str(document_id),
                control_id=data.get("id"),
            )
            return data
        except Exception as exc:
            logger.error(
                "edms_document_get_control_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            return None

    async def get_history(
        self,
        document_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """Fetch document change history (v2).

        Mirrors Java:
            GET /api/document/{id}/history/v2
            DocumentController.getDocumentHistoryV2()
            → documentHistoryServiceV2.findByDocument(docKey)

        Returns List<DocumentHistoryDtoV2> with fields:
            id, documentId, action (CREATE/UPDATE/STATUS_CHANGE/etc.),
            createDate (Instant),
            user: UserInfoDto,
            changes: List<DocumentHistoryChangeDto> {
                fieldName, oldValue, newValue
            }

        Args:
            document_id: UUID of the document.
            token: JWT bearer token.

        Returns:
            List of history event dicts, newest first.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/history/v2",
                token=token,
            )
            result = data if isinstance(data, list) else []
            logger.info(
                "document_history_fetched",
                doc_id=str(document_id),
                events=len(result),
            )
            return result
        except Exception as exc:
            logger.error(
                "edms_document_get_history_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            return []

    async def get_repeat_identical(
        self,
        document_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """Fetch linked repeat and identical appeals.

        Mirrors Java:
            GET /api/document/{documentId}/repeat-identical
            DocumentController.getRepeatIdentical()
            → repeatIdenticalAppealService.findByGroupTypeRegNumNotNull(
                documentId, at, organizationId)
              for each AppealType (REPEAT, IDENTICAL)

        Returns List<RepeatIdenticalAppealDto> with fields:
            docId, regNumber, regDate, shortSummary,
            appealType (REPEAT/IDENTICAL), groupId

        Used for APPEAL category to identify duplicate/identical citizen appeals.

        Args:
            document_id: UUID of the appeal document.
            token: JWT bearer token.

        Returns:
            List of RepeatIdenticalAppealDto dicts.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/repeat-identical",
                token=token,
            )
            result = data if isinstance(data, list) else []
            logger.info(
                "repeat_identical_fetched",
                doc_id=str(document_id),
                count=len(result),
            )
            return result
        except Exception as exc:
            logger.error(
                "edms_repeat_identical_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            return []

    async def get_nomenclature_affairs(
        self,
        document_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """Fetch nomenclature affairs (дела) the document is filed into.

        Mirrors Java:
            GET /api/document/{id}/nomenclature-affair
            DocumentController.affairs()
            → documentService2.findDocumentNomenclatures(documentId, user)

        Returns List<NomenclatureAffairDto> with fields:
            id, number, name, year,
            department: {id, name},
            writeOff (bool — whether document is written off to this affair)

        Used to show archival filing status.

        Args:
            document_id: UUID of the document.
            token: JWT bearer token.

        Returns:
            List of NomenclatureAffairDto dicts.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/nomenclature-affair",
                token=token,
            )
            result = data if isinstance(data, list) else []
            logger.info(
                "nomenclature_affairs_fetched",
                doc_id=str(document_id),
                count=len(result),
            )
            return result
        except Exception as exc:
            logger.error(
                "edms_nomenclature_affairs_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            return []

    async def get_bpmn_activity(
        self,
        document_id: UUID,
        token: str,
    ) -> dict[str, Any] | None:
        """Fetch current BPMN process state and activity.

        Mirrors Java:
            GET /api/document/{id}/bpmn
            DocumentController.getProcessActivity()
            → documentService2.getProcessActivity(id, user)

        Returns BpmnProcessActivityDto with fields:
            activities: List<ActivityDto> [{activityId, name, type}] — активные задачи
            history: Set<String> — все пройденные activity IDs
            transientActivities: List<String> — переходные задачи (next-process)
            transientStart: List<String>
            transientEnd: List<String>
            parsed: List<ParsedFlowElement> [{id, type, name, processType, condition}]
            xml: String — BPMN XML схема процесса

        Used to understand what stage the document is at in the workflow.

        Args:
            document_id: UUID of the document.
            token: JWT bearer token.

        Returns:
            BpmnProcessActivityDto dict or None.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/bpmn",
                token=token,
            )
            if not data or not isinstance(data, dict):
                return None
            logger.info(
                "bpmn_activity_fetched",
                doc_id=str(document_id),
                active_tasks=len(data.get("activities") or []),
            )
            return data
        except Exception as exc:
            logger.error(
                "edms_bpmn_activity_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            return None

    async def get_tasks_and_projects(
        self,
        document_id: UUID,
        token: str,
    ) -> dict[str, Any]:
        """Fetch tasks and project tasks linked to a document.

        Mirrors Java:
            GET /api/document/{id}/task-task-project
            DocumentController.getTasksWithProjects()
            → taskService.findAllByDocumentIdWithExecutors(documentId, user)
            → taskProjectService.findAllByDocumentIdWithExecutors(documentId, user)

        Returns dict with:
            tasks: List<TaskDto> {
                id, name, status (NEW/EXECUTION/EXECUTED/CANCEL),
                deadline, createDate,
                author: UserInfoDto,
                executors: List<UserInfoDto>,
                documentId, description
            }
            taskProjects: List<TaskProjectDto> {
                id, name, status, deadline,
                executors: List<UserInfoDto>,
                projectId, projectName
            }

        Args:
            document_id: UUID of the document.
            token: JWT bearer token.

        Returns:
            Dict with "tasks" and "taskProjects" lists.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/task-task-project",
                token=token,
            )
            if not data or not isinstance(data, dict):
                return {"tasks": [], "taskProjects": []}
            result = {
                "tasks": data.get("tasks") or [],
                "taskProjects": data.get("taskProjects") or [],
            }
            logger.info(
                "tasks_and_projects_fetched",
                doc_id=str(document_id),
                tasks_count=len(result["tasks"]),
                projects_count=len(result["taskProjects"]),
            )
            return result
        except Exception as exc:
            logger.error(
                "edms_tasks_and_projects_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            return {"tasks": [], "taskProjects": []}

    async def get_all_document_with_permissions(
        self,
        document_id: UUID,
        token: str,
    ) -> dict[str, Any] | None:
        """Fetch document with permissions in single call.

        Mirrors Java:
            GET /api/document/{documentId}/all
            DocumentController.getAllWithPermissions()
            → returns {document: DocumentDto, permission: DocPermissionContainer}

        DocPermissionContainer contains allowed operations for current user:
            {DOCUMENT_READ, DOCUMENT_WRITE, DOCUMENT_DELETE, etc.: true/false}

        Args:
            document_id: UUID of the document.
            token: JWT bearer token.

        Returns:
            Dict with "document" and "permission" keys, or None.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/all",
                token=token,
            )
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.error(
                "edms_document_get_all_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            return None

    async def get_document_full(
        self,
        document_id: UUID,
        token: str,
        include_recipients: bool = True,
        include_responsible: bool = True,
        include_control: bool = True,
        include_history: bool = False,
        include_nomenclature: bool = True,
        include_bpmn: bool = False,
        include_tasks: bool = True,
        include_contract_versions: bool = False,
        include_repeat_identical: bool = False,
    ) -> dict[str, Any]:
        """Aggregate all document-related data via parallel API calls.

        Orchestrates multiple EDMS API requests concurrently to build
        a complete picture of the document for LLM context.

        Selective loading: use boolean flags to include only needed data.
        Heavy data (history, BPMN XML, contract versions) disabled by default.

        Pipeline:
            1. Fetch core document (always).
            2. In parallel: fetch all enabled related data.
            3. Return aggregated dict.

        Args:
            document_id: UUID of the document.
            token: JWT bearer token.
            include_recipients: Fetch recipient list (адресаты).
            include_responsible: Fetch contract responsible employees.
            include_control: Fetch control settings.
            include_history: Fetch change history (v2).
            include_nomenclature: Fetch nomenclature affairs.
            include_bpmn: Fetch BPMN process activity.
            include_tasks: Fetch tasks and project tasks.
            include_contract_versions: Fetch contract version history.
            include_repeat_identical: Fetch repeat/identical appeals.

        Returns:
            Dict with keys:
                document: Document | None
                recipients: list[dict]
                responsible: list[dict]
                control: dict | None
                history: list[dict]
                nomenclature_affairs: list[dict]
                bpmn_activity: dict | None
                tasks: list[dict]
                task_projects: list[dict]
                contract_versions: list[dict]
                repeat_identical: list[dict]
        """
        # ── Шаг 1: Загрузка основного документа ──────────────────────
        document = await self.get_by_id(document_id, token)

        # ── Шаг 2: Параллельная загрузка связанных данных ─────────────
        # Строим список корутин с метаданными (name, coro)
        coros: list[tuple[str, Any]] = []

        if include_recipients:
            coros.append(("recipients", self.get_recipients(document_id, token)))
        if include_responsible:
            coros.append(("responsible", self.get_contract_responsible(document_id, token)))
        if include_control:
            coros.append(("control", self.get_control(document_id, token)))
        if include_history:
            coros.append(("history", self.get_history(document_id, token)))
        if include_nomenclature:
            coros.append(("nomenclature", self.get_nomenclature_affairs(document_id, token)))
        if include_bpmn:
            coros.append(("bpmn", self.get_bpmn_activity(document_id, token)))
        if include_tasks:
            coros.append(("tasks", self.get_tasks_and_projects(document_id, token)))
        if include_contract_versions:
            coros.append(("contract_versions", self.get_contract_version_info(document_id, token)))
        if include_repeat_identical:
            coros.append(("repeat_identical", self.get_repeat_identical(document_id, token)))

        names = [name for name, _ in coros]
        gathered = await asyncio.gather(
            *[coro for _, coro in coros],
            return_exceptions=True,
        )

        # ── Шаг 3: Распаковка результатов ────────────────────────────
        results: dict[str, Any] = {}
        for name, raw in zip(names, gathered):
            if isinstance(raw, Exception):
                logger.warning(
                    "document_full_partial_failure",
                    doc_id=str(document_id),
                    key=name,
                    error=str(raw),
                )
                # Graceful degradation: пустые данные вместо исключения
                results[name] = {} if name in ("control", "bpmn") else []
            else:
                results[name] = raw

        tasks_raw = results.get("tasks") or {}
        tasks_list = tasks_raw.get("tasks", []) if isinstance(tasks_raw, dict) else []
        task_projects_list = tasks_raw.get("taskProjects", []) if isinstance(tasks_raw, dict) else []

        logger.info(
            "document_full_fetched",
            doc_id=str(document_id),
            has_document=document is not None,
            recipients=len(results.get("recipients") or []),
            responsible=len(results.get("responsible") or []),
            has_control=results.get("control") is not None,
            tasks=len(tasks_list),
        )

        return {
            "document": document,
            "recipients": results.get("recipients") or [],
            "responsible": results.get("responsible") or [],
            "control": results.get("control"),
            "history": results.get("history") or [],
            "nomenclature_affairs": results.get("nomenclature") or [],
            "bpmn_activity": results.get("bpmn"),
            "tasks": tasks_list,
            "task_projects": task_projects_list,
            "contract_versions": results.get("contract_versions") or [],
            "repeat_identical": results.get("repeat_identical") or [],
        }

    # ------------------------------------------------------------------
    # AbstractDocumentRepository
    # ------------------------------------------------------------------

    async def get_by_reg_number(
        self,
        reg_number: str,
        token: str,
        organization_id: str | None = None,
    ) -> Document | None:
        """Find document by registration number."""
        page = await self.search(
            DocumentFilter(reg_number=reg_number),
            token,
            PageRequest(size=1),
        )
        return page.items[0] if page.items else None

    async def get_with_attachments(
        self,
        document_id: UUID,
        token: str,
    ) -> Document | None:
        """Fetch document with attachments pre-populated."""
        return await self.get_by_id(document_id, token)

    async def get_versions(
        self,
        document_id: UUID,
        token: str,
    ) -> list[Document]:
        """Fetch all historical versions of a document.

        ИСПРАВЛЕНИЕ: Java GET /api/document/{id}/version возвращает
        List<DocumentVersionDto>, где каждый элемент содержит:
          - "version" (int): номер версии
          - "document" (dict): полный DocumentDto вложенного документа
          - "id" (UUID): UUID записи версии (НЕ UUID документа!)
          - "deleted" (bool)

        Args:
            document_id: UUID of the document whose versions to list.
            token: JWT bearer token.

        Returns:
            List of domain Document entities ordered oldest-first.
        """
        try:
            data = await self.http_client._make_request(
                "GET",
                f"api/document/{document_id}/version",
                token=token,
            )

            if not data or not isinstance(data, list):
                logger.info(
                    "document_versions_empty",
                    extra={"document_id": str(document_id)},
                )
                return []

            result: list[Document] = []
            for v in data:
                if not isinstance(v, dict):
                    continue
                if v.get("deleted"):
                    continue
                try:
                    doc = self._map_version_to_entity(v)
                    if doc is not None:
                        result.append(doc)
                except Exception as exc:
                    logger.warning(
                        "document_version_map_failed",
                        extra={
                            "version_id": v.get("id"),
                            "version_number": v.get("version"),
                            "error": str(exc),
                        },
                    )

            result.sort(
                key=lambda d: getattr(d, "version_number_snapshot", 0) or 0
            )

            logger.info(
                "document_versions_fetched",
                extra={
                    "document_id": str(document_id),
                    "count": len(result),
                },
            )
            return result

        except Exception as exc:
            logger.error(
                "edms_document_get_versions_failed",
                doc_id=str(document_id),
                error=str(exc),
            )
            raise

    async def search(
        self,
        filters: DocumentFilter,
        token: str,
        pagination: PageRequest | None = None,
    ) -> Page[Document]:
        """Search documents with rich filtering."""
        filters.validate()
        pag = pagination or PageRequest()
        params = {**filters.as_api_params(), **pag.as_params()}

        data = await self.http_client._make_request(
            "GET", "api/document", token=token, params=params
        )
        data = data or {}

        items: list[Document] = []
        for row in data.get("content", []):
            doc = self._map_to_entity(row)
            if doc is not None:
                items.append(doc)

        return Page(
            items=items,
            page=data.get("number", pag.page),
            size=data.get("size", pag.size),
            has_next=not data.get("last", True),
            total=data.get("totalElements"),
        )

    async def find_by_organization(
        self,
        organization_id: str,
        token: str,
        pagination: PageRequest | None = None,
    ) -> Page[Document]:
        """Fetch documents for a specific organization."""
        return await self.search(
            DocumentFilter(only_user_organization=True),
            token,
            pagination,
        )

    async def get_links(
        self,
        filters: DocumentLinkFilter,
        token: str,
    ) -> list[dict]:
        """Fetch document cross-references (links + nomenclature affairs).

        Mirrors Java:
            GET /api/document/{id}/nomenclature-affair-document-link

        Args:
            filters: Must have doc_id set.
            token: JWT bearer token.

        Returns:
            List of DocumentLinkDto dicts.
        """
        if not filters.doc_id:
            raise ValueError("DocumentLinkFilter.doc_id is required for get_links()")

        params = filters.as_api_params()
        params.pop("docId", None)

        data = await self.http_client._make_request(
            "GET",
            f"api/document/{filters.doc_id}/nomenclature-affair-document-link",
            token=token,
            params=params,
        )
        return data if isinstance(data, list) else []

    async def update_fields(
        self,
        document_id: UUID,
        operation: str,
        payload: dict,
        token: str,
    ) -> bool:
        """Execute named field-update operation via POST /api/document/{id}/execute.

        Mirrors Java:
            POST /api/document/{id}/execute
            DocumentController.executeDocumentOperations()

        Args:
            document_id: UUID of the document.
            operation: Operation type string (DocOperationType).
            payload: Operation parameters.
            token: JWT bearer token.

        Returns:
            True on success, False on failure.
        """
        try:
            await self.http_client._make_request(
                "POST",
                f"api/document/{document_id}/execute",
                token=token,
                json=[{"type": operation, **payload}],
                is_json_response=False,
            )
            return True
        except Exception as exc:
            logger.error(
                "edms_document_update_fields_failed",
                doc_id=str(document_id),
                operation=operation,
                error=str(exc),
            )
            return False