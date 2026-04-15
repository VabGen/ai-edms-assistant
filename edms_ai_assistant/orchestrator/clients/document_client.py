# edms_ai_assistant/orchestrator/clients/document_client.py
"""
EDMS AI Assistant — Document HTTP Client.
──────────────────────────────────────────────────────────────────
  ПОИСК И ЧТЕНИЕ ДОКУМЕНТОВ
  GET  /                              → search_documents()
  GET  /{id}                          → get_document_metadata()
  GET  /{id}/all                      → get_document_with_permissions()
  GET  /{id}/permission               → get_document_permissions()
  GET  /{id}/properties               → get_document_properties()

  ИСТОРИЯ И ПРОЦЕССЫ
  GET  /{id}/history                  → get_document_history()
  GET  /{id}/history/v2               → get_document_history_v2()
  GET  /{id}/bpmn                     → get_process_activity()
  GET  /{id}/task-task-project        → get_tasks_and_projects()

  КОНТРОЛЬ
  GET  /{id}/control                  → get_document_control()
  POST /{id}/control                  → set_document_control()
  PUT  /control                       → remove_document_control()
  DELETE /{id}/control                → delete_document_control()

  АДРЕСАТЫ И КОРРЕСПОНДЕНТЫ
  GET  /{id}/recipient                → get_document_recipients()
  GET  /{id}/responsible              → get_contract_responsible()

  НОМЕНКЛАТУРЫ ДЕЛ
  GET  /{id}/nomenclature-affair      → get_nomenclature_affairs()
  GET  /{id}/repeat-identical         → get_repeat_identical_appeals()

  ВЕРСИИ
  GET  /{id}/version                  → get_document_versions()

  ЖИЗНЕННЫЙ ЦИКЛ
  POST /start                         → start_document()
  POST /cancel                        → cancel_document()
  POST /{id}/execute                  → execute_document_operations()

  СТАТИСТИКА
  GET  /stat/user-executor            → get_stat_user_executor()
  GET  /stat/user-control             → get_stat_user_control()
  GET  /stat/user-author              → get_stat_user_author()
──────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .base_client import EdmsBaseClient, EdmsHttpClient

logger = logging.getLogger(__name__)

_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 10

FULL_DOC_INCLUDES: list[str] = [
    "DOCUMENT_TYPE",
    "DELIVERY_METHOD",
    "CORRESPONDENT",
    "RECIPIENT",
    "PRE_NOMENCLATURE_AFFAIRS",
    "CITIZEN_TYPE",
    "REGISTRATION_JOURNAL",
    "CURRENCY",
    "SOLUTION_RESULT",
    "PARENT_SUBJECT",
    "ADDITIONAL_DOCUMENT_AND_TYPE",
]

SEARCH_DOC_INCLUDES: list[str] = [
    "DOCUMENT_TYPE",
    "CORRESPONDENT",
    "REGISTRATION_JOURNAL",
]


# ══════════════════════════════════════════════════════════════════════════════
# Abstract Interface
# ══════════════════════════════════════════════════════════════════════════════


class EdmsDocumentClient(EdmsBaseClient, ABC):
    """Abstract interface for EDMS Document API clients.

    Определяет контракт для всех методов DocumentController.java.
    Группы методов:
      - Поиск и чтение документов
      - История и процессы
      - Контроль
      - Адресаты
      - Версии
      - Жизненный цикл
      - Статистика
    """

    # ── Поиск и чтение ────────────────────────────────────────────────────────

    @abstractmethod
    async def search_documents(
        self,
        token: str,
        doc_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
        includes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Searches documents using DocumentFilter and Spring Pageable params."""
        raise NotImplementedError

    @abstractmethod
    async def get_document_metadata(
        self,
        token: str,
        document_id: str,
        includes: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Fetches full document metadata (DocumentDto) by UUID."""
        raise NotImplementedError

    @abstractmethod
    async def get_document_with_permissions(
        self,
        token: str,
        document_id: str,
        includes: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Fetches document + permissions bundle in a single request."""
        raise NotImplementedError

    @abstractmethod
    async def get_document_permissions(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches DocPermissionContainer for a document."""
        raise NotImplementedError

    @abstractmethod
    async def get_document_properties(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches extended DocumentPropertiesDto by document UUID."""
        raise NotImplementedError

    # ── История и процессы ────────────────────────────────────────────────────

    @abstractmethod
    async def get_document_history(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches document processing protocol (history v1)."""
        raise NotImplementedError

    @abstractmethod
    async def get_document_history_v2(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches document processing protocol (history v2, preferred)."""
        raise NotImplementedError

    @abstractmethod
    async def get_process_activity(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches current BPMN process with active activities."""
        raise NotImplementedError

    @abstractmethod
    async def get_tasks_and_projects(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches tasks and task-projects linked to a document."""
        raise NotImplementedError

    # ── Контроль ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_document_control(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches current control (ControlDto) for a document."""
        raise NotImplementedError

    @abstractmethod
    async def set_document_control(
        self, token: str, document_id: str, control_request: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Sets a document on control (ControlRequest.CreateControl)."""
        raise NotImplementedError

    @abstractmethod
    async def remove_document_control(self, token: str, document_id: str) -> bool:
        """Removes control mark from a document (снять с контроля)."""
        raise NotImplementedError

    @abstractmethod
    async def delete_document_control(self, token: str, document_id: str) -> bool:
        """Deletes control record for a document."""
        raise NotImplementedError

    # ── Адресаты ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_document_recipients(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches list of document recipients (DocumentRecipientDto)."""
        raise NotImplementedError

    @abstractmethod
    async def get_contract_responsible(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches responsible employees for a contract document."""
        raise NotImplementedError

    # ── Номенклатуры дел ──────────────────────────────────────────────────────

    @abstractmethod
    async def get_nomenclature_affairs(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches nomenclature affairs linked to a document."""
        raise NotImplementedError

    @abstractmethod
    async def get_repeat_identical_appeals(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches repeat and identical appeals linked to a document."""
        raise NotImplementedError

    # ── Версии ────────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_document_versions(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches all versions of a document."""
        raise NotImplementedError

    # ── Жизненный цикл ────────────────────────────────────────────────────────

    @abstractmethod
    async def start_document(self, token: str, document_id: str) -> bool:
        """Starts the document routing process."""
        raise NotImplementedError

    @abstractmethod
    async def cancel_document(
        self, token: str, document_id: str, comment: str | None = None
    ) -> bool:
        """Cancels (annuls) a document."""
        raise NotImplementedError

    @abstractmethod
    async def execute_document_operations(
        self,
        token: str,
        document_id: str,
        operations: list[dict[str, Any]],
    ) -> bool:
        """Executes a list of operations on a document (sign, agree, etc.)."""
        raise NotImplementedError

    # ── Статистика ────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_stat_user_executor(self, token: str) -> dict[str, Any] | None:
        """Fetches execution statistics for the current user."""
        raise NotImplementedError

    @abstractmethod
    async def get_stat_user_control(self, token: str) -> dict[str, Any] | None:
        """Fetches control statistics for the current user."""
        raise NotImplementedError

    @abstractmethod
    async def get_stat_user_author(self, token: str) -> dict[str, Any] | None:
        """Fetches authoring statistics for the current user."""
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _build_includes_params(includes: list[str]) -> dict[str, list[str]]:
    """Converts a list of Include names to Spring multi-value query params."""
    return {"includes": includes}


# ══════════════════════════════════════════════════════════════════════════════
# Concrete Implementation
# ══════════════════════════════════════════════════════════════════════════════


class DocumentClient(EdmsDocumentClient, EdmsHttpClient):
    """Concrete async HTTP client for EDMS Document API.

    Реализует взаимодействие с DocumentController.java.
    Все методы используют async/await и делегируют HTTP-логику в _make_request.

    Соглашения по возвращаемым значениям:
      - dict-эндпоинты  → Optional[Dict]  (None при пустом / 404-ответе)
      - list-эндпоинты  → List[Dict]      ([] при пустом ответе)
      - action-эндпоинты (204 No Content) → bool (True = успех)

    Важно: GET /api/document возвращает Spring Page<DocumentDto> со структурой
      { "content": [...], "totalElements": N, ... }
    Метод search_documents корректно извлекает поле content.

    Важно об includes:
      Методы get_document_metadata и get_document_with_permissions принимают
      параметр includes. По умолчанию используется FULL_DOC_INCLUDES — полный
      набор, необходимый для EDMSNaturalLanguageService.process_document().
      Переопределяйте includes явно только при оптимизации конкретных сценариев.
    """

    # ── Поиск и чтение ────────────────────────────────────────────────────────

    async def search_documents(
        self,
        token: str,
        doc_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
        includes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Searches documents using DocumentFilter and Spring Pageable params.

        Calls GET api/document.
        Spring принимает DocumentFilter и Pageable как query params в одном запросе.

        Параметр includes управляет JOIN-ами на стороне Java.
        По умолчанию — SEARCH_DOC_INCLUDES (лёгкий набор для списка).
        Для детального отображения передавай FULL_DOC_INCLUDES.

        Ответ API — Spring Page<DocumentDto>:
            { "content": [...], "totalElements": N, "totalPages": M, ... }

        Args:
            token: JWT bearer token.
            doc_filter: DocumentFilter fields as query params. None → пустой фильтр.
            pageable: Spring Pageable params. None → дефолты (page=0, size=10).
            includes: Include enum names. None → SEARCH_DOC_INCLUDES.

        Returns:
            List of DocumentDto dicts extracted from Page.content.
        """
        effective_pageable: dict[str, Any] = {
            "page": _DEFAULT_PAGE,
            "size": _DEFAULT_SIZE,
        }
        if pageable:
            effective_pageable.update(pageable)

        effective_includes = includes if includes is not None else SEARCH_DOC_INCLUDES

        params: dict[str, Any] = {
            **(doc_filter or {}),
            **effective_pageable,
            **_build_includes_params(effective_includes),
        }

        logger.debug(
            "Searching documents",
            extra={
                "params_keys": list(params.keys()),
                "includes": effective_includes,
            },
        )

        result = await self._make_request(
            "GET", "api/document", token=token, params=params
        )

        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                logger.debug(
                    "Documents page received",
                    extra={
                        "content_size": len(content),
                        "total_elements": result.get("totalElements"),
                    },
                )
                return content
            logger.warning(
                "Unexpected response from GET api/document: dict without 'content'",
                extra={"response_keys": list(result.keys())},
            )
            return []

        if isinstance(result, list):
            logger.warning(
                "GET api/document returned raw list instead of Page<DocumentDto>."
            )
            return result

        return []

    async def get_document_metadata(
        self,
        token: str,
        document_id: str,
        includes: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Fetches full document metadata by UUID with all nested models.

        Calls GET api/document/{id}?includes=...

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            includes: Include enum names. None → FULL_DOC_INCLUDES.

        Returns:
            DocumentDto as dict with all nested models populated, or None.
        """
        effective_includes = includes if includes is not None else FULL_DOC_INCLUDES

        params = _build_includes_params(effective_includes)

        logger.debug(
            "Fetching document metadata",
            extra={"document_id": document_id, "includes": effective_includes},
        )

        result = await self._make_request(
            "GET",
            f"api/document/{document_id}",
            token=token,
            params=params,
        )
        return result if isinstance(result, dict) and result else None

    async def get_document_with_permissions(
        self,
        token: str,
        document_id: str,
        includes: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Fetches document and its permissions in a single request.

        Calls GET api/document/{id}/all?includes=...
        Returns: { "document": DocumentDto, "permission": DocPermissionContainer }.

        Includes применяются к вложенной части "document".
        По умолчанию — FULL_DOC_INCLUDES.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            includes: Include enum names. None → FULL_DOC_INCLUDES.

        Returns:
            Dict with keys "document" and "permission", or None on failure.
        """
        effective_includes = includes if includes is not None else FULL_DOC_INCLUDES

        params = _build_includes_params(effective_includes)

        logger.debug(
            "Fetching document with permissions",
            extra={"document_id": document_id, "includes": effective_includes},
        )

        result = await self._make_request(
            "GET",
            f"api/document/{document_id}/all",
            token=token,
            params=params,
        )
        return result if isinstance(result, dict) and result else None

    async def get_document_permissions(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches DocPermissionContainer for a document.

        Calls GET api/document/{id}/permission.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            DocPermissionContainer as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/permission", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_document_properties(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches extended document properties (DocumentPropertiesDto).

        Calls GET api/document/{id}/properties.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            DocumentPropertiesDto as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/properties", token=token
        )
        return result if isinstance(result, dict) and result else None

    # ── История и процессы ────────────────────────────────────────────────────

    async def get_document_history(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches document processing protocol (history v1).

        Calls GET api/document/{id}/history.
        Содержит журнал всех действий: кто, что, когда.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            List of DocumentHistoryDto dicts, empty list if no history.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/history", token=token
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            content = result.get("content") or result.get("items")
            if isinstance(content, list):
                return content
            logger.warning(
                "Unexpected history response shape",
                extra={"document_id": document_id, "keys": list(result.keys())},
            )
        return []

    async def get_document_history_v2(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches document processing protocol (history v2, preferred).

        Calls GET api/document/{id}/history/v2.
        Предпочтительная версия: более детальная структура событий.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            List of history event dicts, empty list if no history.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/history/v2", token=token
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            content = result.get("content") or result.get("items")
            if isinstance(content, list):
                return content
            logger.warning(
                "Unexpected history/v2 response shape",
                extra={"document_id": document_id, "keys": list(result.keys())},
            )
        return []

    async def get_process_activity(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches current BPMN process with active activities.

        Calls GET api/document/{id}/bpmn.
        Показывает текущий этап маршрута, активные activity в BPMN-схеме.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            BpmnProcessActivityDto as dict, or None if no active process.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/bpmn", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_tasks_and_projects(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches tasks and task-projects linked to a document.

        Calls GET api/document/{id}/task-task-project.
        Возвращает: { "tasks": [...], "taskProjects": [...] }

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            Dict with keys "tasks" and "taskProjects", or None on failure.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/task-task-project", token=token
        )
        return result if isinstance(result, dict) and result else None

    # ── Контроль ──────────────────────────────────────────────────────────────

    async def get_document_control(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Fetches current control record (ControlDto) for a document.

        Calls GET api/document/{documentId}/control.
        Если документ не на контроле — контроллер возвращает пустой ControlDto.
        Метод возвращает None в этом случае для единообразия.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            ControlDto as dict if control exists, None otherwise.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/control", token=token
        )
        if isinstance(result, dict) and result:
            return result
        return None

    async def set_document_control(
        self,
        token: str,
        document_id: str,
        control_request: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Sets a document on control.

        Calls POST api/document/{docId}/control.
        Минимальный состав:
            { "controlTypeId": UUID, "dateControlEnd": "YYYY-MM-DDTHH:MM:SS" }

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            control_request: ControlRequest payload dict.

        Returns:
            Created ControlDto as dict, or None on failure.
        """
        result = await self._make_request(
            "POST",
            f"api/document/{document_id}/control",
            token=token,
            json=control_request,
        )
        return result if isinstance(result, dict) and result else None

    async def remove_document_control(self, token: str, document_id: str) -> bool:
        """Removes control mark from a document (снять с контроля).

        Calls PUT api/document/control with body { "id": document_id }.
        Возвращает 204 No Content при успехе.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            True on success, False on failure.
        """
        try:
            await self._make_request(
                "PUT",
                "api/document/control",
                token=token,
                json={"id": document_id},
                is_json_response=False,
            )
            return True
        except Exception:
            logger.error(
                "Failed to remove control",
                exc_info=True,
                extra={"document_id": document_id},
            )
            return False

    async def delete_document_control(self, token: str, document_id: str) -> bool:
        """Deletes control record for a document.

        Calls DELETE api/document/{docId}/control.
        Возвращает 204 No Content при успехе.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            True on success, False on failure.
        """
        try:
            await self._make_request(
                "DELETE",
                f"api/document/{document_id}/control",
                token=token,
                is_json_response=False,
            )
            return True
        except Exception:
            logger.error(
                "Failed to delete control",
                exc_info=True,
                extra={"document_id": document_id},
            )
            return False

    # ── Адресаты ──────────────────────────────────────────────────────────────

    async def get_document_recipients(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches list of document recipients.

        Calls GET api/document/{id}/recipient.
        Возвращает список адресатов с информацией о доставке.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            List of DocumentRecipientDto dicts, empty list if none.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/recipient", token=token
        )
        return result if isinstance(result, list) else []

    async def get_contract_responsible(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches responsible employees for a contract document.

        Calls GET api/document/{documentId}/responsible.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            List of ContractResponsibleDto dicts, empty list if none.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/responsible", token=token
        )
        return result if isinstance(result, list) else []

    # ── Номенклатуры дел ──────────────────────────────────────────────────────

    async def get_nomenclature_affairs(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches nomenclature affairs linked to a document.

        Calls GET api/document/{id}/nomenclature-affair.
        Показывает дела (папки архива), в которые списан документ.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            List of NomenclatureAffairDto dicts, empty list if none.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/nomenclature-affair", token=token
        )
        return result if isinstance(result, list) else []

    async def get_repeat_identical_appeals(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches repeat and identical appeals linked to a document.

        Calls GET api/document/{documentId}/repeat-identical.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            List of RepeatIdenticalAppealDto dicts, empty list if none.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/repeat-identical", token=token
        )
        return result if isinstance(result, list) else []

    # ── Версии ────────────────────────────────────────────────────────────────

    async def get_document_versions(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches all versions of a document.

        Calls GET api/document/{id}/version.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            List of DocumentVersionDto dicts, empty list if none.
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/version", token=token
        )
        return result if isinstance(result, list) else []

    # ── Жизненный цикл ────────────────────────────────────────────────────────

    async def start_document(self, token: str, document_id: str) -> bool:
        """Starts the document routing process.

        Calls POST api/document/start with body { "id": document_id }.
        Возвращает 204 No Content при успехе.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            True on success, False on failure.
        """
        try:
            await self._make_request(
                "POST",
                "api/document/start",
                token=token,
                json={"id": document_id},
                is_json_response=False,
            )
            logger.info(
                "Document started successfully",
                extra={"document_id": document_id},
            )
            return True
        except Exception:
            logger.error(
                "Failed to start document",
                exc_info=True,
                extra={"document_id": document_id},
            )
            return False

    async def cancel_document(
        self,
        token: str,
        document_id: str,
        comment: str | None = None,
    ) -> bool:
        """Cancels (annuls) a document.

        Calls POST api/document/cancel.
        Возвращает 204 No Content при успехе.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            comment: Mandatory reason comment for audit trail.

        Returns:
            True on success, False on failure.
        """
        payload: dict[str, Any] = {"id": document_id}
        if comment:
            payload["comment"] = comment.strip()

        try:
            await self._make_request(
                "POST",
                "api/document/cancel",
                token=token,
                json=payload,
                is_json_response=False,
            )
            logger.info(
                "Document cancelled successfully",
                extra={"document_id": document_id},
            )
            return True
        except Exception:
            logger.error(
                "Failed to cancel document",
                exc_info=True,
                extra={"document_id": document_id},
            )
            return False

    async def execute_document_operations(
        self,
        token: str,
        document_id: str,
        operations: list[dict[str, Any]],
    ) -> bool:
        """Executes a list of operations on a document.

        Calls POST api/document/{id}/execute with body List<DocOperation>.
        Операции: подписать, согласовать, отклонить, ознакомиться и др.
        Возвращает 204 No Content при успехе.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            operations: List of DocOperation dicts, e.g.
                        [{"operationType": "SIGN", "comment": "Согласован"}]

        Returns:
            True on success, False on failure.
        """
        if not operations:
            logger.warning(
                "execute_document_operations called with empty operations list",
                extra={"document_id": document_id},
            )
            return False

        try:
            await self._make_request(
                "POST",
                f"api/document/{document_id}/execute",
                token=token,
                json=operations,
                is_json_response=False,
            )
            logger.info(
                "Document operations executed",
                extra={
                    "document_id": document_id,
                    "operation_count": len(operations),
                },
            )
            return True
        except Exception:
            logger.error(
                "Failed to execute document operations",
                exc_info=True,
                extra={"document_id": document_id},
            )
            return False

    # ── Статистика ────────────────────────────────────────────────────────────

    async def get_stat_user_executor(self, token: str) -> dict[str, Any] | None:
        """Fetches document execution statistics for the current user.

        Calls GET api/document/stat/user-executor.
        Содержит количество документов по статусам исполнения.

        Args:
            token: JWT bearer token.

        Returns:
            ExecutionDocumentStatCount as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", "api/document/stat/user-executor", token=token
        )
        return result if isinstance(result, dict) else None

    async def get_stat_user_control(self, token: str) -> dict[str, Any] | None:
        """Fetches document control statistics for the current user.

        Calls GET api/document/stat/user-control.
        Количество документов на контроле по состояниям.

        Args:
            token: JWT bearer token.

        Returns:
            ExecutionDocumentStatCount as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", "api/document/stat/user-control", token=token
        )
        return result if isinstance(result, dict) else None

    async def get_stat_user_author(self, token: str) -> dict[str, Any] | None:
        """Fetches document authoring statistics for the current user.

        Calls GET api/document/stat/user-author.
        Количество документов, созданных текущим пользователем, по статусам.

        Args:
            token: JWT bearer token.

        Returns:
            ExecutionDocumentStatCount as dict, or None on failure.
        """
        result = await self._make_request(
            "GET", "api/document/stat/user-author", token=token
        )
        return result if isinstance(result, dict) else None
