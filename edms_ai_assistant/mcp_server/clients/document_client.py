# mcp-server/clients/document_client.py
"""
EDMS Document HTTP Client.

Перенесён из edms_ai_assistant/clients/document_client.py.
Полный набор методов для работы с документами через Java EDMS API.
"""
from __future__ import annotations

import logging
from abc import abstractmethod
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


def _build_includes_params(includes: list[str]) -> dict[str, list[str]]:
    """Конвертирует список includes в параметры Spring multivalue query."""
    return {"includes": includes}


class DocumentClient(EdmsHttpClient):
    """
    Конкретный async HTTP-клиент для EDMS Document API.

    Реализует все методы DocumentController.java.
    """

    async def search_documents(
        self,
        token: str,
        doc_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
        includes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Поиск документов через GET api/document."""
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

        result = await self._make_request("GET", "api/document", token=token, params=params)
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                return content
        if isinstance(result, list):
            return result
        return []

    async def get_document_metadata(
        self,
        token: str,
        document_id: str,
        includes: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Получить метаданные документа по UUID: GET api/document/{id}."""
        effective_includes = includes if includes is not None else FULL_DOC_INCLUDES
        params = _build_includes_params(effective_includes)
        result = await self._make_request(
            "GET", f"api/document/{document_id}", token=token, params=params
        )
        return result if isinstance(result, dict) and result else None

    async def get_document_with_permissions(
        self,
        token: str,
        document_id: str,
        includes: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Получить документ + права: GET api/document/{id}/all."""
        effective_includes = includes if includes is not None else FULL_DOC_INCLUDES
        params = _build_includes_params(effective_includes)
        result = await self._make_request(
            "GET", f"api/document/{document_id}/all", token=token, params=params
        )
        return result if isinstance(result, dict) and result else None

    async def get_document_history(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """История документа v1: GET api/document/{id}/history."""
        result = await self._make_request(
            "GET", f"api/document/{document_id}/history", token=token
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            content = result.get("content") or result.get("items")
            if isinstance(content, list):
                return content
        return []

    async def get_document_history_v2(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """История документа v2 (предпочтительная): GET api/document/{id}/history/v2."""
        result = await self._make_request(
            "GET", f"api/document/{document_id}/history/v2", token=token
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            content = result.get("content") or result.get("items")
            if isinstance(content, list):
                return content
        return []

    async def get_tasks_and_projects(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Поручения и проекты: GET api/document/{id}/task-task-project."""
        result = await self._make_request(
            "GET", f"api/document/{document_id}/task-task-project", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_document_control(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """Контроль документа: GET api/document/{documentId}/control."""
        result = await self._make_request(
            "GET", f"api/document/{document_id}/control", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def set_document_control(
        self,
        token: str,
        document_id: str,
        control_request: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Поставить на контроль: POST api/document/{docId}/control."""
        result = await self._make_request(
            "POST", f"api/document/{document_id}/control",
            token=token, json=control_request,
        )
        return result if isinstance(result, dict) and result else None

    async def remove_document_control(self, token: str, document_id: str) -> bool:
        """Снять с контроля: PUT api/document/control."""
        try:
            await self._make_request(
                "PUT", "api/document/control",
                token=token, json={"id": document_id}, is_json_response=False,
            )
            return True
        except Exception:
            logger.error("Failed to remove control for %s", document_id, exc_info=True)
            return False

    async def get_document_recipients(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Адресаты документа: GET api/document/{id}/recipient."""
        result = await self._make_request(
            "GET", f"api/document/{document_id}/recipient", token=token
        )
        return result if isinstance(result, list) else []

    async def get_document_versions(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Версии документа: GET api/document/{id}/version."""
        result = await self._make_request(
            "GET", f"api/document/{document_id}/version", token=token
        )
        return result if isinstance(result, list) else []

    async def start_document(self, token: str, document_id: str) -> bool:
        """Запустить маршрут: POST api/document/start."""
        try:
            await self._make_request(
                "POST", "api/document/start",
                token=token, json={"id": document_id}, is_json_response=False,
            )
            return True
        except Exception:
            logger.error("Failed to start document %s", document_id, exc_info=True)
            return False

    async def cancel_document(
        self, token: str, document_id: str, comment: str | None = None
    ) -> bool:
        """Аннулировать документ: POST api/document/cancel."""
        payload: dict[str, Any] = {"id": document_id}
        if comment:
            payload["comment"] = comment.strip()
        try:
            await self._make_request(
                "POST", "api/document/cancel",
                token=token, json=payload, is_json_response=False,
            )
            return True
        except Exception:
            logger.error("Failed to cancel document %s", document_id, exc_info=True)
            return False

    async def execute_document_operations(
        self,
        token: str,
        document_id: str,
        operations: list[dict[str, Any]],
    ) -> bool:
        """Выполнить операции над документом: POST api/document/{id}/execute."""
        if not operations:
            return False
        try:
            await self._make_request(
                "POST", f"api/document/{document_id}/execute",
                token=token, json=operations, is_json_response=False,
            )
            return True
        except Exception:
            logger.error("Failed to execute operations on %s", document_id, exc_info=True)
            return False

    async def get_stat_user_executor(self, token: str) -> dict[str, Any] | None:
        """Статистика исполнения: GET api/document/stat/user-executor."""
        result = await self._make_request("GET", "api/document/stat/user-executor", token=token)
        return result if isinstance(result, dict) else None

    async def get_stat_user_control(self, token: str) -> dict[str, Any] | None:
        """Статистика контроля: GET api/document/stat/user-control."""
        result = await self._make_request("GET", "api/document/stat/user-control", token=token)
        return result if isinstance(result, dict) else None

    async def get_stat_user_author(self, token: str) -> dict[str, Any] | None:
        """Статистика авторства: GET api/document/stat/user-author."""
        result = await self._make_request("GET", "api/document/stat/user-author", token=token)
        return result if isinstance(result, dict) else None