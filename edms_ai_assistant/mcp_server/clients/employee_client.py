# mcp-server/clients/employee_client.py
"""
EDMS Employee HTTP Client.
Перенесён из edms_ai_assistant/clients/employee_client.py.
"""
from __future__ import annotations

import logging
from typing import Any

from .base_client import EdmsHttpClient

logger = logging.getLogger(__name__)

_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 20
_DEFAULT_INCLUDES: list[str] = ["POST", "DEPARTMENT"]


def _ensure_includes(employee_filter: dict[str, Any]) -> dict[str, Any]:
    if not employee_filter.get("includes"):
        return {**employee_filter, "includes": _DEFAULT_INCLUDES}
    return employee_filter


def _extract_slice_content(result: Any, endpoint: str = "api/employee") -> list[dict[str, Any]]:
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            return content
        logger.warning("Unexpected Slice response from %s", endpoint)
        return []
    if isinstance(result, list):
        return result
    return []


class EmployeeClient(EdmsHttpClient):
    """Конкретный async HTTP-клиент для EDMS Employee API."""

    async def search_employees(
        self,
        token: str,
        employee_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Поиск сотрудников: GET api/employee."""
        effective_filter = _ensure_includes(employee_filter or {})
        effective_pageable: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if pageable:
            effective_pageable.update(pageable)
        params = {**effective_filter, **effective_pageable}
        result = await self._make_request("GET", "api/employee", token=token, params=params)
        return _extract_slice_content(result, "GET api/employee")

    async def search_employees_post(
        self,
        token: str,
        employee_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Поиск сотрудников: POST api/employee/search."""
        effective_filter = _ensure_includes(employee_filter or {})
        effective_pageable: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if pageable:
            effective_pageable.update(pageable)
        result = await self._make_request(
            "POST", "api/employee/search",
            token=token, params=effective_pageable, json=effective_filter,
        )
        return _extract_slice_content(result, "POST api/employee/search")

    async def get_employee(self, token: str, employee_id: str) -> dict[str, Any] | None:
        """Получить сотрудника по UUID: GET api/employee/{id}."""
        result = await self._make_request("GET", f"api/employee/{employee_id}", token=token)
        return result if isinstance(result, dict) and result else None

    async def get_current_user(self, token: str) -> dict[str, Any] | None:
        """Текущий пользователь: GET api/employee/me."""
        result = await self._make_request("GET", "api/employee/me", token=token)
        return result if isinstance(result, dict) and result else None

    async def find_by_last_name_fts(
        self, token: str, last_name: str
    ) -> dict[str, Any] | None:
        """FTS-поиск по фамилии, top-1: GET api/employee/fts-lastname."""
        try:
            result = await self._make_request(
                "GET", "api/employee/fts-lastname",
                token=token, params={"fts": last_name.strip()},
            )
            return result if isinstance(result, dict) and result else None
        except Exception:
            logger.warning("FTS search failed for '%s'", last_name, exc_info=True)
            return None