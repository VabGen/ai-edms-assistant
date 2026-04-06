# mcp-server/clients/department_client.py
"""EDMS Department HTTP Client. Перенесён из edms_ai_assistant/clients/department_client.py."""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from .base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class DepartmentClient(EdmsHttpClient):
    """Клиент для работы с подразделениями EDMS."""

    async def find_by_name(
        self, token: str, department_name: str
    ) -> dict[str, Any] | None:
        """FTS-поиск отдела по имени: GET api/department/fts-name."""
        try:
            result = await self._make_request(
                "GET", "api/department/fts-name",
                token=token, params={"fts": department_name},
            )
            if result and isinstance(result, dict):
                return result
            return None
        except Exception as e:
            logger.error("Error searching department '%s': %s", department_name, e)
            return None

    async def get_employees_by_department_id(
        self, token: str, department_id: UUID
    ) -> list[dict[str, Any]]:
        """Сотрудники отдела: GET api/department/{id}/employees/all."""
        try:
            result = await self._make_request(
                "GET", f"api/department/{department_id}/employees/all", token=token
            )
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error("Error fetching employees for department %s: %s", department_id, e)
            return []


# mcp-server/clients/group_client.py
class GroupClient(EdmsHttpClient):
    """Клиент для работы с группами сотрудников."""

    async def find_by_name(self, token: str, group_name: str) -> dict[str, Any] | None:
        """FTS-поиск группы: GET api/group/fts-name."""
        try:
            result = await self._make_request(
                "GET", "api/group/fts-name",
                token=token, params={"fts": group_name},
            )
            if result and isinstance(result, dict):
                return result
            return None
        except Exception as e:
            logger.error("Error searching group '%s': %s", group_name, e)
            return None

    async def get_employees_by_group_ids(
        self, token: str, group_ids: list[UUID]
    ) -> list[dict[str, Any]]:
        """Сотрудники групп: GET api/group/employee/all?ids=..."""
        if not group_ids:
            return []
        endpoint = f"api/group/employee/all?ids={group_ids[0]}"
        for group_id in group_ids[1:]:
            endpoint += f"&ids={group_id}"
        try:
            result = await self._make_request("GET", endpoint, token=token)
            if isinstance(result, list):
                employees = []
                for item in result:
                    if isinstance(item, dict) and "employee" in item:
                        employees.append(item["employee"])
                return employees
            return []
        except Exception as e:
            logger.error("Error fetching employees for groups: %s", e)
            return []