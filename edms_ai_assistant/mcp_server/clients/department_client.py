# edms_ai_assistant/clients/department_client.py
import logging
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from .base_client import EdmsBaseClient, EdmsHttpClient

logger = logging.getLogger(__name__)


class BaseDepartmentClient(EdmsBaseClient, ABC):

    @abstractmethod
    async def find_by_name(
        self, token: str, department_name: str
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def get_employees_by_department_id(
        self, token: str, department_id: UUID
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class DepartmentClient(BaseDepartmentClient, EdmsHttpClient):

    async def find_by_name(
        self, token: str, department_name: str
    ) -> dict[str, Any] | None:
        endpoint = "api/department/fts-name"
        params = {"fts": department_name}

        try:
            result = await self._make_request(
                "GET", endpoint, token=token, params=params
            )
            if result and isinstance(result, dict):
                logger.info(f"Found department: {result.get('name', 'Unknown')}")
                return result
            return None
        except Exception as e:
            logger.error(f"Error searching department '{department_name}': {e}")
            return None

    async def get_employees_by_department_id(
        self, token: str, department_id: UUID
    ) -> list[dict[str, Any]]:
        endpoint = f"api/department/{department_id}/employees/all"

        try:
            result = await self._make_request("GET", endpoint, token=token)
            if isinstance(result, list):
                logger.info(
                    f"Found {len(result)} employees in department {department_id}"
                )
                return result
            return []
        except Exception as e:
            logger.error(
                f"Error fetching employees for department {department_id}: {e}"
            )
            return []
