# src/ai_edms_assistant/infrastructure/edms_api/clients/department_client.py
"""EDMS Department HTTP Client — /api/department/* endpoints."""

from __future__ import annotations

import structlog
from typing import Any
from uuid import UUID

from ..http_client import EdmsHttpClient

logger = structlog.get_logger(__name__)


class EdmsDepartmentClient(EdmsHttpClient):
    """
    Low-level async client for EDMS /api/department/* endpoints.

    Returns raw dicts. Domain mapping is in employee_mapper.py where relevant.
    """

    async def find_by_fts(
        self,
        name: str,
        token: str,
    ) -> dict[str, Any] | None:
        """
        GET /api/department/fts-name?fts={name}

        Returns:
            Single raw DepartmentDto dict or None when not found.
        """
        try:
            data = await self._make_request(
                "GET",
                "api/department/fts-name",
                token=token,
                params={"fts": name},
            )
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning("department_fts_failed", name=name, error=str(exc))
            return None

    async def get_employees(
        self,
        department_id: UUID,
        token: str,
    ) -> list[dict[str, Any]]:
        """GET /api/department/{id}/employees/all"""
        try:
            data = await self._make_request(
                "GET",
                f"api/department/{department_id}/employees/all",
                token=token,
            )
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.error(
                "department_employees_failed",
                dept_id=str(department_id),
                error=str(exc),
            )
            return []
