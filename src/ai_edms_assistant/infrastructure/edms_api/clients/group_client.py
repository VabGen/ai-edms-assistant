# src/ai_edms_assistant/infrastructure/edms_api/clients/group_client.py
from __future__ import annotations

import structlog
from typing import Any
from uuid import UUID

from ..http_client import EdmsHttpClient

logger = structlog.get_logger(__name__)


class EdmsGroupClient(EdmsHttpClient):
    """
    Low-level async client for EDMS /api/group/* endpoints.

    Groups are permission/role groups in EDMS — not calendar groups.
    """

    async def find_by_fts(
        self,
        name: str,
        token: str,
    ) -> dict[str, Any] | None:
        """
        GET /api/group/fts-name?fts={name}

        Returns:
            Single raw GroupDto dict or None.
        """
        try:
            data = await self._make_request(
                "GET",
                "api/group/fts-name",
                token=token,
                params={"fts": name},
            )
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning("group_fts_failed", name=name, error=str(exc))
            return None

    async def get_employees(
        self,
        group_ids: list[UUID],
        token: str,
    ) -> list[dict[str, Any]]:
        """
        GET /api/group/employee/all?ids={id1}&ids={id2}...

        The EDMS API accepts repeated `ids` query params.
        Each item in the response has shape {employee: EmployeeDto, ...}.
        """
        if not group_ids:
            return []
        params = [("ids", str(gid)) for gid in group_ids]
        try:
            data = await self._make_request(
                "GET",
                "api/group/employee/all",
                token=token,
                params=params,
            )
            if not isinstance(data, list):
                return []
            # Unwrap {employee: dto} wrapper
            return [
                item["employee"]
                for item in data
                if isinstance(item, dict) and item.get("employee")
            ]
        except Exception as exc:
            logger.error("group_employees_failed", error=str(exc))
            return []
