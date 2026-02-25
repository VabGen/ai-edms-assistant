# src/ai_edms_assistant/infrastructure/edms_api/clients/employee_client.py
from __future__ import annotations

import structlog
from typing import Any
from uuid import UUID

from ..http_client import EdmsHttpClient
from ....domain.value_objects.filters import EmployeeFilter, UserActionFilter
from ....domain.repositories.base import PageRequest

logger = structlog.get_logger(__name__)


class EdmsEmployeeClient(EdmsHttpClient):
    """
    Low-level async client for EDMS /api/employee/* endpoints.

    Prefers POST /api/employee/search for complex queries.
    GET-based methods exist only for single-entity lookups.

    Returns raw dicts — domain mapping is in employee_mapper.py.
    """

    async def get_by_id(
        self,
        employee_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        GET /api/employee/{id}

        Returns:
            Raw EmployeeDto dict or None when 404.
        """
        params: dict[str, Any] = {}
        if organization_id:
            params["organizationId"] = organization_id
        try:
            return await self._make_request(
                "GET", f"api/employee/{employee_id}", token=token, params=params
            )
        except Exception as exc:
            logger.error(
                "employee_get_by_id_failed", emp_id=str(employee_id), error=str(exc)
            )
            raise

    async def find_by_fts(
        self,
        query: str,
        token: str,
        organization_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        GET /api/employee/fts-lastname?fts={query}

        Returns the single best PostgreSQL FTS+similarity() match.
        searchFtsTopByLastName.

        Returns:
            Single raw EmployeeDto dict or None when no match.
        """
        params: dict[str, Any] = {"fts": query.strip()}
        if organization_id:
            params["organizationId"] = organization_id
        try:
            return await self._make_request(
                "GET", "api/employee/fts-lastname", token=token, params=params
            )
        except Exception as exc:
            logger.warning("employee_fts_no_match", query=query, error=str(exc))
            return None

    async def search(
        self,
        filters: EmployeeFilter,
        token: str,
        pagination: PageRequest | None = None,
    ) -> dict[str, Any]:
        """
        POST /api/employee/search

        Default sort: lastName,ASC (@PageableDefault).

        Returns:
            Spring Page: {content, number, size, last, totalElements}.
        """
        pag = pagination or PageRequest(size=10, sort="lastName,ASC")
        data = await self._make_request(
            "POST",
            "api/employee/search",
            token=token,
            params=pag.as_params(),
            json=filters.as_api_payload(),
        )
        return data or {}

    async def get_activity(
        self,
        employee_id: UUID,
        filters: UserActionFilter,
        token: str,
    ) -> list[dict[str, Any]]:
        """
        GET /api/employee/{id}/user-action

        Raises:
            FilterValidationError: when start/end are missing (Dashboard constraint).
        """
        filters.validate_for_dashboard()
        data = await self._make_request(
            "GET",
            f"api/employee/{employee_id}/user-action",
            token=token,
            params=filters.as_api_params(),
        )
        return data if isinstance(data, list) else []
