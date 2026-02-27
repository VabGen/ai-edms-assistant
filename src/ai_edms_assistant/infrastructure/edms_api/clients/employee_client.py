# src/ai_edms_assistant/infrastructure/edms_api/clients/employee_client.py
"""Low-level typed HTTP client for EDMS /api/employee/* endpoints.

Responsibility: HTTP transport only.
    - Knows API paths and parameter names.
    - Returns raw dicts (no domain mapping here).
    - Uses composition over inheritance — wraps EdmsHttpClient.

Architecture:
    Infrastructure → EdmsHttpClient (shared transport)
    Returns: raw dict | None
    Consumed by: EdmsEmployeeRepository (via EmployeeMapper)
"""

from __future__ import annotations

import structlog
from typing import Any
from uuid import UUID

from ..http_client import EdmsHttpClient
from ....domain.value_objects.filters import EmployeeFilter, UserActionFilter
from ....domain.repositories.base import PageRequest

logger = structlog.get_logger(__name__)


class EdmsEmployeeClient:
    """Typed async client for EDMS /api/employee/* endpoints.

    Uses composition: wraps ``EdmsHttpClient`` (shared transport).
    Returns raw API dicts — domain mapping is EmployeeMapper's responsibility.

    All methods correspond 1:1 to Java ``EmployeeController`` endpoints.

    Attributes:
        _http: Shared async EdmsHttpClient instance.
    """

    def __init__(self, http_client: EdmsHttpClient) -> None:
        """Initialize with shared HTTP transport.

        Args:
            http_client: Configured EdmsHttpClient instance (shared, injected).
        """
        self._http = http_client

    async def get_by_id(
        self,
        employee_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch single employee by UUID.

        GET /api/employee/{id}

        Returns nested ``post`` and ``department`` objects when the employee
        has a linked post/department in the EDMS database.

        Args:
            employee_id: Employee UUID.
            token: JWT bearer token.
            organization_id: Optional org scope for multi-tenant deployments.

        Returns:
            Raw EmployeeDto dict or None on 404.

        Raises:
            Exception: On non-404 HTTP errors (propagated to repository).
        """
        params: dict[str, Any] = {}
        if organization_id:
            params["organizationId"] = organization_id

        try:
            return await self._http._make_request(
                "GET", f"api/employee/{employee_id}", token=token, params=params
            )
        except Exception as exc:
            logger.error(
                "employee_client_get_by_id_failed",
                employee_id=str(employee_id),
                error=str(exc),
            )
            raise

    async def find_by_fts(
        self,
        query: str,
        token: str,
        organization_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Full-text search by last name — returns single best match.

        GET /api/employee/fts-lastname?fts={query}

        Java: ``employeeService.searchFtsTopByLastName(fts, user)``
        Uses PostgreSQL ``ts_rank_cd`` + ``similarity()`` scoring.

        NOTE: Response always has ``post=null``, ``department=null``.
        Callers must enrich via ``get_by_id()`` when full card is needed.

        Args:
            query: Last name query string (partial and fuzzy OK).
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            Single raw EmployeeDto dict or None when no match (404).
        """
        params: dict[str, Any] = {"fts": query.strip()}
        if organization_id:
            params["organizationId"] = organization_id

        try:
            return await self._http._make_request(
                "GET", "api/employee/fts-lastname", token=token, params=params
            )
        except Exception as exc:
            logger.warning(
                "employee_client_fts_no_match",
                query=query,
                error=str(exc),
            )
            return None

    async def search(
        self,
        filters: EmployeeFilter,
        token: str,
        pagination: PageRequest | None = None,
    ) -> dict[str, Any]:
        """Multi-field employee search.

        POST /api/employee/search

        Java default sort: ``lastName,ASC`` (``@PageableDefault``).
        Returns Spring ``SliceDto<EmployeeDto>`` with ``content``, ``number``,
        ``size``, ``last``, ``totalElements``.

        Args:
            filters: Serialized EmployeeFilter (``as_api_payload()`` called here).
            token: JWT bearer token.
            pagination: Page/size/sort params. Default: size=10, sort=lastName,ASC.

        Returns:
            Raw Spring Page dict or empty dict on error.
        """
        pag = pagination or PageRequest(size=10, sort="lastName,ASC")
        data = await self._http._make_request(
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
        """Fetch employee activity log.

        GET /api/employee/{id}/user-action

        Args:
            employee_id: Employee UUID.
            filters: UserActionFilter — ``start`` and ``end`` required.
            token: JWT bearer token.

        Returns:
            List of raw activity dicts.

        Raises:
            FilterValidationError: When ``start`` or ``end`` is missing.
        """
        filters.validate_for_dashboard()
        data = await self._http._make_request(
            "GET",
            f"api/employee/{employee_id}/user-action",
            token=token,
            params=filters.as_api_params(),
        )
        return data if isinstance(data, list) else []

    async def get_by_email(
        self,
        email: str,
        token: str,
    ) -> dict[str, Any] | None:
        """Find employee by corporate email.

        GET /api/employee?email={email}&active=true

        Args:
            email: Corporate email address.
            token: JWT bearer token.

        Returns:
            First matching raw EmployeeDto dict or None.
        """
        data = await self._http._make_request(
            "GET",
            "api/employee",
            token=token,
            params={"email": email, "active": True},
        )
        items = (
            data.get("content", [])
            if isinstance(data, dict)
            else (data if isinstance(data, list) else [])
        )
        return items[0] if items else None

    async def get_by_getu_id(
        self,
        getu_id: str,
        token: str,
    ) -> dict[str, Any] | None:
        """Find employee by GETU external identifier.

        GET /api/employee?uId={getu_id}&fired=false

        Args:
            getu_id: GETU system identifier string.
            token: JWT bearer token.

        Returns:
            First matching raw EmployeeDto dict or None.
        """
        data = await self._http._make_request(
            "GET",
            "api/employee",
            token=token,
            params={"uId": getu_id, "fired": False},
        )
        items = (
            data.get("content", [])
            if isinstance(data, dict)
            else (data if isinstance(data, list) else [])
        )
        return items[0] if items else None
