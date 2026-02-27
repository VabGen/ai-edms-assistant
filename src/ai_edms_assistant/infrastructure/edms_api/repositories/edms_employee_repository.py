# src/ai_edms_assistant/infrastructure/edms_api/repositories/edms_employee_repository.py
"""EDMS Employee Repository — REST API implementation."""

from __future__ import annotations

import structlog
from uuid import UUID
from typing import Any

from ..http_client import EdmsHttpClient
from ..mappers.employee_mapper import EmployeeMapper
from ....domain.entities.employee import Employee
from ....domain.value_objects.filters import EmployeeFilter, UserActionFilter
from ....domain.repositories.employee_repository import AbstractEmployeeRepository
from ....domain.repositories.base import Page, PageRequest

logger = structlog.get_logger(__name__)


class EdmsEmployeeRepository(AbstractEmployeeRepository):
    """Concrete implementation of AbstractEmployeeRepository.

    HTTP calls: EdmsHttpClient._make_request() — shared transport.
    Mapping/normalization: EmployeeMapper — single source of truth.
    No normalization logic in this class.

    Attributes:
        _http: Shared EdmsHttpClient instance.
    """

    def __init__(self, http_client: EdmsHttpClient) -> None:
        """Initialize with shared HTTP client.

        Args:
            http_client: Configured EdmsHttpClient instance (injected).
        """
        self._http = http_client

    # ------------------------------------------------------------------
    # Private HTTP helpers — thin wrappers, return raw dicts
    # ------------------------------------------------------------------

    async def _get(self, path: str, token: str, params: dict | None = None) -> Any:
        """HTTP GET helper.

        Args:
            path: API path relative to base URL.
            token: JWT bearer token.
            params: Optional query parameters.

        Returns:
            Parsed JSON response or None.
        """
        return await self._http._make_request(
            "GET", path, token=token, params=params or {}
        )

    async def _post(
        self, path: str, token: str, params: dict | None = None, json: Any = None
    ) -> Any:
        """HTTP POST helper.

        Args:
            path: API path relative to base URL.
            token: JWT bearer token.
            params: Optional query parameters.
            json: Request body (serialized to JSON).

        Returns:
            Parsed JSON response or None.
        """
        return await self._http._make_request(
            "POST", path, token=token, params=params or {}, json=json
        )

    # ------------------------------------------------------------------
    # AbstractRepository — base lookups
    # ------------------------------------------------------------------

    async def get_by_id(
        self,
        entity_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> Employee | None:
        """Fetch full employee card by UUID.

        GET /api/employee/{id}
        Returns nested post/department objects.
        EmployeeMapper.normalize() extracts them to flat fields.

        Args:
            entity_id: Employee UUID.
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            Fully enriched Employee entity or None.
        """
        params: dict[str, Any] = {}
        if organization_id:
            params["organizationId"] = organization_id

        try:
            raw = await self._get(f"api/employee/{entity_id}", token, params)
            return EmployeeMapper.to_employee(raw)
        except Exception as exc:
            logger.error(
                "edms_employee_get_by_id_failed",
                employee_id=str(entity_id),
                error=str(exc),
            )
            raise

    async def get_by_ids(
        self,
        entity_ids: list[UUID],
        token: str,
        organization_id: str | None = None,
    ) -> list[Employee]:
        """Batch fetch employees by UUID list via search.

        Args:
            entity_ids: List of employee UUIDs.
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            List of mapped Employee entities.
        """
        if not entity_ids:
            return []
        page = await self.search(
            filters=EmployeeFilter(ids=list(entity_ids), org_id=organization_id),
            token=token,
            pagination=PageRequest(size=len(entity_ids)),
        )
        return page.items

    async def find_page(
        self,
        token: str,
        organization_id: str | None = None,
        pagination: PageRequest | None = None,
    ) -> Page[Employee]:
        """Fetch paginated active employee list.

        Args:
            token: JWT bearer token.
            organization_id: Optional org scope.
            pagination: Page/size/sort params.

        Returns:
            Page[Employee].
        """
        return await self.search(
            filters=EmployeeFilter(active=True, org_id=organization_id),
            token=token,
            pagination=pagination,
        )

    # ------------------------------------------------------------------
    # AbstractEmployeeRepository — domain-specific operations
    # ------------------------------------------------------------------

    async def find_by_name_fts(
        self,
        query: str,
        token: str,
        organization_id: str | None = None,
    ) -> Employee | None:
        """Search by last name via FTS with transparent post/dept enrichment.

        Two-step process (invisible to callers):
            Step 1 — GET /api/employee/fts-lastname?fts={query}
                      Best PostgreSQL similarity() match.
                      post=null, department=null always here.
            Step 2 — GET /api/employee/{id}
                      Full card. EmployeeMapper.normalize() extracts
                      post.postName → post_name,
                      department.name → department_name.

        On Step 2 failure: Step 1 result returned (graceful degradation).

        Args:
            query: Last name query string (partial, fuzzy OK).
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            Enriched Employee or None when not found.
        """
        params: dict[str, Any] = {"fts": query.strip()}
        if organization_id:
            params["organizationId"] = organization_id

        try:
            raw_fts = await self._get("api/employee/fts-lastname", token, params)
        except Exception as exc:
            logger.warning("edms_employee_fts_not_found", query=query, error=str(exc))
            return None

        employee = EmployeeMapper.to_employee(raw_fts)
        if employee is None:
            return None

        try:
            enriched = await self.get_by_id(
                entity_id=employee.id,
                token=token,
                organization_id=organization_id,
            )
            if enriched:
                logger.debug(
                    "edms_employee_fts_enriched",
                    employee_id=str(employee.id),
                    post_name=enriched.post_name,
                    department_name=enriched.department_name,
                )
                return enriched
        except Exception as exc:
            logger.warning(
                "edms_employee_fts_enrichment_failed",
                employee_id=str(employee.id),
                error=str(exc),
            )

        return employee

    async def search(
        self,
        filters: EmployeeFilter,
        token: str,
        organization_id: str | None = None,
        pagination: PageRequest | None = None,
    ) -> Page[Employee]:
        """Multi-field employee search via POST /api/employee/search.

        Args:
            filters: EmployeeFilter value object.
            token: JWT bearer token.
            organization_id: Overrides filters.org_id when provided.
            pagination: Default: size=10, sort=lastName,ASC.

        Returns:
            Page[Employee].
        """
        pag = pagination or PageRequest(size=10, sort="lastName,ASC")

        effective = (
            filters.model_copy(update={"org_id": organization_id})
            if organization_id and not filters.org_id
            else filters
        )

        data = (
            await self._post(
                "api/employee/search",
                token,
                params=pag.as_params(),
                json=effective.as_api_payload(),
            )
            or {}
        )

        return Page(
            items=EmployeeMapper.to_employee_list(data.get("content", [])),
            page=data.get("number", pag.page),
            size=data.get("size", pag.size),
            has_next=not data.get("last", True),
            total=data.get("totalElements"),
        )

    async def get_by_email(
        self,
        email: str,
        token: str,
    ) -> Employee | None:
        """Find employee by corporate email.

        GET /api/employee?email={email}&active=true

        Args:
            email: Corporate email address.
            token: JWT bearer token.

        Returns:
            Matching Employee or None.
        """
        raw = await self._get(
            "api/employee", token, params={"email": email, "active": True}
        )
        items = (
            raw.get("content", [])
            if isinstance(raw, dict)
            else (raw if isinstance(raw, list) else [])
        )
        return EmployeeMapper.to_employee(items[0]) if items else None

    async def get_by_getu_id(
        self,
        getu_id: str,
        token: str,
    ) -> Employee | None:
        """Find employee by GETU external identifier.

        GET /api/employee?uId={getu_id}&fired=false

        Args:
            getu_id: GETU system identifier string.
            token: JWT bearer token.

        Returns:
            Active Employee or None.
        """
        raw = await self._get(
            "api/employee", token, params={"uId": getu_id, "fired": False}
        )
        items = (
            raw.get("content", [])
            if isinstance(raw, dict)
            else (raw if isinstance(raw, list) else [])
        )
        return EmployeeMapper.to_employee(items[0]) if items else None

    async def find_by_department(
        self,
        department_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[Employee]:
        """Fetch all active employees in a department.

        Args:
            department_id: Department UUID.
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            List of active Employee objects.
        """
        page = await self.search(
            filters=EmployeeFilter(department_id=[department_id], active=True),
            token=token,
            organization_id=organization_id,
            pagination=PageRequest(size=200),
        )
        return page.items

    async def find_subordinates(
        self,
        leader_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[Employee]:
        """Fetch all employees in a manager's reporting chain.

        Args:
            leader_id: Manager UUID.
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            List of Employee in the reporting chain.
        """
        page = await self.search(
            filters=EmployeeFilter(
                employee_leader_department_all_id=leader_id,
                child_departments=True,
                active=True,
            ),
            token=token,
            organization_id=organization_id,
            pagination=PageRequest(size=1000),
        )
        return page.items

    async def get_activity(
        self,
        employee_id: UUID,
        filters: UserActionFilter,
        token: str,
    ) -> list[dict]:
        """Fetch employee activity log.

        GET /api/employee/{id}/user-action
        Requires filters.start and filters.end.

        Args:
            employee_id: Employee UUID.
            filters: UserActionFilter with start/end required.
            token: JWT bearer token.

        Returns:
            List of raw activity dicts.

        Raises:
            FilterValidationError: When start/end are missing.
        """
        filters.validate_for_dashboard()
        data = await self._get(
            f"api/employee/{employee_id}/user-action",
            token,
            params=filters.as_api_params(),
        )
        return data if isinstance(data, list) else []
