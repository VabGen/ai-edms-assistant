# src/ai_edms_assistant/infrastructure/edms_api/repositories/edms_employee_repository.py
from __future__ import annotations

import structlog
from uuid import UUID
from typing import Any

from ai_edms_assistant.infrastructure.edms_api.http_client import EdmsHttpClient
from ....domain.entities.employee import Employee
from ai_edms_assistant.domain.value_objects.filters import (
    EmployeeFilter,
    UserActionFilter,
)
from ....domain.repositories.employee_repository import AbstractEmployeeRepository
from ....domain.repositories.base import Page, PageRequest

logger = structlog.get_logger(__name__)


class EdmsEmployeeRepository(AbstractEmployeeRepository):
    """
    EDMS REST API implementation of AbstractEmployeeRepository.

    Uses EdmsHttpClient._make_request() exclusively — no .get()/.post() shortcuts.
    All filters come from domain/filters.py; resources_openapi.py is NOT used.

    Key design decisions:
    - find_by_name_fts returns Employee | None (single best FTS match)
      searchFtsTopByLastName → GET /api/employee/fts-lastname
    - search uses POST /api/employee/search for complex queries
    - get_activity calls UserActionFilter.validate_for_dashboard() before request

    Attributes:
        http_client: Shared async EdmsHttpClient instance.
    """

    def __init__(self, http_client: EdmsHttpClient) -> None:
        self.http_client = http_client

    # ------------------------------------------------------------------
    # Mapper: raw API dict → domain Employee
    # ------------------------------------------------------------------

    def _map_to_entity(self, data: dict[str, Any]) -> Employee:
        """
        Maps a raw EDMS API EmployeeDto dict to a domain Employee.

        Handles both flat fields (departmentName, postName) and nested
        post/department objects returned when includes=[POST,DEPARTMENT].
        """
        post_raw = data.get("post") or {}
        dept_raw = data.get("department") or {}

        return Employee(
            id=UUID(data["id"]) if data.get("id") else None,
            first_name=data.get("firstName"),
            last_name=data.get("lastName"),
            middle_name=data.get("middleName"),
            organization_id=data.get("organizationId"),
            department_id=(
                UUID(dept_raw["id"])
                if dept_raw.get("id")
                else (UUID(data["departmentId"]) if data.get("departmentId") else None)
            ),
            department_name=(
                dept_raw.get("departmentName")
                or dept_raw.get("name")
                or data.get("departmentName")
            ),
            post_id=(post_raw.get("id") or data.get("postId")),
            post_name=(
                post_raw.get("postName") or post_raw.get("name") or data.get("postName")
            ),
            email=data.get("email"),
            phone=data.get("phone"),
            is_active=data.get("active", True),
            fired=data.get("fired", False),
            getu_id=data.get("uId"),
        )

    # ------------------------------------------------------------------
    # BaseRepository
    # ------------------------------------------------------------------

    async def get_by_id(
        self,
        entity_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> Employee | None:
        """GET /api/employee/{id}"""
        params: dict = {}
        if organization_id:
            params["organizationId"] = organization_id

        try:
            data = await self.http_client._make_request(
                "GET", f"api/employee/{entity_id}", token=token, params=params
            )
            return self._map_to_entity(data) if data else None
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
        """
        POST /api/employee/search with ids filter.

        findByIdIn / findByIdInAndOrganizationId.
        Uses EmployeeFilter.ids to batch-fetch in a single API request.
        """
        if not entity_ids:
            return []
        f = EmployeeFilter(ids=list(entity_ids))
        if organization_id:
            f = EmployeeFilter(ids=list(entity_ids), org_id=organization_id)
        pg = await self.search(f, token, PageRequest(size=len(entity_ids)))
        return pg.items

    async def find_page(
        self,
        token: str,
        organization_id: str | None = None,
        pagination: PageRequest | None = None,
    ) -> Page[Employee]:
        """Delegates to search() with active=True filter."""
        f = EmployeeFilter(
            active=True,
            org_id=organization_id if organization_id else None,
        )
        return await self.search(f, token, pagination=pagination)

    # ------------------------------------------------------------------
    # EmployeeRepository-specific
    # ------------------------------------------------------------------

    async def find_by_name_fts(
        self,
        query: str,
        token: str,
        organization_id: str | None = None,
    ) -> Employee | None:
        """
        GET /api/employee/fts-lastname?fts={query}

        Returns the single best PostgreSQL FTS match.
        searchFtsTopByLastName (ts_rank_cd + similarity()).
        Returns None when no match found (HTTP 404 from backend).
        """
        params: dict = {"fts": query.strip()}
        if organization_id:
            params["organizationId"] = organization_id

        try:
            data = await self.http_client._make_request(
                "GET", "api/employee/fts-lastname", token=token, params=params
            )
            return self._map_to_entity(data) if data else None
        except Exception as exc:
            # 404 = нет совпадений — не ошибка
            logger.warning("edms_employee_fts_not_found", query=query, error=str(exc))
            return None

    async def search(
        self,
        filters: EmployeeFilter,
        token: str,
        organization_id: str | None = None,
        pagination: PageRequest | None = None,
    ) -> Page[Employee]:
        """
        POST /api/employee/search

        Preferred over GET for complex queries (large id lists, departments).
        Default sort: lastName,ASC — @PageableDefault.
        org_id from parameter overrides filters.org_id when filters.org_id is not set.
        """
        pag = pagination or PageRequest(size=10, sort="lastName,ASC")

        # Применяем organization_id из аргумента если не задан в фильтре
        effective_filters = filters
        if organization_id and not filters.org_id:
            effective_filters = EmployeeFilter(
                first_name=filters.first_name,
                last_name=filters.last_name,
                middle_name=filters.middle_name,
                fired=filters.fired,
                active=filters.active,
                full_post_name=filters.full_post_name,
                post_id=filters.post_id,
                ids=filters.ids,
                department_id=filters.department_id,
                employee_leader_department_id=filters.employee_leader_department_id,
                employee_leader_department_all_id=filters.employee_leader_department_all_id,
                includes=filters.includes,
                org_id=organization_id,
                exclude_role_id=filters.exclude_role_id,
                exclude_group_id=filters.exclude_group_id,
                exclude_ids=filters.exclude_ids,
                all=filters.all,
                child_departments=filters.child_departments,
            )

        data = await self.http_client._make_request(
            "POST",
            "api/employee/search",
            token=token,
            params=pag.as_params(),
            json=effective_filters.as_api_payload(),
        )
        data = data or {}

        items = [self._map_to_entity(row) for row in data.get("content", [])]
        return Page(
            items=items,
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
        """
        GET /api/employee?email={email}

        EDMS has no dedicated /by-email endpoint — uses standard list
        with email param and returns first match.
        """
        data = await self.http_client._make_request(
            "GET", "api/employee", token=token, params={"email": email, "active": True}
        )
        items = (
            data.get("content", [])
            if isinstance(data, dict)
            else (data if isinstance(data, list) else [])
        )
        return self._map_to_entity(items[0]) if items else None

    async def get_by_getu_id(
        self,
        getu_id: str,
        token: str,
    ) -> Employee | None:
        """
        GET /api/employee?uId={getu_id}&fired=false

        findByuIdAndFiredIsFalse — cross-system identity lookup.
        """
        data = await self.http_client._make_request(
            "GET", "api/employee", token=token, params={"uId": getu_id, "fired": False}
        )
        items = (
            data.get("content", [])
            if isinstance(data, dict)
            else (data if isinstance(data, list) else [])
        )
        return self._map_to_entity(items[0]) if items else None

    async def find_by_department(
        self,
        department_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[Employee]:
        """
        POST /api/employee/search with departmentId filter.

        findAllByDepartmentId.
        Uses size=200 — typical department size upper bound.
        """
        f = EmployeeFilter(department_id=[department_id], active=True)
        pg = await self.search(f, token, organization_id, PageRequest(size=200))
        return pg.items

    async def find_subordinates(
        self,
        leader_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[Employee]:
        """
        POST /api/employee/search with employeeLeaderDepartmentAllId + childDepartments=True.

        findAllByLeaderId recursive CTE.
        The EDMS backend resolves the org-tree recursion; we consume the flat list.
        Uses size=1000 for large org structures.
        """
        f = EmployeeFilter(
            employee_leader_department_all_id=leader_id,
            child_departments=True,
            active=True,
        )
        pg = await self.search(f, token, organization_id, PageRequest(size=1000))
        return pg.items

    async def get_activity(
        self,
        employee_id: UUID,
        filters: UserActionFilter,
        token: str,
    ) -> list[dict]:
        """
        GET /api/employee/{id}/user-action

        Requires filters.start and filters.end (Dashboard validation).
        validate_for_dashboard() raises FilterValidationError when missing.
        """
        filters.validate_for_dashboard()
        data = await self.http_client._make_request(
            "GET",
            f"api/employee/{employee_id}/user-action",
            token=token,
            params=filters.as_api_params(),
        )
        return data if isinstance(data, list) else []
