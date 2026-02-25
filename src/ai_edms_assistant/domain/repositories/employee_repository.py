# src/ai_edms_assistant/domain/repositories/employee_repository.py
from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from ..entities.employee import Employee
from ..value_objects.filters import EmployeeFilter, UserActionFilter
from .base import AbstractRepository, Page, PageRequest


class AbstractEmployeeRepository(AbstractRepository[Employee]):
    """Port (interface) for employee and org-structure data access.

    Defines all employee lookup operations available to the application layer.
    The primary resolution path for the AI agent is ``find_by_name_fts`` —
    used when the user refers to a person by name in natural language.

    Implementation:
        ``infrastructure/edms_api/repositories/edms_employee_repository.py``

    Consumers:
        - ``application/tools/employee_tool.py`` — name resolution
        - ``application/tools/introduction_tool.py`` — introduction list
        - ``application/tools/task_tool.py`` — executor assignment
        - ``application/use_cases/create_task.py`` — executor validation

    Note:
        All filters use ``domain/value_objects/filters.py`` value objects.
        ``resources_openapi.py`` types must not appear in this interface.
    """

    # ------------------------------------------------------------------
    # Full-text search — primary lookup path for the AI agent
    # ------------------------------------------------------------------

    @abstractmethod
    async def find_by_name_fts(
        self,
        query: str,
        token: str,
        organization_id: str | None = None,
    ) -> Employee | None:
        """Full-text search for the best-matching employee by last name.

        Mirrors Java ``searchFtsTopByLastName`` — uses PostgreSQL
        ``ts_rank_cd`` + ``similarity()`` to return the single best match.
        Calls ``GET /api/employee/fts-lastname?fts=...``.

        This is the **primary resolution path** used by the AI agent when
        interpreting natural-language executor names like
        "назначь Иванову" or "поручение Петрову из бухгалтерии".

        Args:
            query: Last name string. Supports partial and unaccented input.
            token: JWT bearer token.
            organization_id: Org scope for multi-tenant deployments.

        Returns:
            Best-matching ``Employee``, or ``None`` when no match is found.
        """

    @abstractmethod
    async def search(
        self,
        filters: EmployeeFilter,
        token: str,
        organization_id: str | None = None,
        pagination: PageRequest | None = None,
    ) -> Page[Employee]:
        """Structured multi-field employee search with pagination.

        Calls ``POST /api/employee/search``. Used when the agent needs to:
        - Disambiguate multiple employees with similar names
        - Filter by department or position
        - Fetch employees by a specific list of IDs

        Args:
            filters: ``EmployeeFilter`` value object from
                ``domain/value_objects/filters.py``.
            token: JWT bearer token.
            organization_id: Org scope.
            pagination: Page/size/sort params. Defaults to page 0, size 10,
                sort ``"lastName,ASC"`` (mirrors the Java default).

        Returns:
            ``Page[Employee]`` with Slice semantics.
        """

    # ------------------------------------------------------------------
    # Lookup by contact / external identifier
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_by_email(
        self,
        email: str,
        token: str,
    ) -> Employee | None:
        """Find an employee by their corporate email address.

        Used when the agent receives a direct identifier reference such as
        "найди пользователя ivanov@org.by". Performs exact match,
        case-insensitive.

        Args:
            email: Corporate email address string.
            token: JWT bearer token.

        Returns:
            Matching ``Employee``, or ``None`` when not found.
        """

    @abstractmethod
    async def get_by_getu_id(
        self,
        getu_id: str,
        token: str,
    ) -> Employee | None:
        """Find an employee by their GETU external system identifier.

        Mirrors Java ``findByuId`` / ``findByuIdAndFiredIsFalse``.
        Used for cross-system identity resolution in integration scenarios.

        Args:
            getu_id: GETU system identifier string (``uId`` field in Java).
            token: JWT bearer token.

        Returns:
            Active (non-fired) ``Employee``, or ``None`` when not found.
        """

    # ------------------------------------------------------------------
    # Org-structure queries
    # ------------------------------------------------------------------

    @abstractmethod
    async def find_by_department(
        self,
        department_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[Employee]:
        """Fetch all active employees within a specific department.

        Mirrors Java ``findAllByDepartmentId``. Calls
        ``POST /api/employee/search`` with a ``departmentId`` filter.

        Used when the agent resolves "все сотрудники отдела кадров".

        Args:
            department_id: Department UUID.
            token: JWT bearer token.
            organization_id: Org scope.

        Returns:
            List of active ``Employee`` objects in the department.
            Returns ``[]`` when the department has no active employees.
        """

    @abstractmethod
    async def find_subordinates(
        self,
        leader_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> list[Employee]:
        """Fetch all employees subordinate to a given manager.

        Mirrors Java ``findAllByLeaderId`` — walks the recursive department
        tree to collect all employees in the leader's reporting chain.

        Calls ``POST /api/employee/search`` with
        ``employeeLeaderDepartmentAllId`` + ``childDepartments=True``.

        Args:
            leader_id: UUID of the manager / department leader.
            token: JWT bearer token.
            organization_id: Org scope.

        Returns:
            List of ``Employee`` objects in the leader's chain of command.
        """

    # ------------------------------------------------------------------
    # Activity log
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_activity(
        self,
        employee_id: UUID,
        filters: UserActionFilter,
        token: str,
    ) -> list[dict]:
        """Fetch the activity log for a specific employee.

        Calls ``GET /api/employee/{id}/user-action``. The ``filters.start``
        and ``filters.end`` fields are required by the backend (validated
        by the Dashboard endpoint).

        Args:
            employee_id: Employee UUID.
            filters: ``UserActionFilter`` with ``start`` and ``end`` required.
            token: JWT bearer token.

        Returns:
            List of raw activity dicts from the EDMS API. Mapped to
            value objects by the infrastructure layer if needed.
        """
