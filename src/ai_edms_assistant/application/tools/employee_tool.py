# src/ai_edms_assistant/application/tools/employee_tool.py
"""Employee search and card retrieval tool."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.entities.employee import Employee
from ...domain.repositories import AbstractEmployeeRepository
from ...domain.value_objects.filters import EmployeeFilter
from ...domain.repositories.base import PageRequest
from .base_tool import AbstractEdmsTool

logger = logging.getLogger(__name__)


class EmployeeSearchInput(BaseModel):
    """Input schema for employee search tool.

    Attributes:
        token: JWT auth token (auto-injected by agent).
        employee_id: UUID for direct employee lookup (highest priority).
        last_name: Surname — triggers FTS when provided alone.
        first_name: First name filter for structured search.
        middle_name: Patronymic filter for structured search.
        full_post_name: Position title filter for structured search.
    """

    token: str = Field(..., description="JWT токен авторизации пользователя")
    employee_id: str | None = Field(
        default=None,
        description="UUID конкретного сотрудника для получения карточки",
    )
    last_name: str | None = Field(default=None, description="Фамилия сотрудника")
    first_name: str | None = Field(default=None, description="Имя сотрудника")
    middle_name: str | None = Field(default=None, description="Отчество сотрудника")
    full_post_name: str | None = Field(default=None, description="Должность")


class EmployeeSearchTool(AbstractEdmsTool):
    """Employee search and card retrieval tool.

    Delegates all data access to AbstractEmployeeRepository.
    Tool responsibility: routing to correct repository method + card formatting.

    Architecture:
        Tool (Application) → Repository (Infrastructure) → Java API
        No HTTP calls or enrichment logic in the tool itself.
    """

    name: str = "employee_search_tool"
    description: str = (
        "Поиск сотрудников и получение их детальных карточек. "
        "Если найдено несколько человек — возвращает список для выбора."
    )
    args_schema: type[BaseModel] = EmployeeSearchInput

    def __init__(self, employee_repository: AbstractEmployeeRepository, **kwargs):
        """Initialize with injected employee repository.

        Args:
            employee_repository: Repository implementation for employee access.
            **kwargs: Additional BaseTool arguments.
        """
        super().__init__(**kwargs)
        self._emp_repo = employee_repository

    async def _arun(
        self,
        token: str,
        employee_id: str | None = None,
        last_name: str | None = None,
        first_name: str | None = None,
        middle_name: str | None = None,
        full_post_name: str | None = None,
    ) -> dict[str, Any]:
        """Execute employee search.

        Routing logic:
            employee_id → get_by_id()
            last_name only → find_by_name_fts() (enriched in repository)
            multiple filters → search() with POST

        Args:
            token: JWT token.
            employee_id: Employee UUID string (takes priority over filters).
            last_name: Surname — FTS when provided alone.
            first_name: First name filter.
            middle_name: Patronymic filter.
            full_post_name: Position title filter.

        Returns:
            Success dict with ``employee_card`` or ``choices``, or error dict.
        """
        try:
            # ── 1. Direct lookup by UUID ───────────────────────────────────────
            if employee_id:
                return await self._by_id(employee_id, token)

            if not any([last_name, first_name, middle_name, full_post_name]):
                return self._handle_error(
                    ValueError("Укажите критерии поиска: фамилию, имя или должность")
                )

            # ── 2. FTS by last name (enrichment handled in repository) ─────────
            if last_name and not any([first_name, middle_name, full_post_name]):
                return await self._by_fts(last_name, token)

            # ── 3. Structured multi-field search ───────────────────────────────
            return await self._by_filters(
                token=token,
                last_name=last_name,
                first_name=first_name,
                middle_name=middle_name,
                full_post_name=full_post_name,
            )

        except Exception as exc:
            logger.error("employee_tool_error", exc_info=True)
            return self._handle_error(exc)

    # ------------------------------------------------------------------
    # Private routing methods
    # ------------------------------------------------------------------

    async def _by_id(self, employee_id: str, token: str) -> dict[str, Any]:
        """Fetch employee card by UUID.

        Args:
            employee_id: UUID string.
            token: JWT token.

        Returns:
            Success response with full employee_card, or error.
        """
        try:
            emp_uuid = UUID(str(employee_id))
        except ValueError:
            return self._handle_error(ValueError(f"Неверный UUID: {employee_id}"))

        employee = await self._emp_repo.get_by_id(entity_id=emp_uuid, token=token)
        if not employee:
            return self._handle_error(ValueError(f"Сотрудник {employee_id} не найден"))

        return self._success_response(
            data={"employee_card": self._build_card(employee)},
            message="Сотрудник найден",
        )

    async def _by_fts(self, last_name: str, token: str) -> dict[str, Any]:
        """Search by last name via FTS (repository handles enrichment).

        Repository find_by_name_fts() transparently:
            1. Calls fts-lastname → resolves UUID
            2. Calls get_by_id → enriches with post/department

        Args:
            last_name: Last name query (fuzzy, partial OK).
            token: JWT token.

        Returns:
            Success response with enriched employee_card, or not_found.
        """
        employee = await self._emp_repo.find_by_name_fts(query=last_name, token=token)

        if not employee:
            return self._success_response(
                data={"choices": []},
                message=f"Сотрудник с фамилией '{last_name}' не найден в системе",
            )

        return self._success_response(
            data={"employee_card": self._build_card(employee)},
            message="Сотрудник найден",
        )

    async def _by_filters(
        self,
        token: str,
        last_name: str | None,
        first_name: str | None,
        middle_name: str | None,
        full_post_name: str | None,
    ) -> dict[str, Any]:
        """Structured multi-field search via POST /api/employee/search.

        Args:
            token: JWT token.
            last_name: Surname filter.
            first_name: First name filter.
            middle_name: Patronymic filter.
            full_post_name: Position title filter.

        Returns:
            Single card, disambiguation choices list, or not_found.
        """
        page = await self._emp_repo.search(
            filters=EmployeeFilter(
                last_name=last_name,
                first_name=first_name,
                middle_name=middle_name,
                full_post_name=full_post_name,
                active=True,
                includes=["POST", "DEPARTMENT"],
            ),
            token=token,
            pagination=PageRequest(size=10, sort="lastName,ASC"),
        )
        items = page.items

        if not items:
            return self._success_response(
                data={"choices": []},
                message="Сотрудники не найдены по данным критериям",
            )

        if len(items) == 1:
            return self._success_response(
                data={"employee_card": self._build_card(items[0])},
                message="Сотрудник найден",
            )

        return {
            "status": "requires_action",
            "action_type": "select_employee",
            "message": f"Найдено сотрудников: {len(items)}. Уточните выбор:",
            "data": {
                "choices": [
                    {
                        "id": str(emp.id),
                        "full_name": emp.short_name,
                        "post": emp.post_name or emp.full_post_name or "Не указана",
                        "department": emp.department_name or "Не указан",
                    }
                    for emp in items
                ]
            },
        }

    # ------------------------------------------------------------------
    # Card builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_card(employee: Employee) -> dict[str, Any]:
        """Build LLM-ready employee card from domain entity.

        Includes only fields that have values — avoids cluttering
        the LLM context with "Не указан" placeholders.
        post_name and department_name are populated by repository
        _normalize_raw() from nested Java API objects.

        Args:
            employee: Employee domain entity (post-enrichment).

        Returns:
            Dict with available employee data for LLM context injection.
        """
        card: dict[str, Any] = {
            "ФИО": employee.full_name,
            "Должность": employee.post_name or employee.full_post_name or "Не указана",
            "Подразделение": employee.department_name or "Не указано",
            "Статус": (
                "Уволен"
                if employee.fired
                else ("Активен" if employee.is_active else "Неактивен")
            ),
            "ID": str(employee.id),
        }

        if employee.email:
            card["Email"] = employee.email
        if employee.phone:
            card["Телефон"] = employee.phone
        if employee.address:
            card["Адрес"] = employee.address

        if employee.is_acting:
            card["Примечание"] = "Является И.О."

        return card

    def _run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Synchronous execution not supported.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use _arun for async execution")
