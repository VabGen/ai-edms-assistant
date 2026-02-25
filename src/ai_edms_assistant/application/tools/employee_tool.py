# src/ai_edms_assistant/application/tools/employee_tool.py
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.repositories import AbstractEmployeeRepository
from ...domain.value_objects.filters import EmployeeFilter
from .base_tool import AbstractEdmsTool


class EmployeeSearchInput(BaseModel):
    """Input schema for employee search.

    Attributes:
        token: JWT auth token.
        employee_id: UUID for fetching specific employee card (optional).
        last_name: Surname for fuzzy search (optional).
        first_name: First name (optional).
        middle_name: Middle name (optional).
        full_post_name: Position title (optional).
    """

    token: str = Field(..., description="JWT токен авторизации")
    employee_id: UUID | None = Field(default=None, description="UUID сотрудника")
    last_name: str | None = Field(default=None, description="Фамилия")
    first_name: str | None = Field(default=None, description="Имя")
    middle_name: str | None = Field(default=None, description="Отчество")
    full_post_name: str | None = Field(default=None, description="Должность")


class EmployeeSearchTool(AbstractEdmsTool):
    """Tool for searching employees and retrieving detailed cards.

    Supports two modes:
    1. Get by ID: Returns full employee card when ``employee_id`` provided.
    2. Search: Fuzzy search by name/position, returns list if multiple matches.

    Dependencies:
        - ``AbstractEmployeeRepository``: Search and fetch employees.
    """

    name: str = "employee_search_tool"
    description: str = (
        "Поиск сотрудников и получение их детальных карточек. "
        "Если найдено несколько человек, возвращает список для выбора."
    )
    args_schema: type[BaseModel] = EmployeeSearchInput

    def __init__(self, employee_repository: AbstractEmployeeRepository, **kwargs):
        """Initialize with injected employee repository.

        Args:
            employee_repository: Repository for employee operations.
            **kwargs: Additional BaseTool arguments.
        """
        super().__init__(**kwargs)
        self._emp_repo = employee_repository

    async def _arun(
        self,
        token: str,
        employee_id: UUID | None = None,
        last_name: str | None = None,
        first_name: str | None = None,
        middle_name: str | None = None,
        full_post_name: str | None = None,
    ) -> dict[str, Any]:
        """Execute employee search.

        Args:
            token: JWT token.
            employee_id: Specific employee UUID (takes priority).
            last_name: Surname for search.
            first_name: Name for search.
            middle_name: Middle name for search.
            full_post_name: Position for search.

        Returns:
            Dict with employee card or list of matches.
        """
        try:
            # MODE 1: Get by ID
            if employee_id:
                employee = await self._emp_repo.get_by_id(
                    entity_id=employee_id, token=token
                )
                if not employee:
                    return self._handle_error(ValueError("Сотрудник не найден"))

                card = self._build_employee_card(employee)
                return self._success_response(
                    data={"employee_card": card},
                    message="Сотрудник найден",
                )

            # MODE 2: Search by filters
            filter_obj = EmployeeFilter(
                last_name=last_name,
                first_name=first_name,
                middle_name=middle_name,
                full_post_name=full_post_name,
                active=True,
            )

            # Validate at least one search criterion
            if not any([last_name, first_name, middle_name, full_post_name]):
                return self._handle_error(
                    ValueError("Укажите критерии поиска (например, lastName)")
                )

            results = await self._emp_repo.search(filters=filter_obj, token=token)

            if not results:
                return self._success_response(
                    data={"choices": []},
                    message="Сотрудники не найдены по данным критериям",
                )

            # Single match - return card
            if len(results) == 1:
                card = self._build_employee_card(results[0])
                return self._success_response(
                    data={"employee_card": card},
                    message="Сотрудник найден",
                )

            # Multiple matches - return choices
            choices = [
                {
                    "id": str(emp.id),
                    "full_name": emp.short_name,
                    "post": emp.post.name if emp.post else "Не указана",
                    "department": (
                        emp.department.name if emp.department else "Не указан"
                    ),
                }
                for emp in results
            ]

            return {
                "status": "requires_action",
                "action_type": "select_employee",
                "message": f"Найдено сотрудников: {len(results)}. Выберите нужного:",
                "data": {"choices": choices},
            }

        except Exception as e:
            return self._handle_error(e)

    def _build_employee_card(self, employee: Any) -> dict[str, Any]:
        """Build structured employee card from entity.

        Args:
            employee: Employee domain entity.

        Returns:
            Dict with formatted employee data.
        """
        return {
            "ФИО": employee.short_name,
            "Должность": employee.post.name if employee.post else "Не указана",
            "Подразделение": (
                employee.department.name if employee.department else "Не указано"
            ),
            "Email": employee.email or "Не указан",
            "Телефон": employee.work_phone or "Не указан",
            "Статус": "Активен" if employee.is_active else "Уволен",
        }

    def _run(self, *args, **kwargs) -> dict[str, Any]:
        """Sync execution not supported."""
        raise NotImplementedError("Use _arun for async execution")
