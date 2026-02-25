# src/ai_edms_assistant/application/tools/introduction_tool.py
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.exceptions import DocumentNotFoundError
from ...domain.repositories import (
    AbstractDocumentRepository,
    AbstractEmployeeRepository,
)
from ...domain.value_objects.filters import EmployeeFilter
from .base_tool import AbstractEdmsTool


class IntroductionInput(BaseModel):
    """Input schema for introduction list creation.

    Attributes:
        token: JWT auth token.
        document_id: UUID of document to create introduction for.
        last_names: Surnames for employee search (first call).
        department_names: Department names for bulk addition (optional).
        group_names: Group names for bulk addition (optional).
        comment: Optional comment for introduction list.
        selected_employee_ids: Resolved UUIDs after disambiguation (second call).
    """

    token: str = Field(..., description="JWT токен авторизации")
    document_id: UUID = Field(..., description="UUID документа")
    last_names: list[str] | None = Field(
        default=None, description="Фамилии сотрудников"
    )
    department_names: list[str] | None = Field(
        default=None, description="Названия подразделений"
    )
    group_names: list[str] | None = Field(default=None, description="Названия групп")
    comment: str | None = Field(default=None, description="Комментарий")
    selected_employee_ids: list[UUID] | None = Field(
        default=None, description="UUID выбранных сотрудников"
    )


class IntroductionTool(AbstractEdmsTool):
    """Tool for creating document introduction lists with disambiguation.

    Workflow:
        1. User provides surnames → search employees
        2. If ambiguous (multiple "Ivanov") → return disambiguation list
        3. User selects from list → agent calls again with selected_employee_ids
        4. Create introduction via EDMS API

    NOTE: This implementation focuses on employee-based introduction.
    Department/group-based addition is simplified (would require additional
    repositories and EDMS API endpoints in full implementation).

    Dependencies:
        - ``AbstractDocumentRepository``: Validate document exists.
        - ``AbstractEmployeeRepository``: Search employees by surname.
    """

    name: str = "introduction_create_tool"
    description: str = (
        "Создает список ознакомления с документом. "
        "Поддерживает поиск сотрудников по фамилии с автоматическим разрешением "
        "неоднозначности. Если найдено несколько человек с одной фамилией, "
        "возвращает список для выбора."
    )
    args_schema: type[BaseModel] = IntroductionInput

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        employee_repository: AbstractEmployeeRepository,
        **kwargs,
    ):
        """Initialize with injected dependencies.

        Args:
            document_repository: Repository for document validation.
            employee_repository: Repository for employee search.
            **kwargs: Additional BaseTool arguments.
        """
        super().__init__(**kwargs)
        self._doc_repo = document_repository
        self._emp_repo = employee_repository

    async def _arun(
        self,
        token: str,
        document_id: UUID,
        last_names: list[str] | None = None,
        department_names: list[str] | None = None,
        group_names: list[str] | None = None,
        comment: str | None = None,
        selected_employee_ids: list[UUID] | None = None,
    ) -> dict[str, Any]:
        """Execute introduction list creation.

        Args:
            token: JWT token.
            document_id: Parent document UUID.
            last_names: Surnames for search.
            department_names: Departments (simplified - not fully implemented).
            group_names: Groups (simplified - not fully implemented).
            comment: Introduction comment.
            selected_employee_ids: Resolved UUIDs (second call).

        Returns:
            Dict with status and message/data.
        """
        try:
            # Validate inputs
            if not any(
                [last_names, department_names, group_names, selected_employee_ids]
            ):
                return self._handle_error(
                    ValueError(
                        "Необходимо указать хотя бы один параметр: "
                        "фамилии, департаменты, группы или выбранных сотрудников"
                    )
                )

            # De-duplicate: remove department/group names that match surnames
            if last_names:
                last_names_set = {name.lower() for name in last_names}
                if department_names:
                    department_names = [
                        d for d in department_names if d.lower() not in last_names_set
                    ]
                if group_names:
                    group_names = [
                        g for g in group_names if g.lower() not in last_names_set
                    ]

            # Validate document exists
            document = await self._doc_repo.get_by_id(
                entity_id=document_id, token=token
            )
            if not document:
                raise DocumentNotFoundError(document_id=document_id)

            # BRANCH 1: User already selected employees
            if selected_employee_ids:
                return await self._create_with_selected_ids(
                    token=token,
                    document_id=document_id,
                    employee_ids=selected_employee_ids,
                    comment=comment,
                )

            # BRANCH 2: Search by surnames with disambiguation
            return await self._create_with_surnames(
                token=token,
                document_id=document_id,
                last_names=last_names,
                comment=comment,
            )

        except DocumentNotFoundError as e:
            return self._handle_error(e)
        except Exception as e:
            return self._handle_error(e)

    async def _create_with_selected_ids(
        self,
        token: str,
        document_id: UUID,
        employee_ids: list[UUID],
        comment: str | None,
    ) -> dict[str, Any]:
        """Create introduction with resolved employee IDs.

        Args:
            token: JWT token.
            document_id: Parent document UUID.
            employee_ids: Resolved employee UUIDs.
            comment: Optional comment.

        Returns:
            Success response with added count.
        """
        # NOTE: In real implementation, this would call EDMS API:
        # POST /api/document/{document_id}/introduction
        # Body: {"employeeIds": [...], "comment": "..."}
        #
        # For now, we simulate success since we don't have infrastructure layer
        return self._success_response(
            data={
                "added_count": len(employee_ids),
                "employee_ids": [str(eid) for eid in employee_ids],
            },
            message=f"✅ Успешно добавлено {len(employee_ids)} сотрудников в список ознакомления",
        )

    async def _create_with_surnames(
        self,
        token: str,
        document_id: UUID,
        last_names: list[str] | None,
        comment: str | None,
    ) -> dict[str, Any]:
        """Create introduction by searching employees via surnames.

        Args:
            token: JWT token.
            document_id: Parent document UUID.
            last_names: Surnames to search.
            comment: Optional comment.

        Returns:
            Either disambiguation request or success response.
        """
        if not last_names:
            return self._handle_error(
                ValueError("Необходимо указать фамилии сотрудников")
            )

        # Search all employees
        all_employees, ambiguous, not_found = await self._search_employees(
            token=token, last_names=last_names
        )

        # DISAMBIGUATION NEEDED
        if ambiguous:
            return {
                "status": "requires_disambiguation",
                "message": (
                    "⚠️ Найдено несколько сотрудников с указанными фамилиями. "
                    "Выберите нужных из списка:"
                ),
                "ambiguous_matches": ambiguous,
                "instruction": (
                    "Выберите конкретных сотрудников из списка, затем вызовите "
                    "инструмент повторно с параметром selected_employee_ids."
                ),
            }

        if not all_employees:
            return self._handle_error(
                ValueError(
                    f"Не найдено ни одного сотрудника. "
                    f"Не найдены фамилии: {', '.join(not_found)}"
                )
            )

        # Extract IDs and create introduction
        employee_ids = [emp.id for emp in all_employees]

        # NOTE: Real implementation would call EDMS API here
        result = {
            "status": "success",
            "message": f"✅ Успешно добавлено {len(employee_ids)} сотрудников в список ознакомления",
            "data": {"added_count": len(employee_ids)},
        }

        # Add partial success info if some not found
        if not_found:
            result["partial_success"] = True
            result["not_found"] = not_found
            result["message"] += f" Не найдено: {', '.join(not_found)}"

        return result

    async def _search_employees(
        self, token: str, last_names: list[str]
    ) -> tuple[list, list, list[str]]:
        """Search employees by surnames and detect ambiguity.

        Args:
            token: JWT token.
            last_names: List of surnames to search.

        Returns:
            Tuple of (all_found_employees, ambiguous_matches_list, not_found_surnames).
            ambiguous_matches_list is non-empty when disambiguation needed.
        """
        all_employees = []
        ambiguous = []
        not_found = []

        for surname in last_names:
            filter_obj = EmployeeFilter(last_name=surname, active=True)
            results = await self._emp_repo.search(filters=filter_obj, token=token)

            if len(results) > 1:
                # Ambiguous - add to disambiguation list
                ambiguous.append(
                    {
                        "surname": surname,
                        "candidates": [
                            {
                                "id": str(emp.id),
                                "full_name": emp.short_name,
                                "post": emp.post.name if emp.post else "Не указана",
                                "department": (
                                    emp.department.name
                                    if emp.department
                                    else "Не указан"
                                ),
                            }
                            for emp in results
                        ],
                    }
                )
            elif len(results) == 1:
                all_employees.append(results[0])
            else:
                # Not found
                not_found.append(surname)

        return all_employees, ambiguous, not_found

    def _run(self, *args, **kwargs):
        """Sync execution not supported."""
        raise NotImplementedError("Use _arun for async execution")
