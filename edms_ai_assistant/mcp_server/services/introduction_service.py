# mcp-server/services/introduction_service.py
"""
Introduction Service — создание листов ознакомления.
Перенесён из edms_ai_assistant/services/introduction_service.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..clients.base_client import EdmsHttpClient
from ..clients.department_client import DepartmentClient, GroupClient
from ..clients.employee_client import EmployeeClient

logger = logging.getLogger(__name__)


class PostIntroductionRequest(BaseModel):
    """DTO для создания ознакомления."""
    executorListIds: list[UUID] = Field(...)
    comment: str = Field(default="")

    model_config = ConfigDict(json_encoders={UUID: str})


@dataclass(frozen=True)
class IntroductionResult:
    """Результат создания ознакомления."""
    success: bool
    added_count: int = 0
    error_message: str | None = None


@dataclass(frozen=True)
class EmployeeResolutionResult:
    """Результат резолвинга сотрудников."""
    employee_ids: set[UUID] = field(default_factory=set)
    not_found: list[str] = field(default_factory=list)
    ambiguous: list[dict] = field(default_factory=list)


class IntroductionService:
    """
    Сервис создания листов ознакомления.

    Поддерживает поиск сотрудников по фамилии, отделу и группе.
    """

    def __init__(self) -> None:
        self.employee_client = EmployeeClient()
        self.department_client = DepartmentClient()
        self.group_client = GroupClient()

    async def __aenter__(self) -> "IntroductionService":
        await self.employee_client.__aenter__()
        await self.department_client.__aenter__()
        await self.group_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.employee_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.department_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.group_client.__aexit__(exc_type, exc_val, exc_tb)

    async def resolve_employees(
        self,
        token: str,
        last_names: list[str],
        department_names: list[str],
        group_names: list[str],
    ) -> EmployeeResolutionResult:
        """Резолвит сотрудников по множественным критериям."""
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        ambiguous_results: list[dict] = []

        for last_name in last_names:
            result = await self._resolve_by_last_name(token, last_name)
            if result["status"] == "not_found":
                not_found.append(f"Сотрудник: {last_name}")
            elif result["status"] == "found":
                found_ids.add(result["employee_id"])
            elif result["status"] == "ambiguous":
                ambiguous_results.append(result["ambiguous_data"])

        if department_names:
            dept_ids, dept_not_found = await self._resolve_departments(token, department_names)
            found_ids.update(dept_ids)
            not_found.extend(dept_not_found)

        if group_names:
            group_ids, group_not_found = await self._resolve_groups(token, group_names)
            found_ids.update(group_ids)
            not_found.extend(group_not_found)

        return EmployeeResolutionResult(
            employee_ids=found_ids,
            not_found=not_found,
            ambiguous=ambiguous_results,
        )

    async def _resolve_by_last_name(self, token: str, last_name: str) -> dict:
        employees = await self.employee_client.search_employees(
            token, {"lastName": last_name, "includes": ["POST", "DEPARTMENT"]}
        )
        if not employees:
            return {"status": "not_found"}
        if len(employees) == 1:
            emp = employees[0]
            emp_id = UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
            return {"status": "found", "employee_id": emp_id}
        return {
            "status": "ambiguous",
            "ambiguous_data": {
                "search_query": last_name,
                "matches": [self._format_employee_match(emp) for emp in employees],
            },
        }

    async def _resolve_departments(
        self, token: str, department_names: list[str]
    ) -> tuple[set[UUID], list[str]]:
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        for dept_name in department_names:
            dept = await self.department_client.find_by_name(token, dept_name)
            if not dept:
                not_found.append(f"Департамент: {dept_name}")
                continue
            dept_id = UUID(dept["id"]) if isinstance(dept["id"], str) else dept["id"]
            employees = await self.department_client.get_employees_by_department_id(token, dept_id)
            for emp in employees:
                emp_id = UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
                found_ids.add(emp_id)
        return found_ids, not_found

    async def _resolve_groups(
        self, token: str, group_names: list[str]
    ) -> tuple[set[UUID], list[str]]:
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        group_ids: list[UUID] = []
        for group_name in group_names:
            group = await self.group_client.find_by_name(token, group_name)
            if not group:
                not_found.append(f"Группа: {group_name}")
                continue
            group_id = UUID(group["id"]) if isinstance(group["id"], str) else group["id"]
            group_ids.append(group_id)
        if group_ids:
            employees = await self.group_client.get_employees_by_group_ids(token, group_ids)
            for emp in employees:
                emp_id = UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
                found_ids.add(emp_id)
        return found_ids, not_found

    async def create_introduction(
        self,
        token: str,
        document_id: str,
        employee_ids: list[UUID],
        comment: str | None = None,
    ) -> IntroductionResult:
        """Создать список ознакомления через API EDMS."""
        if not employee_ids:
            return IntroductionResult(success=False, error_message="Не указаны сотрудники.")

        normalized_comment = self._normalize_comment(comment)
        request = PostIntroductionRequest(
            executorListIds=employee_ids, comment=normalized_comment
        )

        try:
            async with EdmsHttpClient() as client:
                payload = request.model_dump(mode="json")
                await client._make_request(
                    "POST", f"api/document/{document_id}/introduction",
                    token=token, json=payload,
                )
            return IntroductionResult(success=True, added_count=len(employee_ids))
        except Exception as e:
            logger.error("Failed to create introduction: %s", e, exc_info=True)
            return IntroductionResult(success=False, error_message=f"Ошибка API: {e!s}")

    @staticmethod
    def _normalize_comment(comment: str | None) -> str:
        if not comment:
            return ""
        comment = comment.strip()
        template_phrases = [
            "не указан комментарий к ознакомлению",
            "не указан комментарий",
            "комментарий к ознакомлению",
        ]
        if comment.lower() in template_phrases:
            return ""
        if len(comment) > 1:
            return comment[0].upper() + comment[1:]
        return comment.upper()

    @staticmethod
    def _format_employee_match(employee: dict) -> dict:
        fn = employee.get("firstName", "")
        ln = employee.get("lastName", "")
        mn = employee.get("middleName", "") or ""
        full_name = f"{ln} {fn} {mn}".strip()
        post_data = employee.get("post", {})
        post_name = (
            post_data.get("postName", "Не указана")
            if isinstance(post_data, dict) else "Не указана"
        )
        dept_data = employee.get("department", {})
        dept_name = (
            dept_data.get("name", "Не указан")
            if isinstance(dept_data, dict) else "Не указан"
        )
        return {
            "id": str(employee["id"]),
            "full_name": full_name,
            "post": post_name,
            "department": dept_name,
        }