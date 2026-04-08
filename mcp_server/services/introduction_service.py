"""
EDMS AI Assistant - Introduction Service.
"""

import logging
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..clients.base_client import EdmsHttpClient
from ..clients.department_client import DepartmentClient
from ..clients.employee_client import EmployeeClient
from ..clients.group_client import GroupClient

logger = logging.getLogger(__name__)


class PostIntroductionRequest(BaseModel):
    """Request DTO для создания ознакомления через API."""

    executorListIds: list[UUID] = Field(
        ..., description="UUID сотрудников для добавления в список ознакомления"
    )
    comment: str = Field(
        default="",
        description="Комментарий к ознакомлению (пустая строка если не указан)",
    )

    model_config = ConfigDict(
        json_encoders={UUID: str},
        use_enum_values=True,
    )


@dataclass(frozen=True)
class IntroductionResult:
    """Immutable результат создания ознакомления."""

    success: bool
    added_count: int = 0
    error_message: str | None = None


@dataclass(frozen=True)
class EmployeeResolutionResult:
    """Результат резолвинга сотрудников с обработкой неоднозначностей."""

    employee_ids: set[UUID] = field(default_factory=set)
    not_found: list[str] = field(default_factory=list)
    ambiguous: list[dict] = field(default_factory=list)


@dataclass
class AmbiguousMatch:
    """Структура для неоднозначного совпадения при поиске сотрудника."""

    search_query: str
    matches: list[dict]


class IntroductionService:
    """
    Сервисный слой для управления списками ознакомления.

    Responsibilities:
    - Резолвинг сотрудников по различным критериям (фамилия, отдел, группа)
    - Обработка неоднозначных совпадений
    - Создание списков ознакомления через API
    - Валидация и нормализация комментариев

    Архитектурные решения:
    - Использование async context manager для автоматического управления клиентами
    """

    def __init__(self):
        """Инициализация с созданием клиентов для внешних API."""
        self.employee_client = EmployeeClient()
        self.department_client = DepartmentClient()
        self.group_client = GroupClient()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.employee_client.__aenter__()
        await self.department_client.__aenter__()
        await self.group_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit с корректным закрытием клиентов."""
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
        """
        Резолвит сотрудников по множественным критериям поиска.

        Обрабатывает:
        - Поиск по фамилиям (с обработкой неоднозначных совпадений)
        - Массовое добавление по отделам
        - Массовое добавление по группам

        Args:
            token: JWT токен авторизации
            last_names: Список фамилий для поиска
            department_names: Названия отделов
            group_names: Названия групп

        Returns:
            EmployeeResolutionResult с найденными UUID, не найденными критериями
            и неоднозначными совпадениями
        """
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        ambiguous_results: list[dict] = []

        if last_names:
            for last_name in last_names:
                result = await self._resolve_by_last_name(token, last_name)

                if result["status"] == "not_found":
                    not_found.append(f"Сотрудник: {last_name}")
                elif result["status"] == "found":
                    found_ids.add(result["employee_id"])
                elif result["status"] == "ambiguous":
                    ambiguous_results.append(result["ambiguous_data"])

        if department_names:
            dept_ids, dept_not_found = await self._resolve_departments(
                token, department_names
            )
            found_ids.update(dept_ids)
            not_found.extend(dept_not_found)

        if group_names:
            group_ids, group_not_found = await self._resolve_groups(token, group_names)
            found_ids.update(group_ids)
            not_found.extend(group_not_found)

        logger.info(
            "Employee resolution complete",
            extra={
                "found_count": len(found_ids),
                "not_found_count": len(not_found),
                "ambiguous_count": len(ambiguous_results),
            },
        )

        return EmployeeResolutionResult(
            employee_ids=found_ids,
            not_found=not_found,
            ambiguous=ambiguous_results,
        )

    async def _resolve_by_last_name(self, token: str, last_name: str) -> dict:
        """
        Резолвит сотрудника по фамилии с обработкой неоднозначностей.

        Returns:
            Dict с ключами:
            - status: "found" | "not_found" | "ambiguous"
            - employee_id: UUID (для "found")
            - ambiguous_data: dict (для "ambiguous")
        """
        employees = await self.employee_client.search_employees(
            token, {"lastName": last_name, "includes": ["POST", "DEPARTMENT"]}
        )

        if not employees:
            logger.warning(f"Employee not found: {last_name}")
            return {"status": "not_found"}

        if len(employees) == 1:
            emp = employees[0]
            emp_id = UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
            logger.debug(f"Found single employee: {last_name} -> {emp_id}")
            return {"status": "found", "employee_id": emp_id}

        logger.info(
            f"Found {len(employees)} employees with last name '{last_name}' - disambiguation required"
        )

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
        """
        Резолвит сотрудников по названиям отделов.

        Returns:
            Tuple[Set[UUID], List[str]]: Найденные ID и не найденные отделы
        """
        found_ids: set[UUID] = set()
        not_found: list[str] = []

        for dept_name in department_names:
            dept = await self.department_client.find_by_name(token, dept_name)

            if not dept:
                not_found.append(f"Департамент: {dept_name}")
                logger.warning(f"Department not found: {dept_name}")
                continue

            dept_id = UUID(dept["id"]) if isinstance(dept["id"], str) else dept["id"]
            employees = await self.department_client.get_employees_by_department_id(
                token, dept_id
            )

            for emp in employees:
                emp_id = UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
                found_ids.add(emp_id)

            logger.info(
                f"Resolved department '{dept_name}': {len(employees)} employees"
            )

        return found_ids, not_found

    async def _resolve_groups(
        self, token: str, group_names: list[str]
    ) -> tuple[set[UUID], list[str]]:
        """
        Резолвит сотрудников по названиям групп.

        Returns:
            Tuple[Set[UUID], List[str]]: Найденные ID и не найденные группы
        """
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        group_ids: list[UUID] = []

        for group_name in group_names:
            group = await self.group_client.find_by_name(token, group_name)

            if not group:
                not_found.append(f"Группа: {group_name}")
                logger.warning(f"Group not found: {group_name}")
                continue

            group_id = (
                UUID(group["id"]) if isinstance(group["id"], str) else group["id"]
            )
            group_ids.append(group_id)

        if group_ids:
            employees = await self.group_client.get_employees_by_group_ids(
                token, group_ids
            )

            for emp in employees:
                emp_id = UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
                found_ids.add(emp_id)

            logger.info(f"Resolved {len(group_ids)} groups: {len(employees)} employees")

        return found_ids, not_found

    async def create_introduction(
        self,
        token: str,
        document_id: str,
        employee_ids: list[UUID],
        comment: str | None = None,
    ) -> IntroductionResult:
        """
        Создает список ознакомления через API EDMS.

        Args:
            token: JWT токен авторизации
            document_id: UUID документа
            employee_ids: Список UUID сотрудников
            comment: Комментарий к ознакомлению (опционально)

        Returns:
            IntroductionResult с информацией об успехе операции
        """
        if not employee_ids:
            logger.warning("Attempted to create introduction with empty employee list")
            return IntroductionResult(
                success=False,
                added_count=0,
                error_message="Не указаны сотрудники для добавления",
            )

        normalized_comment = self._normalize_comment(comment)

        request = PostIntroductionRequest(
            executorListIds=employee_ids, comment=normalized_comment
        )

        try:
            async with EdmsHttpClient() as client:
                endpoint = f"api/document/{document_id}/introduction"
                payload = request.model_dump(mode="json")

                logger.debug(
                    "Creating introduction",
                    extra={
                        "document_id": document_id,
                        "employee_count": len(employee_ids),
                        "has_comment": bool(normalized_comment),
                    },
                )

                await client._make_request("POST", endpoint, token=token, json=payload)

            logger.info(
                "Introduction created successfully",
                extra={"document_id": document_id, "added_count": len(employee_ids)},
            )

            return IntroductionResult(success=True, added_count=len(employee_ids))

        except Exception as e:
            logger.error(
                f"Failed to create introduction: {e}",
                exc_info=True,
                extra={"document_id": document_id},
            )
            return IntroductionResult(
                success=False,
                added_count=0,
                error_message=f"Ошибка API: {e!s}",
            )

    @staticmethod
    def _normalize_comment(comment: str | None) -> str:
        """
        Нормализует комментарий перед отправкой в API.

        Rules:
        - None или пустая строка → ""
        - Убирает шаблонные фразы от LLM
        - Капитализирует первую букву
        """
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
        """
        Форматирует данные сотрудника для disambiguation response.

        Args:
            employee: Сырые данные от API

        Returns:
            Структурированный словарь с полным именем, должностью и отделом
        """
        first_name = employee.get("firstName", "")
        last_name = employee.get("lastName", "")
        middle_name = employee.get("middleName", "") or ""
        full_name = f"{last_name} {first_name} {middle_name}".strip()

        post_data = employee.get("post", {})
        post_name = (
            post_data.get("postName", "Не указана")
            if isinstance(post_data, dict)
            else "Не указана"
        )

        dept_data = employee.get("department", {})
        dept_name = (
            dept_data.get("name", "Не указан")
            if isinstance(dept_data, dict)
            else "Не указан"
        )

        return {
            "id": str(employee["id"]),
            "full_name": full_name,
            "post": post_name,
            "department": dept_name,
        }
