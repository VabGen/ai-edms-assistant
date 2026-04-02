# edms_ai_assistant/tools/employee_search.py
"""
EDMS AI Assistant — Employee Search Tool.

Слой: Infrastructure / Tool.
Поиск сотрудников в реестре EDMS по ФИО, отделу, должности и другим критериям.

Маппинг параметров инструмента → поля EmployeeFilter (EmployeeController.java):
    last_name         → EmployeeFilter.lastName           (String)
    first_name        → EmployeeFilter.firstName          (String)
    middle_name       → EmployeeFilter.middleName         (String)
    full_post_name    → EmployeeFilter.fullPostName       (String — поиск по должности)
    active_only       → EmployeeFilter.active             (Boolean)
    fired_only        → EmployeeFilter.fired              (Boolean)
    department_names  → EmployeeFilter.departmentId       (UUID[] — резолвится автоматически)
    department_ids    → EmployeeFilter.departmentId       (UUID[])
    child_departments → EmployeeFilter.childDepartments   (boolean)
    employee_ids      → EmployeeFilter.ids                (List<UUID>)
    exclude_ids       → EmployeeFilter.excludeIds         (List<UUID>)

Важно:
    departmentId в Java — UUID[] (массив UUID-ов), НЕ названия.
    Если агент получает название отдела («Бухгалтерия»), нужно передать
    через department_names — UUID будет найден через DepartmentClient.

Намеренно не выставлены агенту:
    postId (Long)     — нет справочника ID должностей у агента
    orgId             — агент работает в одной организации
    all (bool)        — опасно: вернёт весь реестр без пагинации
    excludeRole/Group/Grief — управление доступом, не нужно агенту
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.generated.resources_openapi import EmployeeDto
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService

logger = logging.getLogger(__name__)

# Максимальное количество сотрудников в одном ответе агенту
_MAX_RESULTS: int = 20

# Дефолтные includes: без них API не вернёт вложенные объекты post и department
_DEFAULT_INCLUDES: list[str] = ["POST", "DEPARTMENT"]


# ══════════════════════════════════════════════════════════════════════════════
# Input Schema
# ══════════════════════════════════════════════════════════════════════════════


class EmployeeSearchInput(BaseModel):
    """Validated input schema for the employee search tool.

    Поля сгруппированы по семантике:
      - Прямой доступ по UUID
      - ФИО-поиск
      - Должность
      - Статус (активный / уволенный)
      - Структура организации
    """

    token: str = Field(..., description="JWT токен авторизации пользователя")

    # ── Прямой доступ по UUID ─────────────────────────────────────────────────

    employee_id: str | None = Field(
        None,
        description=(
            "UUID конкретного сотрудника для получения полной карточки. "
            "Используй после выбора из списка результатов."
        ),
    )

    # ── ФИО-поиск ─────────────────────────────────────────────────────────────

    last_name: str | None = Field(
        None,
        max_length=150,
        description=(
            "Фамилия сотрудника (частичное совпадение). "
            "Пример: 'Иванов'. Основной параметр поиска по имени."
        ),
    )
    first_name: str | None = Field(
        None,
        max_length=100,
        description="Имя сотрудника. Пример: 'Алексей'.",
    )
    middle_name: str | None = Field(
        None,
        max_length=150,
        description="Отчество сотрудника. Пример: 'Петрович'.",
    )

    # ── Должность ─────────────────────────────────────────────────────────────

    full_post_name: str | None = Field(
        None,
        max_length=300,
        description=(
            "Название должности сотрудника (частичное совпадение по строке). "
            "Пример: 'главный бухгалтер', 'начальник отдела', 'директор'. "
            "Маппится в EmployeeFilter.fullPostName. "
            "Используй когда пользователь ищет по должности, а не по имени."
        ),
    )

    # ── Статус ────────────────────────────────────────────────────────────────

    active_only: bool | None = Field(
        None,
        description=(
            "True — только активные сотрудники (EmployeeFilter.active=true). "
            "Используй для большинства запросов. "
            "None — без фильтра по активности."
        ),
    )
    fired_only: bool | None = Field(
        None,
        description=(
            "True — только уволенные (EmployeeFilter.fired=true). "
            "Используй для поиска бывших сотрудников."
        ),
    )

    # ── Структура организации ─────────────────────────────────────────────────

    department_names: list[str] | None = Field(
        None,
        description=(
            "Список названий отделов/подразделений на русском языке. "
            "Пример: ['Бухгалтерия'], ['Отдел кадров', 'Юридический отдел']. "
            "ПРЕДПОЧТИТЕЛЬНЫЙ параметр когда пользователь называет отдел по имени — "
            "UUID разрешается автоматически через справочник департаментов. "
            "НЕ передавай UUID в это поле."
        ),
    )
    department_ids: list[str] | None = Field(
        None,
        description=(
            "Список UUID отделов/подразделений (EmployeeFilter.departmentId). "
            "Используй ТОЛЬКО когда UUID уже получен из предыдущего API-ответа. "
            "Если известно только название отдела — используй department_names."
        ),
    )
    child_departments: bool | None = Field(
        None,
        description=(
            "True — включить сотрудников всех дочерних подразделений. "
            "Используй совместно с department_names или department_ids."
        ),
    )
    employee_ids: list[str] | None = Field(
        None,
        description=(
            "Список UUID конкретных сотрудников (EmployeeFilter.ids). "
            "Используй для пакетного получения карточек по известным UUID."
        ),
    )
    exclude_ids: list[str] | None = Field(
        None,
        description="Список UUID сотрудников для исключения из результатов.",
    )

    # ── Валидаторы ────────────────────────────────────────────────────────────

    @field_validator(
        "last_name",
        "first_name",
        "middle_name",
        "full_post_name",
        mode="before",
    )
    @classmethod
    def strip_and_none(cls, v: str | None) -> str | None:
        """Strips whitespace; converts empty string to None."""
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None

    @model_validator(mode="after")
    def at_least_one_param(self) -> EmployeeSearchInput:
        """Ensures at least one meaningful search parameter is provided.

        Raises:
            ValueError: If all filter fields are None.
        """
        search_fields = [
            self.employee_id,
            self.last_name,
            self.first_name,
            self.middle_name,
            self.full_post_name,
            self.active_only,
            self.fired_only,
            self.department_names,
            self.department_ids,
            self.employee_ids,
        ]
        if not any(v is not None for v in search_fields):
            raise ValueError(
                "Укажите хотя бы один параметр: ФИО, должность, UUID сотрудника, "
                "название или UUID отдела, или фильтр по статусу."
            )
        return self


# ══════════════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════════════


@tool("employee_search_tool", args_schema=EmployeeSearchInput)
async def employee_search_tool(
    token: str,
    employee_id: str | None = None,
    last_name: str | None = None,
    first_name: str | None = None,
    middle_name: str | None = None,
    full_post_name: str | None = None,
    active_only: bool | None = None,
    fired_only: bool | None = None,
    department_names: list[str] | None = None,
    department_ids: list[str] | None = None,
    child_departments: bool | None = None,
    employee_ids: list[str] | None = None,
    exclude_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Searches for employees in the EDMS directory.

    Use when the user asks:
    - «Найди сотрудника Иванова»               → last_name='Иванов'
    - «Кто такой Алексей Петров?»              → last_name + first_name
    - «Покажи всех сотрудников Бухгалтерии»    → department_names=['Бухгалтерия']
    - «Найди активных в отделе кадров»         → department_names + active_only=True
    - «Кто у нас главный бухгалтер?»           → full_post_name='главный бухгалтер'
    - «Найди уволенного Сидорова»              → last_name + fired_only=True
    - «Покажи карточку сотрудника {uuid}»      → employee_id=uuid
    - «Кто работает в бухгалтерии и дочерних?»→ department_names + child_departments=True

    IMPORTANT:
    - department_names принимает НАЗВАНИЯ отделов (не UUID) — «Бухгалтерия», «ОК».
    - department_ids принимает UUID — только если UUID уже получен из API.
    - Никогда не передавай названия в department_ids — будет 400 Bad Request.

    Returns up to 20 employees. If multiple found — returns selection list.
    """
    nlp = EDMSNaturalLanguageService()

    # ── Прямой запрос по UUID ─────────────────────────────────────────────────
    if employee_id:
        return await _get_employee_card(token, employee_id, nlp)

    # ── Сборка EmployeeFilter ─────────────────────────────────────────────────
    employee_filter: dict[str, Any] = {"includes": _DEFAULT_INCLUDES}

    # ФИО
    if last_name:
        employee_filter["lastName"] = last_name
    if first_name:
        employee_filter["firstName"] = first_name
    if middle_name:
        employee_filter["middleName"] = middle_name

    # Должность — EmployeeFilter.fullPostName (строковый поиск)
    if full_post_name:
        employee_filter["fullPostName"] = full_post_name

    # Статус — передаём только True; False не несёт смысла для фильтрации
    if active_only is True:
        employee_filter["active"] = True
    if fired_only is True:
        employee_filter["fired"] = True

    # Структура: резолвим названия → UUID, затем мержим с явными UUID
    # departmentId в Java: UUID[] — ТОЛЬКО UUID, никаких строк-названий
    resolved_dept_ids: list[str] = list(department_ids or [])

    if department_names:
        newly_resolved, unresolved = await _resolve_department_names(
            token, department_names
        )
        resolved_dept_ids.extend(newly_resolved)

        if unresolved:
            logger.warning(
                "Department names not resolved to UUID",
                extra={"unresolved": unresolved},
            )
            # Если не нашли ни одного отдела и других фильтров нет — ошибка
            has_other_filters = any(
                [
                    last_name,
                    first_name,
                    middle_name,
                    full_post_name,
                    employee_ids,
                ]
            )
            if not resolved_dept_ids and not has_other_filters:
                return {
                    "status": "not_found",
                    "message": (
                        f"Отдел(ы) не найдены: {', '.join(unresolved)}. "
                        "Проверьте название подразделения в системе."
                    ),
                }

    if resolved_dept_ids:
        # Java UUID[] → передаём как список строк; Spring десериализует корректно
        employee_filter["departmentId"] = resolved_dept_ids

    if child_departments is True:
        employee_filter["childDepartments"] = True

    # ids: List<UUID> в Java — пакетный запрос по конкретным UUID
    if employee_ids:
        employee_filter["ids"] = employee_ids

    if exclude_ids:
        employee_filter["excludeIds"] = exclude_ids

    pageable = {"page": 0, "size": _MAX_RESULTS, "sort": "lastName,ASC"}

    logger.info(
        "Employee search requested",
        extra={"filter_keys": list(employee_filter.keys())},
    )

    try:
        async with EmployeeClient() as client:
            # POST /search предпочтителен: нет ограничений размера тела запроса
            results = await client.search_employees_post(
                token=token,
                employee_filter=employee_filter,
                pageable=pageable,
            )

        if not results:
            return {
                "status": "not_found",
                "message": "Сотрудники по данным критериям не найдены.",
                "employees": [],
                "total": 0,
            }

        # Один результат — сразу полная карточка
        if len(results) == 1:
            emp_card = _serialize_employee(results[0], nlp)
            logger.info(
                "Single employee found",
                extra={"id": str(results[0].get("id", ""))[:8]},
            )
            return {
                "status": "found",
                "total": 1,
                "employee_card": emp_card,
            }

        # Несколько — список для уточнения выбора
        choices = [_serialize_employee_brief(r) for r in results[:_MAX_RESULTS]]
        logger.info("Multiple employees found", extra={"count": len(choices)})

        return {
            "status": "requires_action",
            "action_type": "select_employee",
            "message": f"Найдено {len(choices)} сотрудников. Уточните выбор:",
            "total": len(choices),
            "choices": choices,
        }

    except Exception as exc:
        logger.error("Employee search failed", exc_info=True)
        return {
            "status": "error",
            "message": f"Ошибка поиска сотрудников: {exc}",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════


async def _resolve_department_names(
    token: str,
    names: list[str],
) -> tuple[list[str], list[str]]:
    """Resolves department names to UUID strings via DepartmentClient.

    Для каждого названия отдела ищет UUID через GET api/department.
    departmentId в EmployeeFilter — UUID[], строки недопустимы (→ 400).

    Args:
        token: JWT bearer token.
        names: List of department name strings from the agent.

    Returns:
        Tuple of (resolved_uuids: List[str], unresolved_names: List[str]).
    """
    # Импортируем здесь чтобы избежать циклических зависимостей
    from edms_ai_assistant.clients.department_client import DepartmentClient

    resolved: list[str] = []
    unresolved: list[str] = []

    async with DepartmentClient() as dept_client:
        for name in names:
            name_stripped = name.strip()
            if not name_stripped:
                continue
            try:
                dept = await dept_client.find_by_name(token, name_stripped)
                if dept and dept.get("id"):
                    resolved.append(str(dept["id"]))
                    logger.debug(
                        "Department resolved: '%s' → %s",
                        name_stripped,
                        str(dept["id"])[:8],
                    )
                else:
                    unresolved.append(name_stripped)
            except Exception:
                logger.warning(
                    "Failed to resolve department '%s'", name_stripped, exc_info=True
                )
                unresolved.append(name_stripped)

    return resolved, unresolved


async def _get_employee_card(
    token: str,
    employee_id: str,
    nlp: EDMSNaturalLanguageService,
) -> dict[str, Any]:
    """Fetches and formats a single employee card by UUID.

    Args:
        token: JWT bearer token.
        employee_id: Employee UUID string.
        nlp: NLP service for formatting.

    Returns:
        Tool response dict with 'employee_card' key or error message.
    """
    try:
        async with EmployeeClient() as client:
            raw = await client.get_employee(token, employee_id)

        if not raw:
            return {
                "status": "not_found",
                "message": f"Сотрудник с UUID {employee_id} не найден.",
            }

        # nlp.process_employee_info принимает EmployeeDto (Pydantic-объект), не dict
        emp = EmployeeDto.model_validate(raw)
        return {
            "status": "found",
            "total": 1,
            "employee_card": nlp.process_employee_info(emp),
        }
    except Exception as exc:
        logger.error(
            "Failed to fetch employee by id",
            exc_info=True,
            extra={"employee_id": employee_id},
        )
        return {
            "status": "error",
            "message": f"Ошибка получения сотрудника: {exc}",
        }


def _serialize_employee(
    raw: dict[str, Any],
    nlp: EDMSNaturalLanguageService,
) -> dict[str, Any]:
    """Converts raw EmployeeDto dict to full formatted card via NLP service.

    nlp.process_employee_info требует EmployeeDto (Pydantic), не dict.
    Валидируем raw dict перед передачей.

    Args:
        raw: Raw EmployeeDto dict from API.
        nlp: NLP service.

    Returns:
        Formatted employee card dict.
    """
    try:
        emp = EmployeeDto.model_validate(raw)
        return nlp.process_employee_info(emp)
    except Exception:
        logger.warning(
            "NLP/validation failed for employee, falling back to brief format",
            exc_info=True,
        )
        return _serialize_employee_brief(raw)


def _serialize_employee_brief(raw: dict[str, Any]) -> dict[str, Any]:
    """Converts raw EmployeeDto dict to compact list-item representation.

    Используется когда найдено несколько сотрудников (список для выбора).
    Содержит минимум полей для однозначной идентификации человека.

    Args:
        raw: Raw EmployeeDto dict from API.

    Returns:
        Compact dict: id, full_name, post, department, active, fired.
    """
    post: dict[str, Any] = raw.get("post") or {}
    department: dict[str, Any] = raw.get("department") or {}

    parts = [
        raw.get("lastName") or "",
        raw.get("firstName") or "",
        raw.get("middleName") or "",
    ]
    full_name = " ".join(p for p in parts if p).strip() or "—"

    return {
        "id": str(raw.get("id", "")),
        "full_name": full_name,
        "post": post.get("postName") or "—",
        "department": department.get("name") or "—",
        "active": raw.get("active"),
        "fired": raw.get("fired"),
    }
