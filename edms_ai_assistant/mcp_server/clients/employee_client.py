# edms_ai_assistant/clients/employee_client.py
"""
EDMS AI Assistant — Employee HTTP Client.

──────────────────────────────────────────────────────────────────
  ПОИСК И ЧТЕНИЕ
  GET  /                          → search_employees()        (SliceDto<EmployeeDto>)
  POST /search                    → search_employees_post()   (SliceDto<EmployeeDto>)
  GET  /{id}                      → get_employee()            (EmployeeDto)
  GET  /me                        → get_current_user()        (CurrentUser)
  GET  /fts-lastname              → find_by_last_name_fts()   (EmployeeDto, top-1)

  ЖИЗНЕННЫЙ ЦИКЛ
  POST /dismiss                   → dismiss_employee()        (204)
  POST /recover                   → recover_employee()        (204)
  POST /disable                   → disable_employee()        (204)
  POST /enable                    → enable_employee()         (204)

Намеренно НЕ реализованы (не нужны агенту):
  - Аватар / факсимиле (бинарный стриминг — не JSON)
  - Роли / группы / грифы (управление доступом, не входит в сценарии агента)
  - История входов / действий (аудит, не входит в сценарии агента)
──────────────────────────────────────────────────────────────────

Соглашения:
  - Все методы async/await.
  - Клиент — тупой транспорт: никакой бизнес-логики, форматирования или NLP.
  - dict-эндпоинты  → Optional[Dict]  (None при пустом / 404-ответе)
  - list-эндпоинты  → List[Dict]      ([] при пустом ответе)
  - action (204)    → bool            (True = успех)

Параметры EmployeeFilter (resources_openapi.py):
  firstName, lastName, middleName   — ФИО поиск
  fired: bool                       — уволенные
  active: bool                      — активные
  postId: int                       — ID должности
  departmentId: list[UUID]          — отделы
  ids: set[UUID]                    — конкретные UUID
  includes: list[Include1]          — POST | DEPARTMENT (вложенные объекты)
  childDepartments: bool            — включая дочерние отделы
  excludeIds: list[UUID]            — исключить из выборки
  all: bool                         — без пагинации (осторожно: большой ответ)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .base_client import EdmsBaseClient, EdmsHttpClient

logger = logging.getLogger(__name__)

# Дефолтные параметры пагинации
_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 20

# Дефолтные includes для большинства сценариев агента
_DEFAULT_INCLUDES: list[str] = ["POST", "DEPARTMENT"]


# ══════════════════════════════════════════════════════════════════════════════
# Abstract Interface
# ══════════════════════════════════════════════════════════════════════════════


class BaseEmployeeClient(EdmsBaseClient, ABC):
    """Abstract interface for EDMS Employee API clients.

    Определяет контракт для всех read/write методов EmployeeController.java,
    релевантных для агента.
    """

    # ── Поиск и чтение ────────────────────────────────────────────────────────

    @abstractmethod
    async def search_employees(
        self,
        token: str,
        employee_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Searches employees via GET api/employee (EmployeeFilter as query params)."""
        raise NotImplementedError

    @abstractmethod
    async def search_employees_post(
        self,
        token: str,
        employee_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Searches employees via POST api/employee/search (EmployeeFilter as JSON body)."""
        raise NotImplementedError

    @abstractmethod
    async def get_employee(self, token: str, employee_id: str) -> dict[str, Any] | None:
        """Fetches a single EmployeeDto by UUID."""
        raise NotImplementedError

    @abstractmethod
    async def get_current_user(self, token: str) -> dict[str, Any] | None:
        """Fetches CurrentUser info for the authenticated user (GET /me)."""
        raise NotImplementedError

    @abstractmethod
    async def find_by_last_name_fts(
        self, token: str, last_name: str
    ) -> dict[str, Any] | None:
        """Full-text search by last name, returns top-1 EmployeeDto."""
        raise NotImplementedError

    # ── Жизненный цикл ────────────────────────────────────────────────────────

    @abstractmethod
    async def dismiss_employee(
        self, token: str, employee_id: str, delegate_to_id: str
    ) -> bool:
        """Dismisses an employee, delegating their responsibilities."""
        raise NotImplementedError

    @abstractmethod
    async def recover_employee(self, token: str, employee_id: str) -> bool:
        """Recovers a dismissed employee."""
        raise NotImplementedError

    @abstractmethod
    async def disable_employee(
        self, token: str, employee_id: str, delegate_to_id: str
    ) -> bool:
        """Deactivates an employee account, delegating responsibilities."""
        raise NotImplementedError

    @abstractmethod
    async def enable_employee(self, token: str, employee_id: str) -> bool:
        """Activates a previously disabled employee account."""
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Concrete Implementation
# ══════════════════════════════════════════════════════════════════════════════


class EmployeeClient(BaseEmployeeClient, EdmsHttpClient):
    """Concrete async HTTP client for EDMS Employee API.

    Реализует взаимодействие с EmployeeController.java.

    Важно про пагинацию:
      GET  /api/employee          → Spring Slice<EmployeeDto> (НЕ Page — нет totalElements)
      POST /api/employee/search   → Spring Slice<EmployeeDto>

      Структура Slice: { "content": [...], "hasNext": bool, "hasPrevious": bool }
      Метод _extract_slice_content() корректно извлекает content.

    Важно про includes:
      Без includes=[POST, DEPARTMENT] в ответе не будет вложенных объектов.
    """

    # ── Поиск и чтение ────────────────────────────────────────────────────────

    async def search_employees(
        self,
        token: str,
        employee_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Searches employees using EmployeeFilter as GET query params.

        Calls GET api/employee.
        Используй для простых поисков с небольшим числом параметров.
        Для сложных фильтров (большой список ids и пр.) используй search_employees_post.

        EmployeeFilter поля (все опциональны):
            lastName, firstName, middleName  — поиск по ФИО
            active: bool                     — только активные (True для большинства запросов)
            fired: bool                      — только уволенные
            departmentId: list[UUID]         — фильтр по отделу(ам)
            postId: int                      — фильтр по должности (ID, не название)
            includes: list[str]              — POST | DEPARTMENT (вложения)
            childDepartments: bool           — включая дочерние подразделения

        Args:
            token: JWT bearer token.
            employee_filter: EmployeeFilter fields as dict. None → только pageable.
            pageable: {'page': int, 'size': int, 'sort': str}. None → дефолты.

        Returns:
            List of EmployeeDto dicts from Slice.content. Empty list if none found.
        """
        effective_filter = _ensure_includes(employee_filter or {})
        params = _build_params(effective_filter, pageable)

        logger.debug(
            "Searching employees (GET)",
            extra={"filter_keys": list(effective_filter.keys())},
        )

        result = await self._make_request(
            "GET", "api/employee", token=token, params=params
        )
        return _extract_slice_content(result, endpoint="GET api/employee")

    async def search_employees_post(
        self,
        token: str,
        employee_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Searches employees using EmployeeFilter as POST JSON body.

        Calls POST api/employee/search.
        Предпочтительный метод: тело запроса не имеет ограничений на размер,
        в отличие от query params. Используй когда передаёшь ids (set[UUID]) и т.п.

        Параметры Pageable передаются как query params: page, size, sort.
        Тело запроса — EmployeeFilter (JSON).

        Args:
            token: JWT bearer token.
            employee_filter: EmployeeFilter as dict (JSON body). None → пустой фильтр.
            pageable: {'page': int, 'size': int, 'sort': str}. None → дефолты.

        Returns:
            List of EmployeeDto dicts from Slice.content. Empty list if none found.
        """
        effective_filter = _ensure_includes(employee_filter or {})
        pageable_params = _build_pageable_params(pageable)

        logger.debug(
            "Searching employees (POST)",
            extra={"filter_keys": list(effective_filter.keys())},
        )

        result = await self._make_request(
            "POST",
            "api/employee/search",
            token=token,
            params=pageable_params,
            json=effective_filter,
        )
        return _extract_slice_content(result, endpoint="POST api/employee/search")

    async def get_employee(self, token: str, employee_id: str) -> dict[str, Any] | None:
        """Fetches a single employee by UUID.

        Calls GET api/employee/{id}.

        Args:
            token: JWT bearer token.
            employee_id: Employee UUID string.

        Returns:
            EmployeeDto as dict, or None if not found.
        """
        result = await self._make_request(
            "GET", f"api/employee/{employee_id}", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_current_user(self, token: str) -> dict[str, Any] | None:
        """Fetches CurrentUser info for the authenticated user.

        Calls GET api/employee/me.
        Возвращает расширенный профиль текущего пользователя: id, ФИО,
        departmentId, organizationId, subordinates и права доступа.

        Используй для получения UUID текущего пользователя без передачи его явно.

        Args:
            token: JWT bearer token.

        Returns:
            CurrentUser as dict, or None on failure.
        """
        result = await self._make_request("GET", "api/employee/me", token=token)
        return result if isinstance(result, dict) and result else None

    async def find_by_last_name_fts(
        self, token: str, last_name: str
    ) -> dict[str, Any] | None:
        """Full-text search by last name, returns top-1 EmployeeDto.

        Calls GET api/employee/fts-lastname?fts={last_name}.
        Возвращает единственного наиболее релевантного сотрудника.
        Используй для быстрого поиска «найди Иванова» без полного перебора.

        Если нужен список — используй search_employees_post с lastName.

        Args:
            token: JWT bearer token.
            last_name: Last name search string (partial match supported).

        Returns:
            EmployeeDto as dict if found, None if not found (404 = not found).
        """
        try:
            result = await self._make_request(
                "GET",
                "api/employee/fts-lastname",
                token=token,
                params={"fts": last_name.strip()},
            )
            if isinstance(result, dict) and result:
                logger.debug(
                    "FTS employee found",
                    extra={
                        "last_name": last_name,
                        "found_id": str(result.get("id", ""))[:8],
                    },
                )
                return result
            return None
        except Exception:
            logger.warning("FTS search failed for '%s'", last_name, exc_info=True)
            return None

    # ── Жизненный цикл ────────────────────────────────────────────────────────

    async def dismiss_employee(
        self,
        token: str,
        employee_id: str,
        delegate_to_id: str,
    ) -> bool:
        """Dismisses an employee, delegating their responsibilities.

        Calls POST api/employee/dismiss.
        Увольнение передаёт все документы и обязанности сотрудника delegate_to_id.
        Оба параметра обязательны (@NotNull в контроллере).
        Возвращает 204 No Content при успехе.

        Args:
            token: JWT bearer token.
            employee_id: UUID of the employee to dismiss.
            delegate_to_id: UUID of the employee who receives responsibilities.

        Returns:
            True on success, False on failure.
        """
        return await self._action_request(
            "POST",
            "api/employee/dismiss",
            token=token,
            json={"id": employee_id, "to": delegate_to_id},
            log_context={"employee_id": employee_id, "delegate_to": delegate_to_id},
        )

    async def recover_employee(self, token: str, employee_id: str) -> bool:
        """Recovers a previously dismissed employee.

        Calls POST api/employee/recover with body { "id": employee_id }.
        Восстанавливает уволенного сотрудника без делегирования.
        Возвращает 204 No Content при успехе.

        Args:
            token: JWT bearer token.
            employee_id: UUID of the employee to recover.

        Returns:
            True on success, False on failure.
        """
        return await self._action_request(
            "POST",
            "api/employee/recover",
            token=token,
            json={"id": employee_id},
            log_context={"employee_id": employee_id},
        )

    async def disable_employee(
        self,
        token: str,
        employee_id: str,
        delegate_to_id: str,
    ) -> bool:
        """Deactivates an employee account, delegating responsibilities.

        Calls POST api/employee/disable.
        Деактивация (не увольнение): аккаунт блокируется, документы передаются.
        Оба параметра обязательны (@NotNull в контроллере).
        Возвращает 204 No Content при успехе.

        Args:
            token: JWT bearer token.
            employee_id: UUID of the employee to deactivate.
            delegate_to_id: UUID of the employee who receives responsibilities.

        Returns:
            True on success, False on failure.
        """
        return await self._action_request(
            "POST",
            "api/employee/disable",
            token=token,
            json={"id": employee_id, "to": delegate_to_id},
            log_context={"employee_id": employee_id, "delegate_to": delegate_to_id},
        )

    async def enable_employee(self, token: str, employee_id: str) -> bool:
        """Activates a previously disabled employee account.

        Calls POST api/employee/enable with body { "id": employee_id }.
        Активирует заблокированный аккаунт.
        Возвращает 204 No Content при успехе.

        Args:
            token: JWT bearer token.
            employee_id: UUID of the employee to activate.

        Returns:
            True on success, False on failure.
        """
        return await self._action_request(
            "POST",
            "api/employee/enable",
            token=token,
            json={"id": employee_id},
            log_context={"employee_id": employee_id},
        )

    # ── Внутренние хелперы ────────────────────────────────────────────────────

    async def _action_request(
        self,
        method: str,
        endpoint: str,
        token: str,
        json: dict[str, Any],
        log_context: dict[str, Any],
    ) -> bool:
        """Executes a write action that returns 204 No Content.

        Centralizes error handling for all lifecycle action endpoints.
        Поглощает исключения и логирует — вызывающий код получает bool.

        Args:
            method: HTTP method ('POST', 'PUT', 'DELETE').
            endpoint: Relative API endpoint path.
            token: JWT bearer token.
            json: Request body as dict.
            log_context: Extra fields for structured logging.

        Returns:
            True on success (2xx), False on any exception.
        """
        try:
            await self._make_request(
                method,
                endpoint,
                token=token,
                json=json,
                is_json_response=False,
            )
            logger.info(
                "Employee action succeeded: %s %s",
                method,
                endpoint,
                extra=log_context,
            )
            return True
        except Exception:
            logger.error(
                "Employee action failed: %s %s",
                method,
                endpoint,
                exc_info=True,
                extra=log_context,
            )
            return False


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers (pure functions, no state)
# ══════════════════════════════════════════════════════════════════════════════


def _ensure_includes(
    employee_filter: dict[str, Any],
) -> dict[str, Any]:
    """Adds default includes if caller did not specify any.

    Without includes=[POST, DEPARTMENT], the API response will not contain
    nested post and department objects — most agent tools need them.

    Args:
        employee_filter: Raw EmployeeFilter dict from caller.

    Returns:
        Copy of employee_filter with 'includes' guaranteed to be set.
    """
    if not employee_filter.get("includes"):
        return {**employee_filter, "includes": _DEFAULT_INCLUDES}
    return employee_filter


def _build_pageable_params(
    pageable: dict[str, Any] | None,
) -> dict[str, Any]:
    """Builds Spring Pageable query params dict from optional input.

    Args:
        pageable: Optional dict with 'page', 'size', 'sort' keys.

    Returns:
        Pageable params dict with defaults applied.
    """
    effective: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
    if pageable:
        effective.update(pageable)
    return effective


def _build_params(
    employee_filter: dict[str, Any],
    pageable: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merges EmployeeFilter and Pageable into a single query params dict.

    Args:
        employee_filter: EmployeeFilter fields dict.
        pageable: Pageable params dict or None.

    Returns:
        Merged query params dict.
    """
    return {**employee_filter, **_build_pageable_params(pageable)}


def _extract_slice_content(
    result: Any,
    endpoint: str = "api/employee",
) -> list[dict[str, Any]]:
    """Extracts content list from Spring Slice<EmployeeDto> response.

    Spring Slice structure:
        { "content": [...], "hasNext": bool, "hasPrevious": bool }
    Отличие от Page: нет totalElements/totalPages — это намеренно
    (Slice дешевле в SQL, не делает COUNT(*)).

    Args:
        result: Raw response from _make_request (dict, list, or other).
        endpoint: Endpoint name for warning log context.

    Returns:
        List of EmployeeDto dicts. Empty list on any unexpected shape.
    """
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            return content
        logger.warning(
            "Unexpected Slice response from %s: dict without 'content'",
            endpoint,
            extra={"response_keys": list(result.keys())},
        )
        return []

    if isinstance(result, list):
        logger.warning("%s returned raw list instead of Slice<EmployeeDto>", endpoint)
        return result

    return []
