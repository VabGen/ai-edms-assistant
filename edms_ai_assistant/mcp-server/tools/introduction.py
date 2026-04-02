"""
EDMS AI Assistant - Introduction Tool.

Инструмент для создания списков ознакомления с документами.
"""

import logging
from typing import Any
from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.services.introduction_service import IntroductionService

logger = logging.getLogger(__name__)


class IntroductionInput(BaseModel):
    """Валидированная схема входных данных для создания ознакомления."""

    token: str = Field(..., description="JWT токен авторизации пользователя")
    document_id: str = Field(
        ...,
        description="UUID документа для создания ознакомления",
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )
    last_names: list[str] | None = Field(
        None,
        description="Фамилии сотрудников для поиска (например: ['Иванов', 'Петров'])",
        max_length=50,
    )
    department_names: list[str] | None = Field(
        None,
        description="Названия подразделений для массового добавления",
        max_length=20,
    )
    group_names: list[str] | None = Field(
        None,
        description="Названия групп для массового добавления",
        max_length=20,
    )
    comment: str | None = Field(
        None, description="Комментарий к ознакомлению", max_length=500
    )
    selected_employee_ids: list[str] | None = Field(
        None,
        description=(
            "UUID выбранных сотрудников для разрешения disambiguation. "
            "Используется после выбора пользователем из списка неоднозначных совпадений."
        ),
        max_length=100,
    )

    @field_validator("last_names", "department_names", "group_names")
    @classmethod
    def validate_string_lists(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return [s.strip() for s in v if s and s.strip()]

    @field_validator("selected_employee_ids")
    @classmethod
    def validate_employee_ids(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        validated = []
        for emp_id in v:
            try:
                UUID(emp_id)
                validated.append(emp_id)
            except ValueError:
                logger.warning(f"Invalid UUID in selected_employee_ids: {emp_id}")
        return validated if validated else None


@tool("introduction_create_tool", args_schema=IntroductionInput)
async def introduction_create_tool(
    token: str,
    document_id: str,
    last_names: list[str] | None = None,
    department_names: list[str] | None = None,
    group_names: list[str] | None = None,
    comment: str | None = None,
    selected_employee_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Создает список ознакомления с документом через workflow с disambiguation.

    Поддерживаемые сценарии:
    1. Прямое добавление по selected_employee_ids (после выбора пользователя)
    2. Поиск по фамилиям/отделам/группам с автоматическим разрешением
    3. Disambiguation workflow при неоднозначных совпадениях

    Workflow:
    --------
    Шаг 1: Первый вызов
        ```python
        introduction_create_tool(
            token="...",
            document_id="uuid",
            last_names=["Иванов", "Петров"]
        )
        ```
        → Если "Иванов" неоднозначен → возврат requires_disambiguation

    Шаг 2: Пользователь выбирает из списка (UI interaction)

    Шаг 3: Повторный вызов с выбранными ID
        ```python
        introduction_create_tool(
            token="...",
            document_id="uuid",
            selected_employee_ids=["uuid1", "uuid3"]
        )
        ```
        → Создание ознакомления для выбранных сотрудников

    Args:
        token: JWT токен авторизации
        document_id: UUID документа
        last_names: Список фамилий для поиска
        department_names: Список названий отделов
        group_names: Список названий групп
        comment: Комментарий к ознакомлению
        selected_employee_ids: UUID выбранных сотрудников (для disambiguation)

    Returns:
        Dict с ключами:
        - status: "success" | "requires_disambiguation" | "error"
        - message: Информационное сообщение
        - action_type: "select_employee" (для disambiguation)
        - ambiguous_matches: List[Dict] (список неоднозначных совпадений)
        - added_count: int (количество добавленных сотрудников)
        - not_found: List[str] (не найденные критерии)

    Examples:
         # Успешное добавление
         result = await introduction_create_tool(
        ...     token="jwt_token",
        ...     document_id="doc_uuid",
        ...     last_names=["Иванов"]
        ... )
         # {"status": "success", "message": "Успешно добавлено 1 сотрудников", ...}

         # Требуется уточнение
         result = await introduction_create_tool(
        ...     token="jwt_token",
        ...     document_id="doc_uuid",
        ...     last_names=["Иванов"]  # Несколько Ивановых
        ... )
         # {"status": "requires_disambiguation", "ambiguous_matches": [...], ...}
    """
    logger.info(
        "Creating introduction",
        extra={
            "document_id": document_id,
            "last_names": last_names,
            "departments": department_names,
            "groups": group_names,
            "has_selected_ids": bool(selected_employee_ids),
        },
    )

    try:
        async with IntroductionService() as service:
            if selected_employee_ids:
                return await _handle_direct_addition(
                    service=service,
                    token=token,
                    document_id=document_id,
                    employee_ids=selected_employee_ids,
                    comment=comment,
                )

            return await _handle_search_and_create(
                service=service,
                token=token,
                document_id=document_id,
                last_names=last_names,
                department_names=department_names,
                group_names=group_names,
                comment=comment,
            )

    except Exception as e:
        logger.error(
            f"Introduction creation failed: {e}",
            exc_info=True,
            extra={"document_id": document_id},
        )
        return {
            "status": "error",
            "message": f"❌ Произошла ошибка при создании ознакомления: {e!s}",
        }


async def _handle_direct_addition(
    service: IntroductionService,
    token: str,
    document_id: str,
    employee_ids: list[str],
    comment: str | None,
) -> dict[str, Any]:
    """
    Обработка прямого добавления сотрудников по UUID.

    Используется после disambiguation, когда пользователь выбрал конкретных сотрудников.
    """
    logger.info(f"Direct addition of {len(employee_ids)} employees")

    if not employee_ids:
        return {
            "status": "error",
            "message": "Не указаны ID сотрудников для добавления.",
        }

    result = await service.create_introduction(
        token=token,
        document_id=document_id,
        employee_ids=[UUID(emp_id) for emp_id in employee_ids],
        comment=comment,
    )

    if result.success:
        return {
            "status": "success",
            "message": (
                f"✅ Успешно добавлено {result.added_count} сотрудников "
                f"в список ознакомления."
            ),
            "added_count": result.added_count,
        }

    return {
        "status": "error",
        "message": (
            result.error_message
            or "❌ Не удалось создать ознакомление. "
            "Проверьте права доступа или корректность данных."
        ),
    }


async def _handle_search_and_create(
    service: IntroductionService,
    token: str,
    document_id: str,
    last_names: list[str] | None,
    department_names: list[str] | None,
    group_names: list[str] | None,
    comment: str | None,
) -> dict[str, Any]:
    """
    Обработка поиска сотрудников с созданием ознакомления или disambiguation.

    Workflow:
    1. Резолвинг сотрудников по критериям поиска
    2. Если есть неоднозначности → возврат requires_disambiguation
    3. Если все однозначно → создание ознакомления
    """
    resolution_result = await service.resolve_employees(
        token=token,
        last_names=last_names or [],
        department_names=department_names or [],
        group_names=group_names or [],
    )

    employee_ids = resolution_result.employee_ids
    not_found = resolution_result.not_found
    ambiguous_results = resolution_result.ambiguous

    if ambiguous_results:
        logger.info(f"Found {len(ambiguous_results)} ambiguous search terms")
        return _build_disambiguation_response(ambiguous_results)

    if not employee_ids:
        not_found_str = (
            ", ".join(not_found) if not_found else "Критерии поиска не заданы"
        )
        return {
            "status": "error",
            "message": f"❌ Не найдено ни одного сотрудника. Не найдены: {not_found_str}",
            "not_found": not_found,
        }

    logger.info(f"Creating introduction with {len(employee_ids)} employees")

    result = await service.create_introduction(
        token=token,
        document_id=document_id,
        employee_ids=list(employee_ids),
        comment=comment,
    )

    if result.success:
        response = {
            "status": "success",
            "message": (
                f"✅ Успешно добавлено {result.added_count} сотрудников "
                f"в список ознакомления."
            ),
            "added_count": result.added_count,
        }

        if not_found:
            response["partial_success"] = True
            response["not_found"] = not_found
            response["message"] += f" ⚠️ Не найдено: {', '.join(not_found)}."

        return response

    return {
        "status": "error",
        "message": (
            result.error_message
            or "❌ Не удалось создать ознакомление. "
            "Проверьте права доступа или корректность данных."
        ),
    }


def _build_disambiguation_response(
    ambiguous_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Формирует структурированный ответ для disambiguation workflow.

    Args:
        ambiguous_results: Список неоднозначных совпадений из сервиса

    Returns:
        Стандартизированный ответ с action_type для UI
    """
    formatted_choices = []

    for amb in ambiguous_results:
        search_term = amb.get("search_query", "Неизвестно")
        matches = amb.get("matches", [])

        for match in matches:
            formatted_choices.append(
                {
                    "id": match.get("id"),
                    "full_name": match.get("full_name", "Не указано"),
                    "post": match.get("post", "Не указана"),
                    "department": match.get("department", "Не указан"),
                    "search_term": search_term,
                }
            )

    return {
        "status": "requires_disambiguation",
        "action_type": "select_employee",
        "message": "Найдено несколько совпадений. Пожалуйста, уточните выбор:",
        "ambiguous_matches": formatted_choices,
        "instruction": (
            "Выберите нужных сотрудников из списка. "
            "Затем вызовите инструмент повторно с параметром selected_employee_ids."
        ),
    }
