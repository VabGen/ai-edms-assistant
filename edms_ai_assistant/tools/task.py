# edms_ai_assistant/tools/task.py
"""
Task Creation Tool with Disambiguation Workflow.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.models.task_models import TaskType
from edms_ai_assistant.services.task_service import TaskService

logger = logging.getLogger(__name__)


class TaskCreateInput(BaseModel):
    """
    Схема входных данных с поддержкой disambiguation.
    """

    token: str = Field(..., description="Токен авторизации пользователя (JWT)")
    document_id: str = Field(..., description="UUID документа для создания поручения")

    task_text: str = Field(..., description="Текст поручения (обязательно)")

    executor_last_names: list[str] | None = Field(
        None,
        description="Фамилии исполнителей (например: ['Иванов', 'Петров'])",
    )

    selected_employee_ids: list[str] | None = Field(
        None,
        description="UUID сотрудников, выбранных пользователем для разрешения неоднозначности",
    )

    responsible_last_name: str | None = Field(
        None,
        description=(
            "Фамилия ответственного исполнителя. "
            "Если не указано, первый исполнитель будет назначен ответственным."
        ),
    )

    planed_date_end: str | None = Field(
        None,
        description=(
            "Плановая дата окончания в ISO 8601 (например: '2026-02-15T23:59:59Z'). "
            "Если не указано, будет установлен срок +7 дней."
        ),
    )

    task_type: TaskType | None = Field(
        TaskType.GENERAL, description="Тип поручения (по умолчанию: GENERAL)"
    )


@tool("task_create_tool", args_schema=TaskCreateInput)
async def task_create_tool(
    token: str,
    document_id: str,
    task_text: str,
    executor_last_names: list[str] | None = None,
    selected_employee_ids: list[str] | None = None,
    responsible_last_name: str | None = None,
    planed_date_end: str | None = None,
    task_type: TaskType | None = TaskType.GENERAL,
) -> dict[str, Any]:
    """
    Создает поручение с поддержкой disambiguation.

    Workflow:
    1. Если фамилия неоднозначна → возвращает "requires_disambiguation"
    2. Пользователь выбирает из списка
    3. Инструмент вызывается повторно с selected_employee_ids
    """
    logger.info(
        f"[TASK-TOOL] Creating task for document {document_id}. "
        f"Executors: {executor_last_names}, SelectedIDs: {selected_employee_ids}"
    )

    # ═══════════════════════════════════════════════════════════════
    # ВАЛИДАЦИЯ
    # ═══════════════════════════════════════════════════════════════
    if not any([executor_last_names, selected_employee_ids]):
        return {
            "status": "error",
            "message": "Необходимо указать executor_last_names или selected_employee_ids.",
        }

    if not task_text or not task_text.strip():
        return {
            "status": "error",
            "message": "Текст поручения не может быть пустым.",
        }

    # Парсинг даты
    deadline = None
    if planed_date_end:
        try:
            deadline = datetime.fromisoformat(planed_date_end.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Неверный формат даты: {e}",
            }

    # ═══════════════════════════════════════════════════════════════
    # ОСНОВНАЯ ЛОГИКА
    # ═══════════════════════════════════════════════════════════════
    try:
        async with TaskService() as service:

            # ШАГ 3: Если пользователь УЖЕ ВЫБРАЛ из списка
            if selected_employee_ids:
                from uuid import UUID

                employee_uuids = [UUID(emp_id) for emp_id in selected_employee_ids]

                result = await service.create_task_by_employee_ids(
                    token=token,
                    document_id=document_id,
                    task_text=task_text,
                    employee_ids=employee_uuids,
                    planed_date_end=deadline,
                    task_type=task_type or TaskType.GENERAL,
                )

                if result.success:
                    return {
                        "status": "success",
                        "message": f"✅ Поручение успешно создано. Исполнителей: {result.created_count}",
                        "created_count": result.created_count,
                    }
                else:
                    return {
                        "status": "error",
                        "message": result.error_message,
                    }

            # ШАГ 1: Создание с проверкой на неоднозначность
            result = await service.create_task(
                token=token,
                document_id=document_id,
                task_text=task_text,
                executor_last_names=executor_last_names,
                planed_date_end=deadline,
                responsible_last_name=responsible_last_name,
                task_type=task_type or TaskType.GENERAL,
            )

            if result.status == "requires_disambiguation":
                logger.info(
                    f"[TASK-TOOL] Disambiguation required. "
                    f"Ambiguous matches: {len(result.ambiguous_matches)}"
                )

                flat_candidates: list[dict[str, Any]] = []
                for group in result.ambiguous_matches or []:
                    for match in group.get("matches", []):
                        flat_candidates.append(
                            {
                                "id": match.get("id", ""),
                                "full_name": match.get("full_name", "Не указано"),
                                "post": match.get("post", ""),
                                "department": match.get("department", ""),
                            }
                        )

                return {
                    "status": "requires_disambiguation",
                    "message": "⚠️ Найдено несколько сотрудников с указанными фамилиями. Выберите нужного из списка:",
                    "ambiguous_matches": flat_candidates,
                    "instruction": (
                        "Пожалуйста, выберите конкретного сотрудника из списка. "
                        "Затем вызовите инструмент повторно с параметром selected_employee_ids."
                    ),
                }

            if result.success:
                response = {
                    "status": "success",
                    "message": f"✅ Поручение успешно создано. Исполнителей: {result.created_count}",
                    "created_count": result.created_count,
                }

                if result.not_found_employees:
                    response["partial_success"] = True
                    response["not_found"] = result.not_found_employees
                    response[
                        "message"
                    ] += f" Не найдено: {', '.join(result.not_found_employees)}."

                return response

            return {
                "status": "error",
                "message": result.error_message,
                "not_found_employees": result.not_found_employees,
            }

    except Exception as e:
        logger.error(f"[TASK-TOOL] Error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Произошла ошибка при создании поручения: {e!s}",
        }
