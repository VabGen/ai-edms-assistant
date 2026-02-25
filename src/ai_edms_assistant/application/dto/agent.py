# src/ai_edms_assistant/application/dto/agent.py
"""DTOs for agent communication (request/response models).

This module defines the contract between the interface layer (FastAPI endpoints)
and the application layer (agent orchestration). All DTOs are immutable Pydantic
models with comprehensive validation.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentStatus(StrEnum):
    """Статусы обработки запроса агентом.

    Attributes:
        SUCCESS: Запрос успешно обработан, финальный ответ готов.
        ERROR: Произошла ошибка при обработке (техническая или бизнес-логики).
        REQUIRES_ACTION: Требуется действие пользователя (disambiguation, выбор формата).
        PROCESSING: Запрос в процессе обработки (для streaming responses).
    """

    SUCCESS = "success"
    ERROR = "error"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"


class ActionType(StrEnum):
    """Типы интерактивных действий, требующих участия пользователя.

    Attributes:
        SUMMARIZE_SELECTION: Пользователь должен выбрать формат суммаризации
            (extractive, abstractive, bullets).
        DISAMBIGUATION: Требуется уточнение — найдено несколько кандидатов
            (например, несколько сотрудников с похожими именами).
        CONFIRMATION: Требуется подтверждение опасного или необратимого действия
            (например, удаление документа, массовая рассылка).
    """

    SUMMARIZE_SELECTION = "summarize_selection"
    DISAMBIGUATION = "requires_disambiguation"
    CONFIRMATION = "requires_confirmation"


class AgentRequest(BaseModel):
    """Валидированный входящий запрос к агенту.

    Принимается от interface layer (FastAPI endpoint) и передаётся в
    application layer (EdmsDocumentAgent.chat). Содержит все данные,
    необходимые для обработки пользовательского запроса.

    Attributes:
        message: Текст пользовательского запроса на естественном языке.
        user_token: JWT bearer token для аутентификации EDMS API запросов.
        context_ui_id: UUID активного документа в UI (опционально).
            Когда указан, агент работает в режиме "document context".
        thread_id: Идентификатор сессии для LangGraph checkpointer.
            Позволяет агенту сохранять историю conversation state.
        user_context: Дополнительный контекст пользователя (имя, роль, департамент).
            Используется для персонализации ответов и логирования.
        file_path: Путь к загруженному файлу или UUID вложения документа.
            Поддерживает форматы: UUID, Unix path, Windows path.
        human_choice: Выбор пользователя при disambiguation или выборе формата.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="User's natural-language message",
        examples=["Создай поручение для Иванова", "Покажи документы за декабрь"],
    )

    user_token: str = Field(
        ...,
        min_length=20,
        description="JWT bearer token for EDMS API authentication",
        examples=["Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )

    context_ui_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f-]{36}$|^$",
        description="UUID of active document in UI",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )

    thread_id: str | None = Field(
        default=None,
        max_length=255,
        description="Conversation thread ID for LangGraph checkpointer",
        examples=["session-abc-123", "conv-20250220-001"],
    )

    user_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional user metadata (name, role, department)",
        examples=[{"firstName": "Петр", "lastName": "Сидоров", "role": "manager"}],
    )

    file_path: str | None = Field(
        default=None,
        max_length=500,
        description="Uploaded file path or attachment UUID",
        examples=[
            "/tmp/uploaded_file.docx",
            "C:\\Users\\user\\file.pdf",
            "550e8400-e29b-41d4-a716-446655440000",
        ],
    )

    human_choice: str | None = Field(
        default=None,
        max_length=100,
        description="User's choice in disambiguation or format selection",
        examples=["extractive", "employee_1", "confirm"],
    )

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Trim whitespace and validate message is not empty.

        Args:
            v: Raw message string.

        Returns:
            Trimmed message.

        Raises:
            ValueError: When message is empty after trimming.
        """
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Message cannot be empty")
        return trimmed

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str | None) -> str | None:
        """Validate file_path format (UUID, Unix path, or Windows path).

        Поддерживаемые форматы:
        - UUID: 550e8400-e29b-41d4-a716-446655440000
        - Unix absolute path: /tmp/file.docx
        - Unix relative path: uploads/file.docx
        - Windows path: C:\\Users\\user\\file.docx

        Args:
            v: Raw file path or UUID.

        Returns:
            Validated file path or UUID.

        Raises:
            ValueError: When format is invalid.
        """
        if not v:
            return None

        # Check UUID format (8-4-4-4-12 hex digits)
        if re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            v,
            re.IGNORECASE,
        ):
            return v

        # Check file path format (must be under 500 chars)
        if len(v) >= 500:
            raise ValueError(f"File path too long: {len(v)} chars (max 500)")

        # Unix absolute path
        if v.startswith("/"):
            return v

        # Windows path (C:\, D:\, etc.)
        if re.match(r"^[A-Za-z]:\\", v):
            return v

        # Relative path (contains / or \)
        if re.match(r"^[^/\\]+[\\/]", v):
            return v

        raise ValueError(
            f"Invalid file_path format: '{v}'. "
            f"Expected UUID, absolute path (/...), or Windows path (C:\\...)"
        )


class AgentResponse(BaseModel):
    """Стандартизированный ответ агента.

    Возвращается из application layer (EdmsDocumentAgent.chat) в
    interface layer (FastAPI endpoint) для сериализации в JSON.

    Attributes:
        status: Статус обработки запроса (SUCCESS, ERROR, REQUIRES_ACTION, PROCESSING).
        content: Финальный текст ответа агента (для SUCCESS).
            Содержит natural-language ответ на русском языке.
        message: Системное сообщение об ошибке или статусе (для ERROR/PROCESSING).
            Используется когда content отсутствует.
        action_type: Тип требуемого действия пользователя (для REQUIRES_ACTION).
            Указывает, какое конкретно действие требуется (disambiguation, confirmation).
        metadata: Дополнительные метаданные выполнения.
            Может содержать: iterations, tokens_used, latency_ms, tool_calls, warnings.

    Example:
        >>> # Success response
        >>> response = AgentResponse(
        ...     status=AgentStatus.SUCCESS,
        ...     content="Поручение создано для Иванова И.И. со сроком до 15.02.2026",
        ...     metadata={
        ...         "task_id": "uuid-123",
        ...         "iterations": 3,
        ...         "tokens_used": 450,
        ...         "latency_ms": 1200,
        ...     },
        ... )
        >>>
        >>> # Error response
        >>> error_response = AgentResponse(
        ...     status=AgentStatus.ERROR,
        ...     message="Сотрудник 'Иванов' не найден в системе",
        ...     metadata={"error_type": "EmployeeNotFound"},
        ... )
        >>>
        >>> # Requires action response
        >>> action_response = AgentResponse(
        ...     status=AgentStatus.REQUIRES_ACTION,
        ...     action_type=ActionType.DISAMBIGUATION,
        ...     content="Найдено 3 сотрудника: Иванов А.А., Иванов Б.Б., Иванов В.В.",
        ...     metadata={"candidates": ["uuid-1", "uuid-2", "uuid-3"]},
        ... )
    """

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    status: AgentStatus = Field(
        ...,
        description="Agent execution status",
        examples=["success", "error", "requires_action"],
    )

    content: str | None = Field(
        default=None,
        description="Final agent response text (for SUCCESS)",
        examples=["Поручение создано для Иванова И.И. со сроком до 15.02.2026"],
    )

    message: str | None = Field(
        default=None,
        description="System message or error description (for ERROR/PROCESSING)",
        examples=["Превышен лимит итераций обработки", "Сотрудник не найден"],
    )

    action_type: ActionType | None = Field(
        default=None,
        description="Type of required user action (for REQUIRES_ACTION)",
        examples=["requires_disambiguation", "summarize_selection"],
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata",
        examples=[
            {
                "iterations": 3,
                "tokens_used": 450,
                "latency_ms": 1200,
                "tool_calls": ["employee_search", "task_create"],
            }
        ],
    )
