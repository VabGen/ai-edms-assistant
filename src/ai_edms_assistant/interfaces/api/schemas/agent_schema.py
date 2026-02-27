# src/ai_edms_assistant/interfaces/api/schemas/agent_schema.py
"""Pydantic request/response schemas for the Agent API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class UserContextSchema(BaseModel):
    """Контекст текущего пользователя, опционально передаётся фронтендом.

    Attributes:
        firstName: Имя пользователя (для обращения в ответах агента).
        lastName: Фамилия пользователя.
        role: Роль / должность.
    """

    firstName: Optional[str] = None
    lastName: Optional[str] = None
    role: Optional[str] = None


class ChatRequest(BaseModel):
    """Тело запроса POST /chat.

    Attributes:
        message: Сообщение пользователя (1-8000 символов).
        user_token: JWT токен авторизации.
        context_ui_id: UUID активного документа в UI (опционально).
        context: Контекст пользователя (имя, роль). Берётся из фронтенда.
        file_path: Путь к загруженному файлу или UUID вложения документа.
        human_choice: Выбор пользователя при disambiguation или выборе формата.
            Принимает: "1"/"2"/"3", "thesis", "Тезисный план" и т.д.
            Нормализуется агентом через _normalize_choice().
        thread_id: ID треда LangGraph.
            При resume после human_choice: передаётся choice_thread_id
            из предыдущего ChatResponse.
    """

    message: str = Field(..., min_length=1, description="Сообщение пользователя")
    user_token: str = Field(..., description="JWT токен авторизации")
    context_ui_id: Optional[str] = Field(
        None, description="UUID открытого документа в UI"
    )
    context: Optional[UserContextSchema] = Field(
        None, description="Данные текущего пользователя"
    )
    file_path: Optional[str] = Field(None, description="Путь к файлу или UUID вложения")
    human_choice: Optional[str] = Field(
        None,
        description=(
            "Выбор пользователя при disambiguation или выборе формата суммаризации. "
            "Принимает: '1'/'2'/'3', 'thesis', 'Тезисный план' и т.д."
        ),
    )
    thread_id: Optional[str] = Field(
        None,
        description=(
            "ID треда LangGraph. При resume после human_choice — "
            "передавать choice_thread_id из предыдущего ответа."
        ),
    )


class ChatResponse(BaseModel):
    """Ответ агента.

    Attributes:
        status: Статус обработки.
            - 'success': финальный ответ готов (поле response)
            - 'error': техническая ошибка (поле message)
            - 'requires_action': нужен выбор пользователя (поля options, choice_thread_id)
        response: Финальный текст ответа агента (для status=success).
        action_type: Тип требуемого действия (для requires_action).
            - 'summarize_selection': выбор формата суммаризации
            - 'requires_disambiguation': уточнение сотрудника
        message: Системное сообщение или вопрос пользователю.
        thread_id: ID основного треда — сохранить на фронтенде для истории.
        choice_thread_id: ID треда где живёт suspended граф.
            КРИТИЧЕСКИ ВАЖНО: при следующем запросе с human_choice
            передавать этот thread_id как thread_id в ChatRequest.
            Присутствует только при status=requires_action.
        options: Список вариантов выбора (для requires_action).
            Формат: [{id, label, description}, ...].
            id — каноническое значение для human_choice.
    """

    status: str = Field(
        ...,
        description="success | error | requires_action",
    )
    response: Optional[str] = Field(None, description="Финальный текст ответа")
    action_type: Optional[str] = Field(
        None,
        description="Тип действия: summarize_selection | requires_disambiguation",
    )
    message: Optional[str] = Field(
        None, description="Системное сообщение / вопрос пользователю"
    )
    thread_id: Optional[str] = Field(
        None, description="ID основного треда — сохранить на фронтенде"
    )
    choice_thread_id: Optional[str] = Field(
        None,
        description=(
            "ID треда suspended графа. "
            "При следующем запросе с human_choice передавать как thread_id."
        ),
    )
    options: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Варианты выбора [{id, label, description}] для requires_action",
    )


class NewChatRequest(BaseModel):
    """Тело запроса POST /chat/new."""

    user_token: str


class NewChatResponse(BaseModel):
    """Ответ при создании нового треда."""

    status: str
    thread_id: str


class FileUploadResponse(BaseModel):
    """Ответ после успешной загрузки файла."""

    file_path: str = Field(..., description="Абсолютный путь к загруженному файлу")
    file_name: str = Field(..., description="Оригинальное имя файла")


class SummarizeRequest(BaseModel):
    """Тело запроса POST /actions/summarize.

    Attributes:
        message: Дополнительные инструкции пользователя (опционально).
        user_token: JWT токен авторизации.
        context_ui_id: UUID активного документа в UI.
        file_path: UUID вложения EDMS или путь к локальному файлу.
        human_choice: Выбор формата после requires_action.
            При resume: "1"/"2"/"3" или "extractive"/"abstractive"/"thesis".
        thread_id: При resume после human_choice —
            передавать choice_thread_id из предыдущего ответа.
    """

    message: str = Field(
        default="Проанализируй вложение документа",
        description="Инструкции для анализа",
    )
    user_token: str
    context_ui_id: Optional[str] = None
    file_path: Optional[str] = None
    human_choice: Optional[str] = Field(
        None,
        description="Выбор формата: '1'/'2'/'3' или 'extractive'/'abstractive'/'thesis'",
    )
    thread_id: Optional[str] = Field(
        None,
        description="При resume: передавать choice_thread_id из предыдущего ответа",
    )
