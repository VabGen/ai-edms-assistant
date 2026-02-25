# src/ai_edms_assistant/interfaces/api/schemas/agent_schema.py
"""Pydantic request/response schemas for the Agent API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class UserContextSchema(BaseModel):
    """Контекст текущего пользователя, опционально передаётся фронтендом."""

    firstName: Optional[str] = None
    lastName: Optional[str] = None
    role: Optional[str] = None


class ChatRequest(BaseModel):
    """Тело запроса POST /chat."""

    message: str = Field(..., min_length=1, description="Сообщение пользователя")
    user_token: str = Field(..., description="JWT токен авторизации")
    context_ui_id: Optional[str] = Field(
        None, description="ID открытого документа в UI"
    )
    context: Optional[UserContextSchema] = Field(
        None, description="Данные текущего пользователя"
    )
    file_path: Optional[str] = Field(None, description="Путь к загруженному файлу")
    human_choice: Optional[str] = Field(
        None, description="Ответ пользователя на disambiguation-запрос"
    )
    thread_id: Optional[str] = Field(
        None, description="ID треда. Генерируется автоматически если не передан"
    )


class ChatResponse(BaseModel):
    """Ответ агента."""

    status: str = Field(
        ..., description="success | error | requires_action | requires_choice"
    )
    response: Optional[str] = Field(None, description="Текст ответа агента")
    action_type: Optional[str] = Field(
        None, description="Тип запрошенного действия (для requires_action)"
    )
    message: Optional[str] = Field(
        None, description="Системное сообщение или уточняющий вопрос"
    )
    thread_id: Optional[str] = Field(
        None, description="ID треда — сохранить на фронтенде"
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

    file_path: str
    file_name: str


class SummarizeRequest(BaseModel):
    """Тело запроса POST /actions/summarize."""

    message: str = Field(default="Summarize this document")
    user_token: str
    context_ui_id: Optional[str] = None
    file_path: Optional[str] = None
    human_choice: Optional[str] = None
