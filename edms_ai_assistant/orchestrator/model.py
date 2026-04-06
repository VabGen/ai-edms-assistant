# orchestrator/model.py
"""
Публичные контракты данных оркестратора (Pydantic v2).

Экспортирует:
    AgentState        — состояние LangGraph (только для checkpointer)
    UserInput         — входное сообщение от клиента к /chat
    AssistantResponse — ответ агента клиенту
    UserContext       — профиль пользователя
    FileUploadResponse
    NewChatRequest
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ── LangGraph state (только для checkpointer) ────────────────────────────


class AgentState(TypedDict):
    """Состояние LangGraph-треда. Используется только AsyncPostgresSaver."""

    messages: Annotated[list[BaseMessage], add_messages]


# ── Input models ──────────────────────────────────────────────────────────


class UserContext(BaseModel):
    """Профиль пользователя из EDMS."""

    firstName: str | None = Field(None, max_length=100)
    lastName: str | None = Field(None, max_length=100)
    middleName: str | None = Field(None, max_length=100)
    role: str | None = Field(None, max_length=100)
    post: str | None = Field(None, max_length=200)
    department: str | None = Field(None, max_length=200)
    authorPost: str | None = Field(None, max_length=200)


class UserInput(BaseModel):
    """Входящее сообщение от клиента к /chat эндпоинту."""

    message: str = Field(..., min_length=1, max_length=8000)
    user_token: str = Field(..., min_length=10)
    context_ui_id: str | None = Field(
        None,
        description="UUID активного документа в UI EDMS",
    )
    context: UserContext | None = None
    file_path: str | None = Field(None, max_length=500)
    file_name: str | None = None
    human_choice: str | None = Field(None, max_length=200)
    thread_id: str | None = Field(None, max_length=255)
    preferred_summary_format: str | None = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


# ── Response models ───────────────────────────────────────────────────────

ResponseStatus = Literal["success", "error", "requires_action", "processing"]


class AssistantResponse(BaseModel):
    """Стандартизированный ответ агента клиенту."""

    status: ResponseStatus = "success"
    response: str | None = None
    action_type: str | None = None
    message: str | None = None
    thread_id: str | None = None
    requires_reload: bool = Field(default=False)
    navigate_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileUploadResponse(BaseModel):
    """Ответ на загрузку файла."""

    file_path: str
    file_name: str


class NewChatRequest(BaseModel):
    """Запрос на создание нового треда диалога."""

    user_token: str = Field(..., min_length=10)
