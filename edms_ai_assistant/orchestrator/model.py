# edms_ai_assistant/orchestrator/models.py
"""
EDMS AI Assistant — Public data contracts (Pydantic v2).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

# ─────────────────────────────────────────────────────────────
# LangGraph state
# ─────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """
    Состояние LangGraph графа.

    Единственное каноническое определение — используется
    как в _build_graph, так и везде, где нужен тип состояния.
    Reducer add_messages обеспечивает корректное слияние.
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ─────────────────────────────────────────────────────────────
# Input models
# ─────────────────────────────────────────────────────────────


class UserContext(BaseModel):
    """Профиль пользователя из EDMS."""

    firstName: str | None = Field(None, max_length=100)
    lastName: str | None = Field(None, max_length=100)
    middleName: str | None = Field(None, max_length=100)
    role: str | None = Field(None, max_length=100)
    post: str | None = Field(None, max_length=200)


class UserInput(BaseModel):
    """
    Входное сообщение от клиента к /chat эндпоинту.

    Валидация здесь минимальна — детальная валидация
    делегируется AgentRequest в agent.py (Service Layer).
    """

    message: str = Field(..., min_length=1, max_length=8000)
    user_token: str = Field(..., min_length=10)
    context_ui_id: str | None = Field(
        None,
        description="UUID активного документа в UI EDMS",
    )
    context: UserContext | None = None
    file_path: str | None = Field(
        None,
        max_length=500,
        description="UUID вложения EDMS или путь к локальному файлу",
    )
    file_name: str | None = None
    human_choice: str | None = Field(
        None,
        max_length=200,
        description=(
            "Выбор пользователя: тип суммаризации (extractive/abstractive/thesis) "
            "или UUID сотрудников через запятую для disambiguation"
        ),
    )
    thread_id: str | None = Field(None, max_length=255)
    preferred_summary_format: str | None = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        """Removes surrounding whitespace from the message."""
        return v.strip()


# ─────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────

ResponseStatus = Literal["success", "error", "requires_action", "processing"]


class AssistantResponse(BaseModel):
    """
    Стандартизированный ответ агента клиенту.

    Fields:
        status: Машиночитаемый статус выполнения.
        response: Текстовый ответ для отображения пользователю.
        action_type: Тип требуемого действия (если requires_action).
        message: Системное или пользовательское сообщение.
        thread_id: ID треда для продолжения диалога.
        requires_reload: True если EDMS нужно перезагрузить страницу
                         (после мутирующих операций: ознакомление, поручение и т.д.)
    """

    status: ResponseStatus = "success"
    response: str | None = None
    action_type: str | None = None
    message: str | None = None
    thread_id: str | None = None
    requires_reload: bool = Field(
        default=False,
        description=(
            "Сигнал фронтенду перезагрузить страницу EDMS "
            "после успешного выполнения мутирующих операций"
        ),
    )
    navigate_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Auxiliary request/response models
# ─────────────────────────────────────────────────────────────


class FileUploadResponse(BaseModel):
    """Ответ на загрузку файла."""

    file_path: str
    file_name: str


class NewChatRequest(BaseModel):
    """Запрос на создание нового треда диалога."""

    user_token: str = Field(..., min_length=10)
