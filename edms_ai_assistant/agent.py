# edms_ai_assistant/agent.py
"""
EDMS AI Assistant — Core Agent Module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.model import AgentState
from edms_ai_assistant.services.nlp_service import (
    SemanticDispatcher,
    UserIntent,
)
from edms_ai_assistant.tools import all_tools
from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

# ─── Фразы успешных мутирующих операций для requires_reload ──────────────────
_MUTATION_SUCCESS_PHRASES: tuple[str, ...] = (
    "успешно добавлен",
    "успешно создан",
    "список ознакомления",
    "поручение создано",
    "поручение успешно",
    "обращение заполнено",
    "обращение успешно",
    "карточка заполнена",
    "добавлено в список",
    "добавлен в список",
    "ознакомление создано",
    "задача создана",
    "заполнение обращения",
    "обращение автоматически заполнен",
    "карточка обращения заполнен",
    "автозаполнен",
    # Уведомления
    # "уведомление отправлено",
    # "напоминание отправлено",
    # "уведомлен",
)


def _is_valid_uuid(value: str) -> bool:
    """Returns True if *value* matches the canonical UUID4 pattern."""
    return bool(UUID_RE.match(value.strip()))


# ─── Инструменты требующие обязательного document_id из контекста ─────────────
_TOOLS_REQUIRING_DOCUMENT_ID: frozenset[str] = frozenset(
    {
        "doc_get_details",
        "doc_get_versions",
        "doc_compare_documents",
        "doc_get_file_content",
        "doc_compare_attachment_with_local",
        "doc_summarize_text",
        "doc_search_tool",
        "introduction_create_tool",
        "task_create_tool",
        "doc_send_notification",
    }
)

# ─── Placeholder-значения local_file_path для doc_compare_attachment_with_local ──────────
_COMPARE_LOCAL_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "",
        "local_file",
        "local_file_path",
        "/path/to/file",
        "path/to/file",
        "none",
        "null",
        "<local_file_path>",
        "<path>",
    }
)

# ─── Инструменты с disambiguation которые обрабатываются в _handle_human_choice
_DISAMBIGUATION_TOOLS: frozenset[str] = frozenset(
    {
        "introduction_create_tool",
        "task_create_tool",
        "doc_send_notification",
    }
)


def _is_mutation_response(content: str | None) -> bool:
    """
    Returns True if the agent response describes a successful mutating EDMS operation.

    Used to signal the frontend to reload the page so that the EDMS SPA
    reflects the newly created/updated data without a manual refresh.

    Args:
        content: Final agent response text.

    Returns:
        True if the response contains a mutation success phrase.
    """
    if not content:
        return False
    lower = content.lower()
    return any(phrase in lower for phrase in _MUTATION_SUCCESS_PHRASES)


# ─────────────────────────────────────────────────────────────────────────────
# Domain value objects & enumerations
# ─────────────────────────────────────────────────────────────────────────────


class AgentStatus(str, Enum):
    """Agent execution result statuses."""

    SUCCESS = "success"
    ERROR = "error"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"


class ActionType(str, Enum):
    """Types of interactive actions that require user participation."""

    SUMMARIZE_SELECTION = "summarize_selection"
    DISAMBIGUATION = "requires_disambiguation"
    CONFIRMATION = "requires_confirmation"


@dataclass
class ContextParams:
    """
    Immutable execution context passed through the entire agent lifecycle.

    Attributes:
        user_token: JWT authorization token.
        document_id: UUID of the active EDMS document.
        file_path: UUID of an EDMS attachment or local filesystem path.
        thread_id: LangGraph conversation thread identifier.
        user_name: Display name for the system prompt.
        user_first_name: First name for personalized greetings.
        current_date: Formatted date string injected into the prompt.
        user_context: Full user context dict (preferred_summary_format, etc.).
        intent: Primary user intent, set in chat() after semantic analysis.
            Used by tool router to select minimal relevant toolset.
    """

    user_token: str
    document_id: str | None = None
    file_path: str | None = None
    thread_id: str = "default"
    user_name: str = "пользователь"
    user_first_name: str | None = None
    user_last_name: str | None = None
    user_full_name: str | None = None
    user_id: str | None = None
    current_date: str = field(
        default_factory=lambda: datetime.now().strftime("%d.%m.%Y")
    )
    current_year: str = field(default_factory=lambda: str(datetime.now().year))
    uploaded_file_name: str | None = None
    user_context: dict = field(default_factory=dict)
    intent: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")
        if self.file_path and not self.uploaded_file_name:
            fp = str(self.file_path).strip()
            if not _is_valid_uuid(fp):
                self.uploaded_file_name = Path(fp).name
        if not self.user_full_name:
            parts = [p for p in (self.user_last_name, self.user_first_name) if p]
            if parts:
                self.user_full_name = " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Service layer request / response models (Pydantic v2)
# ─────────────────────────────────────────────────────────────────────────────


class AgentRequest(BaseModel):
    """Validated incoming request to the agent (Service Layer boundary)."""

    message: str = Field(default="", max_length=8000)
    user_token: str = Field(..., min_length=10)
    context_ui_id: str | None = Field(
        None,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$|^$",
    )
    thread_id: str | None = Field(None, max_length=255)
    user_context: dict[str, Any] = Field(default_factory=dict)
    file_path: str | None = Field(None, max_length=500)
    file_name: str | None = Field(None, max_length=260)
    human_choice: str | None = Field(None, max_length=200)

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Strips surrounding whitespace."""
        return v.strip()

    @model_validator(mode="after")
    def validate_message_or_choice(self) -> AgentRequest:
        """
        Ensures the request has either a non-empty message or a human_choice.

        Human-in-the-Loop choice flows (summarize type selection, disambiguation)
        send human_choice as the primary payload — message may be empty or
        equal to the choice label. Both cases are valid.

        Raises:
            ValueError: If both message and human_choice are empty.
        """
        has_message = bool(self.message and self.message.strip())
        has_choice = bool(self.human_choice and self.human_choice.strip())
        if not has_message and not has_choice:
            raise ValueError("Either message or human_choice must be provided")
        if not has_message and has_choice:
            self.message = self.human_choice
        return self

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str | None) -> str | None:
        """
        Validates file_path as UUID or filesystem path.

        Accepted formats:
        - UUID v4: ``550e8400-e29b-41d4-a716-446655440000``
        - Unix absolute path: ``/tmp/file.docx``
        - Windows absolute path: ``C:\\Users\\...\\file.docx``

        Args:
            v: Raw file path value.

        Returns:
            Cleaned value or None.

        Raises:
            ValueError: If the format is unrecognized.
        """
        if not v:
            return None
        stripped = v.strip()
        if _is_valid_uuid(stripped):
            return stripped
        if len(stripped) < 500:
            if stripped.startswith("/"):
                return stripped
            if re.match(r"^[A-Za-z]:\\", stripped):
                return stripped
            if re.match(r"^[^/\\]+[\\/]", stripped):
                return stripped
        raise ValueError(f"Invalid file_path format: {v!r}")


class AgentResponse(BaseModel):
    """Standardized agent execution result (internal, not exposed via HTTP)."""

    status: AgentStatus
    content: str | None = None
    message: str | None = None
    action_type: ActionType | None = None
    requires_reload: bool = False
    navigate_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Repository interface & implementation (Dependency Inversion)
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class IDocumentRepository(Protocol):
    """
    Document repository interface.

    Decorated with @runtime_checkable so isinstance() checks work correctly
    in dependency injection and testing scenarios.
    """

    async def get_document(self, token: str, doc_id: str) -> DocumentDto | None:
        """Получить метаданные документа."""
        ...


class DocumentRepository:
    """Production implementation of IDocumentRepository."""

    async def get_document(self, token: str, doc_id: str) -> DocumentDto | None:
        """
        Fetches and validates document metadata from the EDMS REST API.

        Args:
            token: JWT authorization token.
            doc_id: UUID of the document to fetch.

        Returns:
            Validated DocumentDto or None on any error.
        """
        try:
            async with DocumentClient() as client:
                raw_data = await client.get_document_metadata(token, doc_id)
                doc = DocumentDto.model_validate(raw_data)
                logger.info("Document fetched", extra={"doc_id": doc_id})
                return doc
        except Exception as exc:
            logger.error(
                "Failed to fetch document",
                exc_info=True,
                extra={"doc_id": doc_id, "error": str(exc)},
            )
            return None


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder — Strategy pattern для системных промптов
# ─────────────────────────────────────────────────────────────────────────────

USE_LEAN_PROMPT: bool = False
"""
True  → LEAN (~150 токенов)  — Llama 3.2, Mistral 7B, любые ≤13B модели
False → FULL (~2125 токенов) — GPT-4, Claude, Qwen 72B
"""


class PromptBuilder:
    """
    Strategy for building system prompts with dynamic context injection.

    Принципы:
    - Базовый шаблон CORE_TEMPLATE обязателен для всех запросов
    - Дополнительные снипеты добавляются по интенту
    - Промпт НЕ содержит реальных значений token/document_id —
      они инжектируются в _orchestrate через args патчинг
    """

    CORE_TEMPLATE = """<role>
Ты — экспертный ИИ-помощник системы электронного документооборота (EDMS/СЭД).
Специализация: анализ документов, управление персоналом, автоматизация рутинных задач.
</role>

<context>
- Пользователь (имя): {user_name}
- Пользователь (фамилия): {user_last_name}
- Пользователь (полное имя): {user_full_name}
- Текущая дата: {current_date} (год: {current_year})
- Активный документ в EDMS: {context_ui_id}
- Загруженный файл/вложение: {local_file}
- Имя загруженного файла (показывай пользователю): {uploaded_file_name}
<local_file_path>{local_file}</local_file_path>
</context>

<current_user_rules>
Когда пользователь говорит "добавь меня", "я", "моя фамилия" и т.п.:
- Его фамилия: {user_last_name}
- Его полное имя: {user_full_name}
- Используй эти данные напрямую — НЕ спрашивай фамилию у пользователя.
- Передавай фамилию в инструменты поиска сотрудников автоматически.
</current_user_rules>

<critical_rules>
1. **Автоинъекция параметров**: `token` и `document_id` добавляются системой АВТОМАТИЧЕСКИ.
   Не указывай эти параметры явно при вызове инструментов.

2. **Работа с файлом/вложением**:
   - Если "Загруженный файл" — путь (/tmp/...): **ПРИОРИТЕТ 1** — используй ЭТОТ файл.
     Вызови `read_local_file_content(file_path=<путь>)` для анализа или суммаризации.
     **Не спрашивай пользователя — просто сделай это автоматически.**
   - Если "Загруженный файл" — UUID (0c2216e1-...): вызови `doc_get_file_content(attachment_id=<UUID>)`
   - Если "Загруженный файл" — "Не загружен": вызови `doc_get_details()` для поиска вложений в документе
   - **ЗАПРЕЩЕНО**: если указан "Загруженный файл" (путь или UUID) — НИКОГДА не вызывай `doc_get_file_content` с UUID из вложений документа. Работай ТОЛЬКО с указанным файлом.

3. **Строгая последовательность**:
   - Вызывай СТРОГО ОДИН инструмент за раз
   - Дождись результата инструмента, затем вызывай следующий
   - НИКОГДА не вызывай `doc_summarize_text` одновременно с `doc_get_file_content`
   - Правильно: получи текст → получи результат → передай текст в суммаризацию

4. **Disambiguation (requires_disambiguation)**:
   - При получении статуса "requires_disambiguation" — ПОКАЖИ пользователю список вариантов
   - Попроси выбрать конкретную позицию
   - Дождись ответа пользователя ПЕРЕД повторным вызовом инструмента

5. **Финальный ответ**:
   - ВСЕГДА формулируй итоговый ответ на РУССКОМ языке
   - Обращайся к пользователю по имени: {user_name}
   - Ответ должен быть понятен пользователю, без технических деталей API
   - Структурируй ответ: заголовок → ключевые факты → вывод

6. **Язык**: Только русский. Никаких английских терминов в ответе пользователю.

7. **ЗАПРЕТ технических данных в ответах**:
   - НИКОГДА не показывай UUID (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) пользователю.
   - Вместо UUID сотрудника → используй его ФИО.
   - Вместо UUID вложения → используй имя файла.
   - Вместо UUID документа → используй название или номер документа.
   - Технические данные (ID, пути, токены) — только во внутренних вызовах инструментов.

 8. **Создание документа из файла**:
   - Если пользователь загрузил файл (путь в <local_file_path>) И просит
     "создай обращение / входящий / договор" — вызови create_document_from_file.
   - НЕ нужно сначала читать файл — инструмент сам его обработает.
   - После получения navigate_url в ответе — фронтенд откроет новый документ автоматически.
   - Параметр doc_category берётся из запроса пользователя:
     "обращение" → APPEAL, "входящий" → INCOMING, "исходящий" → OUTGOING,
     "внутренний" → INTERN, "договор" → CONTRACT.
</critical_rules>

<available_tools_guide>
| Сценарий                                 | Последовательность инструментов                              |
|------------------------------------------|--------------------------------------------------------------|
| Анализ документа целиком                 | doc_get_details → doc_get_file_content → doc_summarize_text  |
| Анализ конкретного вложения (UUID)       | doc_get_file_content → doc_summarize_text                    |
| Анализ загруженного файла                | read_local_file_content → doc_summarize_text                 |
| Сравнение файла с вложением [ЕСТЬ файл]  | doc_compare_attachment_with_local (приоритет всегда)         |
| Вопрос о документе                       | doc_get_details                                              |
| Сравнение версий документа [НЕТ файла]   | doc_get_versions (возвращает все сравнения)                  |
| Поиск документов в базе EDMS             | doc_search_tool                                              |
| Поиск сотрудника                         | employee_search_tool                                         |
| Добавление в лист ознакомления           | introduction_create_tool                                     |
| Создание поручения                       | task_create_tool                                             |
| Автозаполнение обращения                 | autofill_appeal_document                                     |
| Уведомление / напоминание                | employee_search_tool → doc_send_notification                 |
| Создать документ из файла                | create_document_from_file                                    |
| Вопрос без документа                     | Ответь напрямую из контекста                                 |
</available_tools_guide>

<response_format>
✅ Структурировано, кратко, информативно
✅ Маркированные списки для перечислений
✅ Выделение ключевых данных (суммы, даты, имена)
❌ Технические детали HTTP/API
❌ JSON-структуры в ответе пользователю
❌ Фразы "как ИИ я не могу..." — просто помогай
</response_format>"""

    LEAN_TEMPLATE = """<role>Ты — AI-помощник системы электронного документооборота (EDMS/СЭД).</role>

    <context>
    Пользователь: {user_name} ({user_last_name})
    Дата: {current_date} (год: {current_year})
    Документ: {context_ui_id}
    Файл: {local_file}
    </context>

    <user_self>
    Когда пользователь говорит «я», «добавь меня» — его фамилия: {user_last_name}, полное имя: {user_full_name}.
    </user_self>

    <rules>
    1. token/document_id инжектируются системой — не указывай их при вызове инструментов.
    2. Если есть локальный файл ({local_file}) — работай с ним, не с вложениями документа.
    3. Один инструмент за раз. Дождись результата перед следующим вызовом.
    4. При requires_disambiguation — покажи список, жди выбора пользователя.
    5. Финальный ответ — только на русском, без UUID, без JSON, без технических деталей.
    6. UUID в ответах запрещены. Вместо UUID → имя/название.
    </rules>"""

    _SNIPPETS: dict = {}
    _LEAN_SNIPPETS: dict = {}

    @classmethod
    def build(
        cls,
        context: ContextParams,
        intent: UserIntent,
        semantic_xml: str,
        *,
        lean: bool = False,
    ) -> str:
        """Assembles the full system prompt from context, intent snippet, and semantic XML.

        Args:
            context: Immutable execution context.
            intent: Detected primary user intent for snippet selection.
            semantic_xml: Pre-serialized semantic context XML block.
            lean: When True, uses compact LEAN_TEMPLATE (~150 tokens) instead of
                CORE_TEMPLATE (~2125 tokens). Set True for small LLMs (≤13B).

        Returns:
            Complete system prompt string ready for SystemMessage.
        """
        if lean:
            base = cls.LEAN_TEMPLATE.format(
                user_name=context.user_first_name or context.user_name,
                user_last_name=context.user_last_name or "Не указана",
                user_full_name=context.user_full_name or context.user_name,
                current_date=context.current_date,
                current_year=context.current_year,
                context_ui_id=context.document_id or "Не указан",
                local_file=context.file_path or "Не загружен",
            )
            snippet = cls._LEAN_SNIPPETS.get(intent, "")
            return base + snippet + semantic_xml

        base = cls.CORE_TEMPLATE.format(
            user_name=context.user_first_name or context.user_name,
            user_last_name=context.user_last_name or "Не указана",
            user_full_name=context.user_full_name or context.user_name,
            current_date=context.current_date,
            current_year=context.current_year,
            context_ui_id=context.document_id or "Не указан",
            local_file=context.file_path or "Не загружен",
            uploaded_file_name=context.uploaded_file_name or "Не определено",
        )
        snippet = cls._SNIPPETS.get(intent, "")
        return base + snippet + semantic_xml


PromptBuilder._LEAN_SNIPPETS = {
    UserIntent.SUMMARIZE: """
    <workflow>Суммаризация: получи текст (doc_get_file_content или read_local_file_content) → вызови doc_summarize_text(text=..., summary_type=...). Если summary_type не задан — верни requires_choice.</workflow>""",
    UserIntent.COMPARE: """
    <workflow>Сравнение: если есть локальный файл → doc_compare_attachment_with_local. Если нет → doc_get_versions (сравнивает все версии автоматически). НЕ вызывай doc_get_versions если есть файл.</workflow>""",
    UserIntent.CREATE_INTRODUCTION: """
    <workflow>Ознакомление: introduction_create_tool(last_names=[...]). При requires_disambiguation → покажи список → повторный вызов с selected_employee_ids.</workflow>""",
    UserIntent.CREATE_TASK: """
    <workflow>Поручение: task_create_tool(task_text=..., executor_last_names=[...]). Дата: если упомянута → ISO 8601, иначе не передавай. При disambiguation → покажи список → selected_employee_ids.</workflow>""",
    UserIntent.NOTIFICATION: """
    <workflow>Уведомление: employee_search_tool(last_name=...) → doc_send_notification(recipient_ids=[uuid], message=...).</workflow>""",
    UserIntent.SEARCH: """
    <workflow>Поиск: doc_search_tool(short_summary=...) или employee_search_tool(last_name=...). После поиска можно передать id в doc_get_details.</workflow>""",
    UserIntent.ANALYZE: """
    <workflow>Анализ: doc_get_details → doc_get_file_content → doc_summarize_text(summary_type='thesis').</workflow>""",
    UserIntent.QUESTION: """
    <workflow>Вопрос: doc_get_details для метаданных, doc_get_file_content для содержимого. Ответ — на русском языке без UUID.</workflow>""",
    UserIntent.FILE_ANALYSIS: """
    <workflow>Анализ файла: read_local_file_content(file_path=...) → doc_summarize_text(text=..., summary_type=...). Путь берётся из <context>.</workflow>""",
    UserIntent.CREATE_DOCUMENT: """
    <workflow>Создание документа: create_document_from_file(file_path=<из контекста>, doc_category=<APPEAL/INCOMING/...>). Один вызов — всё остальное автоматически.</workflow>""",
}

PromptBuilder._SNIPPETS = {
    UserIntent.CREATE_INTRODUCTION: """
<introduction_workflow>
Workflow создания листа ознакомления:
1. Вызови introduction_create_tool с last_names сотрудников
2. Если вернулся "requires_disambiguation" → покажи список найденных сотрудников пользователю
3. Попроси пользователя указать ID нужного сотрудника
4. Повторный вызов: introduction_create_tool(selected_employee_ids=["uuid1", "uuid2"])
5. Сообщи пользователю об успехе с именами добавленных сотрудников
</introduction_workflow>""",
    UserIntent.CREATE_TASK: """
<task_creation_guide>
Параметры поручения:
- task_text: текст поручения (обязательно)
- executor_last_names: фамилии исполнителей (обязательно, минимум 1)
- responsible_last_name: ответственный исполнитель (опционально; если не указан → первый из executor_last_names)
- planed_date_end: дата в ISO 8601 (опционально; если не указана → автоматически +7 дней)

КРИТИЧНО — извлечение даты из текста задачи:
Если пользователь упоминает дату или срок В ЛЮБОЙ ФОРМЕ — ОБЯЗАТЕЛЬНО передай planed_date_end.
Примеры (текущий год = {current_year}):
- "к 15 апреля" → "{current_year}-04-15T23:59:59Z"
- "до 1 мая" → "{current_year}-05-01T23:59:59Z"
- "через неделю" → текущая дата ({current_date}) + 7 дней + "T23:59:59Z"
- "до конца месяца" → последний день текущего месяца + "T23:59:59Z"
- "срочно" / без даты → НЕ передавай planed_date_end (сервис поставит +7 дней)
Всегда добавляй суффикс 'Z' (UTC). Год = {current_year} если не указан явно.

Disambiguation: если исполнитель не найден однозначно → покажи список, дождись выбора.
</task_creation_guide>""",
    UserIntent.SUMMARIZE: """
<summarize_guide>
Workflow суммаризации документа:

ШАГ 1 — Получи текст:
  - Локальный файл: read_local_file_content(file_path=<путь>)
  - Вложение EDMS (UUID): doc_get_file_content(attachment_id=<UUID>)

ШАГ 2 — Вызови суммаризацию:
  doc_summarize_text(text=<полученный текст>, summary_type=<тип или None>)
  - Если пользователь явно указал формат ("сделай выжимку фактов", "тезисно", "перескажи") →
    передай соответствующий summary_type: extractive | thesis | abstractive
  - Если формат НЕ указан → передай summary_type=None, инструмент спросит пользователя

ШАГ 3 — Обработай ответ:
  - status=requires_choice → ПОКАЖИ пользователю три варианта и жди его выбора
  - status=success → представь результат структурировано

ЗАПРЕЩЕНО: подставлять summary_type самостоятельно если пользователь не указал формат.
</summarize_guide>""",
    UserIntent.COMPARE: """
<compare_decision_tree>
⚠️ ОБЯЗАТЕЛЬНО прочитай условие ДО выбора инструмента сравнения:

УСЛОВИЕ А: В контексте есть "Загруженный файл" (путь /tmp/... или UUID)?
  → ДА: ИСПОЛЬЗУЙ ТОЛЬКО doc_compare_attachment_with_local. СТОП. doc_get_versions НЕ вызывать.
  → НЕТ: ИСПОЛЬЗУЙ doc_get_versions (сам вернёт все сравнения, doc_compare_documents НЕ нужен).

ЗАПРЕЩЕНО при наличии загруженного файла:
  ❌ doc_get_versions
  ❌ doc_compare_documents
  ❌ предлагать пользователю "выбрать версию"
  ❌ спрашивать "какие версии сравнить"

Если загруженный файл ЕСТЬ, а пользователь говорит "нет" или "не то" —
это значит он хочет другое вложение, а НЕ версии документа.
Покажи список вложений документа через doc_get_details и дай выбрать.
</compare_decision_tree>

<compare_with_local_guide>
ПУТЬ А: Есть загруженный файл → doc_compare_attachment_with_local

ШАГ 1 — Вызови СРАЗУ, без предварительных вызовов:
  doc_compare_attachment_with_local(
      local_file_path=<АВТОМАТИЧЕСКИ из контекста, не спрашивай>,
      attachment_id=<имя или UUID вложения — только если пользователь явно указал>,
      document_id=<АВТОМАТИЧЕСКИ из контекста>
  )
  - Пользователь написал "сравни договор" → attachment_id="договор" (инструмент найдёт)
  - Пользователь написал просто "сравни" → НЕ передавай attachment_id (инструмент найдёт по имени файла)
  - НИКОГДА не вызывай doc_get_details перед этим

ШАГ 2 — Обработай ответ:
  - status=success → покажи результат (схожесть %, различия)
  - status=requires_disambiguation → покажи список вложений, пользователь выберет
  - status=error → сообщи об ошибке, предложи повторить

ШАГ 3 — Формат ответа:
  - "Сравнение: «{имя загруженного файла}» и «{имя вложения}»"
  - Схожесть: X%
  - Различия: что добавлено / что удалено
</compare_with_local_guide>

<compare_versions_guide>
ПУТЬ Б: НЕТ загруженного файла → сравнение версий документа

ШАГ 1 — Вызови doc_get_versions. Инструмент АВТОМАТИЧЕСКИ:
  - Получает ВСЕ N версий документа
  - Сравнивает КАЖДУЮ соседнюю пару: v1↔v2, v2↔v3, ..., v(N-1)↔vN
  - Возвращает поле "comparisons" с результатами всех пар

ШАГ 2 — Ответь пользователю, используя поле "comparisons" из ответа:
  - Для каждой пары: что изменилось в метаданных и вложениях
  - Если "has_any_changes" = false → версии идентичны
  - Если "comparison_complete" = true → НЕ вызывай doc_compare_documents, данные уже есть

⚠️ ЗАПРЕЩЕНО:
  - Спрашивать "какие версии сравнить" — всё уже сравнено автоматически
  - Вызывать doc_compare_documents после doc_get_versions — это дублирование
  - Вызывать doc_get_versions несколько раз

Формат ответа: по каждой паре — секция с изменениями (или "изменений нет").
</compare_versions_guide>""",
    UserIntent.SEARCH: """
<search_guide>
При поиске документов в базе EDMS:
- Поиск по тексту/номеру/категории/дате: doc_search_tool
  Параметры: search, reg_number, doc_category (INTERN/INCOMING/OUTGOING/APPEAL), date_from, date_to
- Поиск сотрудника по фамилии: employee_search_tool
- Информация о текущем документе из контекста: doc_get_details
- Если нужна информация из текста документа: doc_get_file_content → ответь на основе текста
После doc_search_tool можно передать id найденного документа в doc_get_details или doc_get_file_content.
</search_guide>""",
    UserIntent.ANALYZE: """
<analyze_guide>
Для глубокого анализа документа:
1. doc_get_details — структура, метаданные, поручения, процессы
2. doc_get_file_content — текстовое содержимое
3. doc_summarize_text с типом thesis — тезисный разбор
Обязательно укажи: тип документа, статус, ключевые участники, сроки.
</analyze_guide>""",
    UserIntent.QUESTION: """
<question_guide>
Отвечай на вопросы о документе:
- Простые вопросы о метаданных: doc_get_details
- Вопросы о содержимом: doc_get_file_content → ответ на основе текста
- Вопросы о сотрудниках: employee_search_tool
- Общие вопросы без документа: отвечай напрямую из контекста
</question_guide>""",
    UserIntent.NOTIFICATION: """
<notification_guide>
При отправке уведомлений и напоминаний:
- Инструмент: doc_send_notification(document_id=..., recipient_ids=[...], message=..., notification_type=..., deadline=...)
  - recipient_ids — UUID сотрудников (получи через employee_search_tool если не известны)
  - notification_type: REMINDER (напоминание), DEADLINE (срок), CUSTOM (произвольное)
  - deadline — опциональная дата дедлайна в ISO 8601
- Workflow: employee_search_tool → doc_send_notification
- Если сотрудник один и найден однозначно — сразу передавай его UUID.
</notification_guide>""",
    UserIntent.FILE_ANALYSIS: """
<file_analysis_guide>
При анализе загруженного файла:
- Локальный файл (/tmp/...): read_local_file_content → doc_summarize_text
- UUID вложения EDMS: doc_get_file_content → doc_summarize_text
- Сравнение файла с вложением документа: doc_compare_attachment_with_local
Путь к файлу берётся из <local_file_path> в system prompt.
</file_analysis_guide>""",
    UserIntent.CREATE_DOCUMENT: """
<create_document_guide>
    Создание нового документа из загруженного файла:

    ТРИГГЕРЫ (когда вызывать create_document_from_file):
      - Пользователь загрузил файл (есть <local_file_path>) И говорит:
        "создай обращение", "сделай входящий", "оформи договор",
        "зарегистрируй на основе этого файла", "создай документ из файла"

    МАППИНГ категорий:
      - "обращение" / "жалоба" / "заявление" → APPEAL
      - "входящий" / "входящее письмо"       → INCOMING
      - "исходящий" / "исходящее"            → OUTGOING
      - "внутренний"                         → INTERN
      - "договор" / "контракт"               → CONTRACT
      - "совещание"                          → MEETING

    ВЫЗОВ (один инструмент, ничего предварительно):
      create_document_from_file(
          file_path=<из <local_file_path>>,   # АВТОМАТИЧЕСКИ
          doc_category="APPEAL",              # из запроса пользователя
          autofill=True,                      # автозаполнение для APPEAL
      )

    НЕ НУЖНО:
      ❌ Сначала читать файл через read_local_file_content
      ❌ Спрашивать пользователя о пути к файлу
      ❌ Вызывать doc_get_details или autofill_appeal_document отдельно

    ПОСЛЕ получения navigate_url:
      - Скажи пользователю: "Документ создан, открываю карточку..."
      - navigate_url обрабатывается фронтендом автоматически
    </create_document_guide>""",
}


# ─────────────────────────────────────────────────────────────────────────────
# ContentExtractor
# ─────────────────────────────────────────────────────────────────────────────


class ContentExtractor:
    """
    Extracts final human-readable content from a LangGraph message chain.

    Priority chain for extract_final_content:
    1. Last AIMessage with non-trivial text (not a tool-call marker)
    2. Last ToolMessage parsed as JSON (content/message/text fields)
    3. Fallback: any AIMessage with content
    4. Last resort: raw ToolMessage content

    The class is stateless — all methods are classmethods.
    """

    _SKIP_PATTERNS: tuple[str, ...] = (
        "вызвал инструмент",
        "tool call",
        '"name"',
        '"id"',
        '"tool_calls"',
    )
    MIN_CONTENT_LENGTH = 30
    _JSON_PRIORITY_FIELDS: tuple[str, ...] = (
        "content",
        "message",
        "text",
        "text_preview",
        "result",
    )
    _TECHNICAL_FIELDS: frozenset[str] = frozenset(
        {
            "status",
            "meta",
            "format_used",
            "was_truncated",
            "text_length",
            "suggestion",
            "action_type",
            "added_count",
            "not_found",
            "partial_success",
            "attachment_used",
            "warnings",
        }
    )

    @classmethod
    def extract_final_content(cls, messages: list[BaseMessage]) -> str | None:
        """
        Extracts the final user-visible content from the message chain.

        Args:
            messages: Complete LangGraph message chain.

        Returns:
            Cleaned content string, or None if nothing found.
        """
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                text = str(m.content).strip()
                if not cls._is_technical(text) and len(text) >= cls.MIN_CONTENT_LENGTH:
                    logger.debug(
                        "Extracted final AIMessage", extra={"chars": len(text)}
                    )
                    return text

        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._parse_tool_message(m)
                if extracted:
                    logger.debug(
                        "Extracted ToolMessage JSON", extra={"chars": len(extracted)}
                    )
                    return extracted

        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                text = str(m.content).strip()
                if text:
                    logger.debug("Fallback AIMessage", extra={"chars": len(text)})
                    return text

        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._parse_tool_message(m)
                if extracted:
                    return extracted

        return None

    @classmethod
    def extract_last_tool_text(cls, messages: list[BaseMessage]) -> str | None:
        """
        Extracts substantial text content from the most recent ToolMessage.

        Used to feed file content into doc_summarize_text on the next
        orchestration iteration.

        Args:
            messages: Complete LangGraph message chain.

        Returns:
            Text content string (100+ chars) or None.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                raw = str(m.content).strip()
                if raw.startswith("{"):
                    data: dict[str, Any] = json.loads(raw)
                    for key in cls._JSON_PRIORITY_FIELDS:
                        val = data.get(key)
                        if val and len(str(val)) > 100:
                            return str(val)
                if len(raw) > 100:
                    return raw
            except json.JSONDecodeError:
                raw = str(m.content)
                if len(raw) > 100:
                    return raw
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """
        Strips technical JSON wrappers from final content.

        Handles both clean JSON responses and mixed text with embedded JSON.

        Args:
            content: Raw content that may contain JSON envelopes.

        Returns:
            Clean human-readable text.
        """
        stripped = content.strip()

        # Случай 1: весь контент — JSON объект
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                for key in cls._JSON_PRIORITY_FIELDS:
                    val = data.get(key)
                    if (
                        val
                        and isinstance(val, str)
                        and len(val) >= cls.MIN_CONTENT_LENGTH
                    ):
                        return val.replace("\\n", "\n").replace('\\"', '"').strip()
            except (json.JSONDecodeError, ValueError):
                pass

        content = re.sub(
            r'\{"status"\s*:\s*"[^"]*",\s*"(?:content|message|text)"\s*:\s*"',
            "",
            content,
        )
        content = re.sub(r'",\s*"meta"\s*:\s*\{[^}]*\}\s*\}', "", content)
        content = re.sub(
            r'",?\s*"[a-z_]+"\s*:\s*(?:true|false|null|\d+)\s*\}?\s*$', "", content
        )
        content = re.sub(r'"\s*\}$', "", content)

        return content.replace('\\"', '"').replace("\\n", "\n").strip()

    @classmethod
    def _is_technical(cls, content: str) -> bool:
        """Returns True if content is a technical marker not suitable for display."""
        lower = content.lower()
        return any(pattern in lower for pattern in cls._SKIP_PATTERNS)

    @classmethod
    def _parse_tool_message(cls, message: ToolMessage) -> str | None:
        """
        Safely parses human-readable content from a ToolMessage.

        Args:
            message: LangGraph ToolMessage to parse.

        Returns:
            Extracted text or None.
        """
        try:
            raw = str(message.content).strip()
            if raw.startswith("{"):
                data: dict[str, Any] = json.loads(raw)
                if data.get("status") == "error":
                    return None
                for key in cls._JSON_PRIORITY_FIELDS:
                    val = data.get(key)
                    if (
                        val
                        and isinstance(val, str)
                        and len(val) >= cls.MIN_CONTENT_LENGTH
                    ):
                        return val
        except json.JSONDecodeError:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AgentStateManager — управление состоянием LangGraph
# ─────────────────────────────────────────────────────────────────────────────


class AgentStateManager:
    """
    Manages LangGraph graph state: invocation, inspection, and patching.

    Separates LangGraph API calls from orchestration logic,
    making it easier to swap checkpoint backends (MemorySaver → Postgres).
    """

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver) -> None:
        """
        Initializes the state manager with a compiled graph and checkpointer.

        Args:
            graph: Compiled LangGraph state graph.
            checkpointer: Checkpoint backend (MemorySaver in dev, Postgres in prod).

        Raises:
            ValueError: If graph or checkpointer is None.
        """
        if graph is None:
            raise ValueError("Graph cannot be None")
        if checkpointer is None:
            raise ValueError("Checkpointer cannot be None")
        self.graph = graph
        self.checkpointer = checkpointer
        logger.debug(
            "AgentStateManager initialized",
            extra={
                "graph_type": type(graph).__name__,
                "checkpointer_type": type(checkpointer).__name__,
            },
        )

    def _config(self, thread_id: str) -> dict[str, Any]:
        """Builds a LangGraph config dict for the given thread."""
        return {"configurable": {"thread_id": thread_id}}

    async def get_state(self, thread_id: str) -> Any:
        """
        Returns the current graph state snapshot for *thread_id*.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            StateSnapshot with .values (messages) and .next (pending nodes).
        """
        return await self.graph.aget_state(self._config(thread_id))

    async def update_state(
        self,
        thread_id: str,
        messages: list[BaseMessage],
        as_node: str = "agent",
    ) -> None:
        """
        Patches graph state for *thread_id* with new messages.

        Args:
            thread_id: Conversation thread identifier.
            messages: List of messages to merge into state.
            as_node: Node name to attribute the update to.
        """
        await self.graph.aupdate_state(
            self._config(thread_id),
            {"messages": messages},
            as_node=as_node,
        )

    async def invoke(
        self,
        inputs: dict[str, Any],
        thread_id: str,
        timeout: float = 120.0,
    ) -> None:
        """
        Invokes the graph for *thread_id* with optional inputs.

        Passing ``inputs=None`` resumes from the last interrupt point.

        Args:
            inputs: Initial graph inputs, or None to resume from interrupt.
            thread_id: Conversation thread identifier.
            timeout: Maximum wall-clock seconds to wait.

        Raises:
            asyncio.TimeoutError: If execution exceeds *timeout*.
        """
        await asyncio.wait_for(
            self.graph.ainvoke(inputs, config=self._config(thread_id)),
            timeout=timeout,
        )

    async def is_thread_broken(self, thread_id: str) -> bool:
        """Check if a thread has a dangling AIMessage with unresolved tool_calls.

        A thread is "broken" when:
        - The last message is an AIMessage with tool_calls (graph interrupted before tools)
        - AND the message before it is also an AIMessage (i.e. no ToolMessage response exists yet),
          OR the history structure violates the tool_call/tool_response pairing constraint.

        This state causes the LLM API to reject subsequent invocations with
        "An assistant message with tool_calls must be followed by tool messages".

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            True if the thread needs repair before next invocation.
        """
        try:
            state = await self.get_state(thread_id)
            messages = state.values.get("messages", [])
            if not messages:
                return False
            last = messages[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                if len(messages) < 2 or not isinstance(messages[-2], ToolMessage):
                    return True
            return False
        except Exception:
            return False

    async def repair_thread(self, thread_id: str) -> bool:
        """Repair a broken thread by injecting synthetic ToolMessage error responses.

        For each unresolved tool_call in the last AIMessage, injects a synthetic
        ToolMessage with a graceful error payload. This satisfies the LLM API
        constraint and allows the graph to resume normally on the next user message.

        Strategy: inject synthetic ToolMessages → graph can now call agent node
        again → agent sees the errors and formulates a user-facing response.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            True if repair succeeded, False on failure.
        """
        try:
            state = await self.get_state(thread_id)
            messages = state.values.get("messages", [])
            if not messages:
                return False

            last = messages[-1]
            if not isinstance(last, AIMessage):
                return False

            tool_calls = getattr(last, "tool_calls", []) or []
            if not tool_calls:
                return False

            synthetic_tool_msgs = [
                ToolMessage(
                    content=json.dumps(
                        {
                            "status": "error",
                            "message": (
                                "Выполнение инструмента прервано: предыдущий запрос завершился "
                                "некорректно. Пожалуйста, повторите запрос."
                            ),
                        }
                    ),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
                for tc in tool_calls
            ]

            await self.update_state(
                thread_id,
                synthetic_tool_msgs,
                as_node="tools",
            )
            logger.warning(
                "Thread repaired: injected %d synthetic ToolMessage(s)",
                len(synthetic_tool_msgs),
                extra={"thread_id": thread_id},
            )
            return True

        except Exception as exc:
            logger.error(
                "Thread repair failed: %s",
                exc,
                extra={"thread_id": thread_id},
                exc_info=True,
            )
            return False


# ─────────────────────────────────────────────────────────────────────────────
# EdmsDocumentAgent
# ─────────────────────────────────────────────────────────────────────────────


class EdmsDocumentAgent:
    MAX_ITERATIONS: int = 10
    EXECUTION_TIMEOUT: float = 120.0

    def __init__(self, document_repo=None, semantic_dispatcher=None):
        try:
            self.model = get_chat_model()
            self.tools = all_tools
            self.document_repo = document_repo or DocumentRepository()
            self.dispatcher = semantic_dispatcher or SemanticDispatcher()
            self._checkpointer = MemorySaver()
            self._tool_bindings: dict[str, Any] = {}
            self._active_tools: list[Any] = self.tools
            self._model_with_tools = self.model.bind_tools(self.tools)
            self._compiled_graph = self._build_graph()

            if self._compiled_graph is None:
                raise RuntimeError("Graph compilation returned None")

            self.state_manager = AgentStateManager(
                graph=self._compiled_graph,
                checkpointer=self._checkpointer,
            )
            logger.info(
                "EdmsDocumentAgent initialized",
                extra={
                    "tools_count": len(self.tools),
                    "model_type": type(self.model).__name__,
                },
            )
        except Exception as exc:
            logger.error("Failed to initialize EdmsDocumentAgent", exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {exc}") from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def health_check(self) -> dict[str, bool]:
        """
        Returns a shallow health status for each agent component.

        Returns:
            Dict mapping component names to boolean availability flags.
        """
        return {
            "model": self.model is not None,
            "tools": len(self.tools) > 0,
            "document_repo": self.document_repo is not None,
            "dispatcher": self.dispatcher is not None,
            "graph": getattr(self, "_compiled_graph", None) is not None,
            "state_manager": getattr(self, "state_manager", None) is not None,
            "checkpointer": getattr(self, "_checkpointer", None) is not None,
        }

    async def chat(
        self,
        message: str,
        user_token: str,
        context_ui_id: str | None = None,
        thread_id: str | None = None,
        user_context: dict[str, Any] | None = None,
        file_path: str | None = None,
        file_name: str | None = None,
        human_choice: str | None = None,
    ) -> dict[str, Any]:
        """
        Main entry point for agent interaction.

        Handles both fresh conversations and resumptions from Human-in-the-Loop
        interrupts (disambiguation, summarization type selection).

        Args:
            message: User message text.
            user_token: JWT authorization token.
            context_ui_id: UUID of the active EDMS document in the UI.
            thread_id: Conversation thread identifier (auto-generated if None).
            user_context: Optional user profile dict.
            file_path: UUID of an EDMS attachment or local filesystem path.
            human_choice: Disambiguation UUIDs (comma-separated) or summary type.

        Returns:
            Serialized AgentResponse dict suitable for the HTTP layer.
        """
        try:
            request = AgentRequest(
                message=message,
                user_token=user_token,
                context_ui_id=context_ui_id,
                thread_id=thread_id,
                user_context=user_context or {},
                file_path=file_path,
                file_name=file_name,
                human_choice=human_choice,
            )
            context = await self._build_context(request)

            if await self.state_manager.is_thread_broken(context.thread_id):
                repaired = await self.state_manager.repair_thread(context.thread_id)
                logger.warning(
                    "Broken thread detected and %s",
                    "repaired" if repaired else "repair FAILED",
                    extra={"thread_id": context.thread_id},
                )

            state = await self.state_manager.get_state(context.thread_id)

            if human_choice and state.next:
                return await self._handle_human_choice(context, human_choice)

            document: DocumentDto | None = None
            if context.document_id:
                document = await self.document_repo.get_document(
                    context.user_token, context.document_id
                )

            semantic_ctx = self.dispatcher.build_context(
                request.message, document, context.file_path
            )
            logger.info(
                "Semantic analysis complete",
                extra={
                    "intent": semantic_ctx.query.intent.value,
                    "complexity": semantic_ctx.query.complexity.value,
                    "thread_id": context.thread_id,
                },
            )

            context.intent = semantic_ctx.query.intent

            full_prompt = PromptBuilder.build(
                context,
                semantic_ctx.query.intent,
                self._build_semantic_xml(semantic_ctx),
                lean=USE_LEAN_PROMPT,
            )

            inputs: dict[str, Any] = {
                "messages": [
                    SystemMessage(content=full_prompt),
                    HumanMessage(content=semantic_ctx.query.refined),
                ]
            }

            _forced = await self._try_forced_tool_call(context, inputs, request.message)

            return await self._orchestrate(
                context=context,
                inputs=None if _forced else inputs,
                is_choice_active=bool(human_choice),
                iteration=0,
            )

        except Exception as exc:
            logger.error(
                "Chat error", exc_info=True, extra={"user_message": message[:200]}
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {exc}",
            ).model_dump()

    # ── Human-in-the-Loop ─────────────────────────────────────────────────────

    async def _handle_human_choice(
        self, context: ContextParams, human_choice: str
    ) -> dict[str, Any]:
        """
        Resumes a paused graph after the user resolves a disambiguation or
        selects a summarization type.

        Patches the pending AIMessage tool_calls with the user's choice,
        then resumes orchestration.

        Args:
            context: Immutable execution context.
            human_choice: Raw user choice: UUID list or summary type string.

        Returns:
            Serialized AgentResponse dict.
        """
        state = await self.state_manager.get_state(context.thread_id)
        last_msg: AIMessage = state.values["messages"][-1]
        raw_calls = getattr(last_msg, "tool_calls", [])

        patched_calls = []
        for tc in raw_calls:
            t_args = dict(tc["args"])
            t_name: str = tc["name"]

            if t_name == "doc_summarize_text":
                t_args["summary_type"] = human_choice.strip()
                logger.info(
                    "Human choice: summary_type",
                    extra={"type": human_choice, "thread_id": context.thread_id},
                )

            elif t_name in _DISAMBIGUATION_TOOLS:
                raw_ids = [x.strip() for x in human_choice.split(",") if x.strip()]
                valid_ids = []
                for raw_id in raw_ids:
                    try:
                        UUID(raw_id)
                        valid_ids.append(raw_id)
                    except ValueError:
                        logger.warning(
                            "Invalid UUID in human_choice",
                            extra={"raw_id": raw_id},
                        )

                if valid_ids:
                    _TOOL_ID_FIELD: dict[str, str] = {
                        "introduction_create_tool": "selected_employee_ids",
                        "task_create_tool": "selected_employee_ids",
                        "doc_send_notification": "recipient_ids",
                    }
                    id_field = _TOOL_ID_FIELD.get(t_name, "selected_employee_ids")
                    t_args[id_field] = valid_ids
                    t_args.pop("last_names", None)
                    t_args.pop("executor_last_names", None)
                    t_args.pop("recipient_last_names", None)
                    logger.info(
                        "Human choice: employee disambiguation resolved",
                        extra={
                            "tool": t_name,
                            "id_field": id_field,
                            "count": len(valid_ids),
                            "thread_id": context.thread_id,
                        },
                    )

            elif t_name == "doc_compare_attachment_with_local":
                choice = human_choice.strip()

                try:
                    if _is_valid_uuid(choice):
                        t_args["attachment_id"] = choice
                    else:
                        t_args["attachment_id"] = choice
                        logger.info(
                            "Human choice: attachment by name",
                            extra={
                                "attachment_name": choice,
                                "thread_id": context.thread_id,
                            },
                        )

                    if context.document_id:
                        doc_id = str(context.document_id).strip()

                        if _is_valid_uuid(doc_id):
                            t_args["document_id"] = doc_id

                    if context.uploaded_file_name:
                        t_args["original_filename"] = context.uploaded_file_name

                    logger.info(
                        "Human choice: attachment disambiguation resolved for compare",
                        extra={
                            "attachment_id": choice[:8] + "...",
                            "thread_id": context.thread_id,
                            "local_file_path": (
                                t_args.get("local_file_path", "")[:32]
                                if t_args.get("local_file_path")
                                else None
                            ),
                            "doc_id": str(t_args.get("document_id", "?"))[:8],
                        },
                    )

                except ValueError:
                    logger.warning(
                        "Invalid attachment UUID in human_choice for doc_compare_attachment_with_local",
                        extra={"raw_choice": choice},
                    )

            patched_calls.append({"name": t_name, "args": t_args, "id": tc["id"]})

        await self.state_manager.update_state(
            context.thread_id,
            [
                AIMessage(
                    content=last_msg.content or "",
                    tool_calls=patched_calls,
                    id=last_msg.id,
                )
            ],
            as_node="agent",
        )

        return await self._orchestrate(
            context=context,
            inputs=None,
            is_choice_active=True,
            iteration=0,
        )

    # ── Core orchestration loop ───────────────────────────────────────────────

    async def _orchestrate(
        self,
        context: ContextParams,
        inputs: dict[str, Any] | None,
        is_choice_active: bool,
        iteration: int,
    ) -> dict[str, Any]:
        """
        Core recursive orchestration loop.
        """
        if iteration > self.MAX_ITERATIONS:
            logger.error(
                "Max iterations exceeded",
                extra={"thread_id": context.thread_id, "iterations": iteration},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышен лимит итераций обработки.",
            ).model_dump()

        try:
            if iteration == 0:
                from edms_ai_assistant.services.nlp_service import UserIntent
                from edms_ai_assistant.tools.router import (
                    estimate_tools_tokens,
                    get_tools_for_intent,
                )

                active_intent = context.intent or UserIntent.UNKNOWN

                is_appeal_doc = context.user_context.get("doc_category", "") == "APPEAL"

                selected_tools = get_tools_for_intent(
                    active_intent,
                    self.tools,
                    include_appeal=is_appeal_doc,
                )

                cache_key = ",".join(
                    sorted(getattr(t, "name", "") for t in selected_tools)
                )
                if cache_key not in self._tool_bindings:
                    self._tool_bindings[cache_key] = self.model.bind_tools(
                        selected_tools
                    )
                    logger.info(
                        "Tool binding created: intent=%s tools=%d (~%d tokens)",
                        active_intent.value,
                        len(selected_tools),
                        estimate_tools_tokens(selected_tools),
                    )

                self._model_with_tools = self._tool_bindings[cache_key]
                self._active_tools = selected_tools

            await self.state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=self.EXECUTION_TIMEOUT,
            )

            state = await self.state_manager.get_state(context.thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])

            logger.debug(
                "State snapshot",
                extra={
                    "thread_id": context.thread_id,
                    "iteration": iteration,
                    "messages_count": len(messages),
                    "last_type": type(messages[-1]).__name__ if messages else "none",
                    "has_tool_calls": bool(
                        messages
                        and isinstance(messages[-1], AIMessage)
                        and getattr(messages[-1], "tool_calls", None)
                    ),
                    "state_next": list(state.next) if state.next else [],
                },
            )

            if not messages:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Пустое состояние агента.",
                ).model_dump()

            last_msg = messages[-1]

            last_is_tool_msg = isinstance(last_msg, ToolMessage)
            last_has_tool_calls = isinstance(last_msg, AIMessage) and bool(
                getattr(last_msg, "tool_calls", None)
            )
            is_finished = (
                not state.next and not last_is_tool_msg and not last_has_tool_calls
            )
            if is_finished:
                return self._build_final_response(messages, context)

            raw_calls = list(last_msg.tool_calls)

            if len(raw_calls) > 1:
                logger.warning(
                    "Parallel tool_calls detected — keeping only the first",
                    extra={
                        "total": len(raw_calls),
                        "kept": raw_calls[0]["name"],
                        "dropped": [tc["name"] for tc in raw_calls[1:]],
                        "thread_id": context.thread_id,
                    },
                )
                raw_calls = raw_calls[:1]

            last_tool_text = ContentExtractor.extract_last_tool_text(messages)
            patched_calls = []

            _after_compare_disambiguation = False
            for prev_msg in reversed(messages[-15:]):
                if isinstance(prev_msg, ToolMessage):
                    try:
                        prev_data = json.loads(str(prev_msg.content))
                        if (
                            prev_data.get("status") == "requires_disambiguation"
                            and prev_msg.name == "doc_compare_attachment_with_local"
                        ):
                            _after_compare_disambiguation = True
                            logger.debug(
                                "Detected requires_disambiguation from doc_compare_attachment_with_local",
                                extra={"tool_call_id": prev_msg.tool_call_id},
                            )
                            break
                    except (json.JSONDecodeError, AttributeError):
                        continue
                if isinstance(prev_msg, HumanMessage):
                    break

            for tc in raw_calls:
                t_name = tc["name"]
                t_args = dict(tc["args"])
                t_id = tc["id"]

                # ── 1. Инжект токена авторизации ──────────────────────────
                t_args["token"] = context.user_token

                # ── Инжект параметров для create_document_from_file ───────────
                if t_name == "create_document_from_file":
                    if t_args.get("doc_category") is None:
                        from edms_ai_assistant.tools.create_document_from_file import (
                            _extract_category_from_message,
                        )

                        last_human_text = ""
                        for _m in reversed(messages):
                            if isinstance(_m, HumanMessage):
                                last_human_text = str(_m.content)
                                break
                        detected = _extract_category_from_message(last_human_text)
                        if detected:
                            t_args["doc_category"] = detected
                            logger.info(
                                "Injected doc_category=%s for create_document_from_file",
                                detected,
                            )
                    if t_args.get("file_path") is None and context.file_path:
                        _cp = str(context.file_path).strip()
                        if not _is_valid_uuid(_cp):
                            t_args["file_path"] = _cp
                            logger.info(
                                "Injected file_path for create_document_from_file: %s...",
                                _cp[:32],
                            )
                    if t_args.get("file_name") is None and context.uploaded_file_name:
                        t_args["file_name"] = context.uploaded_file_name

                # ── 1а. Инжект document_id ────────────────────────────────
                if context.document_id and t_name in _TOOLS_REQUIRING_DOCUMENT_ID:
                    cur_doc_id = str(t_args.get("document_id", "")).strip()
                    if not cur_doc_id or not _is_valid_uuid(cur_doc_id):
                        t_args["document_id"] = context.document_id
                        logger.debug(
                            "Injected document_id for tool '%s'",
                            t_name,
                            extra={"doc_id_prefix": context.document_id[:8]},
                        )

                # ── 2. Маршрутизация file_path → правильный инструмент ──────
                clean_path = str(context.file_path).strip() if context.file_path else ""
                path_is_uuid = _is_valid_uuid(clean_path)
                path_is_local = bool(clean_path) and not path_is_uuid

                # ── Логика для локальных файлов ───────────────────────────
                if path_is_local:
                    if t_name == "doc_get_versions":
                        t_name = "doc_compare_attachment_with_local"
                        t_args = {
                            "local_file_path": clean_path,
                        }
                        if context.document_id:
                            t_args["document_id"] = context.document_id
                        logger.warning(
                            "GUARD: doc_get_versions blocked (local file present) "
                            "→ redirected to doc_compare_attachment_with_local",
                            extra={"path": clean_path[:32]},
                        )

                    elif t_name == "doc_compare_documents":
                        t_name = "doc_compare_attachment_with_local"
                        t_args["local_file_path"] = clean_path
                        t_args.pop("document_id_1", None)
                        t_args.pop("document_id_2", None)
                        logger.warning(
                            "GUARD: doc_compare_documents blocked (local file present) "
                            "→ redirected to doc_compare_attachment_with_local",
                            extra={"path": clean_path[:32]},
                        )

                    elif t_name == "doc_get_file_content" and not t_args.get(
                        "attachment_id"
                    ):
                        t_name = "read_local_file_content"
                        t_args["file_path"] = clean_path
                        t_args.pop("attachment_id", None)
                        logger.info(
                            "AUTO-PRIORITY: doc_get_file_content → read_local_file_content "
                            "(local file present, no explicit attachment_id)",
                            extra={"path": clean_path[:32]},
                        )

                # ── Логика для UUID-файлов (вложения) ─────────────────────
                elif path_is_uuid:
                    if t_name == "read_local_file_content":
                        t_name = "doc_get_file_content"
                        t_args["attachment_id"] = clean_path
                        t_args.pop("file_path", None)
                        logger.info(
                            "Routed read_local_file_content → doc_get_file_content",
                            extra={"attachment_id": clean_path[:8]},
                        )
                    elif t_name == "doc_get_file_content":
                        cur_att = str(t_args.get("attachment_id", "")).strip()
                        if not cur_att or not _is_valid_uuid(cur_att):
                            t_args["attachment_id"] = clean_path
                            logger.info(
                                "Injected attachment_id from context",
                                extra={"attachment_id": clean_path[:8]},
                            )

                # ── Фолбэк: если нет файла, но вызван compare_with_local ───
                if (
                    t_name == "doc_compare_attachment_with_local"
                    and not clean_path
                    and not _after_compare_disambiguation
                    and not (is_choice_active and t_args.get("attachment_id"))
                ):
                    t_name = "doc_compare_documents"
                    t_args.pop("local_file_path", None)
                    t_args.pop("attachment_id", None)
                    logger.info(
                        "Routed doc_compare_attachment_with_local → doc_compare_documents "
                        "(no file in context → version compare intended)",
                    )

                # ── Инжект local_file_path для compare_with_local ─────────
                if t_name == "doc_compare_attachment_with_local" and path_is_local:
                    cur_local = str(t_args.get("local_file_path", "")).strip()
                    if (
                        not cur_local
                        or cur_local.lower() in _COMPARE_LOCAL_PLACEHOLDERS
                        or not Path(cur_local).exists()
                    ):
                        t_args["local_file_path"] = clean_path
                        logger.info(
                            "Force-injected local_file_path for doc_compare_attachment_with_local",
                            extra={"path": clean_path[:32]},
                        )

                    if context.uploaded_file_name and not t_args.get(
                        "original_filename"
                    ):
                        t_args["original_filename"] = context.uploaded_file_name
                        logger.debug(
                            "Injected original_filename for doc_compare_attachment_with_local",
                            extra={"file_name": context.uploaded_file_name},
                        )

                # ── Блокировка doc_compare_documents после disambiguation ─
                if t_name == "doc_compare_documents":
                    if _after_compare_disambiguation or (
                        is_choice_active and path_is_local
                    ):
                        logger.warning(
                            "GUARD: doc_compare_documents blocked — redirecting to doc_compare_attachment_with_local",
                            extra={
                                "thread_id": context.thread_id,
                                "reason": (
                                    "disambiguation_flow"
                                    if _after_compare_disambiguation
                                    else "choice_active_with_local_file"
                                ),
                            },
                        )
                        t_name = "doc_compare_attachment_with_local"
                        t_args = {
                            "token": context.user_token,
                            "document_id": context.document_id,
                            "local_file_path": (
                                clean_path
                                if path_is_local
                                else t_args.get("local_file_path")
                            ),
                            "attachment_id": t_args.get("document_id_2")
                            or t_args.get("attachment_id"),
                            "original_filename": context.uploaded_file_name,
                        }
                        for key in [
                            "document_id_1",
                            "document_id_2",
                            "comparison_focus",
                        ]:
                            t_args.pop(key, None)
                        if context.uploaded_file_name and not t_args.get(
                            "original_filename"
                        ):
                            t_args["original_filename"] = context.uploaded_file_name

                    else:
                        _versions_result_complete = False
                        for prev_msg in reversed(messages):
                            if isinstance(prev_msg, ToolMessage):
                                try:
                                    prev_data = json.loads(str(prev_msg.content))
                                    if prev_data.get(
                                        "comparison_complete"
                                    ) and prev_data.get("comparisons"):
                                        _versions_result_complete = True
                                        break
                                except (json.JSONDecodeError, AttributeError):
                                    continue
                            if isinstance(prev_msg, HumanMessage):
                                break

                        if _versions_result_complete:
                            logger.warning(
                                "GUARD: doc_compare_documents blocked — doc_get_versions already "
                                "completed all comparisons (comparison_complete=True). "
                                "Replacing with no-op to prevent redundant API call.",
                            )
                            t_name = "doc_get_details"
                            t_args = {}
                            if context.document_id:
                                t_args["document_id"] = context.document_id

                # ── Инжект для read_local_file_content (placeholder replacement) ─
                if path_is_local and t_name == "read_local_file_content":
                    cur_fp = str(t_args.get("file_path", "")).strip()
                    if not cur_fp or cur_fp.lower() in (
                        "local_file",
                        "file_path",
                        "none",
                        "null",
                        "",
                    ):
                        t_args["file_path"] = clean_path
                        logger.info(
                            "Injected local file_path (placeholder replaced)",
                            extra={"path_prefix": clean_path[:32]},
                        )

                # ── Инжект текста для суммаризации ─────────────────────
                if t_name == "doc_summarize_text":
                    if last_tool_text:
                        t_args["text"] = last_tool_text

                    if not t_args.get("summary_type"):
                        if is_choice_active:
                            t_args["summary_type"] = "extractive"
                            logger.warning(
                                "safety-net: summary_type=extractive (is_choice_active but type not set)"
                            )
                        elif (
                            context.user_context.get("preferred_summary_format")
                            and context.user_context["preferred_summary_format"]
                            != "ask"
                        ):
                            t_args["summary_type"] = context.user_context[
                                "preferred_summary_format"
                            ]
                            logger.info(
                                "Using preferred_summary_format from user settings: %s",
                                t_args["summary_type"],
                            )

                patched_calls.append({"name": t_name, "args": t_args, "id": t_id})

            await self.state_manager.update_state(
                context.thread_id,
                [
                    AIMessage(
                        content=last_msg.content or "",
                        tool_calls=patched_calls,
                        id=last_msg.id,
                    )
                ],
                as_node="agent",
            )

            next_is_choice_active = is_choice_active
            if is_choice_active and patched_calls:
                last_tool_name = patched_calls[-1]["name"]
                if last_tool_name in (
                    "doc_compare_attachment_with_local",
                    "doc_summarize_text",
                ):
                    next_is_choice_active = True
                else:
                    next_is_choice_active = False

            return await self._orchestrate(
                context=context,
                inputs=None,
                is_choice_active=next_is_choice_active,
                iteration=iteration + 1,
            )

        except TimeoutError:
            logger.error(
                "Execution timeout",
                extra={
                    "thread_id": context.thread_id,
                    "timeout": self.EXECUTION_TIMEOUT,
                },
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания выполнения.",
            ).model_dump()

        except Exception as exc:
            err_str = str(exc)
            logger.error(
                "Orchestration error",
                exc_info=True,
                extra={"thread_id": context.thread_id, "iteration": iteration},
            )

            _BROKEN_THREAD_SIGNALS = (
                "tool_calls must be followed by tool messages",
                "tool_call_ids did not have response messages",
                "invalid_request_error",
                "messages.[",
            )
            is_broken_thread_error = any(
                sig in err_str for sig in _BROKEN_THREAD_SIGNALS
            )

            if is_broken_thread_error and iteration == 0:
                logger.warning(
                    "Broken thread error detected — attempting auto-repair",
                    extra={"thread_id": context.thread_id},
                )
                repaired = await self.state_manager.repair_thread(context.thread_id)
                if repaired:
                    return AgentResponse(
                        status=AgentStatus.ERROR,
                        message=(
                            "Предыдущий запрос завершился некорректно и был восстановлен. "
                            "Пожалуйста, повторите ваш вопрос."
                        ),
                    ).model_dump()

            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка оркестрации: {exc}",
            ).model_dump()

    # ── Graph compilation ─────────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        """
        Compiles the LangGraph ReAct workflow.

        Nodes:
        - ``agent``: invokes the LLM with bound tools
        - ``tools``: executes tool_calls via ToolNode
        - ``validator``: injects system notifications for tool errors

        Edges:
        - START → agent
        - agent → tools (if tool_calls present) | END
        - tools → validator → agent

        Interrupt:
        - ``interrupt_before=["tools"]`` pauses execution for Human-in-the-Loop
          (human choice injection in _orchestrate / _handle_human_choice)

        Returns:
            Compiled state graph ready for ainvoke/aget_state/aupdate_state.

        Raises:
            RuntimeError: If LangGraph compilation fails.
        """
        workflow: StateGraph = StateGraph(AgentState)

        # ── Нода: вызов LLM ──────────────────────────────────────────────────
        async def call_model(state: AgentState) -> dict[str, Any]:
            """
            Invokes the LLM with bound tools.
            """
            _MAX_HISTORY_MSGS = 40
            sys_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
            non_sys = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
            if len(non_sys) > _MAX_HISTORY_MSGS:
                non_sys = non_sys[-_MAX_HISTORY_MSGS:]
            candidate_msgs = ([sys_msgs[-1]] if sys_msgs else []) + non_sys

            final_msgs = []
            for i, msg in enumerate(candidate_msgs):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    next_msg = (
                        candidate_msgs[i + 1] if i + 1 < len(candidate_msgs) else None
                    )
                    if not isinstance(next_msg, ToolMessage):
                        safe_msg = AIMessage(
                            content=msg.content or "",
                            id=msg.id,
                        )
                        final_msgs.append(safe_msg)
                        logger.warning(
                            "Sanitized dangling AIMessage tool_calls at position %d",
                            i,
                        )
                        continue
                final_msgs.append(msg)

            response = await self._model_with_tools.ainvoke(final_msgs)
            return {"messages": [response]}

        async def validator(state: AgentState) -> dict[str, Any]:
            """
            Post-tool validator: injects system notifications for failed or
            empty tool results as AIMessage with non-empty content.
            """
            last = state["messages"][-1]
            if not isinstance(last, ToolMessage):
                return {"messages": []}

            raw = str(last.content).strip()

            if not raw or raw in ("None", "{}", "null"):
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "⚠️ Системное уведомление (не показывать пользователю): "
                                "Инструмент вернул пустой результат. "
                                "Попробуй другой подход или сообщи пользователю."
                            )
                        )
                    ]
                }

            if raw.startswith("{"):
                try:
                    tool_data = json.loads(raw)
                    interactive_status = tool_data.get("status", "")
                    if interactive_status in (
                        "requires_choice",
                        "requires_disambiguation",
                        "requires_action",
                    ):
                        logger.info(
                            "Validator: interactive status '%s' — stopping graph",
                            interactive_status,
                        )
                        return {"messages": [AIMessage(content="")]}
                except json.JSONDecodeError:
                    pass

            raw_lower = raw.lower()
            if '"status": "error"' in raw_lower or (
                raw_lower.startswith("{") and '"error"' in raw_lower
            ):
                try:
                    err_data = json.loads(raw)
                    err_msg = err_data.get("message", raw[:200])
                except (json.JSONDecodeError, KeyError):
                    err_msg = raw[:200]

                return {
                    "messages": [
                        AIMessage(
                            content=(
                                f"⚠️ Системное уведомление (не показывать пользователю): "
                                f"Инструмент вернул ошибку: {err_msg}. "
                                "Проинформируй пользователя понятным языком."
                            )
                        )
                    ]
                }

            return {"messages": []}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools=self.tools))
        workflow.add_node("validator", validator)
        workflow.add_edge(START, "agent")

        def should_continue(state: AgentState) -> str:
            """Routing function: go to tools if LLM produced tool_calls, else END."""
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "validator")
        workflow.add_edge("validator", "agent")

        try:
            compiled = workflow.compile(
                checkpointer=self._checkpointer,
                interrupt_before=["tools"],
            )
            logger.debug("LangGraph compiled successfully")
            return compiled
        except Exception as exc:
            logger.error("Graph compilation failed", exc_info=True)
            raise RuntimeError(f"Failed to compile graph: {exc}") from exc

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_final_response(
        self, messages: list[BaseMessage], context: ContextParams
    ) -> dict[str, Any]:
        """
        Extracts final content and wraps it into an AgentResponse.

        Before extracting text content, scans the last ToolMessage for
        structured interactive statuses:
        - ``requires_choice``       : Summarisation format selection needed.
        - ``requires_disambiguation``: Attachment or employee disambiguation needed.

        These statuses bypass text extraction and are returned directly to
        the HTTP layer so the frontend can render the appropriate widget.

        Args:
            messages: Complete LangGraph message chain.
            context: Execution context (thread_id, file info for sanitization).

        Returns:
            Serialized AgentResponse dict.
        """
        interactive = self._detect_interactive_status(messages)
        if interactive:
            logger.info(
                "Interactive status detected",
                extra={
                    "status": interactive.get("status"),
                    "thread_id": context.thread_id,
                },
            )
            return interactive

        final_content = ContentExtractor.extract_final_content(messages)
        navigate_url = self._extract_navigate_url(messages)

        if final_content:
            final_content = ContentExtractor.clean_json_artifacts(final_content)
            final_content = self._sanitize_technical_content(final_content, context)
            reload_needed = _is_mutation_response(final_content)

            if navigate_url:
                logger.info(
                    "Execution completed with navigation",
                    extra={
                        "thread_id": context.thread_id,
                        "navigate_url": navigate_url,
                    },
                )
            else:
                logger.info(
                    "Execution completed",
                    extra={
                        "thread_id": context.thread_id,
                        "content_length": len(final_content),
                        "requires_reload": reload_needed,
                    },
                )
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                content=final_content,
                requires_reload=reload_needed,
                navigate_url=navigate_url,
            ).model_dump()

        logger.warning("No final content found", extra={"thread_id": context.thread_id})
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            content="Операция завершена.",
            navigate_url=navigate_url,
        ).model_dump()

    @staticmethod
    def _detect_interactive_status(
        messages: list[BaseMessage],
    ) -> dict[str, Any] | None:
        """
        Сканирует ПОСЛЕДНИЙ ToolMessage на наличие статусов интерактива.

        Статусы 'requires_choice' или 'requires_disambiguation' означают, что
        агенту требуется ввод пользователя для продолжения. Если последний
        ToolMessage не содержит этих статусов, значит инструмент вернул данные
        и мы ждем финального ответа LLM.

        Args:
            messages: Полная цепочка сообщений LangGraph.

        Returns:
            Сериализованный словарь AgentResponse или None.
        """
        last_tool_msg: ToolMessage | None = None
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                last_tool_msg = m
                break

        if last_tool_msg is None:
            return None

        raw = str(last_tool_msg.content).strip()
        if not raw.startswith("{"):
            return None

        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return None

        status = data.get("status", "")
        if status not in (
            "requires_choice",
            "requires_disambiguation",
            "requires_action",
        ):
            return None

        logger.info(
            "Detect interactive: status=%s keys=%s payload_preview=%s",
            status,
            list(data.keys()),
            raw[:1000],
        )

        # ─── REQUIRES_CHOICE: Выбор формата суммаризации ───────────────────
        if status == "requires_choice":
            options = data.get("options", [])
            hint = data.get("hint", "extractive")
            hint_reason = data.get("hint_reason", "")
            msg = data.get("message", "Выберите формат анализа:")

            options_lines = [
                f"- **{opt['key']}** — {opt['label']}: {opt['description']}"
                for opt in options
                if isinstance(opt, dict)
            ]
            options_text = "\n".join(options_lines)
            hint_text = (
                f"\n\n💡 *Рекомендация:* **{hint}** — {hint_reason}"
                if hint_reason
                else ""
            )
            full_message = (
                f"{msg}\n\n{options_text}{hint_text}\n\n"
                "Ответьте: **extractive**, **abstractive** или **thesis**."
            )
            return AgentResponse(
                status=AgentStatus.REQUIRES_ACTION,
                action_type=ActionType.SUMMARIZE_SELECTION,
                message=full_message,
            ).model_dump()

        # ─── REQUIRES_DISAMBIGUATION: Уточнение объекта (сотрудник/файл) ────
        if status == "requires_disambiguation":
            _KNOWN_LIST_KEYS = (
                "available_attachments",
                "available_employees",
                "candidates",
                "employees",
                "results",
                "items",
                "users",
            )

            available: list[dict[str, Any]] = next(
                (
                    v
                    for k in _KNOWN_LIST_KEYS
                    if isinstance(v := data.get(k), list) and v
                ),
                [],
            )

            if not available:
                for _k, _v in data.items():
                    if _k != "options" and isinstance(_v, list) and _v:
                        first_item = _v[0] if _v else {}
                        if (
                            isinstance(first_item, dict)
                            and "matches" in first_item
                            and not first_item.get("id")
                        ):
                            flat: list[dict[str, Any]] = []
                            for group in _v:
                                flat.extend(group.get("matches", []))
                            if flat:
                                available = flat
                                logger.info(
                                    "Disambiguation: unwrapped nested 'matches' "
                                    "from key=%s total=%d",
                                    _k,
                                    len(flat),
                                )
                                break
                        else:
                            available = _v
                            logger.info(
                                "Disambiguation list found via fallback key=%s len=%d",
                                _k,
                                len(_v),
                            )
                            break

            base_msg: str = data.get("message", "Уточните выбор:")
            base_msg = (
                re.sub(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    "",
                    base_msg,
                    flags=re.I,
                )
                .strip()
                .rstrip("с «»")
                .strip()
            )
            if not base_msg:
                base_msg = "Уточните выбор:"
            candidates_structured: list[dict[str, str]] = []

            for item in available:
                if not isinstance(item, dict):
                    continue

                # Извлечение ФИО
                first = (
                    item.get("firstName")
                    or item.get("first_name")
                    or item.get("firstname")
                    or item.get("givenName")
                    or ""
                ).strip()
                last = (
                    item.get("lastName")
                    or item.get("last_name")
                    or item.get("lastname")
                    or item.get("surname")
                    or item.get("familyName")
                    or ""
                ).strip()
                middle = (
                    item.get("middleName")
                    or item.get("middle_name")
                    or item.get("patronymic")
                    or ""
                ).strip()

                display_name = (
                    item.get("fullName")
                    or item.get("full_name")
                    or item.get("fio")
                    or item.get("FIO")
                    or " ".join(filter(None, [last, first, middle]))
                    or item.get("name")
                    or item.get("username")
                    or item.get("login")
                    or item.get("email", "").split("@")[0]
                    or "Без имени"
                ).strip()

                dept = (
                    item.get("department")
                    or item.get("departmentName")
                    or item.get("department_name")
                    or item.get("division")
                    or item.get("post")
                    or item.get("position")
                    or item.get("jobTitle")
                    or item.get("job_title")
                    or item.get("role")
                    or ""
                ).strip()

                item_id = str(
                    item.get("id")
                    or item.get("uuid")
                    or item.get("employeeId")
                    or item.get("employee_id")
                    or item.get("userId")
                    or item.get("user_id")
                    or item.get("personId")
                    or item.get("person_id")
                    or "?"
                )

                candidates_structured.append(
                    {
                        "id": item_id,
                        "name": display_name,
                        "dept": dept,
                    }
                )

            candidates_json = json.dumps(candidates_structured, ensure_ascii=False)
            # Маркер <!--CANDIDATES:[...]---> парсится в AssistantWidget.tsx
            full_msg = f"{base_msg}\n\n<!--CANDIDATES:{candidates_json}-->"

            return AgentResponse(
                status=AgentStatus.REQUIRES_ACTION,
                action_type=ActionType.DISAMBIGUATION,
                message=full_msg,
            ).model_dump()

        # ─── REQUIRES_ACTION: employee_search множественные совпадения ───────
        if status == "requires_action":
            action_type = data.get("action_type", "")
            choices: list[dict[str, Any]] = data.get("choices", [])
            base_msg = data.get("message", "Выберите сотрудника:")

            candidates_structured: list[dict[str, str]] = []
            for item in choices:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id", "?"))
                display_name = (
                    item.get("full_name")
                    or item.get("fullName")
                    or item.get("name")
                    or "Без имени"
                ).strip()
                dept = (item.get("department") or item.get("post") or "").strip()
                candidates_structured.append(
                    {
                        "id": item_id,
                        "name": display_name,
                        "dept": dept,
                    }
                )

            if candidates_structured:
                candidates_json = json.dumps(candidates_structured, ensure_ascii=False)
                full_msg = base_msg + "\n\n<!--CANDIDATES:" + candidates_json + "-->"
                logger.info(
                    "Detect interactive: requires_action/select_employee → %d candidates",
                    len(candidates_structured),
                )
                return AgentResponse(
                    status=AgentStatus.REQUIRES_ACTION,
                    action_type=ActionType.DISAMBIGUATION,
                    message=full_msg,
                ).model_dump()

        return None

    @staticmethod
    def _extract_navigate_url(messages: list) -> str | None:
        """Scan the message chain for a navigate_url in the last ToolMessage.

        create_document_from_file returns ``navigate_url`` in its result dict.
        The LLM agent reformulates this as prose ("Открываю карточку...") but
        the raw URL must still reach the frontend so it can perform navigation.

        Args:
            messages: Complete LangGraph message chain.

        Returns:
            Navigate URL string (e.g. ``"/document-form/uuid"``), or None.
        """
        import json

        from langchain_core.messages import ToolMessage

        for m in reversed(messages[-8:]):
            if not isinstance(m, ToolMessage):
                continue
            try:
                data = json.loads(str(m.content))
                url = data.get("navigate_url")
                if url and isinstance(url, str) and url.startswith("/"):
                    return url
            except (json.JSONDecodeError, AttributeError):
                pass
        return None

    async def _build_context(self, request: AgentRequest) -> ContextParams:
        """Builds an immutable ContextParams from a validated AgentRequest."""
        ctx = request.user_context
        first_name = (ctx.get("firstName") or ctx.get("first_name") or "").strip()
        last_name = (ctx.get("lastName") or ctx.get("last_name") or "").strip()
        full_name = (
            ctx.get("fullName") or ctx.get("full_name") or ctx.get("name") or ""
        ).strip()
        user_id = ctx.get("id") or ctx.get("userId") or ctx.get("user_id")
        display_name = first_name or last_name or full_name or "пользователь"

        return ContextParams(
            user_token=request.user_token,
            document_id=request.context_ui_id,
            file_path=request.file_path,
            uploaded_file_name=request.file_name or None,
            thread_id=request.thread_id or "default",
            user_name=display_name,
            user_first_name=first_name or None,
            user_last_name=last_name or None,
            user_full_name=full_name or None,
            user_id=user_id,
            user_context=ctx,
        )

    def _sanitize_technical_content(self, content: str, context: ContextParams) -> str:
        """
        Removes technical artifacts from user-visible response content.

        Применяет замены в строго определённом порядке, чтобы избежать
        артефактов от частичной замены составных имён temp-файлов.

        Порядок замен:
        1. Абсолютные пути (/tmp/..., C:\\...) → original filename label
        2. Составное имя UUID_hex32.ext (полный паттерн temp-файла) → filename label
        3. hex32.ext без UUID-prefix → filename label
        4. UUID с дефисами (оставшиеся) → «документ»
        5. UUID без дефисов — 32 hex chars → «документ»
        6. Финальная очистка артефактов «документ»«...» → «...»

        Args:
            content: Raw extracted response content.
            context: Execution context with file_path and uploaded_file_name.

        Returns:
            Sanitized content string safe for display to the user.
        """
        file_label = (
            f"«{context.uploaded_file_name}»"
            if context.uploaded_file_name
            else "«загруженный файл»"
        )

        # 1. Абсолютные пути файловой системы
        content = re.sub(
            r"[A-Za-z]:\\[^\s,;)'\"]{3,}|/(?:tmp|var|home|uploads)/[^\s,;)'\"]{3,}",
            file_label,
            content,
        )

        # 2. Составное имя: UUID-с-дефисами_hex32.ext (полный temp-файл паттерн)
        content = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
            r"_[0-9a-f]{32}\.[a-zA-Z]{2,5}",
            file_label,
            content,
            flags=re.I,
        )

        # 3. hex32.ext (с опциональным ведущим _) — частичный temp-файл паттерн
        content = re.sub(
            r"_?[0-9a-f]{32}\.[a-zA-Z]{2,5}\b",
            file_label,
            content,
            flags=re.I,
        )

        # 4. Конкретный UUID загруженного файла или документа (с дефисами)
        if context.file_path and _is_valid_uuid(str(context.file_path).strip()):
            content = content.replace(str(context.file_path).strip(), file_label)
        if context.document_id and _is_valid_uuid(context.document_id):
            content = content.replace(context.document_id, "«текущего документа»")

        # 5. Оставшиеся UUID с дефисами
        content = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "«документ»",
            content,
            flags=re.I,
        )

        # 6. UUID без дефисов — 32 hex chars (uuid4().hex)
        content = re.sub(
            r"(?<![a-zA-Z0-9])[0-9a-f]{32}(?![a-zA-Z0-9])",
            "«документ»",
            content,
            flags=re.I,
        )

        # 7. Артефакты вида "«документ»«имя файла»" → "«имя файла»"
        content = re.sub(r"«документ»\s*(?=«)", "", content)
        content = re.sub(r"«документ»_\s*", "", content)

        return content

    async def _try_forced_tool_call(
        self,
        context: ContextParams,
        inputs: dict,
        original_message: str,
    ) -> bool:
        """Bypass LLM for deterministic intents by injecting a pre-built tool_call.

        Small models (llama3.2, Mistral 7B) often fail to generate a tool_call
        and instead respond with plain text asking for clarification. For intents
        where the tool and its arguments are fully determined by the context
        (file present + clear intent), we skip the LLM entirely and inject the
        AIMessage with tool_calls directly into the graph state.

        Currently handles:
        - CREATE_DOCUMENT + local file → create_document_from_file

        Args:
            context: Immutable execution context.
            inputs: Initial graph inputs (SystemMessage + HumanMessage).
            original_message: Raw user message for category detection.

        Returns:
            True if a forced tool_call was injected (caller must use inputs=None).
            False if normal LLM flow should proceed.
        """
        import uuid as _uuid_module

        from edms_ai_assistant.services.nlp_service import UserIntent
        from edms_ai_assistant.tools.create_document_from_file import (
            _extract_category_from_message,
        )

        # Только для CREATE_DOCUMENT с локальным файлом
        if context.intent != UserIntent.CREATE_DOCUMENT:
            return False

        clean_path = str(context.file_path or "").strip()
        if not clean_path or _is_valid_uuid(clean_path):
            return False  # нет файла или это UUID вложения — пусть LLM решает

        # Определяем категорию из сообщения пользователя
        doc_category = _extract_category_from_message(original_message) or "APPEAL"

        logger.info(
            "Forced tool call: create_document_from_file " "file=%s... category=%s",
            clean_path[:32],
            doc_category,
        )

        tool_call_id = f"forced_{_uuid_module.uuid4().hex[:12]}"
        forced_ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_document_from_file",
                    "args": {
                        "token": context.user_token,
                        "file_path": clean_path,
                        "doc_category": doc_category,
                        "file_name": context.uploaded_file_name or "",
                        "autofill": True,
                    },
                    "id": tool_call_id,
                }
            ],
        )

        sys_msg = inputs["messages"][0]
        human_msg = inputs["messages"][1]

        await self.state_manager.update_state(
            context.thread_id,
            [sys_msg, human_msg, forced_ai_msg],
            as_node="agent",
        )

        logger.info(
            "Forced tool call injected: id=%s thread=%s",
            tool_call_id,
            context.thread_id,
        )
        return True

    @staticmethod
    def _build_semantic_xml(semantic_context: Any) -> str:
        """
        Serializes semantic context into an XML block for prompt injection.

        Args:
            semantic_context: Output of SemanticDispatcher.build_context().

        Returns:
            XML string block appended to the system prompt.
        """
        return (
            "\n<semantic_context>\n"
            f"  <intent>{semantic_context.query.intent.value}</intent>\n"
            f"  <complexity>{semantic_context.query.complexity.value}</complexity>\n"
            f"  <original>{semantic_context.query.original}</original>\n"
            f"  <refined>{semantic_context.query.refined}</refined>\n"
            "</semantic_context>"
        )
