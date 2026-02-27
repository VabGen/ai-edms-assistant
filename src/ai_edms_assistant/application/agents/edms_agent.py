# src/ai_edms_assistant/application/agents/edms_agent.py
"""
EDMS Document Agent — production-grade LangGraph оркестратор
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import structlog
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

from ...domain.entities.document import Document
from ...infrastructure.llm.providers.openai_provider import get_chat_model
from ..dto.agent import ActionType, AgentRequest, AgentResponse, AgentStatus
from ..services.semantic_dispatcher import SemanticDispatcher, UserIntent
from ..tools import create_all_tools
from .agent_config import AgentConfig
from .agent_state import AgentStateWithCounter
from .base_agent import AbstractAgent


log = structlog.get_logger(__name__)
logger = logging.getLogger(__name__)

_MAX_GRAPH_ITERATIONS: int = 10

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# ── Human-choice constants ────────────────────────────────────────────────────
_CHOICE_THREAD_KEY: str = "choice_thread_id"

_CHOICE_MAP: dict[str, str] = {
    # Numeric shortcuts (пользователь нажал 1/2/3)
    "1": "extractive",
    "2": "abstractive",
    "3": "thesis",
    # Full names / label aliases
    "extractive": "extractive",
    "abstractive": "abstractive",
    "thesis": "thesis",
    "ключевые факты": "extractive",
    "факты": "extractive",
    "ключевые": "extractive",
    "краткий пересказ": "abstractive",
    "пересказ": "abstractive",
    "тезисный план": "thesis",
    "тезисы": "thesis",
    "тезис": "thesis",
    "план": "thesis",
}


def _normalize_choice(raw: str) -> str:
    """Normalize human_choice to canonical SummaryType value.

    Фронтенд может прислать "3", "Тезисный план" или "thesis" — всё маппится
    в каноническое значение ``SummaryType`` ("extractive" / "abstractive" / "thesis").

    Args:
        raw: Raw choice string from frontend.

    Returns:
        Canonical SummaryType string. Falls back to ``raw`` when unrecognized.
    """
    normalized = raw.strip().lower()
    return _CHOICE_MAP.get(normalized, raw.strip())


# ─────────────────────────────────────────────────────────────────────────────
# ContextParams — immutable value object per request
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContextParams:
    """Immutable execution context for a single agent invocation.

    Constructed once per ``chat()`` from ``AgentRequest``.
    Passed through the whole pipeline without mutation (frozen dataclass).

    Attributes:
        user_token: JWT bearer token for all EDMS API calls.
        document_id: UUID of the active document in UI (optional).
        file_path: Local file path or EDMS attachment UUID from frontend.
        thread_id: LangGraph checkpointer session identifier.
        user_name: Display name for prompt personalization.
        user_first_name: First name for informal address.
        current_date: Date string for system prompt.
    """

    user_token: str
    document_id: str | None = None
    file_path: str | None = None
    thread_id: str = "default"
    user_name: str = "пользователь"
    user_first_name: str | None = None
    user_last_name: str | None = None
    user_department: str | None = None
    user_post: str | None = None
    current_date: str = field(
        default_factory=lambda: datetime.now().strftime("%d.%m.%Y")
    )
    preselected_summary_type: str | None = None
    """Когда фронтенд передаёт human_choice В ПЕРВОМ запросе (до suspend графа),
    это значение инъектируется напрямую в doc_summarize_text.summary_type.
    Сценарий: пользователь выбрал тип из выпадающего списка вложений."""

    def __post_init__(self) -> None:
        """Validate token.

        Raises:
            ValueError: When ``user_token`` is empty or not a string.
        """
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")


# ─────────────────────────────────────────────────────────────────────────────
# IDocumentRepository — Dependency Inversion port
# ─────────────────────────────────────────────────────────────────────────────


class IDocumentRepository(Protocol):
    """Minimal port for document metadata used by EdmsDocumentAgent.
    """

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        """Fetch document with all fields including attachments.

        Args:
            token: JWT bearer token.
            doc_id: Document UUID string.

        Returns:
            ``Document`` entity or ``None`` when not found / API error.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# DocumentRepositoryAdapter — default infrastructure adapter
# ─────────────────────────────────────────────────────────────────────────────


class DocumentRepositoryAdapter:
    """Default IDocumentRepository via EdmsDocumentClient + DocumentMapper.

    Failures silently absorbed — missing document is NOT fatal.
    Agent continues with empty <document_context> and LLM may call
    doc_get_details() if needed.
    """

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        """Fetch document via REST API.

        Args:
            token: JWT bearer token.
            doc_id: Document UUID string.

        Returns:
            ``Document`` or ``None``.
        """
        try:
            from uuid import UUID as _UUID

            from ...infrastructure.edms_api.clients.document_client import (
                EdmsDocumentClient,
            )
            from ...infrastructure.edms_api.mappers.document_mapper import (
                DocumentMapper,
            )

            async with EdmsDocumentClient() as client:
                raw = await client.get_by_id(
                    document_id=_UUID(str(doc_id)),
                    token=token,
                )
                if not raw:
                    log.warning("document_not_found", doc_id=doc_id)
                    return None

                doc = DocumentMapper.from_dto(raw)
                log.debug(
                    "document_fetched",
                    doc_id=doc_id,
                    reg_number=getattr(doc, "reg_number", None),
                    attachments_count=len(getattr(doc, "attachments", []) or []),
                )
                return doc

        except Exception as exc:
            log.warning("document_fetch_failed", doc_id=doc_id, error=str(exc))
            return None


# ─────────────────────────────────────────────────────────────────────────────
# DocumentContextBuilder — Document entity → XML for LLM system prompt
# ─────────────────────────────────────────────────────────────────────────────


class DocumentContextBuilder:
    """Converts a domain ``Document`` into a ``<document_context>`` XML block.

    Единственная ответственность — сериализация Document в XML.
    Встраивается в system prompt, чтобы LLM знала все поля документа
    и могла отвечать на вопросы без лишних tool calls.

    Покрывает:
        - Все стандартные поля: reg_number, reg_date, status, author...
        - custom_fields: сумма договора, дата подписания, контрагент...
        - Вложения с UUID, именем, размером
        - Graceful degradation: None → "—"
    """

    @staticmethod
    def build(document: Document | None) -> str:
        """Build ``<document_context>`` XML from a ``Document`` entity.

        Args:
            document: Domain ``Document`` or ``None``.

        Returns:
            XML string. When ``document`` is ``None`` — returns note
            instructing LLM to call ``doc_get_details()``.
        """
        if document is None:
            return (
                "\n<document_context>"
                "\n  <note>Метаданные недоступны. "
                "Вызови doc_get_details() для получения данных документа.</note>"
                "\n</document_context>"
            )

        def _s(val: Any, default: str = "—") -> str:
            return str(val).strip() if val is not None else default

        def _d(val: Any) -> str:
            if val is None:
                return "—"
            try:
                return val.strftime("%d.%m.%Y") if hasattr(val, "strftime") else str(val)[:10]
            except Exception:
                return str(val)

        attachments: list = getattr(document, "attachments", None) or []
        att_lines = [
            f'    <attachment id="{_s(getattr(a, "id", None))}"'
            f' name="{_s(getattr(a, "file_name", None))}"'
            f' size="{getattr(a, "file_size", 0) or 0}"/>'
            for a in attachments
        ]
        att_xml = "\n".join(att_lines) if att_lines else "    <none/>"

        author = getattr(document, "author", None)
        resp = getattr(document, "responsible_executor", None)

        status = getattr(document, "status", None)
        status_str = status.value if hasattr(status, "value") else _s(status)

        custom: dict = getattr(document, "custom_fields", None) or {}
        custom_lines = [
            f'    <field name="{k}">{_s(v)}</field>'
            for k, v in custom.items()
            if v is not None
        ]
        custom_block = (
            "\n  <custom_fields>\n" + "\n".join(custom_lines) + "\n  </custom_fields>"
            if custom_lines
            else ""
        )

        return (
            "\n<document_context>"
            f"\n  <id>{_s(getattr(document, 'id', None))}</id>"
            f"\n  <reg_number>{_s(getattr(document, 'reg_number', None))}</reg_number>"
            f"\n  <reg_date>{_d(getattr(document, 'reg_date', None))}</reg_date>"
            f"\n  <create_date>{_d(getattr(document, 'create_date', None))}</create_date>"
            f"\n  <short_summary>{_s(getattr(document, 'short_summary', None))}</short_summary>"
            f"\n  <summary>{_s(getattr(document, 'summary', None))}</summary>"
            f"\n  <status>{status_str}</status>"
            f"\n  <document_type>{_s(getattr(document, 'document_type_name', None))}</document_type>"
            f"\n  <profile>{_s(getattr(document, 'profile_name', None))}</profile>"
            f"\n  <author>{_s(getattr(author, 'name', None) if author else None)}</author>"
            f"\n  <responsible>{_s(getattr(resp, 'name', None) if resp else None)}</responsible>"
            f"\n  <correspondent>{_s(getattr(document, 'correspondent_name', None))}</correspondent>"
            f"\n  <pages>{_s(getattr(document, 'pages_count', None))}</pages>"
            f"\n  <out_reg_number>{_s(getattr(document, 'out_reg_number', None))}</out_reg_number>"
            f"\n  <out_reg_date>{_d(getattr(document, 'out_reg_date', None))}</out_reg_date>"
            f"\n  <days_execution>{_s(getattr(document, 'days_execution', None))}</days_execution>"
            f"\n  <control>{_s(getattr(document, 'control_flag', None))}</control>"
            f"\n  <dsp>{_s(getattr(document, 'dsp_flag', None))}</dsp>"
            f"\n  <attachments_count>{len(attachments)}</attachments_count>"
            f"\n  <attachments>\n{att_xml}\n  </attachments>"
            + custom_block
            + "\n</document_context>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder — Strategy Pattern для system prompt
# ─────────────────────────────────────────────────────────────────────────────


class PromptBuilder:
    """Assembles the LangGraph system prompt.

    Strategy Pattern: base template + intent guide + document_context_xml
    + semantic_xml. Никакой бизнес-логики — только сборка строк.

    Принцип: LLM получает ВСЁ необходимое в одном system prompt.
    Нет скрытых данных, нет магических переменных.
    """

    # ── Core template ─────────────────────────────────────────────────────────
    _CORE: str = """\
<role>
Ты — интеллектуальный помощник системы электронного документооборота (EDMS/СЭД).
Помогаешь пользователям работать с документами: анализировать, суммаризировать,
создавать поручения, управлять ознакомлением и отвечать на любые вопросы.
</role>

<session>
- Пользователь: {user_name}
- Должность: {user_post}
- Подразделение: {user_department}
- Дата: {current_date}
- Активный документ в UI: {document_id}
- Загруженный файл: {file_path}

ВАЖНО: поля "Должность" и "Подразделение" выше — это данные ТЕКУЩЕГО пользователя
({user_name}). Используй их для вопросов "кто я", "мой отдел", "моя должность".
НЕ вызывай employee_search_tool для получения своих данных.
</session>

<rules>
## Автоинъекция параметров
Параметры `token` и `document_id` передаются системой АВТОМАТИЧЕСКИ.
Ты НИКОГДА не указываешь их явно в tool_calls.

## Работа с document_context
Блок `<document_context>` ниже содержит все поля активного документа.
- Отвечай на вопросы о полях НАПРЯМУЮ из него (рег.номер, дата, автор, статус).
- Поле "—" = данные отсутствуют. Скажи об этом честно.
- НЕ вызывай doc_get_details() если ответ уже есть в document_context.
- Если document_context содержит <note> об ошибке → вызови doc_get_details().

## Вопросы о ТЕКУЩЕМ пользователе (СЕБЕ)
Вопросы "кто я", "мой отдел", "моя должность", "где я работаю" —
отвечай ИСКЛЮЧИТЕЛЬНО из <session>. НЕ используй инструменты.

## Поиск ДРУГОГО сотрудника (employee_search_tool)
Используй employee_search_tool когда нужна информация о ДРУГОМ человеке:
  - "найди Иванова" / "кто такой Петров" / "номер телефона Сидоровой"
  - создание поручения или ознакомления с указанием ФИО

СТРАТЕГИИ поиска (выбирай минимально достаточный):
  1. employee_search_tool(last_name="Иванов")            — по фамилии
  2. employee_search_tool(last_name="Иванов", first_name="Николай")  — уточненный
  3. employee_search_tool(full_post_name="директор")     — по должности
  4. employee_search_tool(employee_id="<uuid>")          — прямой по UUID

ЕСЛИ найдено НЕСКОЛЬКО сотрудников:
  → Выведи НУМЕРОВАННЫЙ список:
    "Найдено N сотрудников с фамилией Иванов, уточните выбор:
     1. Иванов Н. — аккумуляторщик, Технический отдел
     2. Иванов И.А. — администратор, Бухгалтерия
     ..."
  → Попроси выбрать номер ИЛИ уточнить имя/должность
  → После выбора: employee_search_tool(employee_id="<uuid выбранного>")

## Работа с файлами (LOCAL_FILE)
- UUID (xxxxxxxx-xxxx-...) → doc_get_file_content(attachment_id=UUID)
- Путь /tmp/... или C:\\... → read_local_file_content(file_path=PATH)
- "Не загружен" → список вложений из <attachments> или вызови doc_get_details()

## Суммаризация
Всегда два шага:
  1. doc_get_file_content(attachment_id=<uuid из attachments>) — получить текст
  2. doc_summarize_text(text=<текст>) — ТОЛЬКО текст, summary_type НЕ указывай
     (инструмент сам предложит выбор пользователю)

## Вопросы о конкретных полях
Проверь <custom_fields> в document_context. Не придумывай данные.

## Общие вопросы
Если вопрос не связан с EDMS (наука, программирование, общие знания) —
отвечай как умный ассистент. Используй все свои знания.

## Формат ответов
- Финальный ответ всегда на русском языке
- Структурировано и по делу
- Обращайся к пользователю по имени: {user_name}
- После каждого tool call формулируй понятный ответ пользователю
</rules>

<tools_guide>
ИНСТРУМЕНТ                  КОГДА ИСПОЛЬЗОВАТЬ
─────────────────────────────────────────────────────────────────
doc_get_details()           Детали документа если context пустой
doc_get_file_content        Содержимое вложения по UUID
doc_summarize_text          Суммаризация текста (после get_file_content)
doc_compare(id1, id2)       Сравнение двух документов
employee_search_tool        Поиск ДРУГОГО сотрудника по ФИО/должности/UUID
introduction_create_tool    Создать ознакомление с документом
task_create_tool            Создать поручение по документу
read_local_file_content     Прочитать локальный файл с диска

НЕ использовать инструменты для:
  × Данные текущего пользователя → из <session>
  × Поля документа → из <document_context>
  × Общие знания не связанные с EDMS
</tools_guide>"""

    # ── Intent-specific guides ────────────────────────────────────────────────
    _GUIDES: dict[UserIntent, str] = {
        UserIntent.CREATE_INTRODUCTION: """
<introduction_guide>
При "requires_disambiguation": перечисли найденных сотрудников, дождись выбора.
Повторный вызов: introduction_create_tool(selected_employee_ids=[uuid1, uuid2])
</introduction_guide>""",

        UserIntent.CREATE_TASK: """
<task_guide>
executor_last_names: обязателен (минимум 1).
planed_date_end: ISO 8601 UTC — "2026-03-01T23:59:59Z".
Если дата не указана → текущая дата + 7 дней.
При "requires_disambiguation" → перечисли найденных, дождись выбора.
</task_guide>""",

        UserIntent.SUMMARIZE: """
<summarize_guide>
Шаг 1: doc_get_file_content(attachment_id=<uuid>) — извлечь текст
Шаг 2: doc_summarize_text(text=<текст>) — summary_type оставь пустым
Инструмент сам предложит пользователю выбрать формат (факты / пересказ / тезисы).
</summarize_guide>""",
    }

    @classmethod
    def build(
        cls,
        context: ContextParams,
        intent: UserIntent,
        document_context_xml: str,
        semantic_xml: str,
    ) -> str:
        """Assemble the complete system prompt.

        Args:
            context: Immutable request context.
            intent: Detected user intent.
            document_context_xml: ``<document_context>`` from DocumentContextBuilder.
            semantic_xml: ``<semantic_context>`` from SemanticDispatcher.

        Returns:
            Complete system prompt string.
        """

        base = cls._CORE.format(
            user_name=context.user_name,
            user_post=context.user_post or "Не указана",
            user_department=context.user_department or "Не указано",
            current_date=context.current_date,
            document_id=context.document_id or "Не указан",
            file_path=context.file_path or "Не загружен",
        )
        guide = cls._GUIDES.get(intent, "")
        return base + guide + document_context_xml + semantic_xml


# ─────────────────────────────────────────────────────────────────────────────
# ContentExtractor — финальный ответ из цепочки сообщений
# ─────────────────────────────────────────────────────────────────────────────


class ContentExtractor:
    """Extracts the final user-facing answer from a LangGraph message chain.

    Priority:
        1. Last AIMessage без tool_calls — финальный ответ LLM
        2. Structured text from last ToolMessage JSON
        3. Any AIMessage with content (fallback)
        4. Raw ToolMessage (last resort; пропускает JSON-конверты)
    """

    _SKIP_PATTERNS: tuple[str, ...] = (
        "вызвал инструмент",
        "tool call",
        '"name"',
        '"id"',
    )
    _MIN_LENGTH: int = 30
    _JSON_TEXT_FIELDS: tuple[str, ...] = (
        "content",
        "text",
        "text_preview",
        "message",
    )

    @classmethod
    def extract_final_content(cls, messages: list[BaseMessage]) -> str | None:
        """Extract final user-facing response.

        Args:
            messages: Full LangGraph message chain.

        Returns:
            Content string or ``None``.
        """
        # 1. Last AIMessage without tool_calls (real final answer)
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                if getattr(m, "tool_calls", None):
                    continue  # intermediate step
                content = str(m.content).strip()
                if not cls._is_artifact(content) and len(content) > cls._MIN_LENGTH:
                    return content

        # 2. Structured text from last ToolMessage
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._from_tool_json(m)
                if extracted:
                    return extracted

        # 3. Any AIMessage (fallback)
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                content = str(m.content).strip()
                if len(content) > cls._MIN_LENGTH:
                    return content

        # 4. Raw ToolMessage (last resort — skip JSON envelopes)
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                content = str(m.content).strip()
                if content.startswith('{"status"') or content.startswith('{"error"'):
                    continue
                if len(content) > cls._MIN_LENGTH:
                    return content

        return None

    @classmethod
    def extract_last_tool_text(cls, messages: list[BaseMessage]) -> str | None:
        """Extract raw text from the last ToolMessage for summarization injection.

        Args:
            messages: Full message chain.

        Returns:
            Text string or ``None``.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                raw = str(m.content)
                if raw.startswith("{"):
                    data = json.loads(raw)
                    nested = data.get("data") or {}
                    text = (
                        nested.get("content")
                        or nested.get("text_preview")
                        or data.get("content")
                        or data.get("text_preview")
                        or data.get("text")
                    )
                    if text and len(str(text)) > 100:
                        return str(text)
                if len(raw) > 100:
                    return raw
            except (json.JSONDecodeError, TypeError):
                if len(str(m.content)) > 100:
                    return str(m.content)
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """Strip JSON status-envelope wrappers.

        Args:
            content: Raw content string.

        Returns:
            Clean human-readable string.
        """
        if content.startswith('{"status"'):
            try:
                data = json.loads(content)
                if "content" in data:
                    content = data["content"]
                elif isinstance(data.get("data"), dict):
                    content = data["data"].get("content", content)
            except json.JSONDecodeError:
                pass

        return (
            content
            .replace('{"status": "success", "content": "', "")
            .replace('"}', "")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .strip()
        )

    @classmethod
    def _is_artifact(cls, content: str) -> bool:
        lower = content.lower()
        return any(p in lower for p in cls._SKIP_PATTERNS)

    @classmethod
    def _from_tool_json(cls, message: ToolMessage) -> str | None:
        try:
            raw = str(message.content).strip()
            if not raw.startswith("{"):
                return None
            data = json.loads(raw)
            nested = data.get("data") or {}
            for key in cls._JSON_TEXT_FIELDS:
                val = nested.get(key) or data.get(key)
                if val:
                    text = str(val).strip()
                    if len(text) > cls._MIN_LENGTH:
                        return text
        except (json.JSONDecodeError, TypeError):
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ToolParameterInjector — инъекция параметров в tool_calls перед выполнением
# ─────────────────────────────────────────────────────────────────────────────


class ToolParameterInjector:
    """Инъектирует обязательные параметры в tool_calls перед исполнением.

    Единственный класс, который знает о правилах инъекции:
        1. token  — всегда, для всех инструментов
        2. document_id — для doc_* и task/introduction tools
        3. UUID file_path → attachment_id conversion для doc_get_file_content
        4. text — для doc_summarize_text из последнего ToolMessage

    summary_type НЕ инъектируется — SummarizationTool управляет форматом сам.
    """

    @staticmethod
    def inject(
        tool_calls: list[dict],
        context: ContextParams,
        last_tool_text: str | None,
    ) -> list[dict]:
        """Apply parameter injection to all pending tool_calls.

        Injection rules:
            1. ``token``                — всегда, для всех инструментов
            2. ``document_id``          — для doc_* / task / introduction tools
            3. UUID file_path           → attachment_id (redirect to doc_get_file_content)
            4. ``text``                 — для doc_summarize_text из последнего ToolMessage
            5. ``summary_type``         — ТОЛЬКО если context.preselected_summary_type задан.
               Это сценарий «выбор из UI выпадающего списка»: фронтенд передаёт human_choice
               в первом запросе (до suspend графа), агент инъектирует тип сразу.
               Иначе — SummarizationTool сам предлагает выбор (requires_choice).

        Args:
            tool_calls: Raw tool_calls list from AIMessage.
            context: Immutable execution context.
            last_tool_text: Text from last ToolMessage (for summarization).

        Returns:
            New list of tool_calls with injected parameters.
        """
        fixed: list[dict] = []

        for tc in tool_calls:
            t_name: str = tc["name"]
            t_args: dict = dict(tc["args"])
            t_id: str = tc["id"]

            # 1. JWT — всегда
            t_args["token"] = context.user_token

            # 2. document_id — для doc_*, task_create_tool, introduction_create_tool
            if context.document_id and (
                t_name.startswith("doc_")
                or t_name in ("introduction_create_tool", "task_create_tool")
                or "document_id" in t_args
            ):
                if not t_args.get("document_id"):
                    t_args["document_id"] = context.document_id

            # 3. UUID file_path → attachment_id
            clean_path = str(context.file_path or "").strip()
            if _UUID_RE.match(clean_path) and t_name == "read_local_file_content":
                t_name = "doc_get_file_content"
                t_args["attachment_id"] = clean_path
                t_args.pop("file_path", None)
                log.info(
                    "uuid_path_to_attachment_converted",
                    attachment_id_prefix=clean_path[:8],
                )

            # 4 + 5. Параметры doc_summarize_text
            if t_name == "doc_summarize_text":
                # 4. Текст из последнего ToolMessage
                if last_tool_text and not t_args.get("text"):
                    t_args["text"] = str(last_tool_text)

                # 5. Preselected summary_type (выбор из UI до запуска графа).
                if not t_args.get("summary_type") and context.preselected_summary_type:
                    t_args["summary_type"] = context.preselected_summary_type
                    log.info(
                        "preselected_summary_type_injected",
                        summary_type=context.preselected_summary_type,
                    )

            fixed.append({"name": t_name, "args": t_args, "id": t_id})

        return fixed


# ─────────────────────────────────────────────────────────────────────────────
# AgentStateManager — тонкая обёртка над CompiledStateGraph API
# ─────────────────────────────────────────────────────────────────────────────


class AgentStateManager:
    """Thin wrapper encapsulating LangGraph CompiledStateGraph API.

    Provides: get_state / update_state / invoke.
    Используется исключительно в EdmsDocumentAgent._orchestrate().
    """

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver) -> None:
        """Initialize.

        Args:
            graph: Compiled LangGraph StateGraph.
            checkpointer: MemorySaver for in-process persistence.

        Raises:
            ValueError: When either argument is ``None``.
        """
        if graph is None:
            raise ValueError("graph cannot be None")
        if checkpointer is None:
            raise ValueError("checkpointer cannot be None")
        self._graph = graph
        self._checkpointer = checkpointer

    async def get_state(self, thread_id: str) -> Any:
        """Retrieve current StateSnapshot for a thread.

        Args:
            thread_id: LangGraph session identifier.

        Returns:
            ``StateSnapshot`` with ``.values`` and ``.next``.
        """
        config = {"configurable": {"thread_id": thread_id}}
        return await self._graph.aget_state(config)

    async def update_state(
        self,
        thread_id: str,
        messages: list[BaseMessage],
        as_node: str = "agent",
    ) -> None:
        """Inject messages into state (parameter injection pattern).

        Args:
            thread_id: LangGraph session identifier.
            messages: Messages to inject.
            as_node: Node context for the update.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await self._graph.aupdate_state(
            config, {"messages": messages}, as_node=as_node
        )

    async def invoke(
        self,
        inputs: dict[str, Any] | None,
        thread_id: str,
        timeout: float = 120.0,
    ) -> None:
        """Run one graph step with timeout.

        Args:
            inputs: Graph inputs (``None`` = resume from interrupt).
            thread_id: LangGraph session identifier.
            timeout: Execution timeout in seconds.

        Raises:
            asyncio.TimeoutError: On timeout.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await asyncio.wait_for(
            self._graph.ainvoke(inputs, config=config),
            timeout=timeout,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EdmsDocumentAgent — production LangGraph orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class EdmsDocumentAgent(AbstractAgent):
    """Production-grade LangGraph agent for EDMS document operations.

    Attributes:
        config: Agent behavior configuration (``AgentConfig``).
        model: LangChain chat model.
        tools: List of ``AbstractEdmsTool`` instances.
        document_repo: Port for document metadata.
        dispatcher: Semantic intent classifier.
        state_manager: LangGraph state wrapper.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        document_repo: IDocumentRepository | None = None,
        semantic_dispatcher: SemanticDispatcher | None = None,
    ) -> None:
        """Initialize agent with optional dependency injection.

        All dependencies have safe defaults — zero-argument construction works.

        Args:
            config: Agent configuration. Defaults to ``AgentConfig()``.
            document_repo: Document metadata port.
                Defaults to ``DocumentRepositoryAdapter()``.
            semantic_dispatcher: Intent classifier.
                Defaults to ``SemanticDispatcher()``.

        Raises:
            RuntimeError: On initialization or graph compilation failure.
        """
        try:
            self.config: AgentConfig = config or AgentConfig()
            self.model = get_chat_model()
            self.document_repo: IDocumentRepository = (
                document_repo or DocumentRepositoryAdapter()
            )
            self.dispatcher: SemanticDispatcher = (
                semantic_dispatcher or SemanticDispatcher()
            )

            log.debug("base_components_initialized")

            self.tools = self._create_tools()
            self._checkpointer = MemorySaver()
            self._graph = self._build_graph()

            self.state_manager = AgentStateManager(
                graph=self._graph,
                checkpointer=self._checkpointer,
            )

            log.info(
                "edms_agent_ready",
                tools_count=len(self.tools),
                tool_names=[t.name for t in self.tools],
                max_iterations=self.config.max_iterations,
                timeout=self.config.execution_timeout,
            )

        except Exception as exc:
            log.error("edms_agent_init_failed", error=str(exc), exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {exc}") from exc

    # ── Initialization helpers ────────────────────────────────────────────────

    def _create_tools(self) -> list:
        """Instantiate all EDMS tools via DI factory.

        Lazy imports ensure infrastructure is loaded only at startup.
        ``storage=None``: AttachmentTool downloads via EdmsAttachmentClient REST API.

        Returns:
            List of ``AbstractEdmsTool`` instances.
        """
        from ...domain.services.appeal_validator import AppealValidator
        from ...domain.services.document_comparer import DocumentComparer
        from ...domain.services.task_assigner import TaskAssigner
        from ...infrastructure.edms_api.http_client import EdmsHttpClient
        from ...infrastructure.edms_api.repositories.edms_document_repository import (
            EdmsDocumentRepository,
        )
        from ...infrastructure.edms_api.repositories.edms_employee_repository import (
            EdmsEmployeeRepository,
        )
        from ...infrastructure.edms_api.repositories.edms_task_repository import (
            EdmsTaskRepository,
        )
        from ...infrastructure.llm.providers.openai_provider import OpenAIProvider

        http_client = EdmsHttpClient()

        tools = create_all_tools(
            document_repository=EdmsDocumentRepository(http_client),
            employee_repository=EdmsEmployeeRepository(http_client),
            task_repository=EdmsTaskRepository(http_client),
            llm_provider=OpenAIProvider(),
            nlp_extractor=self._try_create_nlp_extractor(),
            document_comparer=DocumentComparer(),
            appeal_validator=AppealValidator(),
            task_assigner=TaskAssigner(),
            storage=None,
            attachment_client=None,
        )

        log.info("tools_initialized", count=len(tools), names=[t.name for t in tools])
        return tools

    def _try_create_nlp_extractor(self) -> Any | None:
        """Create NLP extractor with graceful degradation.

        Returns:
            ``AppealExtractor`` or ``None``.
        """
        try:
            from ...infrastructure.nlp.extractors.appeal_extractor import AppealExtractor
            return AppealExtractor()
        except Exception as exc:
            log.warning("nlp_extractor_unavailable", error=str(exc))
            return None

    def health_check(self) -> dict[str, Any]:
        """Return component health dict for monitoring/readiness probes.

        Returns:
            Dict with boolean flags per component.
        """
        return {
            "model": self.model is not None,
            "tools_count": len(self.tools),
            "tool_names": [t.name for t in self.tools],
            "document_repo": self.document_repo is not None,
            "dispatcher": self.dispatcher is not None,
            "graph": self._graph is not None,
            "state_manager": self.state_manager is not None,
            "max_iterations": self.config.max_iterations,
            "execution_timeout": self.config.execution_timeout,
        }

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        """Build and compile the LangGraph workflow.

        Nodes:
            agent:      LLM call — increments graph_iterations.
            tools:      ToolNode — исполнение tool_calls.
            validator:  Проверяет результат; инъектирует уведомление при ошибке.

        Returns:
            ``CompiledStateGraph`` с ``interrupt_before=["tools"]``.

        Raises:
            RuntimeError: On compilation failure.
        """
        workflow = StateGraph(AgentStateWithCounter)

        # ── Node: agent ───────────────────────────────────────────────────────

        async def call_model(state: AgentStateWithCounter) -> dict:
            """Call LLM with bound tools.

            Passes exactly ONE SystemMessage (the latest) + all non-system
            messages. Increments ``graph_iterations`` via operator.add.

            Args:
                state: Current graph state.

            Returns:
                Dict with AIMessage and graph_iterations increment.
            """
            model_with_tools = self.model.bind_tools(self.tools)

            all_msgs = state["messages"]
            sys_msgs = [m for m in all_msgs if isinstance(m, SystemMessage)]
            non_sys = [m for m in all_msgs if not isinstance(m, SystemMessage)]
            llm_input = ([sys_msgs[-1]] if sys_msgs else []) + non_sys

            response = await model_with_tools.ainvoke(llm_input)
            return {"messages": [response], "graph_iterations": 1}

        # ── Node: validator ───────────────────────────────────────────────────

        async def validate_tool_result(state: AgentStateWithCounter) -> dict:
            """Validate tool execution result; inject error notification if needed.

            Trigger conditions:
                - Empty result (None / {} / empty string)
                - BOTH "error" AND "exception" in content (avoids false positives)

            Skipped when ``config.enable_tool_validation`` is False.

            Args:
                state: Current state after tool execution.

            Returns:
                Dict with optional system HumanMessage.
            """
            if not self.config.enable_tool_validation:
                return {"messages": []}

            last_msg = state["messages"][-1]
            if not isinstance(last_msg, ToolMessage):
                return {"messages": []}

            content = str(last_msg.content).strip()

            if not content or content in ("None", "{}"):
                return {
                    "messages": [
                        HumanMessage(
                            content=(
                                "[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Инструмент вернул пустой результат. "
                                "Сообщи пользователю понятным языком."
                            )
                        )
                    ]
                }

            c_lower = content.lower()
            if "error" in c_lower and "exception" in c_lower:
                return {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Техническая ошибка: "
                                f"{content[:400]}"
                            )
                        )
                    ]
                }

            return {"messages": []}

        # ── Edges ─────────────────────────────────────────────────────────────

        def should_continue(state: AgentStateWithCounter) -> str:
            """Route agent → tools or END.

            NOTE: AgentStateWithCounter MUST be at module level.
            LangGraph resolves this type from module globals().

            Args:
                state: Current graph state.

            Returns:
                ``"tools"`` or ``END``.
            """
            last_msg = state["messages"][-1]
            if (
                isinstance(last_msg, AIMessage)
                and getattr(last_msg, "tool_calls", None)
            ):
                return "tools"
            return END

        def route_after_validator(state: AgentStateWithCounter) -> str:
            """Route validator → agent or END (infinite loop guard).

            Args:
                state: State with ``graph_iterations`` counter.

            Returns:
                ``"agent"`` or ``END``.
            """
            iterations = state.get("graph_iterations", 0)
            if iterations >= _MAX_GRAPH_ITERATIONS:
                log.warning(
                    "graph_max_iterations_reached",
                    iterations=iterations,
                    max=_MAX_GRAPH_ITERATIONS,
                )
                return END
            return "agent"

        # ── Wiring ────────────────────────────────────────────────────────────

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("validator", validate_tool_result)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "validator")
        workflow.add_conditional_edges(
            "validator", route_after_validator, {"agent": "agent", END: END}
        )

        try:
            compiled = workflow.compile(
                checkpointer=self._checkpointer,
                interrupt_before=["tools"],
            )
            log.debug("graph_compiled_successfully")
            return compiled
        except Exception as exc:
            log.error("graph_compilation_failed", error=str(exc), exc_info=True)
            raise RuntimeError(f"Graph compilation failed: {exc}") from exc

    # ── Public API ────────────────────────────────────────────────────────────

    async def chat(self, request: AgentRequest) -> AgentResponse:
        """Process user message — main entry point.

        Args:
            request: Validated ``AgentRequest`` from interface layer.

        Returns:
            ``AgentResponse`` — ВСЕГДА, даже при ошибках.
        """
        try:
            context = await self._build_context(request)

            # ── Human-in-the-loop: resume suspended graph ─────────────────────
            if request.human_choice:
                normalized_choice = _normalize_choice(request.human_choice)
                state = await self.state_manager.get_state(context.thread_id)
                if state.next:
                    # ── Сценарий A: resume suspended графа ───────────────────
                    log.info(
                        "human_choice_resume",
                        raw=request.human_choice,
                        normalized=normalized_choice,
                        thread_id=context.thread_id,
                    )
                    return await self._handle_human_choice(context, normalized_choice)
                else:
                    # ── Сценарий B: preselected из UI (1 шаг) ────────────────
                    log.info(
                        "human_choice_preselected",
                        raw=request.human_choice,
                        normalized=normalized_choice,
                        thread_id=context.thread_id,
                    )
                    context = ContextParams(
                        user_token=context.user_token,
                        document_id=context.document_id,
                        file_path=context.file_path,
                        thread_id=context.thread_id,
                        user_name=context.user_name,
                        user_first_name=context.user_first_name,
                        user_last_name=context.user_last_name,
                        user_department=context.user_department,
                        user_post=context.user_post,
                        current_date=context.current_date,
                        preselected_summary_type=normalized_choice,
                    )

            # ── Fetch document (graceful: None if unavailable) ─────────────────
            document: Document | None = None
            if context.document_id:
                document = await self.document_repo.get_document(
                    context.user_token, context.document_id
                )

            # ── Semantic analysis ─────────────────────────────────────────────
            semantic_ctx = self.dispatcher.build_context(request.message, document)
            log.info(
                "semantic_analysis_complete",
                intent=semantic_ctx.query.intent.value,
                complexity=semantic_ctx.query.complexity.value,
                thread_id=context.thread_id,
            )

            # ── Assemble system prompt ────────────────────────────────────────
            system_prompt = PromptBuilder.build(
                context=context,
                intent=semantic_ctx.query.intent,
                document_context_xml=DocumentContextBuilder.build(document),
                semantic_xml=self._build_semantic_xml(semantic_ctx),
            )

            inputs: dict[str, Any] = {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=semantic_ctx.query.refined),
                ]
            }

            # ── Per-request isolated LangGraph thread ─────────────────────────
            request_thread_id = f"{context.thread_id}_req_{uuid.uuid4().hex[:8]}"
            log.debug(
                "request_thread_created",
                base_thread=context.thread_id,
                request_thread=request_thread_id,
            )

            request_context = ContextParams(
                user_token=context.user_token,
                document_id=context.document_id,
                file_path=context.file_path,
                thread_id=request_thread_id,
                user_name=context.user_name,
                user_first_name=context.user_first_name,
                user_last_name=context.user_last_name,
                user_department=context.user_department,
                user_post=context.user_post,
                current_date=context.current_date,
                preselected_summary_type=context.preselected_summary_type,
            )

            return await self._orchestrate(
                context=request_context,
                inputs=inputs,
                iteration=0,
            )

        except Exception as exc:
            log.error("chat_error", error=str(exc), exc_info=True)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {exc}",
            )

    # ── Human-in-the-loop ─────────────────────────────────────────────────────

    async def _handle_human_choice(
        self,
        context: ContextParams,
        human_choice: str,
    ) -> AgentResponse:
        """Resume suspended graph after user's selection.

        Injects ``human_choice`` into the pending tool_call args,
        then resumes via ``_orchestrate(inputs=None)``.

        Supported:
            - doc_summarize_text: injects ``summary_type``
            - Employee disambiguation: propagated as-is

        Args:
            context: Context with ORIGINAL thread_id (suspension lives there).
            human_choice: User's selection (e.g. ``"extractive"``).

        Returns:
            ``AgentResponse`` after resumed execution.
        """
        state = await self.state_manager.get_state(context.thread_id)
        last_msg = state.values["messages"][-1]

        fixed_calls = []
        for tc in getattr(last_msg, "tool_calls", []):
            t_args = dict(tc["args"])
            t_name = tc["name"]

            if t_name == "doc_summarize_text":
                t_args["summary_type"] = human_choice
                log.info("human_choice_injected", tool=t_name, choice=human_choice)

            fixed_calls.append({"name": t_name, "args": t_args, "id": tc["id"]})

        await self.state_manager.update_state(
            context.thread_id,
            [
                AIMessage(
                    content=last_msg.content or "",
                    tool_calls=fixed_calls,
                    id=last_msg.id,
                )
            ],
            as_node="agent",
        )

        return await self._orchestrate(
            context=context,
            inputs=None,
            iteration=0,
        )

    # ── Orchestration loop ────────────────────────────────────────────────────

    async def _orchestrate(
        self,
        context: ContextParams,
        inputs: dict[str, Any] | None,
        iteration: int = 0,
    ) -> AgentResponse:
        """LangGraph orchestration loop with ToolParameterInjector.

        На каждой паузе ``interrupt_before["tools"]`` вызывает
        ``ToolParameterInjector.inject()`` для добавления token / document_id /
        text в tool_calls, затем возобновляет граф.

        Guards:
            1. ``iteration > config.max_iterations`` — application layer
            2. ``graph_iterations >= _MAX_GRAPH_ITERATIONS`` — graph layer

        Args:
            context: Immutable execution context.
            inputs: Graph inputs (``None`` = resume from interrupt).
            iteration: Recursion depth counter.

        Returns:
            ``AgentResponse``.
        """
        if iteration > self.config.max_iterations:
            log.error(
                "max_iterations_exceeded",
                thread_id=context.thread_id,
                iteration=iteration,
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышен лимит итераций обработки.",
            )

        try:
            # ── Execute graph step ────────────────────────────────────────────
            await self.state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=self.config.execution_timeout,
            )

            state = await self.state_manager.get_state(context.thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])

            log.debug(
                "state_snapshot",
                thread_id=context.thread_id,
                iteration=iteration,
                messages_count=len(messages),
                last_type=type(messages[-1]).__name__ if messages else None,
                state_next=list(state.next) if state.next else [],
            )

            if not messages:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Пустое состояние агента.",
                )

            last_msg = messages[-1]

            # ── Detect requires_choice from ToolMessage ────────────────────────
            if isinstance(last_msg, ToolMessage):
                try:
                    raw = str(last_msg.content).strip()
                    if raw.startswith("{"):
                        tool_data = json.loads(raw)
                        if tool_data.get("status") == "requires_choice":
                            choices_data = tool_data.get("data", {})
                            options = choices_data.get("options", [])
                            message_text = tool_data.get(
                                "message", "Выберите формат анализа:"
                            )

                            log.info(
                                "requires_choice_detected",
                                thread_id=context.thread_id,
                                options_count=len(options),
                            )

                            return AgentResponse(
                                status=AgentStatus.REQUIRES_ACTION,
                                message=message_text,
                                action_type=ActionType.SUMMARIZE_SELECTION,
                                metadata={
                                    _CHOICE_THREAD_KEY: context.thread_id,
                                    "options": options,
                                },
                            )
                except (json.JSONDecodeError, TypeError):
                    pass

            # ── Graph done → extract final answer ─────────────────────────────
            if (
                not state.next
                or not isinstance(last_msg, AIMessage)
                or not getattr(last_msg, "tool_calls", None)
            ):
                final = ContentExtractor.extract_final_content(messages)
                if final:
                    final = ContentExtractor.clean_json_artifacts(final)
                    log.info(
                        "execution_completed",
                        thread_id=context.thread_id,
                        chars=len(final),
                        iterations=iteration + 1,
                    )
                    return AgentResponse(
                        status=AgentStatus.SUCCESS,
                        content=final,
                    )

                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    content="Анализ завершён.",
                )

            # ── Graph paused → inject parameters → resume ─────────────────────
            last_tool_text = ContentExtractor.extract_last_tool_text(messages)

            fixed_calls = ToolParameterInjector.inject(
                tool_calls=last_msg.tool_calls,
                context=context,
                last_tool_text=last_tool_text,
            )

            await self.state_manager.update_state(
                context.thread_id,
                [
                    AIMessage(
                        content=last_msg.content or "",
                        tool_calls=fixed_calls,
                        id=last_msg.id,
                    )
                ],
                as_node="agent",
            )

            # Recurse
            return await self._orchestrate(
                context=context,
                inputs=None,
                iteration=iteration + 1,
            )

        except asyncio.TimeoutError:
            log.error("execution_timeout", thread_id=context.thread_id)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания выполнения.",
            )

        except Exception as exc:
            log.error("orchestration_error", error=str(exc), exc_info=True)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка оркестрации: {exc}",
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _build_context(self, request: AgentRequest) -> ContextParams:
        """Build ``ContextParams`` from ``AgentRequest``.

        Handles ``user_context`` as plain dict or Pydantic model.

        Args:
            request: Validated ``AgentRequest``.

        Returns:
            ``ContextParams`` for this request.
        """
        ctx = request.user_context or {}
        if hasattr(ctx, "model_dump"):
            ctx = ctx.model_dump(exclude_none=True)

        user_name: str = (
                ctx.get("firstName")
                or ctx.get("first_name")
                or ctx.get("name")
                or "пользователь"
        ).strip()

        # ── Фамилия ───────────────────────────────────────────────────────────
        user_last_name: str | None = (
                ctx.get("lastName") or ctx.get("last_name")
        )

        # ── Подразделение (EmployeeRepository → agent_routes → user_context) ──
        user_department: str | None = (
                ctx.get("departmentName")
                or ctx.get("department_name")
                or ctx.get("authorDepartmentName")
        )

        # ── Должность ─────────────────────────────────────────────────────────
        user_post: str | None = (
                ctx.get("postName")
                or ctx.get("post_name")
                or ctx.get("authorPost")
        )

        return ContextParams(
            user_token=request.user_token,
            document_id=request.context_ui_id,
            file_path=request.file_path,
            thread_id=request.thread_id or "default",
            user_name=user_name,
            user_first_name=ctx.get("firstName") or ctx.get("first_name"),
            user_last_name=user_last_name,
            user_department=user_department,
            user_post=user_post,
        )

    @staticmethod
    def _build_semantic_xml(semantic_ctx: Any) -> str:
        """Build ``<semantic_context>`` XML block.

        Args:
            semantic_ctx: ``SemanticContext`` from ``SemanticDispatcher``.

        Returns:
            XML string for system prompt.
        """
        return (
            "\n<semantic_context>"
            f"\n  <intent>{semantic_ctx.query.intent.value}</intent>"
            f"\n  <original>{semantic_ctx.query.original}</original>"
            f"\n  <refined>{semantic_ctx.query.refined}</refined>"
            f"\n  <complexity>{semantic_ctx.query.complexity.value}</complexity>"
            "\n</semantic_context>"
        )