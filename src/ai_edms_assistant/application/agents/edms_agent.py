# src/ai_edms_assistant/application/agents/edms_agent.py
"""EDMS Document Agent — production-ready autonomous multi-agent orchestrator.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
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
from langgraph.types import RunnableConfig

from ...domain.entities.document import Document
from ...infrastructure.edms_api.http_client import EdmsHttpClient
from ...infrastructure.llm.providers.openai_provider import get_chat_model
from ..dto.agent import AgentRequest, AgentResponse, AgentStatus
from ..services.semantic_dispatcher import SemanticDispatcher, UserIntent
from ..tools import create_all_tools
from .agent_config import AgentConfig

from .agent_state import AgentStateWithCounter

# ── Module-level loggers ──────────────────────────────────────────────────────
log = structlog.get_logger(__name__)
logger = logging.getLogger(__name__)  # stdlib fallback

# ── Graph-level constants (visible in module globalns for LangGraph) ──────────
MAX_GRAPH_ITERATIONS: int = 10
MAX_CONTEXT_MESSAGES: int = 20
SUMMARIZE_TEXT_LIMIT: int = 10_000

# ── UUID patterns (compiled once at module load) ──────────────────────────────
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# ── Summary type mappings ─────────────────────────────────────────────────────
_SUMMARY_TYPE_LABELS: dict[str, str] = {
    "extractive": "ключевые факты",
    "abstractive": "краткий пересказ",
    "thesis": "тезисный план",
}

_CHOICE_NORM: dict[str, str] = {
    "1": "extractive",
    "факты": "extractive",
    "ключевые факты": "extractive",
    "extractive": "extractive",
    "2": "abstractive",
    "пересказ": "abstractive",
    "краткий пересказ": "abstractive",
    "abstractive": "abstractive",
    "3": "thesis",
    "тезисы": "thesis",
    "тезисный план": "thesis",
    "thesis": "thesis",
}



@dataclass(frozen=True)
class ContextParams:
    """Immutable execution context for one agent invocation.

    Built once per ``chat()`` call from ``AgentRequest`` and passed through
    the entire orchestration pipeline without mutation. ``frozen=True`` ensures
    thread-safety in concurrent async environments.

    Attributes:
        user_token: JWT bearer token for all EDMS API calls.
        thread_id: LangGraph checkpointer session identifier.
        document_id: UUID of the active document in UI context (optional).
        file_path: Local temp path or EDMS attachment UUID from frontend.
        summary_type: Canonical SummaryType from human_choice (optional).
        user_name: Display name for prompt personalization.
        user_first_name: First name extracted from employee API.
        current_date: Date string injected into system prompt.
    """

    user_token: str
    thread_id: str
    document_id: str | None = None
    file_path: str | None = None
    summary_type: str | None = None
    user_name: str = "пользователь"
    user_first_name: str | None = None
    current_date: str = field(
        default_factory=lambda: datetime.now().strftime("%d.%m.%Y")
    )

    def __post_init__(self) -> None:
        """Validate required fields on construction.

        Raises:
            ValueError: When user_token or thread_id is empty.
        """
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")
        if not self.thread_id or not isinstance(self.thread_id, str):
            raise ValueError("thread_id must be a non-empty string")


class IDocumentRepository(Protocol):
    """Port: abstract contract for document metadata access."""

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        """Fetch document metadata by ID.

        Args:
            token: JWT bearer token.
            doc_id: Document UUID string.

        Returns:
            Document entity or None when not found / on error.
        """


class DocumentRepository:
    """Default adapter: fetches via EdmsDocumentRepository (EDMS REST API).

    Failures are swallowed — a missing document is NOT a fatal error.
    The agent continues operating without document context.
    """

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        """Fetch document metadata. Graceful on any error.

        Args:
            token: JWT bearer token.
            doc_id: Document UUID string.

        Returns:
            Document entity or None.
        """
        try:
            from uuid import UUID

            from ...infrastructure.edms_api.repositories.edms_document_repository import (
                EdmsDocumentRepository,
            )

            repo = EdmsDocumentRepository(http_client=EdmsHttpClient())
            return await repo.get_by_id(entity_id=UUID(doc_id), token=token)
        except Exception as exc:  # noqa: BLE001
            log.warning("document_fetch_failed", doc_id=doc_id, error=str(exc))
            return None



class NLPHelperService:
    """Recommends summarization format based on text characteristics.

    Decision matrix:
        chars > 5000 OR digit_groups > 20  →  thesis     (объёмный / числовой)
        chars > 2000                        →  extractive (средний объём)
        chars <= 2000                       →  abstractive (краткий)
    """

    @staticmethod
    def suggest_summarize_format(text: str) -> dict[str, Any]:
        """Analyze text and recommend optimal summary type.

        Args:
            text: Input document text to analyze.

        Returns:
            Dict with keys: recommended (str), reason (str), stats (dict).
        """
        if not text:
            return {
                "recommended": "abstractive",
                "reason": "Текст пустой — используется пересказ по умолчанию",
                "stats": {"chars": 0},
            }

        chars = len(text)
        digit_groups = len(re.findall(r"\d+", text))

        if chars > 5000 or digit_groups > 20:
            return {
                "recommended": "thesis",
                "reason": (
                    f"Объёмный текст ({chars} симв.) или много чисел "
                    f"({digit_groups}) — тезисы оптимальны"
                ),
                "stats": {"chars": chars},
            }
        if chars > 2000:
            return {
                "recommended": "extractive",
                "reason": f"Средний объём ({chars} симв.) — ключевые факты предпочтительны",
                "stats": {"chars": chars},
            }
        return {
            "recommended": "abstractive",
            "reason": f"Краткий текст ({chars} симв.) — пересказ своими словами",
            "stats": {"chars": chars},
        }



class DocumentContextService:
    """Builds deterministic XML context block from Document entity.

    Injected into SystemMessage ONCE before graph execution.
    LLM reads exact attachment UUIDs from XML → no hallucination →
    no extra API calls to doc_get_details() for UUID discovery.
    """

    @staticmethod
    def build_context_block(
        document: Document | None,
        context: ContextParams,
    ) -> str:
        """Build XML document context block for system prompt injection.

        Args:
            document: Fetched Document entity (may be None).
            context: Immutable execution context.

        Returns:
            XML string with document metadata and attachment UUIDs.
            Empty string when no document and no context IDs.
        """
        if not document and not context.document_id and not context.file_path:
            return ""

        parts: list[str] = ["<document_context>"]

        if document:
            parts.append(f"  <id>{document.id}</id>")
            if document.short_summary:
                # XML-экранирование заголовка
                title = (
                    str(document.short_summary)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                parts.append(f"  <title>{title}</title>")
            if document.document_category:
                parts.append(f"  <type>{document.document_category.value}</type>")
            if document.status:
                parts.append(f"  <status>{document.status.value}</status>")
            if document.create_date:
                parts.append(f"  <created>{document.create_date.isoformat()}</created>")

            attachments = getattr(document, "attachments", None) or []
            if attachments:
                parts.append("  <attachments>")
                for idx, att in enumerate(attachments, start=1):
                    att_id = str(att.id)
                    att_name = (getattr(att, "file_name", "") or "").replace('"', "&quot;")
                    att_type = getattr(att, "content_type", "") or ""
                    parts.append(
                        f'    <attachment index="{idx}" id="{att_id}"'
                        f' name="{att_name}" type="{att_type}"/>'
                    )
                parts.append("  </attachments>")
                parts.append(f"  <active_file>{str(attachments[0].id)}</active_file>")
            else:
                parts.append(
                    "  <attachments>"
                    "<!-- вызови doc_get_details() для получения списка -->"
                    "</attachments>"
                )
        else:
            if context.document_id:
                parts.append(f"  <id>{context.document_id}</id>")
            parts.append(
                "  <attachments>"
                "<!-- вызови doc_get_details() для получения списка -->"
                "</attachments>"
            )

        if context.file_path:
            is_uuid = bool(_UUID_RE.match(context.file_path))
            if is_uuid:
                parts.append(f"  <attachment_uuid>{context.file_path}</attachment_uuid>")
            else:
                parts.append(f"  <local_file_path>{context.file_path}</local_file_path>")

        parts.append("</document_context>")
        return "\n".join(parts)


class PromptBuilder:
    """Builds system prompts with dynamic context injection.
    """

    _CORE_TEMPLATE = """\
<role>
Ты — экспертный помощник системы электронного документооборота (EDMS/СЭД).
Помогаешь с анализом документов, управлением персоналом и делегированием задач.
</role>

<context>
- Пользователь: {user_name}
- Текущая дата: {current_date}
- Активный документ ID: {document_id}
- Файл / вложение: {file_hint}
</context>

{document_context}

<critical_rules>
1. АВТОИНЪЕКЦИЯ ПАРАМЕТРОВ:
   - `token` передаётся в каждый вызов инструмента АВТОМАТИЧЕСКИ из <injected_auth>
   - `document_id` также добавляется АВТОМАТИЧЕСКИ — НЕ передавай вручную
   - `attachment_id` бери ТОЛЬКО из блока <document_context><attachments> выше

2. ПРАВИЛА РАБОТЫ С ВЛОЖЕНИЯМИ:
   - UUID вложений указаны в <attachments>. Используй их напрямую.
   - Если <attachments/> пустой → вызови doc_get_details() → возьми UUID из результата
   - НЕЛЬЗЯ: придумывать UUID, изменять document_id на +1, использовать document_id как attachment_id
   - НЕЛЬЗЯ: повторно вызывать doc_get_file_content с тем же UUID если уже получил ошибку

3. ПРАВИЛА РАБОТЫ С ФАЙЛАМИ:
   - UUID вложения в <attachment_uuid> → doc_get_file_content(attachment_id=ТОТ_UUID)
   - Локальный путь в <local_file_path> → read_local_file_content(file_path=ТОТ_ПУТЬ)
   - Если вложений нет → честно сообщи пользователю

4. ЯЗЫК И СТИЛЬ:
   - Отвечай ТОЛЬКО на русском языке
   - Обращайся по имени: {user_name}
   - После каждого вызова инструмента формулируй итоговый ответ пользователю

5. ПАМЯТЬ ДИАЛОГА:
   - История доступна — не спрашивай повторно уже упомянутое
   - "этот документ", "его" → document_id из <context> выше
</critical_rules>

<tool_selection_guide>
Задача                               Инструменты
────────────────────────────         ──────────────────────────────────────────────
Узнать о документе                   doc_get_details()
Вложение (UUID в <attachments>)      doc_get_file_content(attachment_id=UUID)
Вложение (UUID неизвестен)           doc_get_details() → выбрать UUID → doc_get_file_content()
Локальный файл (<local_file_path>)   read_local_file_content(file_path=PATH)
Суммаризация                         [чтение файла] → doc_summarize_text(text=..., summary_type=...)
Найти сотрудника                     employee_search_tool(query="ФИО или должность")
Список ознакомления                  introduction_create_tool(employee_names=[...])
Создать поручение                    task_create_tool(executor_last_names=[...], text="...")
</tool_selection_guide>"""

    _INTENT_SNIPPETS: dict[UserIntent, str] = {
        UserIntent.CREATE_INTRODUCTION: """
<introduction_guide>
При requires_disambiguation → покажи список, дождись выбора пользователя.
Повторный вызов: introduction_create_tool(selected_employee_ids=[uuid1, uuid2])
</introduction_guide>""",
        UserIntent.CREATE_TASK: """
<task_guide>
executor_last_names обязателен (минимум 1 фамилия).
planed_date_end: ISO 8601 UTC — "2026-03-01T23:59:59Z".
Если дата не указана → +7 дней от текущей даты.
</task_guide>""",
        UserIntent.SUMMARIZE: """
<summarize_guide>
ОБЯЗАТЕЛЬНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ:
  Шаг 1: Прочитай содержимое файла (doc_get_file_content или read_local_file_content)
  Шаг 2: Вызови doc_summarize_text(text=<текст из шага 1>, summary_type=...)
НЕДОПУСТИМО: отвечать "вот ваш документ" без суммаризации.
</summarize_guide>""",
    }

    @classmethod
    def build(
        cls,
        context: ContextParams,
        intent: UserIntent,
        semantic_xml: str,
        document_context_xml: str,
        injected_token: str,
    ) -> str:
        """Build complete system prompt with all context blocks.

        Args:
            context: Immutable execution context.
            intent: Detected user intent from SemanticDispatcher.
            semantic_xml: XML block from semantic analysis.
            document_context_xml: XML with document metadata & attachment UUIDs.
            injected_token: JWT token for explicit auth block in prompt.

        Returns:
            Full system prompt string ready for SystemMessage.
        """
        file_hint = "Не указан"
        if context.file_path:
            if _UUID_RE.match(context.file_path):
                file_hint = f"UUID вложения: {context.file_path}"
            else:
                file_hint = f"Локальный файл: {context.file_path}"

        base = cls._CORE_TEMPLATE.format(
            user_name=context.user_name,
            current_date=context.current_date,
            document_id=context.document_id or "Не указан",
            file_hint=file_hint,
            document_context=document_context_xml,
        )

        auth_block = (
            "\n<injected_auth>"
            f"\n  <token>{injected_token}</token>"
            f"\n  <document_id>{context.document_id or ''}</document_id>"
            "\n</injected_auth>"
        )

        snippet = cls._INTENT_SNIPPETS.get(intent, "")
        return base + auth_block + snippet + semantic_xml



class ContentExtractor:
    """Extracts final user-facing content from LangGraph message chain.

    Priority order for extract_final_content():
        1. Last AIMessage without tool_calls AFTER last HumanMessage
        2. ToolMessage JSON content from current window
        3. Any AIMessage without tool_calls in full history (fallback)
        4. Long ToolMessage raw text (last resort)
    """

    _SKIP_PATTERNS: tuple[str, ...] = ("вызвал инструмент", "tool call", '"name"', '"id"')
    _MIN_CONTENT_LENGTH: int = 50
    _JSON_FIELDS: tuple[str, ...] = ("content", "text", "text_preview", "message")

    @classmethod
    def extract_final_content(cls, messages: list[BaseMessage]) -> str | None:
        """Extract final user-facing response from message chain.

        Args:
            messages: Full message chain from LangGraph state.

        Returns:
            Final content string or None.
        """
        last_human_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage) and not str(msg.content).startswith(
                "[СИСТЕМНОЕ"
            ):
                last_human_idx = i

        current_window = messages[last_human_idx:] if last_human_idx >= 0 else messages

        for msg in reversed(current_window):
            if isinstance(msg, AIMessage) and msg.content:
                if getattr(msg, "tool_calls", None):
                    continue
                content = str(msg.content).strip()
                if (
                    not cls._is_internal_artifact(content)
                    and len(content) > cls._MIN_CONTENT_LENGTH
                ):
                    return content

        for msg in reversed(current_window):
            if isinstance(msg, ToolMessage):
                extracted = cls._extract_from_tool_message(msg)
                if extracted:
                    return extracted

        for msg in reversed(messages):
            if (
                isinstance(msg, AIMessage)
                and msg.content
                and not getattr(msg, "tool_calls", None)
            ):
                content = str(msg.content).strip()
                if len(content) > cls._MIN_CONTENT_LENGTH:
                    return content

        # 4. Длинный ToolMessage (last resort)
        for msg in reversed(current_window):
            if isinstance(msg, ToolMessage):
                content = str(msg.content).strip()
                if len(content) > cls._MIN_CONTENT_LENGTH:
                    return content

        return None

    @classmethod
    def extract_last_tool_text(cls, messages: list[BaseMessage]) -> str | None:
        """Extract text content from last ToolMessage for summarization input.

        Args:
            messages: Full message chain.

        Returns:
            Extracted text or None.
        """
        for msg in reversed(messages):
            if not isinstance(msg, ToolMessage):
                continue
            try:
                raw = str(msg.content)
                if raw.startswith("{"):
                    data = json.loads(raw)
                    nested = data.get("data", {}) or {}
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
            except json.JSONDecodeError:
                if len(str(msg.content)) > 100:
                    return str(msg.content)
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """Strip JSON status envelope artifacts from content.

        Args:
            content: Raw content string possibly wrapped in JSON envelope.

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
            content.replace('{"status": "success", "content": "', "")
            .replace('"}', "")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .strip()
        )

    @classmethod
    def _is_internal_artifact(cls, content: str) -> bool:
        """Check if content is an internal system artifact, not user-facing.

        Args:
            content: Content string to check.

        Returns:
            True if content should be skipped.
        """
        lower = content.lower()
        return any(skip in lower for skip in cls._SKIP_PATTERNS)

    @classmethod
    def _extract_from_tool_message(cls, message: ToolMessage) -> str | None:
        """Try to extract meaningful text from ToolMessage JSON payload.

        Args:
            message: ToolMessage to parse.

        Returns:
            Extracted text or None.
        """
        try:
            raw = str(message.content).strip()
            if not raw.startswith("{"):
                return None
            data = json.loads(raw)
            nested = data.get("data", {}) or {}
            for json_field in cls._JSON_FIELDS:
                val = nested.get(json_field) or data.get(json_field)
                if val:
                    content = str(val).strip()
                    if len(content) > cls._MIN_CONTENT_LENGTH:
                        return content
        except json.JSONDecodeError:
            pass
        return None



class AgentStateManager:
    """Thin wrapper around CompiledStateGraph for state read and graph invoke.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        checkpointer: MemorySaver,
    ) -> None:
        """Initialize state manager.

        Args:
            graph: Compiled LangGraph StateGraph.
            checkpointer: MemorySaver for in-process persistence.

        Raises:
            ValueError: When graph or checkpointer is None.
        """
        if graph is None:
            raise ValueError("Graph cannot be None")
        if checkpointer is None:
            raise ValueError("Checkpointer cannot be None")
        self.graph = graph
        self.checkpointer = checkpointer

    @staticmethod
    def _make_config(thread_id: str) -> RunnableConfig:
        """Build typed RunnableConfig for LangGraph API calls.

        LangGraph API expects ``RunnableConfig``, not a bare dict.
        Static typing is satisfied, runtime dict-like access preserved.

        Args:
            thread_id: LangGraph session identifier.

        Returns:
            RunnableConfig with configurable.thread_id set.
        """
        return RunnableConfig(configurable={"thread_id": thread_id})

    async def get_state(self, thread_id: str) -> Any:
        """Retrieve current state snapshot for a thread.

        Args:
            thread_id: LangGraph session identifier.

        Returns:
            StateSnapshot with .values and .next attributes.
        """
        return await self.graph.aget_state(self._make_config(thread_id))

    async def invoke(
        self,
        inputs: dict[str, Any] | None,
        thread_id: str,
        timeout: float = 120.0,
    ) -> None:
        """Run graph to completion with timeout protection.

        Args:
            inputs: Graph inputs dict (messages list).
            thread_id: LangGraph session identifier.
            timeout: Execution timeout in seconds.

        Raises:
            asyncio.TimeoutError: When execution exceeds timeout.
        """
        config = self._make_config(thread_id)
        await asyncio.wait_for(
            self.graph.ainvoke(inputs, config=config),
            timeout=timeout,
        )



def _normalize_human_choice(raw: str) -> str:
    """Normalize human_choice to canonical SummaryType value.

    Args:
        raw: Raw string from frontend (e.g. "2", "abstractive", "Краткий пересказ").

    Returns:
        Canonical value: "extractive" | "abstractive" | "thesis".
    """
    return _CHOICE_NORM.get(raw.strip().lower(), raw.strip())


def _build_human_message(
    context: ContextParams,
    refined_query: str,
) -> str:
    """Build HumanMessage content based on context and intent.

    Args:
        context: Immutable execution context.
        refined_query: Refined user query from SemanticDispatcher.

    Returns:
        Human message content string.
    """
    if context.summary_type:
        type_label = _SUMMARY_TYPE_LABELS.get(context.summary_type, context.summary_type)

        if context.file_path and _UUID_RE.match(context.file_path):
            return (
                f"Выполни суммаризацию вложения в формате «{type_label}».\n\n"
                f"Шаг 1: Вызови doc_get_file_content с attachment_id из "
                f"<document_context><attachment_uuid> в системном промпте.\n"
                f"Шаг 2: Вызови doc_summarize_text(text=<текст>, "
                f"summary_type='{context.summary_type}').\n\n"
                f"[ЗАПРОС]: {refined_query}"
            )

        if context.file_path and not _UUID_RE.match(context.file_path):
            from pathlib import Path as _Path

            fname = _Path(context.file_path).name
            return (
                f"Выполни суммаризацию файла «{fname}» в формате «{type_label}».\n\n"
                f"Шаг 1: Вызови read_local_file_content(file_path='{context.file_path}').\n"
                f"Шаг 2: Вызови doc_summarize_text(text=<текст>, "
                f"summary_type='{context.summary_type}').\n\n"
                f"[ЗАПРОС]: {refined_query}"
            )

        if context.document_id:
            return (
                f"Выполни суммаризацию вложения документа в формате «{type_label}».\n\n"
                f"Шаг 1: Возьми attachment_id из <document_context><attachments> "
                f"(или вызови doc_get_details() если список пуст).\n"
                f"Шаг 2: Вызови doc_get_file_content(attachment_id=<UUID из шага 1>).\n"
                f"Шаг 3: Вызови doc_summarize_text(text=<текст>, "
                f"summary_type='{context.summary_type}').\n\n"
                f"[ЗАПРОС]: {refined_query}"
            )

    if context.file_path and not _UUID_RE.match(context.file_path):
        from pathlib import Path as _Path

        fname = _Path(context.file_path).name
        base = refined_query or f"Проанализируй файл «{fname}»"
        return (
            f"{base}\n\n"
            f"[ОБЯЗАТЕЛЬНО]: Используй read_local_file_content("
            f"file_path='{context.file_path}') — НЕ doc_get_file_content."
        )

    return refined_query


def _truncate_text(text: str, limit: int = SUMMARIZE_TEXT_LIMIT) -> str:
    """Truncate long text preserving head + tail structure.

    Args:
        text: Source text to truncate.
        limit: Maximum character count.

    Returns:
        Truncated text with ellipsis marker or original if within limit.
    """
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.3) :]
    return head + "\n\n[...середина пропущена...]\n\n" + tail


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent
# ─────────────────────────────────────────────────────────────────────────────


class EdmsDocumentAgent:
    """Production-ready autonomous multi-agent system for EDMS operations.

    Attributes:
        config: Agent behavior configuration (AgentConfig).
        model: LLM instance.
        tools: List of registered tool instances.
        document_repo: Repository for document metadata fetching.
        dispatcher: SemanticDispatcher for intent analysis.
        state_manager: AgentStateManager wrapping compiled LangGraph graph.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        document_repo: IDocumentRepository | None = None,
        semantic_dispatcher: SemanticDispatcher | None = None,
    ) -> None:
        """Initialize EDMS Document Agent.

        Args:
            config: Agent configuration (creates default if None).
            document_repo: Document repository (creates default if None).
            semantic_dispatcher: Semantic analyzer (creates default if None).

        Raises:
            RuntimeError: When initialization or graph compilation fails.
        """
        try:
            self.config = config or AgentConfig(timeout_seconds=300.0, max_iterations=10)
            self.model = get_chat_model()
            self.document_repo: IDocumentRepository = document_repo or DocumentRepository()
            self.dispatcher = semantic_dispatcher or SemanticDispatcher()

            log.debug("base_components_initialized")

            self.tools = self._create_tools()
            self._checkpointer = MemorySaver()
            self._compiled_graph = self._build_graph()

            if self._compiled_graph is None:
                raise RuntimeError("Graph compilation returned None")

            self.state_manager = AgentStateManager(
                graph=self._compiled_graph,
                checkpointer=self._checkpointer,
            )

            log.info(
                "edms_agent_ready",
                version="v3",
                tools_count=len(self.tools),
                tool_names=[t.name for t in self.tools],
                max_iterations=self.config.max_iterations,
                timeout=self.config.execution_timeout,
                interrupt_before=False,
            )

        except Exception as exc:  # noqa: BLE001
            log.error("edms_agent_init_failed", error=str(exc), exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {exc}") from exc

    # ── Initialization helpers ────────────────────────────────────────────────

    def _create_tools(self) -> list[Any]:
        """Create all EDMS tools via DI factory.

        Returns:
            List of instantiated tool objects.
        """
        from ...domain.services.appeal_validator import AppealValidator
        from ...domain.services.document_comparer import DocumentComparer
        from ...domain.services.task_assigner import TaskAssigner
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
        )
        log.info("tools_initialized", count=len(tools), names=[t.name for t in tools])
        return tools

    @staticmethod
    def _try_create_nlp_extractor() -> Any | None:
        """Try to create NLP extractor. Graceful degradation on failure.

        Returns:
            AppealExtractor instance or None.
        """
        try:
            from ...infrastructure.nlp.extractors.appeal_extractor import AppealExtractor

            return AppealExtractor()
        except Exception as exc:  # noqa: BLE001
            log.warning("nlp_extractor_unavailable", error=str(exc))
            return None

    def health_check(self) -> dict[str, Any]:
        """Return component health status dictionary.

        Returns:
            Dict with boolean/count flags per component.
        """
        return {
            "version": "v3",
            "model": self.model is not None,
            "tools_count": len(self.tools),
            "tool_names": [t.name for t in self.tools],
            "document_repo": self.document_repo is not None,
            "dispatcher": self.dispatcher is not None,
            "graph": (
                hasattr(self, "_compiled_graph") and self._compiled_graph is not None
            ),
            "state_manager": (
                hasattr(self, "state_manager") and self.state_manager is not None
            ),
            "max_iterations": self.config.max_iterations,
            "execution_timeout": self.config.execution_timeout,
            "interrupt_before": False,
        }

    # ── Graph Construction ────────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        """Build and compile autonomous ReAct LangGraph workflow.

        Returns:
            CompiledStateGraph (no interrupt_before).

        Raises:
            RuntimeError: When graph compilation fails.
        """
        workflow: Any = StateGraph(AgentStateWithCounter)

        # ── Node: agent (LLM call) ────────────────────────────────────────────

        async def call_model(state: AgentStateWithCounter) -> dict[str, Any]:
            """Invoke LLM with bound tools.

            Applies sliding window trimming to prevent context bloat.
            Increments graph_iterations via operator.add reducer.

            Args:
                state: Current graph state.

            Returns:
                Dict with updated messages list and graph_iterations increment.
            """
            model_with_tools = self.model.bind_tools(self.tools)

            all_msgs: list[BaseMessage] = state["messages"]
            sys_msgs = [m for m in all_msgs if isinstance(m, SystemMessage)]
            non_sys = [m for m in all_msgs if not isinstance(m, SystemMessage)]

            if len(non_sys) > MAX_CONTEXT_MESSAGES:
                window = non_sys[-MAX_CONTEXT_MESSAGES:]
                if window and not isinstance(window[0], HumanMessage):
                    for earlier in reversed(non_sys[: -MAX_CONTEXT_MESSAGES]):
                        if isinstance(earlier, HumanMessage):
                            window = [earlier, *window]
                            break
                non_sys = window

            final_messages: list[BaseMessage] = (
                [sys_msgs[-1]] if sys_msgs else []
            ) + non_sys

            response: BaseMessage = await model_with_tools.ainvoke(final_messages)
            return {"messages": [response], "graph_iterations": 1}

        # ── Node: validator ───────────────────────────────────────────────────

        async def validator(state: AgentStateWithCounter) -> dict[str, Any]:
            """Validate tool execution result.

            Injects system notification on empty/error tool result.
            Agent then generates user-friendly error explanation.

            Args:
                state: Current graph state after ToolNode execution.

            Returns:
                Dict with optional HumanMessage notification.
            """
            messages: list[BaseMessage] = state["messages"]
            last_msg = messages[-1]

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

            content_lower = content.lower()
            if "error" in content_lower and "exception" in content_lower:
                return {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Техническая ошибка инструмента: "
                                f"{content[:500]}"
                            )
                        )
                    ]
                }

            return {"messages": []}

        # ── Conditional Edges ─────────────────────────────────────────────────

        def should_continue(state: AgentStateWithCounter) -> str:
            """Route from agent node: to tools or END.

            AgentStateWithCounter MUST be at module level — LangGraph resolves
            this type hint from globalns of the edms_agent module.

            Args:
                state: Current state with messages list.

            Returns:
                "tools" if last AIMessage has tool_calls, else END.
            """
            messages: list[BaseMessage] = state["messages"]
            if not messages:
                return END
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                return "tools"
            return END

        def after_validator(state: AgentStateWithCounter) -> str:
            """Route from validator: continue agent or stop (infinite loop guard).

            graph_iterations accumulates via operator.add reducer within this
            invocation. Counter is correct here because each chat() call passes
            fresh inputs — LangGraph appends, but iterations from prior turns
            also accumulate. The application-level guard in _run_graph handles
            cross-turn protection if needed.

            Args:
                state: Current state with graph_iterations counter.

            Returns:
                "agent" to continue, END when max iterations reached.
            """
            iterations: int = state.get("graph_iterations", 0)  # type: ignore[call-overload]
            if iterations >= MAX_GRAPH_ITERATIONS:
                log.warning(
                    "graph_max_iterations_reached",
                    iterations=iterations,
                    max=MAX_GRAPH_ITERATIONS,
                )
                return END
            return "agent"

        # ── Wiring ────────────────────────────────────────────────────────────

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("validator", validator)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "validator")
        workflow.add_conditional_edges(
            "validator",
            after_validator,
            {"agent": "agent", END: END},
        )

        try:
            compiled = workflow.compile(checkpointer=self._checkpointer)
            log.debug("graph_compiled_v3_autonomous")
            return compiled
        except Exception as exc:  # noqa: BLE001
            log.error("graph_compilation_failed", error=str(exc), exc_info=True)
            raise RuntimeError(f"Failed to compile graph: {exc}") from exc

    # ── Public API ────────────────────────────────────────────────────────────

    async def chat(self, request: AgentRequest) -> dict[str, Any]:
        """Process user message through the full agent pipeline.

        Args:
            request: Validated AgentRequest DTO.

        Returns:
            AgentResponse dict (status + content/message).
        """
        try:
            context = await self._build_context(request)

            document: Document | None = None
            if context.document_id:
                document = await self.document_repo.get_document(
                    context.user_token, context.document_id
                )

            doc_context_xml = DocumentContextService.build_context_block(
                document=document,
                context=context,
            )

            semantic_ctx = self.dispatcher.build_context(request.message, document)
            intent = semantic_ctx.query.intent

            log.info(
                "semantic_analysis_complete",
                intent=intent.value,
                complexity=semantic_ctx.query.complexity.value,
                thread_id=context.thread_id,
            )

            semantic_xml = self._build_semantic_xml(semantic_ctx)

            system_prompt = PromptBuilder.build(
                context=context,
                intent=intent,
                semantic_xml=semantic_xml,
                document_context_xml=doc_context_xml,
                injected_token=context.user_token,
            )

            human_content = _build_human_message(
                context=context,
                refined_query=semantic_ctx.query.refined,
            )

            inputs: dict[str, Any] = {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_content),
                ]
            }

            return await self._run_graph(context=context, inputs=inputs)

        except ValueError as exc:
            log.error("chat_validation_error", error=str(exc))
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка валидации запроса: {exc}",
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            log.error("chat_error", error=str(exc), exc_info=True)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {exc}",
            ).model_dump()

    # ── Graph Execution ───────────────────────────────────────────────────────

    async def _run_graph(
        self,
        context: ContextParams,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute LangGraph autonomously and extract final response.

        Args:
            context: Immutable execution context.
            inputs: Graph inputs with SystemMessage + HumanMessage.

        Returns:
            AgentResponse dict.
        """
        try:
            await self.state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=self.config.execution_timeout,
            )

            state = await self.state_manager.get_state(context.thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])

            log.debug(
                "graph_execution_complete",
                thread_id=context.thread_id,
                messages_count=len(messages),
                last_type=type(messages[-1]).__name__ if messages else "none",
            )

            if not messages:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Пустое состояние агента.",
                ).model_dump()

            last_msg = messages[-1]

            if isinstance(last_msg, ToolMessage):
                synthesis = await self._synthesize_from_tool_message(last_msg)
                if synthesis:
                    return AgentResponse(
                        status=AgentStatus.SUCCESS,
                        content=synthesis,
                    ).model_dump()

            final = ContentExtractor.extract_final_content(messages)
            if final:
                final = ContentExtractor.clean_json_artifacts(final)

                if context.summary_type and len(final) > 3000:
                    summarized = await self._direct_summarize(
                        text=final,
                        summary_type=context.summary_type,
                        thread_id=context.thread_id,
                    )
                    if summarized:
                        log.info(
                            "execution_completed_with_direct_summarization",
                            thread_id=context.thread_id,
                            summary_type=context.summary_type,
                            input_chars=len(final),
                            output_chars=len(summarized),
                        )
                        return AgentResponse(
                            status=AgentStatus.SUCCESS,
                            content=summarized,
                        ).model_dump()

                log.info(
                    "execution_completed",
                    thread_id=context.thread_id,
                    chars=len(final),
                )
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    content=final,
                ).model_dump()

            return AgentResponse(
                status=AgentStatus.SUCCESS,
                content="Анализ завершён.",
            ).model_dump()

        except asyncio.TimeoutError:
            log.error("execution_timeout", thread_id=context.thread_id)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания выполнения.",
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            log.error("graph_execution_error", error=str(exc), exc_info=True)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка выполнения графа: {exc}",
            ).model_dump()

    async def _synthesize_from_tool_message(
        self,
        tool_msg: ToolMessage,
    ) -> str | None:
        """Synthesize final answer when graph ended on a ToolMessage.

        Called when MAX_GRAPH_ITERATIONS reached and graph terminated without
        a final AIMessage. Without this, ContentExtractor returns stale content.

        Args:
            tool_msg: Last ToolMessage in the message chain.

        Returns:
            Synthesized answer string or None.
        """
        tool_content = str(tool_msg.content).strip()
        if len(tool_content) < 50:
            return None

        try:
            model_plain = self.model.bind_tools([])
            resp = await asyncio.wait_for(
                model_plain.ainvoke([
                    SystemMessage(
                        content=(
                            "На основе результата инструмента дай краткий финальный "
                            "ответ пользователю на русском языке. "
                            "Не повторяй технические детали."
                        )
                    ),
                    HumanMessage(
                        content=f"Результат инструмента:\n{tool_content[:6000]}"
                    ),
                ]),
                timeout=60.0,
            )
            content = str(resp.content).strip()
            if content and len(content) > 20:
                log.info("synthesis_from_tool_message_ok", chars=len(content))
                return content
        except asyncio.TimeoutError:
            log.warning("synthesis_timeout")
        except Exception as exc:  # noqa: BLE001
            log.warning("synthesis_failed", error=str(exc))
        return None

    async def _direct_summarize(
        self,
        text: str,
        summary_type: str,
        thread_id: str,
    ) -> str | None:
        """Directly summarize text via LLM bypass when agent skipped doc_summarize_text.

        Edge case: LLM after reading 18K+ chars responds with raw content instead
        of calling doc_summarize_text. This method forces the summarization.

        Args:
            text: Raw text to summarize.
            summary_type: Canonical SummaryType string.
            thread_id: Thread ID for logging context.

        Returns:
            Summarized text string or None.
        """
        _instructions: dict[str, str] = {
            "extractive": (
                "Выдели ключевые факты, даты, суммы и конкретные обязательства. "
                "Оформи списком с краткими пояснениями."
            ),
            "abstractive": (
                "Напиши связный краткий пересказ сути документа своими словами "
                "(1-2 абзаца). Сохрани ключевую информацию, перефразируй."
            ),
            "thesis": (
                "Сформируй структурированный тезисный план с выделением главных "
                "мыслей. Используй нумерацию."
            ),
        }
        instruction = _instructions.get(summary_type, "Кратко изложи суть.")
        truncated = _truncate_text(text, SUMMARIZE_TEXT_LIMIT)

        try:
            model_plain = self.model.bind_tools([])
            resp = await asyncio.wait_for(
                model_plain.ainvoke([
                    SystemMessage(
                        content=(
                            f"Ты — ведущий аналитик СЭД. Задача: {instruction} "
                            f"Пиши строго по делу, на русском языке."
                        )
                    ),
                    HumanMessage(
                        content=f"ИСХОДНЫЙ ТЕКСТ:\n{truncated}\n\nРЕЗУЛЬТАТ:"
                    ),
                ]),
                timeout=120.0,
            )
            content = str(resp.content).strip()
            if content and len(content) > 100:
                return content
        except asyncio.TimeoutError:
            log.warning("direct_summarization_timeout", thread_id=thread_id)
        except Exception as exc:
            log.warning(
                "direct_summarization_failed", error=str(exc), thread_id=thread_id
            )
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    async def _build_context(request: AgentRequest) -> ContextParams:
        """Build immutable ContextParams from AgentRequest.

        Safely extracts user_context fields: handles None, dict, Pydantic model.

        Args:
            request: Validated AgentRequest DTO.

        Returns:
            Immutable ContextParams.
        """
        ctx: dict[str, Any] = request.user_context or {}
        if hasattr(ctx, "model_dump"):
            ctx = ctx.model_dump(exclude_none=True)

        user_name = str(
            ctx.get("firstName")
            or ctx.get("first_name")
            or ctx.get("name")
            or "пользователь"
        ).strip()

        return ContextParams(
            user_token=request.user_token,
            thread_id=request.thread_id or "default",
            document_id=request.context_ui_id,
            file_path=request.file_path,
            summary_type=(
                _normalize_human_choice(request.human_choice)
                if request.human_choice
                else None
            ),
            user_name=user_name,
            user_first_name=str(
                ctx.get("firstName") or ctx.get("first_name") or ""
            )
            or None,
        )

    @staticmethod
    def _build_semantic_xml(semantic_ctx: Any) -> str:
        """Build XML block from SemanticContext for prompt injection.

        Args:
            semantic_ctx: SemanticContext returned by SemanticDispatcher.

        Returns:
            XML string to append to system prompt.
        """
        return (
            "\n<semantic_context>"
            f"\n  <intent>{semantic_ctx.query.intent.value}</intent>"
            f"\n  <original>{semantic_ctx.query.original}</original>"
            f"\n  <refined>{semantic_ctx.query.refined}</refined>"
            f"\n  <complexity>{semantic_ctx.query.complexity.value}</complexity>"
            "\n</semantic_context>"
        )