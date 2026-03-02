# src/ai_edms_assistant/application/agents/edms_agent.py
"""
EDMS Document Agent — основной оркестратор мультиагентной системы.
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

from ...domain.entities.document import Document
from ...infrastructure.edms_api.http_client import EdmsHttpClient
from ...infrastructure.llm.providers.openai_provider import get_chat_model
from ..dto.agent import AgentRequest, AgentResponse, AgentStatus
from ..services.semantic_dispatcher import SemanticDispatcher, UserIntent
from ..tools import create_all_tools

from .agent_config import AgentConfig
from .agent_state import AgentState, AgentStateWithCounter

log = structlog.get_logger(__name__)
logger = logging.getLogger(__name__)

MAX_GRAPH_ITERATIONS: int = 8


# ─────────────────────────────────────────────────────────────────────────────
# Domain Value Objects
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContextParams:
    """Immutable execution context for one agent invocation.

    Built once per chat() call from AgentRequest and passed through
    the entire orchestration pipeline without mutation.

    Attributes:
        user_token: JWT bearer token for all EDMS API calls.
        document_id: UUID of the active document in UI context (optional).
        file_path: Local temp path or EDMS attachment UUID from frontend.
        thread_id: LangGraph checkpointer session identifier.
        user_name: Display name for prompt personalization.
        user_first_name: First name extracted from employee API.
        current_date: Date string injected into system prompt.
    """

    user_token: str
    document_id: str | None = None
    file_path: str | None = None
    thread_id: str = "default"
    user_name: str = "пользователь"
    user_first_name: str | None = None
    current_date: str = field(
        default_factory=lambda: datetime.now().strftime("%d.%m.%Y")
    )

    def __post_init__(self) -> None:
        """Validate required fields on construction.

        Raises:
            ValueError: When user_token is empty or not a string.
        """
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")


# ─────────────────────────────────────────────────────────────────────────────
# Repository (Dependency Inversion)
# ─────────────────────────────────────────────────────────────────────────────


class IDocumentRepository(Protocol):
    """Port: contract for document metadata access."""

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        """Fetch document metadata.

        Args:
            token: JWT bearer token.
            doc_id: Document UUID string.

        Returns:
            Document entity or None.
        """
        ...


class DocumentRepository:
    """Default adapter: fetches documents via EdmsDocumentClient (EDMS REST API).

    Used when no custom repository injected at construction.
    Failures are swallowed — a missing document is NOT a fatal error;
    the agent continues without document context.
    """

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        """Fetch document metadata. Graceful on error.

        Args:
            token: JWT bearer token.
            doc_id: Document UUID string.

        Returns:
            Document entity or None on any error.
        """
        try:
            from uuid import UUID
            from ...infrastructure.edms_api.repositories.edms_document_repository import (
                EdmsDocumentRepository,
            )
            repo = EdmsDocumentRepository(http_client=EdmsHttpClient())
            return await repo.get_by_id(entity_id=UUID(doc_id), token=token)
        except Exception as exc:
            log.warning("document_fetch_failed", doc_id=doc_id, error=str(exc))
            return None


# ─────────────────────────────────────────────────────────────────────────────
# NLP Helper — автоматический выбор типа суммаризации
# ─────────────────────────────────────────────────────────────────────────────


class NLPHelperService:
    """Recommends summarization format based on text characteristics.

    Replaces old EDMSNaturalLanguageService.suggest_summarize_format()
    without the external service dependency.

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
            Dict with keys:
                recommended (str): 'thesis' | 'extractive' | 'abstractive'
                reason (str): Human-readable explanation in Russian.
                stats (dict): {'chars': int}
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
                "reason": f"Объёмный текст ({chars} симв.) или много чисел ({digit_groups}) — тезисы оптимальны",
                "stats": {"chars": chars},
            }
        elif chars > 2000:
            return {
                "recommended": "extractive",
                "reason": f"Средний объём ({chars} симв.) — ключевые факты предпочтительны",
                "stats": {"chars": chars},
            }
        else:
            return {
                "recommended": "abstractive",
                "reason": f"Краткий текст ({chars} симв.) — пересказ своими словами",
                "stats": {"chars": chars},
            }


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builder (Strategy Pattern)
# ─────────────────────────────────────────────────────────────────────────────


class PromptBuilder:
    """Builds system prompts with dynamic context injection.

    Uses Strategy pattern: base template + intent-specific snippets + semantic XML.
    """

    CORE_TEMPLATE = """<role>
Ты — экспертный помощник системы электронного документооборота (EDMS/СЭД).
Помогаешь с анализом документов, управлением персоналом и делегированием задач.
</role>

<context>
- Пользователь: {user_name}
- Текущая дата: {current_date}
- Активный документ: {context_ui_id}
- Загруженный файл: {local_file}
</context>

<critical_rules>
1. **Автоинъекция**: `token` и `document_id` добавляются АВТОМАТИЧЕСКИ — не передавай их явно.

2. **Обработка LOCAL_FILE**:
   - UUID (xxxxxxxx-xxxx-...) → `doc_get_file_content(attachment_id=LOCAL_FILE)`
   - Путь /tmp/... или C:\\... → `read_local_file_content(file_path=LOCAL_FILE)`
   - "Не загружен" → `doc_get_details()` для получения списка вложений

3. **После requires_action**:
   - "summarize_selection" → Предложи формат (факты/пересказ/тезисы), дождись выбора
   - "requires_disambiguation" → Покажи список кандидатов, дождись выбора пользователя

4. После каждого вызова инструмента ОБЯЗАТЕЛЬНО формулируй финальный ответ на русском языке.
5. Язык ответов: только русский. Обращайся по имени: {user_name}
</critical_rules>

<tool_selection>
- Анализ документа:   doc_get_details → doc_get_file_content → doc_summarize_text
- Файл (UUID):        doc_get_file_content(attachment_id=UUID) → doc_summarize_text
- Файл (путь):        read_local_file_content(file_path=PATH) → doc_summarize_text
- Сотрудник:          employee_search_tool
- Ознакомление:       introduction_create_tool
- Поручение:          task_create_tool
</tool_selection>

<attachment_rules>
КРИТИЧНО: attachment_id — это UUID вложения файла, НЕ UUID документа.
Они РАЗНЫЕ: document_id="{context_ui_id}", attachment_id — другое значение.

Если пользователь просит "первое", "второе", "последнее" вложение или называет имя файла:
  Шаг 1: ОБЯЗАТЕЛЬНО вызови doc_get_details() — получи список вложений с их UUID.
  Шаг 2: Выбери нужное вложение из списка по порядку или по имени файла.
  Шаг 3: Вызови doc_get_file_content(attachment_id=<UUID из шага 2>).
  НИКОГДА не придумывай attachment_id самостоятельно — это вызовет ошибку.
</attachment_rules>"""

    CONTEXT_SNIPPETS: dict[UserIntent, str] = {
        UserIntent.CREATE_INTRODUCTION: """
<introduction_guide>
При "requires_disambiguation" → покажи список найденных сотрудников, дождись выбора.
Повторный вызов: introduction_create_tool(selected_employee_ids=[uuid1, uuid2])
</introduction_guide>""",

        UserIntent.CREATE_TASK: """
<task_guide>
executor_last_names: обязателен (минимум 1 фамилия).
planed_date_end: формат ISO 8601 UTC — "2026-03-01T23:59:59Z".
Если дата не указана → +7 дней от текущей.
</task_guide>""",

        UserIntent.SUMMARIZE: """
<date_parsing>
Преобразование дат в ISO 8601:
- "до 15 февраля" → "2026-02-15T23:59:59Z"
- "через неделю"  → +7 дней от текущей даты с суффиксом Z (UTC).
</date_parsing>""",
    }

    @classmethod
    def build(
        cls,
        context: ContextParams,
        intent: UserIntent,
        semantic_xml: str,
    ) -> str:
        """Build complete system prompt.

        Args:
            context: Immutable execution context.
            intent: Detected user intent from SemanticDispatcher.
            semantic_xml: XML block from semantic analysis.

        Returns:
            Full system prompt string.
        """
        base = cls.CORE_TEMPLATE.format(
            user_name=context.user_name,
            current_date=context.current_date,
            context_ui_id=context.document_id or "Не указан",
            local_file=context.file_path or "Не загружен",
        )
        snippet = cls.CONTEXT_SNIPPETS.get(intent, "")
        return base + snippet + semantic_xml


# ─────────────────────────────────────────────────────────────────────────────
# Content Extractor
# ─────────────────────────────────────────────────────────────────────────────


class ContentExtractor:
    """Extracts final user-facing content from LangGraph message chain.
    """

    SKIP_PATTERNS: list[str] = ["вызвал инструмент", "tool call", '"name"', '"id"']
    MIN_CONTENT_LENGTH: int = 50
    JSON_FIELDS: list[str] = ["content", "text", "text_preview", "message"]

    @classmethod
    def extract_final_content(cls, messages: list[BaseMessage]) -> str | None:
        """Extract final user-facing response from message chain.

        Args:
            messages: Full message chain from LangGraph state.

        Returns:
            Final content string or None.
        """
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                if getattr(m, "tool_calls", None):
                    continue
                content = str(m.content).strip()
                if not cls._is_skip_content(content) and len(content) > cls.MIN_CONTENT_LENGTH:
                    return content

        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._extract_from_tool_message(m)
                if extracted:
                    return extracted

        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                return str(m.content).strip()

        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                content = str(m.content).strip()
                if len(content) > cls.MIN_CONTENT_LENGTH:
                    return content

        return None

    @classmethod
    def extract_last_text(cls, messages: list[BaseMessage]) -> str | None:
        """Extract text content from last ToolMessage for summarization input.

        Looks in data.content, data.text_preview, nested data dict, then raw.

        Args:
            messages: Full message chain.

        Returns:
            Extracted text or None.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                raw = str(m.content)
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
                if len(str(m.content)) > 100:
                    return str(m.content)
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """Strip JSON status envelope artifacts from LLM-generated content.

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
            content
            .replace('{"status": "success", "content": "', "")
            .replace('"}', "")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .strip()
        )

    @classmethod
    def _is_skip_content(cls, content: str) -> bool:
        """Check if content is an internal system artifact, not user-facing.

        Args:
            content: Content string to check.

        Returns:
            True if content should be skipped.
        """
        return any(skip in content.lower() for skip in cls.SKIP_PATTERNS)

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
            for f in cls.JSON_FIELDS:
                val = nested.get(f) or data.get(f)
                if val:
                    content = str(val).strip()
                    if len(content) > cls.MIN_CONTENT_LENGTH:
                        return content
        except json.JSONDecodeError:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Agent State Manager
# ─────────────────────────────────────────────────────────────────────────────


class AgentStateManager:
    """Thin wrapper around CompiledStateGraph for state operations.

    Provides get / update / invoke methods used by EdmsDocumentAgent._orchestrate().
    """

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver) -> None:
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

    async def get_state(self, thread_id: str) -> Any:
        """Retrieve current state snapshot for a thread.

        Args:
            thread_id: LangGraph session identifier.

        Returns:
            StateSnapshot with .values and .next.
        """
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aget_state(config)

    async def update_state(
        self,
        thread_id: str,
        messages: list[BaseMessage],
        as_node: str = "agent",
    ) -> None:
        """Inject messages into state (human-in-the-loop parameter injection).

        Args:
            thread_id: LangGraph session identifier.
            messages: Messages to inject into state.
            as_node: Node context for the update.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await self.graph.aupdate_state(config, {"messages": messages}, as_node=as_node)

    async def invoke(
        self,
        inputs: dict[str, Any] | None,
        thread_id: str,
        timeout: float = 120.0,
    ) -> None:
        """Run graph with timeout protection.

        Args:
            inputs: Graph inputs (None to resume from interrupt_before["tools"]).
            thread_id: LangGraph session identifier.
            timeout: Execution timeout in seconds.

        Raises:
            asyncio.TimeoutError: When execution exceeds timeout.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await asyncio.wait_for(
            self.graph.ainvoke(inputs, config=config),
            timeout=timeout,
        )


_UUID_PAT = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_SUMMARY_TYPE_LABELS: dict[str, str] = {
    "extractive": "ключевые факты",
    "abstractive": "краткий пересказ",
    "thesis": "тезисный план",
}

_CHOICE_NORM: dict[str, str] = {
    "1": "extractive", "факты": "extractive", "ключевые факты": "extractive",
    "extractive": "extractive",
    "2": "abstractive", "пересказ": "abstractive", "краткий пересказ": "abstractive",
    "abstractive": "abstractive",
    "3": "thesis", "тезисы": "thesis", "тезисный план": "thesis",
    "thesis": "thesis",
}


def _normalize_human_choice(raw: str) -> str:
    """Normalize human_choice to canonical SummaryType value.

    Args:
        raw: Raw string from frontend (e.g. "2", "abstractive", "Краткий пересказ").

    Returns:
        Canonical value: "extractive" | "abstractive" | "thesis".
    """
    return _CHOICE_NORM.get(raw.strip().lower(), raw.strip())


def _build_forced_pipeline_message(
    file_path: str | None,
    summary_type: str,
    document_id: str | None,
    original: str | None,
) -> str:
    """Build mandatory two-stage pipeline instruction for LLM.


    Args:
        file_path: Local path or EDMS attachment UUID (may be None).
        summary_type: Canonical SummaryType string.
        document_id: Active document UUID (fallback if no file_path).
        original: Original user message text for context.

    Returns:
        Strict two-stage pipeline instruction string for LLM HumanMessage.
    """
    type_label = _SUMMARY_TYPE_LABELS.get(summary_type, summary_type)

    if file_path and _UUID_PAT.match(str(file_path)):
        stage1 = (
            f"Шаг 1: Вызови doc_get_file_content(attachment_id=\'{file_path}\') "
            f"для получения текста вложения."
        )
        source = f"вложение EDMS ({file_path[:8]}...)"
    elif file_path:
        from pathlib import Path as _Path
        fname = _Path(file_path).name
        stage1 = (
            f"Шаг 1: Вызови read_local_file_content(file_path=\'{file_path}\') "
            f"для чтения файла «{fname}». "
            f"НЕ используй doc_get_file_content."
        )
        source = f"локальный файл «{fname}»"
    elif document_id:
        import re as _re
        _orig = str(original or "").strip()

        _fname_match = _re.search(
            r"(?:вложение|файл|Анализ файла).*?«(.+?)»|"
            r"(?:Анализ файла|файл):\s*(.+?)(?:\s*$)",
            _orig,
            _re.IGNORECASE,
        )
        target_name: str | None = None
        if _fname_match:
            target_name = (_fname_match.group(1) or _fname_match.group(2) or "").strip() or None

        if not target_name:
            _bare = _re.match(
                r"^([^/\\<>:\"|?*\r\n]+\.(?:docx?|pdf|txt|rtf|html?|xlsx?|pptx?))$",
                _orig,
                _re.IGNORECASE,
            )
            if _bare:
                target_name = _bare.group(1).strip()

        if target_name:
            stage1 = (
                f"Шаг 1: Вызови doc_get_details() чтобы получить список вложений. "
                f"Затем вызови doc_get_file_content с attachment_id вложения "
                f"с именем «{target_name}» (ищи совпадение по имени файла). "
                f"НЕ бери первое вложение автоматически — найди именно «{target_name}»."
            )
            source = f"вложение «{target_name}»"
        else:
            stage1 = (
                f"Шаг 1: Вызови doc_get_file_content(document_id=\'{document_id}\') "
                f"для получения текста первого вложения."
            )
            source = "первое вложение документа"
    else:
        return original or "Файл для анализа не указан."

    stage2 = (
        f"Шаг 2: Вызови doc_summarize_text(text=<текст из шага 1>, "
        f"summary_type=\'{summary_type}\') — формат «{type_label}»."
    )
    hint = f"\n\nЗапрос: {original}" if original else ""

    return (
        f"Выполни анализ {source} в формате «{type_label}».\n\n"
        f"{stage1}\n{stage2}{hint}\n\n"
        f"[ОБЯЗАТЕЛЬНО]: Выполни оба шага. "
        f"НЕ вызывай doc_get_details — нужно содержимое файла, не метаданные."
    )


class EdmsDocumentAgent:
    """Production-ready multi-agent system for EDMS document operations.

    Attributes:
        config: Agent behavior configuration (AgentConfig).
        model: LLM with bound tools.
        tools: List of registered AbstractEdmsTool instances.
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

        Compared to old agent: takes AgentConfig instead of class constants,
        uses _create_tools() DI factory instead of global all_tools import.

        Args:
            config: Agent configuration (defaults if None).
            document_repo: Document repository (creates default if None).
            semantic_dispatcher: Semantic analyzer (creates default if None).

        Raises:
            RuntimeError: When initialization or graph compilation fails.
        """
        try:
            self.config = config or AgentConfig(timeout_seconds=300.0)
            self.model = get_chat_model()
            self.document_repo = document_repo or DocumentRepository()
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
                tools_count=len(self.tools),
                tool_names=[t.name for t in self.tools],
                max_iterations=self.config.max_iterations,
                timeout=self.config.execution_timeout,
            )

        except Exception as exc:
            log.error("edms_agent_init_failed", error=str(exc), exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {exc}") from exc

    # ── Initialization Helpers ────────────────────────────────────────────────

    def _create_tools(self) -> list:
        """Create all tools via DI factory.

        No MinIO/storage required — AttachmentTool downloads directly via
        EDMS REST API (EdmsAttachmentClient.get_content()).

        Returns:
            List of instantiated AbstractEdmsTool objects (8-9 tools).
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
            # storage=None → AttachmentTool использует EdmsAttachmentClient напрямую
        )

        log.info("tools_initialized", count=len(tools), names=[t.name for t in tools])
        return tools

    def _try_create_nlp_extractor(self) -> Any | None:
        """Try to create NLP extractor. Returns None on failure (graceful degradation).

        Returns:
            AppealExtractor instance or None.
        """
        try:
            from ...infrastructure.nlp.extractors.appeal_extractor import AppealExtractor
            return AppealExtractor()
        except Exception as exc:
            log.warning("nlp_extractor_unavailable", error=str(exc))
            return None

    def health_check(self) -> dict[str, Any]:
        """Return component health status.

        Returns:
            Dict with boolean/count flags per component.
        """
        return {
            "model": self.model is not None,
            "tools_count": len(self.tools),
            "tool_names": [t.name for t in self.tools],
            "document_repo": self.document_repo is not None,
            "dispatcher": self.dispatcher is not None,
            "graph": hasattr(self, "_compiled_graph") and self._compiled_graph is not None,
            "state_manager": hasattr(self, "state_manager") and self.state_manager is not None,
            "max_iterations": self.config.max_iterations,
            "execution_timeout": self.config.execution_timeout,
        }

    # ── Graph Construction ────────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        """Build and compile LangGraph workflow.

        Returns:
            CompiledStateGraph with interrupt_before=["tools"].

        Raises:
            RuntimeError: When graph compilation fails.
        """
        workflow = StateGraph(AgentStateWithCounter)

        # ── Node: agent (LLM call) ────────────────────────────────────────────

        async def call_model(state: AgentStateWithCounter) -> dict:
            """Invoke LLM with bound tools.

            Increments graph_iterations by 1 via operator.add reducer.

            Args:
                state: Current graph state.

            Returns:
                Dict with updated messages and incremented graph_iterations.
            """
            model_with_tools = self.model.bind_tools(self.tools)

            sys_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
            non_sys = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
            final_messages = ([sys_msgs[-1]] if sys_msgs else []) + non_sys

            response = await model_with_tools.ainvoke(final_messages)
            return {"messages": [response], "graph_iterations": 1}

        # ── Node: validator (tool result check) ──────────────────────────────

        async def validator(state: AgentStateWithCounter) -> dict:
            """Validate tool execution result.

            Injects error notification into state when tool returned
            empty result or raised an exception. Agent then generates
            user-friendly error message.

            Args:
                state: Current graph state after tool execution.

            Returns:
                Dict with optional error notification message.
            """
            messages = state["messages"]
            last_msg = messages[-1]

            if not isinstance(last_msg, ToolMessage):
                return {"messages": []}

            content = str(last_msg.content).strip()

            if not content or content in ("None", "{}"):
                return {
                    "messages": [
                        HumanMessage(
                            content="[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Инструмент вернул пустой результат. "
                                    "Сообщи пользователю понятным языком."
                        )
                    ]
                }

            if "error" in content.lower() and "exception" in content.lower():
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Техническая ошибка инструмента: "
                                    f"{content[:500]}"
                        )
                    ]
                }

            return {"messages": []}

        # ── Conditional Edges ─────────────────────────────────────────────────

        def should_continue(state: AgentStateWithCounter) -> str:
            """Route from agent: tools or END.

            NOTE: AgentStateWithCounter MUST be at module level.
            LangGraph resolves this type hint from module globalns.

            Args:
                state: Current state with messages.

            Returns:
                "tools" if last AIMessage has tool_calls, else END.
            """
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                return "tools"
            return END

        def after_validator(state: AgentStateWithCounter) -> str:
            """Route from validator: continue agent or stop (infinite loop guard).

            Uses graph_iterations counter from AgentStateWithCounter state.
            MAX_GRAPH_ITERATIONS is a module-level constant — visible here.

            Args:
                state: Current state with graph_iterations counter.

            Returns:
                "agent" to continue, or END when max iterations reached.
            """
            iterations = state.get("graph_iterations", 0)
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
            "agent", should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "validator")
        workflow.add_conditional_edges(
            "validator", after_validator, {"agent": "agent", END: END}
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
            raise RuntimeError(f"Failed to compile graph: {exc}") from exc

    # ── Public API ────────────────────────────────────────────────────────────

    async def chat(self, request: AgentRequest) -> dict:
        """Process user message through the full agent pipeline.

        Difference from old agent:
            Old: chat(message, user_token, ...) — positional/keyword args
            New: chat(request: AgentRequest)    — single validated DTO

        Pipeline:
            1. Build immutable ContextParams from request
            2. Check for human-in-the-loop continuation (human_choice)
            3. Fetch document metadata for semantic analysis
            4. SemanticDispatcher → intent + refined query
            5. Build system prompt (PromptBuilder)
            6. _orchestrate() → LangGraph execution loop

        Args:
            request: Validated AgentRequest DTO.

        Returns:
            AgentResponse dict (status + content/message + optional action_type).
        """
        try:
            context = await self._build_context(request)
            state = await self.state_manager.get_state(context.thread_id)

            if request.human_choice and state.next:
                return await self._handle_human_choice(context, request.human_choice)

            document = None
            if context.document_id:
                document = await self.document_repo.get_document(
                    context.user_token, context.document_id
                )

            semantic_ctx = self.dispatcher.build_context(request.message, document)

            log.info(
                "semantic_analysis_complete",
                intent=semantic_ctx.query.intent.value,
                complexity=semantic_ctx.query.complexity.value,
                thread_id=context.thread_id,
            )

            semantic_xml = self._build_semantic_xml(semantic_ctx)
            full_prompt = PromptBuilder.build(context, semantic_ctx.query.intent, semantic_xml)

            if request.human_choice and not state.next:
                normalized_type = _normalize_human_choice(request.human_choice)
                human_content = _build_forced_pipeline_message(
                    file_path=context.file_path,
                    summary_type=normalized_type,
                    document_id=context.document_id,
                    original=request.message,
                )
                log.info(
                    "forced_pipeline_injected",
                    summary_type=normalized_type,
                    file_path=str(context.file_path or "")[:40],
                    thread_id=context.thread_id,
                )
            elif context.file_path and not _UUID_PAT.match(str(context.file_path)):
                from pathlib import Path as _Path
                fname = _Path(str(context.file_path)).name
                base = request.message or f"Проанализируй файл «{fname}»"
                human_content = (
                    f"{base}\n\n"
                    f"[ОБЯЗАТЕЛЬНО]: Используй read_local_file_content("
                    f"file_path=\'{context.file_path}\') — НЕ doc_get_file_content."
                )
            else:
                human_content = semantic_ctx.query.refined

            inputs: dict[str, Any] = {
                "messages": [
                    SystemMessage(content=full_prompt),
                    HumanMessage(content=human_content),
                ]
            }

            return await self._orchestrate(
                context=context,
                inputs=inputs,
                is_choice_active=False,
                iteration=0,
            )

        except Exception as exc:
            log.error("chat_error", error=str(exc), exc_info=True)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {exc}",
            ).model_dump()

    # ── Human-in-the-Loop ─────────────────────────────────────────────────────

    async def _handle_human_choice(
        self,
        context: ContextParams,
        human_choice: str,
    ) -> dict:
        """Handle user's disambiguation or format selection.

        Injects human_choice into pending tool_call args and resumes graph.
        Used for: summarize format selection, employee disambiguation.

        Args:
            context: Immutable execution context.
            human_choice: User's selection string (e.g. 'extractive', 'thesis').

        Returns:
            AgentResponse dict.
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
            is_choice_active=True,
            iteration=0,
        )

    # ── Orchestration Loop ────────────────────────────────────────────────────

    async def _orchestrate(
        self,
        context: ContextParams,
        inputs: dict[str, Any] | None,
        is_choice_active: bool = False,
        iteration: int = 0,
    ) -> dict:
        """Main LangGraph orchestration loop with automatic parameter injection.

        On each interrupt_before["tools"] pause injects:
            - token: JWT for all tool calls
            - document_id: for doc_* and task/introduction tools
            - attachment_id: when file_path is UUID (UUID → attachment conversion)
            - text + summary_type: for doc_summarize_text (via NLPHelperService)

        Recursion protected by:
            1. iteration > config.max_iterations (application layer guard)
            2. graph_iterations >= MAX_GRAPH_ITERATIONS (graph layer guard)

        Args:
            context: Immutable execution context.
            inputs: Graph inputs dict (None = resume from interrupt).
            is_choice_active: True after user made a disambiguation choice.
            iteration: Recursion depth counter.

        Returns:
            AgentResponse dict.
        """
        # ── Iteration Guard (application layer) ──────────────────────────────
        if iteration > self.config.max_iterations:
            log.error(
                "max_iterations_exceeded",
                thread_id=context.thread_id,
                iteration=iteration,
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышен лимит итераций обработки.",
            ).model_dump()

        try:
            # ── Graph Execution ───────────────────────────────────────────────
            await self.state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=self.config.execution_timeout,
            )

            state = await self.state_manager.get_state(context.thread_id)
            messages = state.values.get("messages", [])

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
                ).model_dump()

            last_msg = messages[-1]

            # ── Graph Done — финальный ответ ──────────────────────────────────
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
                    ).model_dump()

                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    content="Анализ завершён.",
                ).model_dump()

            last_text = ContentExtractor.extract_last_text(messages)
            fixed_calls = []

            for tc in last_msg.tool_calls:
                t_name: str = tc["name"]
                t_args: dict = dict(tc["args"])
                t_id: str = tc["id"]

                t_args["token"] = context.user_token

                if context.document_id and (
                    t_name.startswith("doc_")
                    or t_name in ("introduction_create_tool", "task_create_tool")
                    or "document_id" in t_args
                ):
                    if not t_args.get("document_id"):
                        t_args["document_id"] = context.document_id

                clean_path = str(context.file_path or "").strip()
                is_uuid = bool(
                    re.match(
                        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                        clean_path,
                        re.I,
                    )
                )
                if is_uuid and t_name == "read_local_file_content":
                    t_name = "doc_get_file_content"
                    t_args["attachment_id"] = clean_path
                    t_args.pop("file_path", None)
                    log.info(
                        "uuid_to_attachment_converted",
                        attachment_id_prefix=clean_path[:8],
                    )

                if (
                    t_name == "read_local_file_content"
                    and clean_path
                    and not is_uuid
                ):
                    t_args["file_path"] = clean_path
                    log.debug(
                        "local_file_path_injected",
                        tool=t_name,
                        path_suffix=clean_path[-40:],
                    )

                if t_name == "doc_get_file_content":
                    att_id = str(t_args.get("attachment_id") or "").strip()
                    doc_id = str(context.document_id or "").strip()

                    if att_id and doc_id and att_id == doc_id:
                        log.warning(
                            "attachment_id_equals_document_id_cleared",
                            attachment_id=att_id[:8],
                            document_id=doc_id[:8],
                        )
                        t_args.pop("attachment_id", None)

                    elif att_id and doc_id and len(att_id) == len(doc_id) == 36:
                        # Считаем совпадающие символы
                        matching = sum(a == b for a, b in zip(att_id, doc_id))
                        if matching >= 32:
                            log.warning(
                                "attachment_id_hallucination_cleared",
                                attachment_id=att_id[:8],
                                document_id=doc_id[:8],
                                matching_chars=matching,
                            )
                            t_args.pop("attachment_id", None)

                # 4. Суммаризация — инъекция текста + умный выбор формата
                if t_name == "doc_summarize_text":
                    if last_text and not t_args.get("text"):
                        raw_text = str(last_text)
                        if len(raw_text) > 12000:
                            log.info(
                                "summarize_text_truncated",
                                original_chars=len(raw_text),
                                truncated_chars=12000,
                            )
                            raw_text = raw_text[:12000]
                        t_args["text"] = raw_text

                    if not t_args.get("summary_type") and not is_choice_active:
                        suggestion = NLPHelperService.suggest_summarize_format(
                            str(last_text) if last_text else ""
                        )
                        t_args["summary_type"] = suggestion["recommended"]
                        log.info(
                            "auto_summary_type_selected",
                            summary_type=t_args["summary_type"],
                            reason=suggestion["reason"],
                            chars=suggestion["stats"]["chars"],
                        )
                    elif not t_args.get("summary_type"):
                        t_args["summary_type"] = "extractive"

                fixed_calls.append({"name": t_name, "args": t_args, "id": t_id})

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
                is_choice_active=True,
                iteration=iteration + 1,
            )

        except asyncio.TimeoutError:
            log.error("execution_timeout", thread_id=context.thread_id)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания выполнения.",
            ).model_dump()

        except Exception as exc:
            log.error("orchestration_error", error=str(exc), exc_info=True)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка оркестрации: {exc}",
            ).model_dump()

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _build_context(self, request: AgentRequest) -> ContextParams:
        """Build immutable ContextParams from AgentRequest.

        Safely extracts user_context fields — handles None, dict, or Pydantic model.

        Args:
            request: Validated AgentRequest DTO.

        Returns:
            Immutable ContextParams.
        """
        ctx = request.user_context or {}
        if hasattr(ctx, "model_dump"):
            ctx = ctx.model_dump(exclude_none=True)

        user_name = (
            ctx.get("firstName")
            or ctx.get("first_name")
            or ctx.get("name")
            or "пользователь"
        ).strip()

        return ContextParams(
            user_token=request.user_token,
            document_id=request.context_ui_id,
            file_path=request.file_path,
            thread_id=request.thread_id or "default",
            user_name=user_name,
            user_first_name=ctx.get("firstName") or ctx.get("first_name"),
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