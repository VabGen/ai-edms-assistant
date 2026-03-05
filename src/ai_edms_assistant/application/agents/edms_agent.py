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
logger = logging.getLogger(__name__)

# ── Graph-level constants ──────────────────────────────────────────────────────
MAX_GRAPH_ITERATIONS: int = 10
MAX_CONTEXT_MESSAGES: int = 20
SUMMARIZE_TEXT_LIMIT: int = 10_000

# ── UUID patterns ──────────────────────────────────────────────────────
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
    "1": "extractive", "факты": "extractive", "ключевые факты": "extractive", "extractive": "extractive",
    "2": "abstractive", "пересказ": "abstractive", "краткий пересказ": "abstractive", "abstractive": "abstractive",
    "3": "thesis", "тезисы": "thesis", "тезисный план": "thesis", "thesis": "thesis",
}


@dataclass(frozen=True)
class ContextParams:
    """Immutable execution context for one agent invocation."""
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
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")
        if not self.thread_id or not isinstance(self.thread_id, str):
            raise ValueError("thread_id must be a non-empty string")


class IDocumentRepository(Protocol):
    """Port: abstract contract for document metadata access."""

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        pass


class DocumentRepository:
    """Default adapter: fetches via EdmsDocumentRepository."""

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        try:
            from uuid import UUID
            from ...infrastructure.edms_api.repositories.edms_document_repository import EdmsDocumentRepository
            repo = EdmsDocumentRepository(http_client=EdmsHttpClient())
            return await repo.get_by_id(entity_id=UUID(doc_id), token=token)
        except Exception as exc:
            log.warning("document_fetch_failed", doc_id=doc_id, error=str(exc))
            return None


class NLPHelperService:
    """Recommends summarization format based on text characteristics."""

    @staticmethod
    def suggest_summarize_format(text: str) -> dict[str, Any]:
        if not text:
            return {"recommended": "abstractive", "reason": "Текст пустой", "stats": {"chars": 0}}

        chars = len(text)
        digit_groups = len(re.findall(r"\d+", text))

        if chars > 5000 or digit_groups > 20:
            return {"recommended": "thesis", "reason": f"Объёмный текст ({chars}) или много чисел",
                    "stats": {"chars": chars}}
        if chars > 2000:
            return {"recommended": "extractive", "reason": f"Средний объём ({chars})", "stats": {"chars": chars}}
        return {"recommended": "abstractive", "reason": f"Краткий текст ({chars})", "stats": {"chars": chars}}


class DocumentContextService:
    """Builds deterministic XML context block from Document entity."""

    @staticmethod
    def build_context_block(document: Document | None, context: ContextParams) -> str:
        if not document and not context.document_id and not context.file_path:
            return ""

        parts: list[str] = ["<document_context>"]

        if document:
            parts.append(f"  <id>{document.id}</id>")
            if document.short_summary:
                title = str(document.short_summary).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
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
                    parts.append(f'    <attachment index="{idx}" id="{att_id}" name="{att_name}" type="{att_type}"/>')
                parts.append("  </attachments>")
                parts.append(f"  <active_file>{str(attachments[0].id)}</active_file>")
            else:
                parts.append("  <attachments><!-- вызови doc_get_details() для получения списка --></attachments>")
        else:
            if context.document_id:
                parts.append(f"  <id>{context.document_id}</id>")
                parts.append("  <attachments><!-- вызови doc_get_details() для получения списка --></attachments>")

            if context.file_path:
                is_uuid = bool(_UUID_RE.match(context.file_path))
                if is_uuid:
                    parts.append(f"  <attachment_uuid>{context.file_path}</attachment_uuid>")
                else:
                    parts.append(f"  <local_file_path>{context.file_path}</local_file_path>")

        parts.append("</document_context>")
        return "\n".join(parts)


class PromptBuilder:
    """Builds system prompts with dynamic context injection."""
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
            f"\n<token>{injected_token}</token>"
            f"\n<document_id>{context.document_id or ''}</document_id>"
            "\n</injected_auth>"
        )

        snippet = cls._INTENT_SNIPPETS.get(intent, "")
        return base + auth_block + snippet + semantic_xml


class ContentExtractor:
    """Extracts final user-facing content from LangGraph message chain."""
    _SKIP_PATTERNS: tuple[str, ...] = ("вызвал инструмент", "tool call", '"name"', '"id"', "invoke_tool")
    _MIN_CONTENT_LENGTH: int = 20
    _JSON_FIELDS: tuple[str, ...] = ("content", "text", "text_preview", "message", "result")

    @classmethod
    def extract_final_content(cls, messages: list[BaseMessage]) -> str | None:
        """Extract final user-facing response from message chain."""
        if not messages:
            return None

        last_human_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage) and not str(msg.content).startswith("[СИСТЕМНОЕ"):
                last_human_idx = i

        current_window = messages[last_human_idx:] if last_human_idx >= 0 else messages

        for msg in reversed(current_window):
            if isinstance(msg, AIMessage) and msg.content:
                if getattr(msg, "tool_calls", None):
                    continue
                content = str(msg.content).strip()
                if not cls._is_internal_artifact(content) and len(content) >= cls._MIN_CONTENT_LENGTH:
                    log.debug("found_final_ai_message", length=len(content))
                    return content

        for msg in reversed(current_window):
            if isinstance(msg, ToolMessage):
                extracted = cls._extract_from_tool_message(msg)
                if extracted:
                    log.debug("found_fallback_tool_message", length=len(extracted))
                    return extracted

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                content = str(msg.content).strip()
                if len(content) >= cls._MIN_CONTENT_LENGTH:
                    return content

        return None

    @classmethod
    def _is_internal_artifact(cls, content: str) -> bool:
        lower = content.lower()
        return any(skip in lower for skip in cls._SKIP_PATTERNS)

    @classmethod
    def _extract_from_tool_message(cls, message: ToolMessage) -> str | None:
        try:
            raw = str(message.content).strip()
            if not raw.startswith("{"):
                if len(raw) > cls._MIN_CONTENT_LENGTH:
                    return raw
                return None

            data = json.loads(raw)
            nested = data.get("data", {}) or {}

            for json_field in cls._JSON_FIELDS:
                val = nested.get(json_field) or data.get(json_field)
                if val:
                    content = str(val).strip()
                    if len(content) >= cls._MIN_CONTENT_LENGTH:
                        return content

            if len(raw) > 100:
                return raw
        except json.JSONDecodeError:
            if len(str(message.content)) > cls._MIN_CONTENT_LENGTH:
                return str(message.content)
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
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
            .replace("\n", "\n")
            .strip()
        )


class AgentStateManager:
    """Wrapper around CompiledStateGraph."""

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver) -> None:
        if graph is None or checkpointer is None:
            raise ValueError("Graph and Checkpointer cannot be None")
        self.graph = graph
        self.checkpointer = checkpointer

    @staticmethod
    def _make_config(thread_id: str) -> RunnableConfig:
        return RunnableConfig(configurable={"thread_id": thread_id})

    async def get_state(self, thread_id: str) -> Any:
        return await self.graph.aget_state(self._make_config(thread_id))

    async def invoke(self, inputs: dict[str, Any] | None, thread_id: str, timeout: float = 120.0) -> None:
        config = self._make_config(thread_id)
        await asyncio.wait_for(self.graph.ainvoke(inputs, config=config), timeout=timeout)


def _normalize_human_choice(raw: str) -> str:
    return _CHOICE_NORM.get(raw.strip().lower(), raw.strip())


def _build_human_message(context: ContextParams, refined_query: str) -> str:
    if context.summary_type:
        type_label = _SUMMARY_TYPE_LABELS.get(context.summary_type, context.summary_type)
        if context.file_path and _UUID_RE.match(context.file_path):
            return (f"Выполни суммаризацию вложения в формате «{type_label}».\n"
                    f"Шаг 1: Вызови doc_get_file_content с attachment_id из промпта.\n"
                    f"Шаг 2: Вызови doc_summarize_text.\n[ЗАПРОС]: {refined_query}")
        if context.file_path:
            from pathlib import Path as _Path
            fname = _Path(context.file_path).name
            return (f"Выполни суммаризацию файла «{fname}» в формате «{type_label}».\n"
                    f"Шаг 1: read_local_file_content.\nШаг 2: doc_summarize_text.\n[ЗАПРОС]: {refined_query}")
        if context.document_id:
            return (f"Выполни суммаризацию вложения документа в формате «{type_label}».\n"
                    f"Шаг 1: Возьми attachment_id из контекста.\nШаг 2: doc_get_file_content.\nШаг 3: doc_summarize_text.\n[ЗАПРОС]: {refined_query}")

    if context.file_path and not _UUID_RE.match(context.file_path):
        from pathlib import Path as _Path
        fname = _Path(context.file_path).name
        return f"{refined_query or f'Проанализируй файл «{fname}»'}\n[ОБЯЗАТЕЛЬНО]: Используй read_local_file_content."

    return refined_query


def _truncate_text(text: str, limit: int = SUMMARIZE_TEXT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.3):]
    return head + "\n[...середина пропущена...]\n" + tail


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent
# ─────────────────────────────────────────────────────────────────────────────
class EdmsDocumentAgent:
    def __init__(
            self,
            config: AgentConfig | None = None,
            document_repo: IDocumentRepository | None = None,
            semantic_dispatcher: SemanticDispatcher | None = None,
    ) -> None:
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

            self.state_manager = AgentStateManager(graph=self._compiled_graph, checkpointer=self._checkpointer)

            log.info(
                "edms_agent_ready",
                version="v3",
                tools_count=len(self.tools),
                tool_names=[t.name for t in self.tools],
                max_iterations=self.config.max_iterations,
                timeout=self.config.execution_timeout,
            )
        except Exception as exc:
            log.error("edms_agent_init_failed", error=str(exc), exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {exc}") from exc

    def _create_tools(self) -> list[Any]:
        from ...domain.services.appeal_validator import AppealValidator
        from ...domain.services.document_comparer import DocumentComparer
        from ...domain.services.task_assigner import TaskAssigner
        from ...infrastructure.edms_api.repositories.edms_document_repository import EdmsDocumentRepository
        from ...infrastructure.edms_api.repositories.edms_employee_repository import EdmsEmployeeRepository
        from ...infrastructure.edms_api.repositories.edms_task_repository import EdmsTaskRepository
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
        try:
            from ...infrastructure.nlp.extractors.appeal_extractor import AppealExtractor
            return AppealExtractor()
        except Exception:
            return None

    def health_check(self) -> dict[str, Any]:
        return {
            "version": "v3",
            "model": self.model is not None,
            "tools_count": len(self.tools),
            "graph": hasattr(self, "_compiled_graph") and self._compiled_graph is not None,
            "state_manager": hasattr(self, "state_manager") and self.state_manager is not None,
        }

    def _build_graph(self) -> CompiledStateGraph:
        workflow: Any = StateGraph(AgentStateWithCounter)

        async def call_model(state: AgentStateWithCounter) -> dict[str, Any]:
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

            final_messages: list[BaseMessage] = ([sys_msgs[-1]] if sys_msgs else []) + non_sys
            response: BaseMessage = await model_with_tools.ainvoke(final_messages)
            return {"messages": [response], "graph_iterations": 1}

        async def validator(state: AgentStateWithCounter) -> dict[str, Any]:
            messages: list[BaseMessage] = state["messages"]
            last_msg = messages[-1]
            if not isinstance(last_msg, ToolMessage):
                return {"messages": []}

            content = str(last_msg.content).strip()
            if not content or content in ("None", "{}"):
                return {
                    "messages": [HumanMessage(content="[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Инструмент вернул пустой результат.")]}

            if "error" in content.lower() and "exception" in content.lower():
                return {
                    "messages": [HumanMessage(content=f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Ошибка инструмента: {content[:500]}")]}
            return {"messages": []}

        def should_continue(state: AgentStateWithCounter) -> str:
            messages: list[BaseMessage] = state["messages"]
            if not messages:
                return END
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                return "tools"
            return END

        def after_validator(state: AgentStateWithCounter) -> str:
            iterations: int = state.get("graph_iterations", 0)
            if iterations >= MAX_GRAPH_ITERATIONS:
                log.warning("graph_max_iterations_reached", iterations=iterations)
                return END
            return "agent"

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("validator", validator)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "validator")
        workflow.add_conditional_edges("validator", after_validator, {"agent": "agent", END: END})

        try:
            compiled = workflow.compile(checkpointer=self._checkpointer)
            log.debug("graph_compiled_v3_autonomous")
            return compiled
        except Exception as exc:
            log.error("graph_compilation_failed", error=str(exc), exc_info=True)
            raise RuntimeError(f"Failed to compile graph: {exc}") from exc

    async def chat(self, request: AgentRequest) -> dict[str, Any]:
        try:
            context = await self._build_context(request)
            document: Document | None = None
            if context.document_id:
                document = await self.document_repo.get_document(context.user_token, context.document_id)

            doc_context_xml = DocumentContextService.build_context_block(document=document, context=context)
            semantic_ctx = self.dispatcher.build_context(request.message, document)
            intent = semantic_ctx.query.intent

            log.info("semantic_analysis_complete", intent=intent.value, complexity=semantic_ctx.query.complexity.value)

            semantic_xml = self._build_semantic_xml(semantic_ctx)
            system_prompt = PromptBuilder.build(
                context=context, intent=intent, semantic_xml=semantic_xml,
                document_context_xml=doc_context_xml, injected_token=context.user_token,
            )

            human_content = _build_human_message(context=context, refined_query=semantic_ctx.query.refined)

            inputs: dict[str, Any] = {
                "messages": [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
            }

            return await self._run_graph(context=context, inputs=inputs)

        except ValueError as exc:
            log.error("chat_validation_error", error=str(exc))
            return AgentResponse(status=AgentStatus.ERROR, message=f"Ошибка валидации: {exc}").model_dump()
        except Exception as exc:
            log.error("chat_error", error=str(exc), exc_info=True)
            return AgentResponse(status=AgentStatus.ERROR, message=f"Ошибка обработки: {exc}").model_dump()

    async def _run_graph(self, context: ContextParams, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            await self.state_manager.invoke(inputs=inputs, thread_id=context.thread_id,
                                            timeout=self.config.execution_timeout)
            state = await self.state_manager.get_state(context.thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])

            log.debug("graph_execution_complete", thread_id=context.thread_id, messages_count=len(messages),
                      last_type=type(messages[-1]).__name__ if messages else "none")

            if not messages:
                return AgentResponse(status=AgentStatus.ERROR, message="Пустое состояние агента.").model_dump()

            last_msg = messages[-1]

            if isinstance(last_msg, ToolMessage):
                synthesis = await self._synthesize_from_tool_message(last_msg)
                if synthesis:
                    return AgentResponse(status=AgentStatus.SUCCESS, content=synthesis).model_dump()
                tool_text = ContentExtractor._extract_from_tool_message(last_msg)
                if tool_text:
                    return AgentResponse(status=AgentStatus.SUCCESS, content=tool_text).model_dump()

            final = ContentExtractor.extract_final_content(messages)

            if final:
                final = ContentExtractor.clean_json_artifacts(final)

                if context.summary_type and len(final) > 3000:
                    summarized = await self._direct_summarize(text=final, summary_type=context.summary_type,
                                                              thread_id=context.thread_id)
                    if summarized:
                        return AgentResponse(status=AgentStatus.SUCCESS, content=summarized).model_dump()

                return AgentResponse(status=AgentStatus.SUCCESS, content=final).model_dump()

            log.warning("failed_to_extract_any_content", thread_id=context.thread_id)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Агент выполнил действия, но не смог сформировать текстовый ответ. Проверьте логи."
            ).model_dump()

        except asyncio.TimeoutError:
            log.error("execution_timeout", thread_id=context.thread_id)
            return AgentResponse(status=AgentStatus.ERROR, message="Превышено время ожидания.").model_dump()
        except Exception as exc:
            log.error("graph_execution_error", error=str(exc), exc_info=True)
            return AgentResponse(status=AgentStatus.ERROR, message=f"Ошибка выполнения графа: {exc}").model_dump()

    async def _synthesize_from_tool_message(self, tool_msg: ToolMessage) -> str | None:
        """Synthesize final answer when graph ended on a ToolMessage."""
        tool_content = str(tool_msg.content).strip()
        if len(tool_content) < 30:
            return None

        try:
            model_plain = self.model.bind_tools([])
            resp = await asyncio.wait_for(
                model_plain.ainvoke([
                    SystemMessage(
                        content="На основе данных инструмента дай краткий, понятный ответ пользователю на русском языке."),
                    HumanMessage(content=f"Данные инструмента:\n{tool_content[:6000]}")
                ]),
                timeout=60.0,
            )
            content = str(resp.content).strip()
            if content and len(content) > 10:
                log.info("synthesis_from_tool_message_ok", chars=len(content))
                return content
        except Exception as exc:
            log.warning("synthesis_failed", error=str(exc))
        return None

    async def _direct_summarize(self, text: str, summary_type: str, thread_id: str) -> str | None:
        _instructions = {
            "extractive": "Выдели ключевые факты списком.",
            "abstractive": "Напиши краткий пересказ своими словами.",
            "thesis": "Сформируй тезисный план.",
        }
        instruction = _instructions.get(summary_type, "Кратко изложи суть.")
        truncated = _truncate_text(text, SUMMARIZE_TEXT_LIMIT)

        try:
            model_plain = self.model.bind_tools([])
            resp = await asyncio.wait_for(
                model_plain.ainvoke([
                    SystemMessage(content=f"Ты аналитик СЭД. Задача: {instruction}"),
                    HumanMessage(content=f"ТЕКСТ:\n{truncated}\nОТВЕТ:")
                ]),
                timeout=120.0,
            )
            content = str(resp.content).strip()
            if content and len(content) > 50:
                return content
        except Exception:
            pass
        return None

    @staticmethod
    async def _build_context(request: AgentRequest) -> ContextParams:
        ctx: dict[str, Any] = request.user_context or {}
        if hasattr(ctx, "model_dump"):
            ctx = ctx.model_dump(exclude_none=True)

        user_name = str(ctx.get("firstName") or ctx.get("first_name") or ctx.get("name") or "пользователь").strip()

        return ContextParams(
            user_token=request.user_token,
            thread_id=request.thread_id or "default",
            document_id=request.context_ui_id,
            file_path=request.file_path,
            summary_type=_normalize_human_choice(request.human_choice) if request.human_choice else None,
            user_name=user_name,
            user_first_name=str(ctx.get("firstName") or ctx.get("first_name") or "") or None,
        )

    @staticmethod
    def _build_semantic_xml(semantic_ctx: Any) -> str:
        return (
            "\n<semantic_context>"
            f"\n<intent>{semantic_ctx.query.intent.value}</intent>"
            f"\n<original>{semantic_ctx.query.original}</original>"
            f"\n<refined>{semantic_ctx.query.refined}</refined>"
            f"\n<complexity>{semantic_ctx.query.complexity.value}</complexity>"
            "\n</semantic_context>"
        )