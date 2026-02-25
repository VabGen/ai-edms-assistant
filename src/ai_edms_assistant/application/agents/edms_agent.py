# src/ai_edms_assistant/application/agents/edms_agent.py
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

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
from ...infrastructure.edms_api.clients.document_client import EdmsDocumentClient
from ...infrastructure.llm.providers.openai_provider import get_chat_model
from ..dto.agent import AgentRequest, AgentResponse, AgentStatus
from ..services.semantic_dispatcher import SemanticDispatcher, UserIntent
from ..tools import LocalFileTool  # FIXED: Changed from all_tools
from .agent_config import AgentConfig
from .agent_state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextParams:
    """Immutable контекст для выполнения агента.

    Attributes:
        user_token: JWT bearer token для API запросов.
        document_id: UUID активного документа в UI.
        file_path: Путь к загруженному файлу или UUID вложения.
        thread_id: Идентификатор сессии для LangGraph checkpointer.
        user_name: Имя пользователя для обращения.
        user_first_name: Имя пользователя (если доступно).
        current_date: Текущая дата в формате dd.mm.yyyy.
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

    def __post_init__(self):
        """Validate required fields."""
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")


# ---------------------------------------------------------------------------
# Repository Protocol (Dependency Inversion)
# ---------------------------------------------------------------------------


class IDocumentRepository(Protocol):
    """Интерфейс для работы с документами (Dependency Inversion Principle)."""

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        """Получить документ по ID.

        Args:
            token: JWT bearer token.
            doc_id: UUID документа.

        Returns:
            Document entity или None если не найден.
        """
        ...


class DocumentRepository:
    """Конкретная реализация репозитория документов."""

    async def get_document(self, token: str, doc_id: str) -> Document | None:
        """Получить документ через EDMS API client.

        Args:
            token: JWT bearer token.
            doc_id: UUID документа.

        Returns:
            Document entity или None.
        """
        try:
            async with EdmsDocumentClient() as client:
                raw_data = await client.get_by_id(
                    document_id=doc_id,
                    token=token,
                )
                if not raw_data:
                    return None

                from ...infrastructure.edms_api.mappers.document_mapper import (
                    DocumentMapper,
                )

                doc = DocumentMapper.from_dto(raw_data)
                logger.info(f"Document fetched: {doc_id}", extra={"doc_id": doc_id})
                return doc

        except Exception as e:
            logger.error(
                f"Failed to fetch document {doc_id}: {e}",
                exc_info=True,
                extra={"doc_id": doc_id, "error": str(e)},
            )
            return None


# ---------------------------------------------------------------------------
# Prompt Builder (Strategy Pattern)
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Strategy для построения системных промптов с динамическим контекстом."""

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
1. **Автоинъекция**: Параметры `token` и `document_id` добавляются АВТОМАТИЧЕСКИ системой. Не указывай их явно.

2. **Обработка LOCAL_FILE**:
   - UUID формат (например: 0c2216e1-...) → Вызови `doc_get_file_content(attachment_id=LOCAL_FILE)`
   - Путь к файлу (/tmp/...) → Вызови `read_local_file_content(file_path=LOCAL_FILE)`
   - Пустое значение ("Не загружен") → Вызови `doc_get_details()` для поиска вложений

3. **Обработка requires_action**:
   - Статус "summarize_selection" → Предложи формат анализа (факты/пересказ/тезисы)
   - Статус "requires_disambiguation" → Покажи список, дождись выбора пользователя

4. **ВАЖНО**: После вызова инструментов ВСЕГДА формулируй финальный ответ на русском языке.

5. **Язык**: Только русский. Обращайся к пользователю по имени: {user_name}
</critical_rules>

<tool_selection>
**Типичные сценарии**:
- Анализ документа: doc_get_details → doc_get_file_content → doc_summarize_text
- Анализ файла (UUID): doc_get_file_content → doc_summarize_text
- Поиск сотрудника: employee_search_tool
- Список ознакомления: introduction_create_tool
- Создание поручения: task_create_tool
</tool_selection>

<response_format>
✅ Структурировано, кратко, по делу
❌ Многословие, технические детали API
</response_format>"""

    CONTEXT_SNIPPETS = {
        UserIntent.CREATE_INTRODUCTION: """
<introduction_guide>
При создании списка ознакомления:
- Если статус "requires_disambiguation" → Покажи список найденных сотрудников
- Дождись выбора пользователя
- Повторный вызов: introduction_create_tool(selected_employee_ids=[uuid1, uuid3])
</introduction_guide>""",
        UserIntent.CREATE_TASK: """
<task_guide>
При создании поручения:
- executor_last_names: обязательно (минимум 1)
- responsible_last_name: опционально (если НЕ указан → первый исполнитель)
- planed_date_end: опционально (если НЕ указан → +7 дней)
- Даты должны быть в формате ISO 8601 с timezone (например: "2026-02-15T23:59:59Z")
</task_guide>""",
        UserIntent.SUMMARIZE: """
<date_parsing>
Преобразование дат в ISO 8601:
- "до 15 февраля" → "2026-02-15T23:59:59Z"
- "через неделю" → +7 дней от текущей даты
Всегда добавляй суффикс 'Z' (UTC timezone).
</date_parsing>""",
    }

    @classmethod
    def build(
        cls,
        context: ContextParams,
        intent: UserIntent,
        semantic_xml: str,
    ) -> str:
        """Построить полный системный промпт.

        Args:
            context: Контекст выполнения.
            intent: Распознанный intent пользователя.
            semantic_xml: XML с семантическим анализом.

        Returns:
            Полный системный промпт.
        """
        base_prompt = cls.CORE_TEMPLATE.format(
            user_name=context.user_name,
            current_date=context.current_date,
            context_ui_id=context.document_id or "Не указан",
            local_file=context.file_path or "Не загружен",
        )

        dynamic_context = cls.CONTEXT_SNIPPETS.get(intent, "")
        return base_prompt + dynamic_context + semantic_xml


# ---------------------------------------------------------------------------
# Content Extractor
# ---------------------------------------------------------------------------


class ContentExtractor:
    """Извлечение финального контента из цепочки сообщений."""

    SKIP_PATTERNS = ["вызвал инструмент", "tool call", '"name"', '"id"']
    MIN_CONTENT_LENGTH = 50
    JSON_FIELDS = ["content", "text", "text_preview", "message"]

    @classmethod
    def extract_final_content(cls, messages: list[BaseMessage]) -> str | None:
        """Извлечь финальный ответ агента из истории сообщений.

        Args:
            messages: История сообщений LangGraph.

        Returns:
            Финальный текст ответа или None.
        """
        # 1. Поиск AIMessage с контентом
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                content = str(m.content).strip()
                if cls._is_skip_content(content):
                    continue
                if len(content) > cls.MIN_CONTENT_LENGTH:
                    logger.debug(f"Extracted AIMessage: {len(content)} chars")
                    return content

        # 2. Поиск в ToolMessage (JSON extract)
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._extract_from_tool_message(m)
                if extracted:
                    return extracted

        # 3. Fallback: любой AIMessage
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                content = str(m.content).strip()
                if content:
                    logger.debug(f"Fallback AIMessage: {len(content)} chars")
                    return content

        # 4. Last resort: ToolMessage raw content
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                content = str(m.content).strip()
                if len(content) > cls.MIN_CONTENT_LENGTH:
                    logger.debug(f"Fallback ToolMessage: {len(content)} chars")
                    return content

        return None

    @classmethod
    def extract_last_text(cls, messages: list[BaseMessage]) -> str | None:
        """Извлечь последний текстовый контент из ToolMessage.

        Args:
            messages: История сообщений.

        Returns:
            Текстовый контент или None.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                if isinstance(m.content, str) and m.content.startswith("{"):
                    data = json.loads(m.content)
                    text = (
                        data.get("content")
                        or data.get("text_preview")
                        or data.get("text")
                    )
                    if text and len(str(text)) > 100:
                        return str(text)
                if len(str(m.content)) > 100:
                    return str(m.content)
            except json.JSONDecodeError:
                if len(str(m.content)) > 100:
                    return str(m.content)
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """Очистить JSON артефакты из финального контента.

        Args:
            content: Сырой контент.

        Returns:
            Очищенный текст.
        """
        if content.startswith('{"status"'):
            try:
                data = json.loads(content)
                if "content" in data:
                    content = data["content"]
            except json.JSONDecodeError:
                pass

        content = content.replace('{"status": "success", "content": "', "")
        content = content.replace('"}', "")
        content = content.replace('\\"', '"')
        content = content.replace("\\n", "\n")
        return content.strip()

    @classmethod
    def _is_skip_content(cls, content: str) -> bool:
        """Проверить, является ли контент служебным (skip).

        Args:
            content: Проверяемый текст.

        Returns:
            True если контент нужно пропустить.
        """
        return any(skip in content.lower() for skip in cls.SKIP_PATTERNS)

    @classmethod
    def _extract_from_tool_message(cls, message: ToolMessage) -> str | None:
        """Извлечь контент из JSON ToolMessage.

        Args:
            message: ToolMessage для анализа.

        Returns:
            Извлеченный текст или None.
        """
        try:
            if isinstance(message.content, str) and message.content.strip().startswith(
                "{"
            ):
                data = json.loads(message.content)
                for field in cls.JSON_FIELDS:
                    if field in data and data[field]:
                        content = str(data[field]).strip()
                        if len(content) > cls.MIN_CONTENT_LENGTH:
                            logger.debug(
                                f"ToolMessage JSON[{field}]: {len(content)} chars"
                            )
                            return content
        except json.JSONDecodeError:
            pass
        return None


# ---------------------------------------------------------------------------
# Agent State Manager
# ---------------------------------------------------------------------------


class AgentStateManager:
    """Управление состоянием LangGraph агента."""

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver):
        """Initialize state manager.

        Args:
            graph: Compiled LangGraph StateGraph.
            checkpointer: MemorySaver для persistence.

        Raises:
            ValueError: When graph or checkpointer is None.
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

    async def get_state(self, thread_id: str) -> Any:
        """Получить текущее состояние по thread_id.

        Args:
            thread_id: Идентификатор сессии.

        Returns:
            Текущее состояние GraphState.
        """
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aget_state(config)

    async def update_state(
        self,
        thread_id: str,
        messages: list[BaseMessage],
        as_node: str = "agent",
    ) -> None:
        """Обновить состояние (например, для human-in-the-loop).

        Args:
            thread_id: Идентификатор сессии.
            messages: Новые сообщения для добавления.
            as_node: Имя node для контекста обновления.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await self.graph.aupdate_state(config, {"messages": messages}, as_node=as_node)

    async def invoke(
        self,
        inputs: dict[str, Any],
        thread_id: str,
        timeout: float = 120.0,
    ) -> None:
        """Запустить выполнение графа.

        Args:
            inputs: Входные данные для графа.
            thread_id: Идентификатор сессии.
            timeout: Таймаут выполнения в секундах.

        Raises:
            asyncio.TimeoutError: When execution exceeds timeout.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await asyncio.wait_for(
            self.graph.ainvoke(inputs, config=config),
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------


class EdmsDocumentAgent:
    """Production-ready мультиагентная система для EDMS.

    Использует LangGraph для оркестрации, SemanticDispatcher для анализа
    intent'ов, и автоматическую инъекцию параметров в tool calls.

    Attributes:
        config: Конфигурация агента (AgentConfig).
        model: LLM модель для рассуждений.
        tools: Список LangChain tools.
        document_repo: Репозиторий для работы с документами.
        dispatcher: Семантический диспетчер для анализа intent.
        state_manager: Менеджер состояния LangGraph.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        document_repo: IDocumentRepository | None = None,
        semantic_dispatcher: SemanticDispatcher | None = None,
    ):
        """Initialize EDMS Document Agent.

        Args:
            config: Agent configuration (uses defaults if None).
            document_repo: Document repository (uses default if None).
            semantic_dispatcher: Semantic analyzer (uses default if None).

        Raises:
            RuntimeError: When graph compilation fails.
        """
        try:
            self.config = config or AgentConfig()
            self.model = get_chat_model()
            self.tools = [LocalFileTool()]  # FIXED: Minimal tools without dependencies
            self.document_repo = document_repo or DocumentRepository()
            self.dispatcher = semantic_dispatcher or SemanticDispatcher()

            logger.debug("Base components initialized")

            self._checkpointer = MemorySaver()
            logger.debug("Checkpointer created")

            self._compiled_graph = self._build_graph()

            if self._compiled_graph is None:
                raise RuntimeError("Graph compilation returned None")

            logger.debug(
                "Graph compiled successfully",
                extra={"graph_type": type(self._compiled_graph).__name__},
            )

            self.state_manager = AgentStateManager(
                graph=self._compiled_graph,
                checkpointer=self._checkpointer,
            )

            logger.info(
                "EdmsDocumentAgent initialized successfully",
                extra={
                    "tools_count": len(self.tools),
                    "model": str(self.model),
                    "max_iterations": self.config.max_iterations,
                },
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize EdmsDocumentAgent: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Agent initialization failed: {e}") from e

    def health_check(self) -> dict[str, Any]:
        """Проверка здоровья агента.

        Returns:
            Dict с состоянием компонентов.
        """
        return {
            "model": self.model is not None,
            "tools": len(self.tools) > 0,
            "document_repo": self.document_repo is not None,
            "dispatcher": self.dispatcher is not None,
            "graph": hasattr(self, "_compiled_graph")
            and self._compiled_graph is not None,
            "state_manager": hasattr(self, "state_manager")
            and self.state_manager is not None,
            "checkpointer": hasattr(self, "_checkpointer")
            and self._checkpointer is not None,
        }

    def _build_graph(self) -> CompiledStateGraph:
        """Построить и скомпилировать LangGraph workflow.

        Returns:
            Compiled StateGraph.

        Raises:
            RuntimeError: When compilation fails.
        """
        workflow = StateGraph(AgentState)

        async def call_model(state: AgentState) -> dict:
            """Node: вызов LLM с tool binding."""
            model_with_tools = self.model.bind_tools(self.tools)
            non_sys = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
            sys_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
            final_messages = ([sys_msgs[-1]] if sys_msgs else []) + non_sys

            response = await model_with_tools.ainvoke(final_messages)
            return {"messages": [response]}

        async def validator(state: AgentState) -> dict:
            """Node: валидация результатов tool execution."""
            messages = state["messages"]
            last_message = messages[-1]

            if not isinstance(last_message, ToolMessage):
                return {"messages": []}

            content_raw = str(last_message.content).strip()

            if not content_raw or content_raw in ("None", "{}"):
                return {
                    "messages": [
                        HumanMessage(
                            content="[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Инструмент вернул пустой результат."
                        )
                    ]
                }

            if "error" in content_raw.lower() or "exception" in content_raw.lower():
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Техническая ошибка: {content_raw}"
                        )
                    ]
                }

            return {"messages": []}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("validator", validator)
        workflow.add_edge(START, "agent")

        def should_continue(state: AgentState) -> str:
            """Conditional edge: продолжить или завершить."""
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and getattr(
                last_message, "tool_calls", None
            ):
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

            logger.debug("Graph compiled successfully")
            return compiled

        except Exception as e:
            logger.error(f"Graph compilation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to compile graph: {e}") from e

    async def chat(self, request: AgentRequest) -> dict:
        """Основной метод обработки пользовательского запроса.

        Args:
            request: Валидированный AgentRequest.

        Returns:
            AgentResponse dict.
        """
        try:
            context = await self._build_context(request)
            state = await self.state_manager.get_state(context.thread_id)

            # Handle human choice (disambiguation)
            if request.human_choice and state.next:
                return await self._handle_human_choice(context, request.human_choice)

            # Fetch document if available
            document = None
            if context.document_id:
                document = await self.document_repo.get_document(
                    context.user_token,
                    context.document_id,
                )

            # Semantic analysis
            semantic_context = self.dispatcher.build_context(request.message, document)
            logger.info(
                f"Semantic analysis complete",
                extra={
                    "intent": semantic_context.query.intent.value,
                    "complexity": semantic_context.query.complexity.value,
                    "thread_id": context.thread_id,
                },
            )

            refined_message = semantic_context.query.refined
            user_intent = semantic_context.query.intent

            # Build full prompt
            semantic_xml = self._build_semantic_xml(semantic_context)
            full_prompt = PromptBuilder.build(context, user_intent, semantic_xml)

            sys_msg = SystemMessage(content=full_prompt)
            hum_msg = HumanMessage(content=refined_message)
            inputs = {"messages": [sys_msg, hum_msg]}

            return await self._orchestrate(
                context=context,
                inputs=inputs,
                is_choice_active=bool(request.human_choice),
                iteration=0,
            )

        except Exception as e:
            logger.error(
                f"Chat error: {e}",
                exc_info=True,
                extra={"user_message": request.message},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {str(e)}",
            ).model_dump()

    async def _handle_human_choice(
        self,
        context: ContextParams,
        human_choice: str,
    ) -> dict:
        """Обработка выбора пользователя (disambiguation).

        Args:
            context: Контекст выполнения.
            human_choice: Выбор пользователя.

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

    async def _orchestrate(
        self,
        context: ContextParams,
        inputs: dict | None,
        is_choice_active: bool = False,
        iteration: int = 0,
    ) -> dict:
        """Основной цикл оркестрации (recursive execution).

        Args:
            context: Контекст выполнения.
            inputs: Входные сообщения (None для продолжения).
            is_choice_active: Активен ли выбор пользователя.
            iteration: Номер итерации.

        Returns:
            AgentResponse dict.
        """
        if iteration > self.config.max_iterations:
            logger.error(
                f"Max iterations exceeded",
                extra={"thread_id": context.thread_id},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышен лимит итераций обработки.",
            ).model_dump()

        try:
            # Execute graph
            await self.state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=self.config.timeout_seconds,
            )

            state = await self.state_manager.get_state(context.thread_id)
            messages = state.values.get("messages", [])

            logger.debug(
                "State snapshot",
                extra={
                    "thread_id": context.thread_id,
                    "iteration": iteration,
                    "messages_count": len(messages),
                    "last_message_type": (
                        type(messages[-1]).__name__ if messages else None
                    ),
                    "state_next": state.next,
                },
            )

            if not messages:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Пустое состояние агента.",
                ).model_dump()

            last_msg = messages[-1]

            # Check if execution complete
            if (
                not state.next
                or not isinstance(last_msg, AIMessage)
                or not getattr(last_msg, "tool_calls", None)
            ):
                final_content = ContentExtractor.extract_final_content(messages)
                if final_content:
                    final_content = ContentExtractor.clean_json_artifacts(final_content)
                    logger.info(
                        f"Execution completed successfully",
                        extra={
                            "thread_id": context.thread_id,
                            "content_length": len(final_content),
                            "iterations": iteration + 1,
                        },
                    )
                    return AgentResponse(
                        status=AgentStatus.SUCCESS,
                        content=final_content,
                    ).model_dump()

                logger.warning(
                    "No final content found",
                    extra={"thread_id": context.thread_id},
                )
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    content="Анализ завершен.",
                ).model_dump()

            # Auto-inject parameters
            last_extracted_text = ContentExtractor.extract_last_text(messages)
            fixed_calls = []

            for tc in last_msg.tool_calls:
                t_name, t_args, t_id = tc["name"], dict(tc["args"]), tc["id"]

                # Inject token
                t_args["token"] = context.user_token

                # Handle UUID vs file path
                clean_path = str(context.file_path).strip() if context.file_path else ""
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

                # Inject document_id
                if context.document_id and (
                    t_name.startswith("doc_")
                    or "document_id" in t_args
                    or t_name in ["introduction_create_tool", "task_create_tool"]
                ):
                    t_args["document_id"] = context.document_id

                # Auto-inject text for summarization
                if t_name == "doc_summarize_text":
                    if last_extracted_text:
                        t_args["text"] = str(last_extracted_text)

                    if not t_args.get("summary_type") and not is_choice_active:
                        # Use simple default instead of NLP service
                        t_args["summary_type"] = "extractive"

                fixed_calls.append({"name": t_name, "args": t_args, "id": t_id})

            # Update state with fixed tool calls
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
                is_choice_active=True,
                iteration=iteration + 1,
            )

        except asyncio.TimeoutError:
            logger.error("Execution timeout", extra={"thread_id": context.thread_id})
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания выполнения.",
            ).model_dump()
        except Exception as e:
            logger.error(
                f"Orchestration error: {e}",
                exc_info=True,
                extra={"thread_id": context.thread_id},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка оркестрации: {str(e)}",
            ).model_dump()

    async def _build_context(self, request: AgentRequest) -> ContextParams:
        """Построить контекст из request.

        Args:
            request: Входящий запрос.

        Returns:
            ContextParams.
        """
        user_name = (
            request.user_context.get("firstName")
            or request.user_context.get("name")
            or "пользователь"
        ).strip()

        return ContextParams(
            user_token=request.user_token,
            document_id=request.context_ui_id,
            file_path=request.file_path,
            thread_id=request.thread_id or "default",
            user_name=user_name,
            user_first_name=request.user_context.get("firstName"),
        )

    @staticmethod
    def _build_semantic_xml(semantic_context) -> str:
        """Построить XML с семантическим контекстом.

        Args:
            semantic_context: Результат semantic dispatcher.

        Returns:
            XML string.
        """
        return f"""
<semantic_context>
  <user_query>
    <original>{semantic_context.query.original}</original>
    <refined>{semantic_context.query.refined}</refined>
    <intent>{semantic_context.query.intent.value}</intent>
    <complexity>{semantic_context.query.complexity.value}</complexity>
  </user_query>
</semantic_context>"""