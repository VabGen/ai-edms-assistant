# orchestrator/agent.py
"""
EDMS AI Assistant — главный агент на нативном Anthropic SDK + tool_use.

Архитектура:
    EdmsDocumentAgent         — публичный класс, используется из main.py
    _build_tools_for_claude() — конвертирует MCP-инструменты в формат Anthropic
    _run_react_loop()         — ReAct-цикл: reasoning → tool_call → result → repeat
    StateManager              — персистентное хранилище состояния (AsyncPostgresSaver)

Персистентность истории:
    Используется AsyncPostgresSaver из langgraph-checkpoint-postgres.
    База данных: CHECKPOINT_DB_URL (отдельная от основной БД).
    При недоступности PostgreSQL — fallback на MemorySaver с WARNING.

Инструменты:
    Агент работает с MCP-сервером через HTTP (не импортирует инструменты напрямую).
    Список доступных инструментов загружается из MCP_URL/tools при старте.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx

from config import settings

logger = logging.getLogger(__name__)

# ── Константы ─────────────────────────────────────────────────────────────
_MAX_TOOL_ITERATIONS = 10
_MCP_CALL_TIMEOUT = 30.0


# ── Датаклассы ────────────────────────────────────────────────────────────


@dataclass
class AgentResponse:
    """Стандартизированный ответ агента для main.py."""

    status: str = "success"
    content: str | None = None
    message: str | None = None
    action_type: str | None = None
    requires_reload: bool = False
    navigate_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Менеджер состояния с AsyncPostgresSaver ────────────────────────────────


class StateManager:
    """
    Персистентное хранилище состояния тредов через PostgreSQL.

    Использует AsyncPostgresSaver из langgraph-checkpoint-postgres.
    При недоступности — fallback на MemorySaver с предупреждением.

    Хранит только метаданные тредов (не историю сообщений —
    история сохраняется в conversation_logs через LongTermMemory).
    """

    def __init__(self) -> None:
        self._saver: Any = None
        self._conn: Any = None
        self._postgres_available = False

    async def initialize(self) -> None:
        """
        Инициализирует подключение к PostgreSQL для хранения checkpoint.

        Использует CHECKPOINT_DB_URL (отдельная схема от основной БД).
        Fallback на MemorySaver если PostgreSQL недоступен.
        """
        checkpoint_url = settings.CHECKPOINT_DB_URL or settings.DATABASE_URL

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            import psycopg

            # Конвертируем asyncpg URL → psycopg URL
            psycopg_url = checkpoint_url.replace(
                "postgresql+asyncpg://", "postgresql://"
            ).replace("postgresql+psycopg2://", "postgresql://")

            self._conn = await psycopg.AsyncConnection.connect(
                psycopg_url,
                autocommit=True,
            )
            self._saver = AsyncPostgresSaver(self._conn)
            await self._saver.setup()
            self._postgres_available = True
            logger.info(
                "StateManager: AsyncPostgresSaver initialized",
                extra={"db": psycopg_url.split("@")[-1] if "@" in psycopg_url else "local"},
            )

        except ImportError:
            logger.warning(
                "langgraph-checkpoint-postgres not installed. "
                "Install: pip install langgraph-checkpoint-postgres psycopg[binary]. "
                "Falling back to MemorySaver (non-persistent)."
            )
            self._init_memory_fallback()

        except Exception as exc:
            logger.warning(
                "AsyncPostgresSaver init failed: %s. Falling back to MemorySaver.",
                exc,
            )
            self._init_memory_fallback()

    def _init_memory_fallback(self) -> None:
        """Инициализирует MemorySaver как запасной вариант."""
        from langgraph.checkpoint.memory import MemorySaver

        self._saver = MemorySaver()
        self._postgres_available = False
        logger.warning(
            "StateManager: using MemorySaver (in-memory, non-persistent). "
            "Thread history will be lost on restart."
        )

    def get_saver(self) -> Any:
        """Возвращает checkpointer для передачи в граф."""
        if self._saver is None:
            self._init_memory_fallback()
        return self._saver

    async def close(self) -> None:
        """Закрывает соединение с PostgreSQL."""
        if self._conn is not None:
            try:
                await self._conn.close()
                logger.info("StateManager: PostgreSQL connection closed")
            except Exception as exc:
                logger.warning("StateManager: error closing connection: %s", exc)

    @property
    def is_persistent(self) -> bool:
        """True если используется персистентное PostgreSQL-хранилище."""
        return self._postgres_available


# ── MCP клиент ────────────────────────────────────────────────────────────


class MCPClient:
    """
    HTTP-клиент для взаимодействия с MCP-сервером.

    Загружает список инструментов при инициализации.
    Вызывает инструменты через POST /call.
    """

    def __init__(self, mcp_url: str) -> None:
        self._url = mcp_url.rstrip("/")
        self._tools_cache: list[dict[str, Any]] = []

    async def load_tools(self) -> list[dict[str, Any]]:
        """
        Загружает список инструментов с MCP-сервера.

        Возвращает список в формате Anthropic tools.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._url}/tools")
                resp.raise_for_status()
                data = resp.json()

            # FastMCP возвращает список инструментов в поле tools или напрямую
            raw_tools = data.get("tools", data) if isinstance(data, dict) else data

            # Конвертируем в формат Anthropic tool_use
            anthropic_tools = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("inputSchema") or t.get("input_schema") or {"type": "object", "properties": {}},
                }
                for t in raw_tools
                if isinstance(t, dict) and t.get("name")
            ]

            self._tools_cache = anthropic_tools
            logger.info("MCPClient: loaded %d tools from %s", len(anthropic_tools), self._url)
            return anthropic_tools

        except Exception as exc:
            logger.error("MCPClient: failed to load tools from %s: %s", self._url, exc)
            return []

    async def call_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """
        Вызывает MCP-инструмент через HTTP POST /call.

        Args:
            tool_name:  Имя инструмента (как зарегистрирован в MCP).
            tool_input: Аргументы инструмента.

        Returns:
            Результат выполнения инструмента или словарь с ошибкой.
        """
        try:
            async with httpx.AsyncClient(timeout=_MCP_CALL_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._url}/call",
                    json={"tool": tool_name, "args": tool_input},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException:
            logger.error("MCPClient: timeout calling tool '%s'", tool_name)
            return {"success": False, "error": {"code": "TIMEOUT", "message": f"Инструмент '{tool_name}' не ответил за {_MCP_CALL_TIMEOUT}s"}}
        except Exception as exc:
            logger.error("MCPClient: error calling tool '%s': %s", tool_name, exc)
            return {"success": False, "error": {"code": "CALL_ERROR", "message": str(exc)}}

    @property
    def cached_tools(self) -> list[dict[str, Any]]:
        """Кэшированный список инструментов в формате Anthropic."""
        return self._tools_cache


# ── Построитель системного промпта ────────────────────────────────────────

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _build_system_prompt(
    user_context: dict[str, Any],
    context_ui_id: str | None,
    file_path: str | None,
    file_name: str | None,
    human_choice: str | None,
    user_token: str,
) -> str:
    """
    Строит системный промпт с инжекцией контекста.

    ВАЖНО: токен авторизации передаётся в системный промпт,
    чтобы инструменты могли его использовать при вызове через MCP.
    """
    parts: list[str] = [
        "Ты — ИИ-ассистент корпоративной системы электронного документооборота (EDMS).",
        "Язык ответов: русский. Технические термины допустимы на английском.",
        "",
        "ПРАВИЛА:",
        "1. Всегда начинай с reasoning: <reasoning>что хочет пользователь → какой инструмент → какие параметры</reasoning>",
        "2. Используй только доступные тебе инструменты.",
        "3. При нехватке данных — задай уточняющий вопрос, не угадывай.",
        "4. Никогда не показывай пользователю HTTP-коды, stack traces, UUID внутренних объектов.",
        "5. Структура ответа: краткий итог → детали → следующие шаги (если нужно).",
    ]

    # Профиль пользователя
    first = user_context.get("firstName", "")
    last = user_context.get("lastName", "")
    name = f"{last} {first}".strip() or "Коллега"
    role = user_context.get("role") or user_context.get("authorPost") or ""
    dept = user_context.get("department") or ""
    profile = f"\nПОЛЬЗОВАТЕЛЬ: {name}"
    if role:
        profile += f" | {role}"
    if dept:
        profile += f" | {dept}"
    parts.append(profile)

    # Токен авторизации (нужен инструментам)
    parts.append(f"\n<auth_token>{user_token}</auth_token>")

    # Активный документ
    if context_ui_id:
        parts.append(
            f"\n<context_document_id>{context_ui_id}</context_document_id>\n"
            "Это UUID активного документа в UI. Используй его при запросах о 'текущем документе'."
        )

    # Загруженный файл
    if file_path:
        is_uuid = bool(_UUID_RE.match(file_path.strip()))
        display = file_name or file_path.split("/")[-1]
        if is_uuid:
            parts.append(
                f"\n<attachment_id>{file_path}</attachment_id>\n"
                "UUID вложения из EDMS для работы с содержимым."
            )
        else:
            parts.append(
                f"\n<local_file_path>{file_path}</local_file_path>\n"
                f"<local_file_name>{display}</local_file_name>\n"
                "Пользователь загрузил локальный файл."
            )

    # Явный выбор пользователя
    if human_choice:
        parts.append(
            f"\n<human_choice>{human_choice}</human_choice>\n"
            "Явный выбор пользователя из предложенных вариантов."
        )

    return "\n".join(parts)


# ── ReAct-цикл ────────────────────────────────────────────────────────────


async def _run_react_loop(
    client: anthropic.AsyncAnthropic,
    mcp_client: MCPClient,
    system_prompt: str,
    user_message: str,
    history: list[dict[str, Any]],
    model: str,
    max_iterations: int = _MAX_TOOL_ITERATIONS,
) -> tuple[str, list[dict[str, Any]]]:
    """
    ReAct-цикл: reasoning → tool_use → tool_result → repeat → end_turn.

    Args:
        client:         Anthropic AsyncAnthropic клиент.
        mcp_client:     MCP HTTP-клиент для вызова инструментов.
        system_prompt:  Системный промпт с контекстом.
        user_message:   Текущее сообщение пользователя.
        history:        История предыдущих сообщений треда.
        model:          Название модели Anthropic.
        max_iterations: Максимальное количество tool_use итераций.

    Returns:
        Кортеж (финальный_ответ_текст, обновлённая_история).
    """
    tools = mcp_client.cached_tools
    messages = list(history) + [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        logger.debug("ReAct iteration %d/%d", iteration + 1, max_iterations)

        # Вызов LLM
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await client.messages.create(**kwargs)

        # Добавляем ответ ассистента в историю
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Проверяем stop_reason
        if response.stop_reason == "end_turn":
            # Извлекаем финальный текст
            final_text = ""
            for block in assistant_content:
                if hasattr(block, "text") and block.text:
                    final_text += block.text
            return final_text.strip() or "Готово.", messages

        if response.stop_reason != "tool_use":
            # Неожиданный stop_reason — возвращаем что есть
            text = ""
            for block in assistant_content:
                if hasattr(block, "text") and block.text:
                    text += block.text
            return text.strip() or "Готово.", messages

        # Обрабатываем tool_use блоки
        tool_results: list[dict[str, Any]] = []

        for block in assistant_content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input or {}

            logger.info(
                "Calling MCP tool: %s",
                tool_name,
                extra={"tool": tool_name, "input_keys": list(tool_input.keys())},
            )

            tool_result = await mcp_client.call_tool(tool_name, tool_input)

            # Сериализуем результат для Anthropic
            import json as _json
            result_content = _json.dumps(tool_result, ensure_ascii=False, default=str)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_content,
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            # tool_use без результатов — выходим
            break

    # Превышен лимит итераций
    logger.warning("ReAct loop reached max iterations (%d)", max_iterations)
    # Запрашиваем финальный ответ без инструментов
    final_response = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt + "\n\nПодведи итог выполненных действий кратко.",
        messages=messages,
    )
    text = ""
    for block in final_response.content:
        if hasattr(block, "text"):
            text += block.text
    return text.strip() or "Операция выполнена.", messages


# ── Маршрутизатор моделей ─────────────────────────────────────────────────


def _select_model(
    intent: str,
    confidence: float,
    bypass_llm: bool,
    is_write: bool,
) -> str:
    """
    Выбирает оптимальную модель Anthropic по сложности запроса.

    Логика:
        bypass_llm или (высокая уверенность + один инструмент + не write)
            → claude-haiku-4-5  (быстро, дёшево)
        write-операции или средняя сложность
            → claude-sonnet-4-6 (баланс)
        неизвестный intent или сложный workflow
            → claude-opus-4-5   (максимальное качество)
    """
    _SIMPLE_INTENTS = {
        "get_document", "get_history", "get_workflow_status", "get_analytics",
    }

    if bypass_llm or (
        intent in _SIMPLE_INTENTS
        and confidence > 0.85
        and not is_write
    ):
        return "claude-haiku-4-5"

    if is_write or intent in {"update_status", "assign_document", "search_documents"}:
        return "claude-sonnet-4-6"

    # unknown, composite, create_document, complex workflows
    return "claude-opus-4-5"


# ── Главный класс агента ──────────────────────────────────────────────────


class EdmsDocumentAgent:
    """
    Главный ИИ-агент EDMS AI Assistant.

    Использует нативный Anthropic SDK (tool_use) для взаимодействия с LLM.
    Инструменты вызываются через MCP HTTP-клиент (не импортируются напрямую).
    История хранится в AsyncPostgresSaver (PostgreSQL).

    Публичный API:
        await agent.initialize()   — вызвать один раз при старте приложения
        await agent.chat(...)      — обработать сообщение пользователя
        await agent.close()        — освободить ресурсы при остановке
        agent.health_check()       — статус компонентов
    """

    def __init__(self) -> None:
        self.state_manager = StateManager()
        self._mcp_client = MCPClient(str(settings.MCP_URL))
        self._anthropic: anthropic.AsyncAnthropic | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Инициализирует агент: PostgreSQL, Anthropic клиент, MCP инструменты.

        Вызывается один раз из lifespan() в main.py.
        """
        # Anthropic клиент
        api_key = None
        if settings.ANTHROPIC_API_KEY:
            api_key = settings.ANTHROPIC_API_KEY.get_secret_value()

        self._anthropic = anthropic.AsyncAnthropic(api_key=api_key)

        # Персистентное состояние
        await self.state_manager.initialize()

        # Загружаем инструменты с MCP-сервера
        tools = await self._mcp_client.load_tools()
        if not tools:
            logger.warning(
                "No tools loaded from MCP server at %s. "
                "Agent will work without tools.",
                settings.MCP_URL,
            )

        self._initialized = True
        logger.info(
            "EdmsDocumentAgent initialized",
            extra={
                "mcp_tools": len(tools),
                "persistent_state": self.state_manager.is_persistent,
                "mcp_url": str(settings.MCP_URL),
            },
        )

    async def close(self) -> None:
        """Освобождает ресурсы. Вызывается из lifespan() при остановке."""
        await self.state_manager.close()
        if self._anthropic:
            await self._anthropic.close()
        logger.info("EdmsDocumentAgent closed")

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
        Обрабатывает входящее сообщение пользователя.

        Args:
            message:       Текст запроса.
            user_token:    JWT-токен для авторизации в EDMS API.
            context_ui_id: UUID активного документа в UI.
            thread_id:     ID треда для персистентной истории.
            user_context:  Профиль пользователя {firstName, lastName, role, ...}.
            file_path:     Путь к файлу или UUID вложения.
            file_name:     Отображаемое имя файла.
            human_choice:  Явный выбор пользователя (disambiguation).

        Returns:
            dict со статусом и контентом ответа.
        """
        if not self._initialized:
            await self.initialize()

        start_ts = time.monotonic()
        ctx = user_context or {}

        # NLU-анализ для маршрутизации модели
        from services.nlp_service import SemanticDispatcher

        dispatcher = SemanticDispatcher()
        semantic = dispatcher.build_context(message=message, file_path=file_path)
        intent = semantic.query.intent.value
        confidence = semantic.query.confidence
        bypass_llm = semantic.query.intent.value in {
            "get_document", "get_history", "get_workflow_status",
        } and confidence > 0.92

        _WRITE_INTENTS = {"create_document", "update_status", "assign_document"}
        is_write = intent in _WRITE_INTENTS

        model = _select_model(intent, confidence, bypass_llm, is_write)

        logger.info(
            "Agent chat",
            extra={
                "thread_id": thread_id,
                "intent": intent,
                "confidence": round(confidence, 2),
                "model": model,
                "has_file": bool(file_path),
                "bypass_llm": bypass_llm,
            },
        )

        # Загружаем историю треда из PostgreSQL
        history = await self._load_thread_history(thread_id)

        # Системный промпт
        system_prompt = _build_system_prompt(
            user_context=ctx,
            context_ui_id=context_ui_id,
            file_path=file_path,
            file_name=file_name,
            human_choice=human_choice,
            user_token=user_token,
        )

        # ReAct-цикл
        try:
            final_text, updated_history = await asyncio.wait_for(
                _run_react_loop(
                    client=self._anthropic,
                    mcp_client=self._mcp_client,
                    system_prompt=system_prompt,
                    user_message=message,
                    history=history,
                    model=model,
                    max_iterations=settings.AGENT_MAX_ITERATIONS,
                ),
                timeout=settings.AGENT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("Agent timeout after %.1fs", settings.AGENT_TIMEOUT)
            return {
                "status": "error",
                "content": "Превышено время ожидания. Попробуйте переформулировать запрос.",
                "requires_reload": False,
                "metadata": {},
            }
        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc, exc_info=True)
            return {
                "status": "error",
                "content": "Ошибка связи с языковой моделью. Попробуйте позже.",
                "requires_reload": False,
                "metadata": {},
            }
        except Exception as exc:
            logger.error("Agent unexpected error: %s", exc, exc_info=True)
            return {
                "status": "error",
                "content": "Произошла ошибка при обработке запроса. Попробуйте ещё раз.",
                "requires_reload": False,
                "metadata": {},
            }

        # Сохраняем историю треда
        await self._save_thread_history(thread_id, updated_history)

        # Извлекаем навигационные данные из истории tool_results
        navigate_url, requires_reload = self._extract_nav_meta(updated_history)

        elapsed_ms = round((time.monotonic() - start_ts) * 1000)
        return {
            "status": "success",
            "content": final_text,
            "navigate_url": navigate_url,
            "requires_reload": requires_reload,
            "metadata": {
                "latency_ms": elapsed_ms,
                "model": model,
                "intent": intent,
                "mcp_tools_available": len(self._mcp_client.cached_tools),
            },
        }

    async def _load_thread_history(
        self,
        thread_id: str | None,
    ) -> list[dict[str, Any]]:
        """
        Загружает историю сообщений треда из checkpointer.

        Возвращает список сообщений в формате Anthropic API
        (без системного промпта — он передаётся отдельно).
        Обрезает историю до AGENT_MAX_CONTEXT_MESSAGES.
        """
        if not thread_id:
            return []

        try:
            saver = self.state_manager.get_saver()

            # MemorySaver и AsyncPostgresSaver имеют разный API
            if hasattr(saver, "aget"):
                from langchain_core.runnables import RunnableConfig
                config = RunnableConfig(configurable={"thread_id": thread_id})
                state = await saver.aget(config)
            elif hasattr(saver, "get"):
                from langchain_core.runnables import RunnableConfig
                import asyncio as _asyncio
                config = RunnableConfig(configurable={"thread_id": thread_id})
                loop = _asyncio.get_event_loop()
                state = await loop.run_in_executor(None, saver.get, config)
            else:
                return []

            if not state:
                return []

            # Извлекаем историю из состояния checkpoint
            messages = state.values.get("messages", []) if hasattr(state, "values") else []

            # Конвертируем LangChain messages → Anthropic messages
            history: list[dict[str, Any]] = []
            for msg in messages:
                role = getattr(msg, "type", None)
                content = getattr(msg, "content", "")
                if role == "human":
                    history.append({"role": "user", "content": content})
                elif role == "ai" and content:
                    history.append({"role": "assistant", "content": content})

            # Обрезаем до лимита
            max_msgs = settings.AGENT_MAX_CONTEXT_MESSAGES
            if len(history) > max_msgs:
                history = history[-max_msgs:]

            return history

        except Exception as exc:
            logger.warning("Failed to load thread history for '%s': %s", thread_id, exc)
            return []

    async def _save_thread_history(
        self,
        thread_id: str | None,
        messages: list[dict[str, Any]],
    ) -> None:
        """
        Сохраняет историю сообщений в checkpointer.

        Конвертирует Anthropic messages → LangChain messages для совместимости.
        """
        if not thread_id or not messages:
            return

        try:
            from langchain_core.messages import AIMessage, HumanMessage
            from langgraph.graph import StateGraph, END
            from model import AgentState

            lc_messages = []
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        lc_messages.append(HumanMessage(content=content))
                elif msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content:
                        lc_messages.append(AIMessage(content=content))

            if not lc_messages:
                return

            # Сохраняем через checkpointer напрямую
            saver = self.state_manager.get_saver()
            from langchain_core.runnables import RunnableConfig

            config = RunnableConfig(configurable={"thread_id": thread_id})
            checkpoint = {
                "v": 1,
                "ts": str(time.time()),
                "channel_values": {"messages": lc_messages},
                "channel_versions": {"messages": 1},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata = {"source": "agent", "step": len(lc_messages)}

            if hasattr(saver, "aput"):
                await saver.aput(config, checkpoint, metadata, {})
            elif hasattr(saver, "put"):
                import asyncio as _asyncio
                loop = _asyncio.get_event_loop()
                await loop.run_in_executor(None, saver.put, config, checkpoint, metadata, {})

        except Exception as exc:
            logger.warning("Failed to save thread history for '%s': %s", thread_id, exc)

    @staticmethod
    def _extract_nav_meta(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, bool]:
        """
        Ищет navigate_url и requires_reload в результатах tool_use.

        Сканирует tool_result блоки в истории сообщений.
        """
        import json as _json

        navigate_url: str | None = None
        requires_reload = False

        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                raw = block.get("content", "")
                if not isinstance(raw, str):
                    continue
                try:
                    data = _json.loads(raw)
                    if isinstance(data, dict):
                        if data.get("navigate_url") and not navigate_url:
                            navigate_url = data["navigate_url"]
                        if data.get("requires_reload"):
                            requires_reload = True
                except (ValueError, TypeError):
                    pass

        return navigate_url, requires_reload

    def health_check(self) -> dict[str, Any]:
        """Возвращает статус всех компонентов агента."""
        return {
            "initialized": self._initialized,
            "mcp_url": str(settings.MCP_URL),
            "mcp_tools": len(self._mcp_client.cached_tools),
            "persistent_state": self.state_manager.is_persistent,
            "max_iterations": settings.AGENT_MAX_ITERATIONS,
            "agent_timeout": settings.AGENT_TIMEOUT,
        }
