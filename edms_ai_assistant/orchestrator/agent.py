# edms_ai_assistant/orchestrator/agent.py
"""
EDMS AI Assistant — главный агент оркестратора.

Использует LLMClient (llm_client.py) — поддерживает Ollama и Anthropic.
Модели берутся из .env: MODEL_PLANNER, MODEL_EXECUTOR, MODEL_EXPLAINER и т.д.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from edms_ai_assistant.config import settings
from edms_ai_assistant.llm_client import LLMClient, LLMResponse, get_llm_client

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 10
_MCP_CALL_TIMEOUT = 30.0

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


# ── Датаклассы ────────────────────────────────────────────────────────────


@dataclass
class AgentResponse:
    """Стандартизированный ответ агента."""

    status: str = "success"
    content: str | None = None
    message: str | None = None
    action_type: str | None = None
    requires_reload: bool = False
    navigate_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── StateManager ──────────────────────────────────────────────────────────


class StateManager:
    """Персистентное хранилище истории диалогов.

    Primary:  AsyncPostgresSaver (LangGraph checkpoint-postgres)
    Fallback: MemorySaver (in-memory)
    """

    def __init__(self) -> None:
        self._saver: Any = None
        self._conn: Any = None
        self._postgres_available = False

    async def initialize(self) -> None:
        """Инициализирует хранилище состояний."""
        checkpoint_url = settings.CHECKPOINT_DB_URL or settings.DATABASE_URL
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            import psycopg

            psycopg_url = checkpoint_url.replace(
                "postgresql+asyncpg://", "postgresql://"
            ).replace("postgresql+psycopg2://", "postgresql://")

            self._conn = await psycopg.AsyncConnection.connect(
                psycopg_url, autocommit=True
            )
            self._saver = AsyncPostgresSaver(self._conn)
            await self._saver.setup()
            self._postgres_available = True
            logger.info("StateManager: AsyncPostgresSaver initialized")

        except ImportError:
            logger.warning(
                "langgraph-checkpoint-postgres не установлен. Используется MemorySaver."
            )
            self._init_memory_fallback()

        except Exception as exc:
            exc_str = str(exc)
            if "ProactorEventLoop" in exc_str:
                logger.warning(
                    "Windows ProactorEventLoop несовместим с psycopg. "
                    "Запустите через main_entrypoint.py для PostgreSQL checkpoint. "
                    "Используется MemorySaver."
                )
            else:
                logger.warning(
                    "AsyncPostgresSaver init failed: %s. Используется MemorySaver.", exc
                )
            self._init_memory_fallback()

    def _init_memory_fallback(self) -> None:
        """Инициализирует in-memory fallback."""
        from langgraph.checkpoint.memory import MemorySaver

        self._saver = MemorySaver()
        self._postgres_available = False
        logger.warning("StateManager: используется MemorySaver (не персистентен)")

    def get_saver(self) -> Any:
        if self._saver is None:
            self._init_memory_fallback()
        return self._saver

    async def get_state(self, thread_id: str) -> Any:
        """Возвращает текущий snapshot состояния для треда."""
        saver = self.get_saver()
        try:
            config = {"configurable": {"thread_id": thread_id}}
            if hasattr(saver, "aget"):
                state = await saver.aget(config)
            elif hasattr(saver, "get"):
                loop = asyncio.get_event_loop()
                state = await loop.run_in_executor(None, saver.get, config)
            else:
                return _EmptyState()
            return state if state is not None else _EmptyState()
        except Exception as exc:
            logger.warning(
                "StateManager.get_state failed for thread '%s': %s", thread_id, exc
            )
            return _EmptyState()

    async def close(self) -> None:
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass

    @property
    def is_persistent(self) -> bool:
        return self._postgres_available


class _EmptyState:
    values: dict[str, Any] = {}
    next: tuple[()] = ()


# ── MCPClient ─────────────────────────────────────────────────────────────


class MCPClient:
    """HTTP-клиент для взаимодействия с MCP-сервером.

    fastmcp 2.x изменил эндпоинты:
      - /tools     → не существует
      - /mcp       → основной MCP эндпоинт (StreamableHTTP)
      - Список инструментов получаем через MCP initialize/tools/list
    """

    def __init__(self, mcp_url: str) -> None:
        self._base_url = mcp_url.rstrip("/")
        self._tools_cache: list[dict[str, Any]] = []

    async def load_tools(self) -> list[dict[str, Any]]:
        """Загружает список инструментов из MCP-сервера.

        fastmcp 2.x: инструменты доступны через POST /mcp с MCP-протоколом
        или через GET /mcp/tools (зависит от версии).
        Пробуем несколько вариантов эндпоинтов.
        """
        # Список эндпоинтов для попытки — от новых к старым
        candidates = [
            f"{self._base_url}/mcp",         # fastmcp 2.x StreamableHTTP
            f"{self._base_url}/tools",        # старый формат / кастомный
            f"{self._base_url}/",             # корневой
        ]

        # Сначала пробуем получить через MCP JSON-RPC протокол
        tools = await self._load_via_mcp_protocol()
        if tools:
            self._tools_cache = tools
            logger.info("MCPClient: загружено %d инструментов (MCP protocol)", len(tools))
            return tools

        # Fallback: пробуем REST-эндпоинты
        for url in candidates:
            tools = await self._try_rest_endpoint(url)
            if tools is not None:
                self._tools_cache = tools
                logger.info(
                    "MCPClient: загружено %d инструментов из %s", len(tools), url
                )
                return tools

        logger.error(
            "MCPClient: не удалось загрузить инструменты ни с одного эндпоинта"
        )
        return []

    async def _load_via_mcp_protocol(self) -> list[dict[str, Any]] | None:
        """Загружает инструменты через MCP JSON-RPC протокол (StreamableHTTP).

        fastmcp 2.x использует StreamableHTTP транспорт.
        Отправляем tools/list запрос.
        """
        mcp_endpoint = f"{self._base_url}/mcp"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # MCP initialize
                init_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "edms-orchestrator", "version": "1.0"},
                    },
                }
                resp = await client.post(
                    mcp_endpoint,
                    json=init_payload,
                    headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                )
                if resp.status_code not in (200, 201):
                    return None

                # MCP tools/list
                list_payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                }
                resp2 = await client.post(
                    mcp_endpoint,
                    json=list_payload,
                    headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                )
                if resp2.status_code not in (200, 201):
                    return None

                # Ответ может быть SSE или plain JSON
                content = resp2.text
                data = _parse_mcp_response(content)
                if data is None:
                    return None

                raw_tools = (
                    data.get("result", {}).get("tools", [])
                    if isinstance(data, dict)
                    else []
                )
                return _normalize_tools(raw_tools)

        except Exception as exc:
            logger.debug("MCP protocol load failed: %s", exc)
            return None

    async def _try_rest_endpoint(self, url: str) -> list[dict[str, Any]] | None:
        """Пробует загрузить инструменты с REST-эндпоинта."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return None
                data = resp.json()
            raw_tools = data.get("tools", data) if isinstance(data, dict) else data
            if not isinstance(raw_tools, list):
                return None
            result = _normalize_tools(raw_tools)
            return result if result else None
        except Exception as exc:
            logger.debug("REST endpoint %s failed: %s", url, exc)
            return None

    async def call_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Вызывает инструмент MCP-сервера через JSON-RPC протокол."""
        mcp_endpoint = f"{self._base_url}/mcp"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_input,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=_MCP_CALL_TIMEOUT) as client:
                resp = await client.post(
                    mcp_endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                )
                resp.raise_for_status()
                data = _parse_mcp_response(resp.text)
                if data is None:
                    return {"success": False, "error": "Empty response"}

                # Извлекаем результат из MCP-ответа
                if isinstance(data, dict):
                    if "error" in data:
                        return {
                            "success": False,
                            "error": data["error"].get("message", "MCP error"),
                        }
                    result = data.get("result", data)
                    # MCP возвращает content как список блоков
                    if isinstance(result, dict) and "content" in result:
                        content_blocks = result["content"]
                        if isinstance(content_blocks, list) and content_blocks:
                            text = content_blocks[0].get("text", "")
                            try:
                                return _json.loads(text)
                            except Exception:
                                return {"success": True, "data": text}
                    return result

                return {"success": True, "data": data}

        except httpx.TimeoutException:
            logger.error("MCPClient: таймаут при вызове инструмента '%s'", tool_name)
            return {
                "success": False,
                "error": {"code": "TIMEOUT", "message": f"Tool '{tool_name}' timeout"},
            }
        except Exception as exc:
            logger.error("MCPClient: ошибка вызова '%s': %s", tool_name, exc)
            return {"success": False, "error": {"code": "CALL_ERROR", "message": str(exc)}}

    @property
    def cached_tools(self) -> list[dict[str, Any]]:
        return self._tools_cache


def _parse_mcp_response(text: str) -> Any:
    """Парсит ответ MCP-сервера — может быть plain JSON или SSE."""
    text = text.strip()
    if not text:
        return None
    # SSE формат: строки начинаются с "data: "
    if text.startswith("data:"):
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        return _json.loads(payload)
                    except Exception:
                        continue
        return None
    # Plain JSON
    try:
        return _json.loads(text)
    except Exception:
        return None


def _normalize_tools(raw_tools: list) -> list[dict[str, Any]]:
    """Нормализует инструменты в формат Anthropic tool schema."""
    result = []
    for t in raw_tools:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        result.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": (
                t.get("inputSchema")
                or t.get("input_schema")
                or {"type": "object", "properties": {}}
            ),
        })
    return result


# ── Системный промпт ──────────────────────────────────────────────────────


def _build_system_prompt(
    user_context: dict[str, Any],
    context_ui_id: str | None,
    file_path: str | None,
    file_name: str | None,
    human_choice: str | None,
    user_token: str,
) -> str:
    """Собирает системный промпт для LLM из контекста запроса."""
    parts: list[str] = [
        "Ты — ИИ-ассистент корпоративной системы электронного документооборота (EDMS).",
        "Язык ответов: русский. Технические термины допустимы на английском.",
        "",
        "ПРАВИЛА:",
        "1. Всегда начинай с reasoning: "
        "<reasoning>что хочет пользователь → какой инструмент → какие параметры</reasoning>",
        "2. Используй только доступные тебе инструменты.",
        "3. При нехватке данных — задай уточняющий вопрос, не угадывай.",
        "4. Никогда не показывай пользователю HTTP-коды, stack traces, UUID внутренних объектов.",
        "5. Структура ответа: краткий итог → детали → следующие шаги (если нужно).",
    ]

    first = user_context.get("firstName", "")
    last = user_context.get("lastName", "")
    name = f"{last} {first}".strip() or "Коллега"
    role = user_context.get("role") or user_context.get("authorPost") or ""
    dept = user_context.get("department") or user_context.get("authorDepartmentName") or ""

    profile = f"\nПОЛЬЗОВАТЕЛЬ: {name}"
    if role:
        profile += f" | {role}"
    if dept:
        profile += f" | {dept}"
    parts.append(profile)

    parts.append(f"\n<auth_token>{user_token}</auth_token>")

    if context_ui_id:
        parts.append(
            f"\n<context_document_id>{context_ui_id}</context_document_id>\n"
            "Это UUID активного документа в UI. "
            "Используй его при запросах о 'текущем документе'."
        )

    if file_path:
        is_uuid = bool(_UUID_RE.match(file_path.strip()))
        display = file_name or file_path.split("/")[-1]
        if is_uuid:
            parts.append(
                f"\n<attachment_id>{file_path}</attachment_id>\nUUID вложения из EDMS."
            )
        else:
            parts.append(
                f"\n<local_file_path>{file_path}</local_file_path>\n"
                f"<local_file_name>{display}</local_file_name>\n"
                "Пользователь загрузил локальный файл."
            )

    if human_choice:
        parts.append(
            f"\n<human_choice>{human_choice}</human_choice>\nЯвный выбор пользователя."
        )

    return "\n".join(parts)


# ── Выбор модели ──────────────────────────────────────────────────────────


def _select_model_role(
    intent: str,
    confidence: float,
    bypass_llm: bool,
    is_write: bool,
) -> str:
    """Определяет роль агента для выбора LLM-модели."""
    _SIMPLE_INTENTS = {
        "get_document", "get_history", "get_workflow_status", "get_analytics",
    }
    if bypass_llm or (intent in _SIMPLE_INTENTS and confidence > 0.85 and not is_write):
        return "explainer"
    if is_write or intent in {"update_status", "assign_document", "search_documents"}:
        return "executor"
    return "planner"


# ── ReAct-цикл ────────────────────────────────────────────────────────────


async def _run_react_loop(
    llm_client: LLMClient,
    mcp_client: MCPClient,
    system_prompt: str,
    user_message: str,
    history: list[dict[str, Any]],
    model_role: str,
    max_iterations: int = _MAX_TOOL_ITERATIONS,
) -> tuple[str, list[dict[str, Any]]]:
    """ReAct-цикл: reasoning → tool_use → tool_result → repeat → end_turn."""
    tools = mcp_client.cached_tools
    model_name = llm_client.model_for_role(model_role)
    messages = list(history) + [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        logger.debug(
            "ReAct iteration %d/%d model=%s", iteration + 1, max_iterations, model_name
        )

        response: LLMResponse = await llm_client.create(
            model=model_name,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=system_prompt,
            messages=messages,
            tools=tools if tools else None,
        )

        assistant_content: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text" and block.text:
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input or {},
                })

        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            final_text = "".join(
                b["text"] for b in assistant_content if b.get("type") == "text"
            )
            return final_text.strip() or "Готово.", messages

        if response.stop_reason != "tool_use":
            text = "".join(
                b["text"] for b in assistant_content if b.get("type") == "text"
            )
            return text.strip() or "Готово.", messages

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            logger.info("Вызов MCP tool: %s args=%s", block.name, block.input)
            tool_result = await mcp_client.call_tool(block.name, block.input or {})
            result_content = _json.dumps(tool_result, ensure_ascii=False, default=str)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_content,
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    # Превышен лимит — финальный ответ
    logger.warning("ReAct loop достиг максимума итераций (%d)", max_iterations)
    final_response = await llm_client.create(
        model=model_name,
        max_tokens=1024,
        system=system_prompt + "\n\nПодведи итог выполненных действий кратко.",
        messages=messages,
    )
    text = "".join(b.text for b in final_response.content if b.type == "text")
    return text.strip() or "Операция выполнена.", messages


# ── Главный класс агента ──────────────────────────────────────────────────


class EdmsDocumentAgent:
    """Главный ИИ-агент EDMS AI Assistant."""

    def __init__(self) -> None:
        self.state_manager = StateManager()
        self._mcp_client = MCPClient(str(settings.MCP_URL))
        self._llm: LLMClient = get_llm_client()
        self._initialized = False

    async def initialize(self) -> None:
        """Инициализирует агент: StateManager + загрузка MCP-инструментов."""
        await self.state_manager.initialize()
        tools = await self._mcp_client.load_tools()
        if not tools:
            logger.warning(
                "Инструменты не загружены из MCP-сервера %s. "
                "Агент будет работать без инструментов.",
                settings.MCP_URL,
            )
        self._initialized = True
        logger.info(
            "EdmsDocumentAgent инициализирован: backend=%s mcp_tools=%d persistent=%s",
            self._llm.backend,
            len(tools),
            self.state_manager.is_persistent,
        )

    async def close(self) -> None:
        await self.state_manager.close()
        logger.info("EdmsDocumentAgent остановлен")

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
        """Основная точка входа для диалога с агентом."""
        if not self._initialized:
            await self.initialize()

        start_ts = time.monotonic()
        ctx = user_context or {}

        from edms_ai_assistant.orchestrator.services.nlp_service import SemanticDispatcher

        dispatcher = SemanticDispatcher()
        semantic = dispatcher.build_context(message=message, file_path=file_path)
        intent = semantic.query.intent.value
        confidence = semantic.query.confidence

        _WRITE_INTENTS = {"create_document", "update_status", "assign_document"}
        is_write = intent in _WRITE_INTENTS
        bypass_llm = (
            intent in {"get_document", "get_history", "get_workflow_status"}
            and confidence > 0.92
        )

        model_role = _select_model_role(intent, confidence, bypass_llm, is_write)

        logger.info(
            "Agent chat: thread=%s intent=%s confidence=%.2f model_role=%s backend=%s",
            thread_id, intent, round(confidence, 2), model_role, self._llm.backend,
        )

        history = await self._load_thread_history(thread_id)

        system_prompt = _build_system_prompt(
            user_context=ctx,
            context_ui_id=context_ui_id,
            file_path=file_path,
            file_name=file_name,
            human_choice=human_choice,
            user_token=user_token,
        )

        try:
            final_text, updated_history = await asyncio.wait_for(
                _run_react_loop(
                    llm_client=self._llm,
                    mcp_client=self._mcp_client,
                    system_prompt=system_prompt,
                    user_message=message,
                    history=history,
                    model_role=model_role,
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
        except Exception as exc:
            logger.error("Agent error: %s", exc, exc_info=True)
            return {
                "status": "error",
                "content": "Произошла ошибка при обработке запроса. Попробуйте ещё раз.",
                "requires_reload": False,
                "metadata": {},
            }

        await self._save_thread_history(thread_id, updated_history)
        navigate_url, requires_reload = self._extract_nav_meta(updated_history)

        elapsed_ms = round((time.monotonic() - start_ts) * 1000)
        model_used = self._llm.model_for_role(model_role)

        return {
            "status": "success",
            "content": final_text,
            "navigate_url": navigate_url,
            "requires_reload": requires_reload,
            "metadata": {
                "latency_ms": elapsed_ms,
                "model": model_used,
                "backend": self._llm.backend,
                "intent": intent,
                "mcp_tools_available": len(self._mcp_client.cached_tools),
            },
        }

    async def _load_thread_history(
        self, thread_id: str | None
    ) -> list[dict[str, Any]]:
        """Загружает историю диалога из StateManager."""
        if not thread_id:
            return []
        try:
            saver = self.state_manager.get_saver()
            config = {"configurable": {"thread_id": thread_id}}

            if hasattr(saver, "aget"):
                state = await saver.aget(config)
            elif hasattr(saver, "get"):
                loop = asyncio.get_event_loop()
                state = await loop.run_in_executor(None, saver.get, config)
            else:
                return []

            if not state:
                return []

            messages_raw = (
                state.values.get("messages", []) if hasattr(state, "values") else []
            )
            history: list[dict[str, Any]] = []
            for msg in messages_raw:
                role = getattr(msg, "type", None)
                content = getattr(msg, "content", "")
                if role == "human":
                    history.append({"role": "user", "content": content})
                elif role == "ai" and content:
                    history.append({"role": "assistant", "content": content})

            max_msgs = settings.AGENT_MAX_CONTEXT_MESSAGES
            if len(history) > max_msgs:
                history = history[-max_msgs:]
            return history

        except Exception as exc:
            logger.warning(
                "Не удалось загрузить историю треда '%s': %s", thread_id, exc
            )
            return []

    async def _save_thread_history(
        self, thread_id: str | None, messages: list[dict[str, Any]]
    ) -> None:
        """Сохраняет историю диалога в StateManager.

        Фикс 'checkpoint_ns': новые версии langgraph MemorySaver требуют
        поле checkpoint_ns в конфиге. Используем прямое хранение в dict
        вместо формата checkpoint для MemorySaver.
        """
        if not thread_id or not messages:
            return

        saver = self.state_manager.get_saver()

        # Для MemorySaver используем упрощённое хранение через внутренний storage
        # вместо формата langgraph checkpoint (который меняется между версиями)
        if not self.state_manager.is_persistent:
            self._save_to_memory_saver(saver, thread_id, messages)
            return

        # Для AsyncPostgresSaver — полный формат checkpoint
        try:
            from langchain_core.messages import AIMessage, HumanMessage

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

            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",   # обязательное поле в новых версиях
                    "checkpoint_id": "",
                }
            }
            checkpoint = {
                "v": 1,
                "ts": str(time.time()),
                "id": thread_id,
                "channel_values": {"messages": lc_messages},
                "channel_versions": {"messages": 1},
                "versions_seen": {"__input__": {}, "__start__": {"__start__": 1}},
                "pending_sends": [],
            }
            metadata = {"source": "agent", "step": len(lc_messages), "writes": {}}

            if hasattr(saver, "aput"):
                await saver.aput(config, checkpoint, metadata, {})
            elif hasattr(saver, "put"):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, saver.put, config, checkpoint, metadata, {})

        except Exception as exc:
            logger.warning(
                "Не удалось сохранить историю треда '%s': %s", thread_id, exc
            )

    def _save_to_memory_saver(
        self,
        saver: Any,
        thread_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Сохраняет историю в MemorySaver через его внутренний storage.

        MemorySaver хранит данные в self.storage (dict).
        Мы записываем напрямую, минуя версионированный checkpoint-формат,
        который несовместим между версиями langgraph.
        """
        try:
            # MemorySaver.storage — это dict вида {thread_id: {ns: {cid: checkpoint}}}
            storage = getattr(saver, "storage", None)
            if storage is None:
                return

            from langchain_core.messages import AIMessage, HumanMessage

            lc_messages = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if not isinstance(content, str):
                    continue
                if role == "user":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant" and content:
                    lc_messages.append(AIMessage(content=content))

            if not lc_messages:
                return

            checkpoint_id = f"chk_{thread_id}"
            checkpoint = {
                "v": 1,
                "ts": str(time.time()),
                "id": checkpoint_id,
                "channel_values": {"messages": lc_messages},
                "channel_versions": {"messages": len(lc_messages)},
                "versions_seen": {},
                "pending_sends": [],
            }

            # Структура storage: {thread_id: {"": {checkpoint_id: (checkpoint, metadata, {})}}}
            if thread_id not in storage:
                storage[thread_id] = {}
            if "" not in storage[thread_id]:
                storage[thread_id][""] = {}
            storage[thread_id][""][checkpoint_id] = (checkpoint, {"source": "agent"}, {})

        except Exception as exc:
            logger.debug("_save_to_memory_saver failed: %s", exc)

    @staticmethod
    def _extract_nav_meta(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, bool]:
        """Извлекает navigate_url и requires_reload из результатов инструментов."""
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
        return {
            "initialized": self._initialized,
            "mcp_url": str(settings.MCP_URL),
            "mcp_tools": len(self._mcp_client.cached_tools),
            "persistent_state": self.state_manager.is_persistent,
            "llm_backend": self._llm.backend,
            "max_iterations": settings.AGENT_MAX_ITERATIONS,
            "agent_timeout": settings.AGENT_TIMEOUT,
        }