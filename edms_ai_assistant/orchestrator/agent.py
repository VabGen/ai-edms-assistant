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
    Fallback: SimpleMemoryStore — простое in-memory хранилище без langgraph,
              чтобы избежать несовместимости форматов между версиями.
    """

    def __init__(self) -> None:
        self._saver: Any = None
        self._conn: Any = None
        self._postgres_available = False
        # Простое fallback-хранилище: {thread_id: list[dict]}
        self._simple_store: dict[str, list[dict[str, Any]]] = {}

    async def initialize(self) -> None:
        """Инициализирует хранилище состояний."""
        checkpoint_url = settings.CHECKPOINT_DB_URL or settings.DATABASE_URL
        try:
            import psycopg
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

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
                "langgraph-checkpoint-postgres не установлен. Используется SimpleMemoryStore."
            )
        except Exception as exc:
            exc_str = str(exc)
            if "ProactorEventLoop" in exc_str:
                logger.warning(
                    "Windows ProactorEventLoop несовместим с psycopg. "
                    "Запустите через main_entrypoint.py для PostgreSQL checkpoint. "
                    "Используется SimpleMemoryStore."
                )
            else:
                logger.warning(
                    "AsyncPostgresSaver init failed: %s. Используется SimpleMemoryStore.",
                    exc,
                )

        if not self._postgres_available:
            logger.warning(
                "StateManager: используется SimpleMemoryStore (не персистентен)"
            )

    def get_saver(self) -> Any:
        return self._saver

    async def get_state(self, thread_id: str) -> Any:
        """Возвращает текущий snapshot состояния для треда."""
        if not self._postgres_available or self._saver is None:
            return _EmptyState()
        try:
            config = {"configurable": {"thread_id": thread_id}}
            if hasattr(self._saver, "aget"):
                state = await self._saver.aget(config)
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
    """HTTP-клиент для взаимодействия с MCP-сервером (fastmcp 2.x StreamableHTTP)."""

    def __init__(self, mcp_url: str) -> None:
        self._base_url = mcp_url.rstrip("/")
        self._tools_cache: list[dict[str, Any]] = []
        self._session_id: str | None = None

    async def load_tools(self) -> list[dict[str, Any]]:
        """Загружает список инструментов из MCP-сервера через MCP JSON-RPC протокол."""
        tools = await self._load_via_mcp_protocol()
        if tools:
            self._tools_cache = tools
            logger.info(
                "MCPClient: загружено %d инструментов (MCP protocol)", len(tools)
            )
            return tools

        logger.error(
            "MCPClient: не удалось загрузить инструменты ни с одного эндпоинта"
        )
        return []

    async def _load_via_mcp_protocol(self) -> list[dict[str, Any]] | None:
        """Загружает инструменты через MCP JSON-RPC протокол (StreamableHTTP).

        fastmcp 2.x требует:
        1. POST /mcp initialize → получить session_id из заголовка ответа
        2. POST /mcp notifications/initialized с mcp-session-id
        3. POST /mcp tools/list с mcp-session-id → SSE или JSON ответ
        """
        mcp_endpoint = f"{self._base_url}/mcp"
        headers_base = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:

                # ── Шаг 1: initialize ─────────────────────────────────────────
                init_resp = await client.post(
                    mcp_endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {
                                "name": "edms-orchestrator",
                                "version": "1.0",
                            },
                        },
                    },
                    headers=headers_base,
                )

                if init_resp.status_code not in (200, 201):
                    logger.debug(
                        "MCP initialize failed: HTTP %d", init_resp.status_code
                    )
                    return None

                # Извлекаем session_id из заголовков ответа
                session_id = init_resp.headers.get(
                    "mcp-session-id"
                ) or init_resp.headers.get("x-session-id")

                if session_id:
                    self._session_id = session_id
                    logger.debug("MCP session_id established: %s", session_id[:8])
                else:
                    logger.warning(
                        "MCP server did not return session_id in response headers"
                    )

                # Заголовки для последующих запросов — с session_id
                session_headers = dict(headers_base)
                if session_id:
                    session_headers["mcp-session-id"] = session_id

                # ── Шаг 2: notifications/initialized ──────────────────────────
                notif_resp = await client.post(
                    mcp_endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                        "params": {},
                    },
                    headers=session_headers,
                )
                logger.debug(
                    "MCP notifications/initialized: HTTP %d", notif_resp.status_code
                )

                # ── Шаг 3: tools/list ─────────────────────────────────────────
                list_resp = await client.post(
                    mcp_endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/list",
                        "params": {},
                    },
                    headers=session_headers,
                )

                if list_resp.status_code not in (200, 201):
                    logger.debug(
                        "MCP tools/list failed: HTTP %d body=%s",
                        list_resp.status_code,
                        list_resp.text[:300],
                    )
                    return None

                logger.debug(
                    "MCP tools/list response: HTTP %d content_type=%s body_start=%s",
                    list_resp.status_code,
                    list_resp.headers.get("content-type", ""),
                    list_resp.text[:200],
                )

                data = _parse_mcp_response(list_resp.text)
                if not isinstance(data, dict):
                    logger.debug(
                        "MCP tools/list: failed to parse response as dict, got: %s",
                        type(data),
                    )
                    return None

                if "error" in data:
                    logger.warning("MCP tools/list error: %s", data["error"])
                    return None

                raw_tools = data.get("result", {}).get("tools", [])
                if not raw_tools:
                    logger.warning("MCP tools/list returned empty tools list")
                    return None

                tools = _normalize_tools(raw_tools)
                logger.info("MCP tools loaded: %d", len(tools))
                return tools if tools else None

        except Exception as exc:
            logger.warning("MCP protocol load failed: %s", exc, exc_info=True)
            return None

    async def call_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Вызывает MCP-инструмент, переиспользуя session_id."""
        mcp_endpoint = f"{self._base_url}/mcp"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": tool_input},
        }
        try:
            async with httpx.AsyncClient(timeout=_MCP_CALL_TIMEOUT) as client:
                resp = await client.post(mcp_endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = _parse_mcp_response(resp.text)
                if data is None:
                    return {"success": False, "error": "Empty response"}

                if isinstance(data, dict):
                    if "error" in data:
                        return {
                            "success": False,
                            "error": data["error"].get("message", "MCP error"),
                        }
                    result = data.get("result", data)
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
            return {
                "success": False,
                "error": {"code": "CALL_ERROR", "message": str(exc)},
            }

    @property
    def cached_tools(self) -> list[dict[str, Any]]:
        return self._tools_cache


def _parse_mcp_response(text: str) -> Any:
    """Парсит ответ MCP-сервера.

    fastmcp 2.x возвращает SSE формат:
        event: message
        data: {"jsonrpc":"2.0","id":2,"result":{...}}

    или plain JSON. Обрабатывает multiline SSE.
    """
    text = text.strip()
    if not text:
        return None

    # SSE: ищем строки "data: {...}"
    if "data:" in text:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        parsed = _json.loads(payload)
                        if isinstance(parsed, dict):
                            return parsed
                    except _json.JSONDecodeError:
                        continue
        return None

    # Plain JSON
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        return None


def _normalize_tools(raw_tools: list) -> list[dict[str, Any]]:
    """Нормализует инструменты в формат Anthropic tool schema."""
    result = []
    for t in raw_tools:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        result.append(
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": (
                    t.get("inputSchema")
                    or t.get("input_schema")
                    or {"type": "object", "properties": {}}
                ),
            }
        )
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
        "1. Свои рассуждения веди внутренне — НЕ выводи блок <reasoning> пользователю.",
        "2. Используй только реальные инструменты через tool_use. "
        "НИКОГДА не пиши JSON вызова инструмента в текст ответа.",
        "3. Если инструменты недоступны — честно скажи что не можешь выполнить действие сейчас.",
        "4. При нехватке данных — задай уточняющий вопрос, не угадывай.",
        "5. ЗАПРЕЩЕНО показывать пользователю: UUID, токены, HTTP-коды, "
        "stack traces, JSON, технические идентификаторы любого вида.",
        "6. Структура ответа пользователю: краткий итог → детали → следующие шаги (если нужно).",
        "7. Обращайся к документу по его названию, не по UUID.",
    ]

    first = user_context.get("firstName", "")
    last = user_context.get("lastName", "")
    name = f"{last} {first}".strip() or "Коллега"
    role = user_context.get("role") or user_context.get("authorPost") or ""
    dept = (
        user_context.get("department") or user_context.get("authorDepartmentName") or ""
    )

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
        "get_document",
        "get_history",
        "get_workflow_status",
        "get_analytics",
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

    # Если инструменты недоступны — предупреждаем модель чтобы не галлюцинировала JSON
    if not tools:
        system_prompt += (
            "\n\nВАЖНО: инструменты сейчас недоступны. "
            "Не пиши JSON и не имитируй вызовы инструментов в тексте. "
            "Вежливо сообщи пользователю что функция временно недоступна."
        )

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
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input or {},
                    }
                )

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
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                }
            )

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
        """Инициализирует агент: только StateManager.

        Загрузка MCP-инструментов отложена до первого запроса через
        _ensure_tools_loaded(), чтобы не зависеть от порядка запуска сервисов.
        """
        await self.state_manager.initialize()
        self._initialized = True
        logger.info(
            "EdmsDocumentAgent инициализирован: backend=%s persistent=%s "
            "(MCP tools загрузятся при первом запросе)",
            self._llm.backend,
            self.state_manager.is_persistent,
        )

    async def _ensure_tools_loaded(self) -> None:
        """Загружает MCP-инструменты если ещё не загружены.

        Вызывается при каждом chat() — повторная загрузка происходит только
        если кэш пуст (например, после перезапуска MCP-сервера).
        При неудаче делает 3 попытки с нарастающей задержкой (2s, 4s).
        """
        if self._mcp_client.cached_tools:
            return

        for attempt in range(1, 4):
            logger.info("Загрузка MCP инструментов, попытка %d/3...", attempt)
            tools = await self._mcp_client.load_tools()
            if tools:
                logger.info("MCP инструменты загружены: %d инструментов", len(tools))
                return
            if attempt < 3:
                delay = 2.0 * attempt  # 2s, 4s
                logger.warning(
                    "MCP недоступен, следующая попытка через %.0fs...", delay
                )
                await asyncio.sleep(delay)

        logger.warning(
            "MCP инструменты недоступны после 3 попыток. "
            "Агент работает без инструментов."
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

        # Lazy-load: подключаемся к MCP при первом запросе
        await self._ensure_tools_loaded()

        start_ts = time.monotonic()
        ctx = user_context or {}

        from edms_ai_assistant.orchestrator.services.nlp_service import (
            SemanticDispatcher,
        )

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
            "Agent chat: thread=%s intent=%s confidence=%.2f model_role=%s backend=%s tools=%d",
            thread_id,
            intent,
            round(confidence, 2),
            model_role,
            self._llm.backend,
            len(self._mcp_client.cached_tools),
        )

        history = self._load_thread_history(thread_id)

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

        self._save_thread_history(thread_id, updated_history)
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

    def _load_thread_history(self, thread_id: str | None) -> list[dict[str, Any]]:
        """Загружает историю диалога из простого in-memory хранилища."""
        if not thread_id:
            return []
        history = self.state_manager._simple_store.get(thread_id, [])
        max_msgs = settings.AGENT_MAX_CONTEXT_MESSAGES
        if len(history) > max_msgs:
            history = history[-max_msgs:]
        return list(history)

    def _save_thread_history(
        self, thread_id: str | None, messages: list[dict[str, Any]]
    ) -> None:
        """Сохраняет историю диалога в простом in-memory хранилище.

        Сохраняет только role/content пары — tool_use блоки отбрасываются,
        они не нужны в истории для LLM-контекста следующего запроса.
        """
        if not thread_id or not messages:
            return

        clean: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and isinstance(content, str) and content:
                clean.append({"role": "user", "content": content})
            elif role == "assistant":
                if isinstance(content, list):
                    text = "".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    if text:
                        clean.append({"role": "assistant", "content": text})
                elif isinstance(content, str) and content:
                    clean.append({"role": "assistant", "content": content})

        if clean:
            self.state_manager._simple_store[thread_id] = clean
            logger.debug(
                "Thread history saved: thread=%s messages=%d",
                thread_id,
                len(clean),
            )

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
