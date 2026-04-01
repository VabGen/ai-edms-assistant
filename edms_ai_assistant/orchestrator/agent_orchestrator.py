"""
Agent Orchestrator — Главный оркестратор EDMS AI Ассистента.

Реализует:
1. ReAct цикл (Reasoning + Acting): LLM рассуждает → выбирает инструмент → анализирует результат
2. Plan and Execute: для сложных задач — планирование → пошаговое исполнение → проверка
3. Multi-agent координация: Planner, Researcher, Executor, Reviewer, Explainer
4. NLU предобработка с fast-path (без LLM для простых запросов)
5. Мульти-роутинг: выбор модели (лёгкая/тяжёлая) по сложности запроса
6. Интеграция памяти (short/medium/long-term) и RAG
7. Кэширование частых запросов с TTL
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from memory import MemoryManager, Message, UserProfile
from nlp_preprocessor import Intent, NLPPreprocessor, NLUResult
from rag_module import RAGModule
from tools.router import get_tools_for_intent
from utils.format_utils import format_document_response, sanitize_file_paths, sanitize_uuids

logger = logging.getLogger("orchestrator")


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class OrchestratorConfig:
    """Конфигурация оркестратора из переменных окружения."""
    # LLM — лёгкая модель (для простых задач)
    llm_light_url: str = field(
        default_factory=lambda: os.getenv("LLM_LIGHT_URL", "http://localhost:11434/v1")
    )
    llm_light_model: str = field(
        default_factory=lambda: os.getenv("LLM_LIGHT_MODEL", "llama3.2:3b")
    )
    # LLM — тяжёлая модель (для сложных задач)
    llm_heavy_url: str = field(
        default_factory=lambda: os.getenv("LLM_HEAVY_URL", "http://localhost:11434/v1")
    )
    llm_heavy_model: str = field(
        default_factory=lambda: os.getenv("LLM_HEAVY_MODEL", "gpt-oss:120b-cloud")
    )
    # Внешние модели (при наличии ключей)
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    # MCP
    mcp_server_url: str = field(
        default_factory=lambda: os.getenv("MCP_SERVER_URL", "http://localhost:8001")
    )
    # Память
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    pg_dsn: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://edms:edms@localhost:5432/edms_ai"
        )
    )
    # Агент
    max_react_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_REACT_ITERATIONS", "8"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60"))
    )
    cache_ttl: int = 300  # секунд
    max_context_tokens: int = 8000

    # Порог сложности для выбора тяжёлой модели
    complexity_threshold: float = 0.7


# ── Agent roles ───────────────────────────────────────────────────────────────


class AgentRole(str, Enum):
    """Роли агентов в мульти-агентной координации."""
    PLANNER = "planner"
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    EXPLAINER = "explainer"
    REACT = "react"      # Единый ReAct агент для простых задач


# ── Response models ───────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """Вызов MCP инструмента."""
    tool_name: str
    args: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class ToolResult:
    """Результат выполнения MCP инструмента."""
    call_id: str
    tool_name: str
    success: bool
    data: Any
    error: str = ""
    latency_ms: int = 0


@dataclass
class AgentResponse:
    """Финальный ответ ассистента пользователю."""
    content: str
    session_id: str
    dialog_id: str | None = None
    intent: str = ""
    tools_used: list[str] = field(default_factory=list)
    model_used: str = ""
    latency_ms: int = 0
    requires_clarification: bool = False
    clarification_question: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Plan models ───────────────────────────────────────────────────────────────


@dataclass
class PlanStep:
    """Один шаг плана выполнения."""
    step_number: int
    description: str
    agent_role: AgentRole
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    status: str = "pending"   # pending | running | done | failed | skipped
    result: Any = None
    error: str = ""


@dataclass
class ExecutionPlan:
    """План выполнения сложного запроса."""
    goal: str
    steps: list[PlanStep]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"


# ── LLM Client (multi-routing) ────────────────────────────────────────────────


class LLMRouter:
    """
    Мульти-роутинг LLM: выбирает провайдера и модель по сложности запроса.

    Порядок выбора:
    1. Если complexity_score < threshold → лёгкая локальная модель (Ollama)
    2. Если complexity_score >= threshold → тяжёлая модель (Ollama big или Claude/GPT)
    3. Если доступен Anthropic API → Claude для самых сложных задач
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config

    def select_model(self, complexity_score: float, force_heavy: bool = False) -> tuple[str, str]:
        """
        Выбрать URL и модель на основе сложности.

        Returns: (base_url, model_name)
        """
        use_heavy = force_heavy or complexity_score >= self.config.complexity_threshold

        if use_heavy:
            return self.config.llm_heavy_url, self.config.llm_heavy_model
        return self.config.llm_light_url, self.config.llm_light_model

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        complexity_score: float = 0.3,
        force_heavy: bool = False,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        Вызов LLM с автоматическим выбором модели.

        Returns: dict с полем 'content' (текст) и опционально 'tool_calls'
        """
        base_url, model = self.select_model(complexity_score, force_heavy)

        # Пробуем через OpenAI-совместимый эндпоинт
        try:
            return await self._call_openai_compat(
                base_url=base_url,
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning(
                "LLM call failed (%s/%s): %s — trying fallback",
                base_url, model, exc,
            )
            # Fallback: если тяжёлая упала — попробовать лёгкую
            if force_heavy or complexity_score >= self.config.complexity_threshold:
                try:
                    return await self._call_openai_compat(
                        base_url=self.config.llm_light_url,
                        model=self.config.llm_light_model,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                    )
                except Exception as exc2:
                    raise RuntimeError(f"All LLM endpoints failed: {exc2}") from exc2
            raise

    async def _call_openai_compat(
        self,
        base_url: str,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
    ) -> dict[str, Any]:
        """Вызов через OpenAI-совместимый API (/v1/chat/completions)."""
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"

        api_key = self.config.openai_api_key or self.config.anthropic_api_key or "ollama"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        result: dict[str, Any] = {
            "content": msg.get("content") or "",
            "model": data.get("model", model),
            "finish_reason": choice.get("finish_reason", ""),
        }
        if msg.get("tool_calls"):
            result["tool_calls"] = msg["tool_calls"]
        return result


# ── MCP Client ────────────────────────────────────────────────────────────────


class MCPClient:
    """
    HTTP-клиент к MCP серверу EDMS.

    Поддерживает как HTTP/SSE транспорт (для развёрнутого сервера),
    так и прямые вызовы функций (для тестирования).
    """

    def __init__(self, server_url: str, timeout: int = 30) -> None:
        self._url = server_url.rstrip("/")
        self._timeout = timeout
        self._tools_cache: list[dict] | None = None

    async def list_tools(self) -> list[dict]:
        """Получить список доступных инструментов MCP."""
        if self._tools_cache:
            return self._tools_cache
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._url}/tools/list",
                    json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                self._tools_cache = data.get("result", {}).get("tools", [])
                return self._tools_cache
        except Exception as exc:
            logger.warning("MCPClient.list_tools error: %s", exc)
            # Fallback: загрузить из локального реестра
            registry_path = Path(__file__).parent.parent / "mcp-server" / "tools_registry.json"
            if registry_path.exists():
                with open(registry_path, encoding="utf-8") as f:
                    registry = json.load(f)
                return registry.get("tools", [])
            return []

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Вызвать MCP инструмент."""
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._url}/tools/call",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "id": uuid.uuid4().hex[:8],
                        "params": {"name": tool_name, "arguments": args},
                    },
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                result = data.get("result", {})
                content = result.get("content", [{}])
                tool_text = content[0].get("text", json.dumps(result)) if content else ""
                latency = int((time.monotonic() - start) * 1000)
                return {
                    "success": True,
                    "data": json.loads(tool_text) if tool_text.startswith("{") else {"raw": tool_text},
                    "latency_ms": latency,
                }
        except Exception as exc:
            latency = int((time.monotonic() - start) * 1000)
            logger.error("MCPClient.call_tool '%s' error: %s", tool_name, exc)
            return {
                "success": False,
                "error": str(exc),
                "latency_ms": latency,
            }

    def get_tools_for_llm(self, tools: list[dict]) -> list[dict]:
        """Конвертировать MCP инструменты в формат OpenAI function calling."""
        result = []
        for tool in tools:
            result.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                }),
            })
        return result


# ── Prompt loader ─────────────────────────────────────────────────────────────


class PromptLoader:
    """Загрузчик промптов из файлов с кэшированием."""

    def __init__(self, prompts_dir: str) -> None:
        self._dir = Path(prompts_dir)
        self._cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        """Получить промпт по имени файла (без расширения)."""
        if name in self._cache:
            return self._cache[name]
        for ext in (".txt", ".md"):
            path = self._dir / f"{name}{ext}"
            if path.exists():
                content = path.read_text(encoding="utf-8")
                self._cache[name] = content
                return content
        logger.warning("Prompt '%s' not found in %s", name, self._dir)
        return ""

    def reload(self) -> None:
        """Перезагрузить все промпты (для обновления без рестарта)."""
        self._cache.clear()


# ── Main Orchestrator ─────────────────────────────────────────────────────────


class EdmsAgentOrchestrator:
    """
    Главный оркестратор EDMS AI Ассистента.

    Точки входа:
        response = await orchestrator.process(
            user_message="Найди договоры за последний месяц",
            user_id="user-uuid",
            session_id="session-uuid",
            token="jwt-token",
            context={"document_id": "doc-uuid"},  # опционально
        )
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        prompts_dir: str = "prompts",
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.nlu = NLPPreprocessor()
        self.llm = LLMRouter(self.config)
        self.mcp = MCPClient(self.config.mcp_server_url, timeout=self.config.request_timeout)
        self.prompts = PromptLoader(prompts_dir)
        self.memory = MemoryManager(
            redis_url=self.config.redis_url,
            pg_dsn=self.config.pg_dsn,
            max_context_tokens=self.config.max_context_tokens,
        )
        self.rag = RAGModule(dsn=self.config.pg_dsn)
        self._initialized = False

    async def init(self) -> None:
        """Инициализация всех компонентов."""
        await self.memory.init()
        await self.rag.init()
        self._initialized = True
        logger.info("EdmsAgentOrchestrator initialized")

    async def close(self) -> None:
        """Graceful shutdown."""
        await self.memory.close()
        await self.rag.close()

    # ── Main entry point ──────────────────────────────────────────────────────

    async def process(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        token: str = "",
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Обработать запрос пользователя.

        Pipeline:
        1. NLU предобработка
        2. Fast-path (если возможно — без LLM)
        3. Загрузка контекста (профиль, сессия, RAG)
        4. Определение сложности → выбор агентного режима
        5. ReAct цикл или Plan+Execute
        6. Формирование ответа
        7. Логирование и обновление памяти
        """
        start_time = time.monotonic()
        ctx = context or {}
        ctx["token"] = token
        ctx["document_id"] = ctx.get("document_id", "")

        logger.info(
            "Processing request: user=%s session=%s query='%s'",
            user_id[:8], session_id[:8], user_message[:80],
        )

        # ── Кэш ──────────────────────────────────────────────────────────────
        cache_key = f"{user_id}:{hash(user_message)}"
        cached = await self.memory.medium.cache_get(cache_key)
        if cached:
            logger.info("Cache hit for query: %s", user_message[:40])
            return AgentResponse(**cached)

        # ── 1. NLU ────────────────────────────────────────────────────────────
        nlu_result = self.nlu.preprocess(user_message, ctx)
        logger.info(
            "NLU: intent=%s confidence=%.2f fast_path=%s",
            nlu_result.intent.value, nlu_result.confidence, nlu_result.can_skip_llm,
        )

        # ── 2. Fast-path ──────────────────────────────────────────────────────
        if nlu_result.can_skip_llm and nlu_result.suggested_tool:
            return await self._fast_path(
                nlu_result, user_id, session_id, start_time,
            )

        # ── 3. Загрузка контекста ─────────────────────────────────────────────
        await self.memory.get_or_create_session(session_id, user_id)
        profile = await self.memory.get_user_profile(user_id)

        # RAG: похожие примеры
        rag_results = await self.rag.search(
            user_message, intent=nlu_result.intent.value, top_k=3,
        )
        anti_results = await self.rag.search_anti_examples(user_message, top_k=2)

        few_shot = self.rag.format_few_shot(rag_results)
        anti_examples = self.rag.format_anti_examples(anti_results)
        nlu_context = self.nlu.build_enriched_prompt_context(nlu_result)

        system_prompt = await self.memory.build_system_context(
            base_system_prompt=self.prompts.get("system_prompt"),
            nlu_context=nlu_context,
            rag_examples=few_shot,
            anti_examples=anti_examples,
        )

        # ── 4. Выбор агентного режима ─────────────────────────────────────────
        complexity = self._estimate_complexity(nlu_result, user_message)
        use_plan_execute = (
            complexity >= self.config.complexity_threshold
            and nlu_result.intent in {
                Intent.CREATE_TASK, Intent.AGREE_DOCUMENT,
                Intent.START_ROUTING, Intent.ANALYZE,
            }
        )

        # ── Добавление сообщения в краткосрочную память ───────────────────────
        self.memory.short.set_system(system_prompt)
        self.memory.short.add_user(user_message)

        # ── 5. Агентный цикл ──────────────────────────────────────────────────
        if use_plan_execute:
            response_text, tools_used, model_used = await self._plan_and_execute(
                nlu_result=nlu_result,
                user_message=user_message,
                token=token,
                ctx=ctx,
                complexity=complexity,
            )
        else:
            response_text, tools_used, model_used = await self._react_cycle(
                nlu_result=nlu_result,
                token=token,
                ctx=ctx,
                complexity=complexity,
            )

        # ── 6. Формирование ответа ────────────────────────────────────────────
        # Очищаем технический мусор из ответа (UUID, пути к файлам)
        if tools_used and response_text:
            if "doc_get_details" in tools_used or "doc_get_file_content" in tools_used:
                response_text = format_document_response(response_text)
            response_text = sanitize_file_paths(response_text)
        self.memory.short.add_assistant(response_text)
        latency_ms = int((time.monotonic() - start_time) * 1000)

        # ── 7. Логирование ────────────────────────────────────────────────────
        dialog_id = await self.memory.long.log_dialog(
            session_id=session_id,
            user_id=user_id,
            user_query=user_message,
            normalized_query=nlu_result.normalized,
            intent=nlu_result.intent.value,
            entities={
                k: [{"value": str(e.value), "raw": e.raw_text} for e in elist]
                for k, elist in nlu_result.entities.items()
            },
            selected_tool=tools_used[0] if tools_used else "",
            tool_args={},
            final_response=response_text,
            model_used=model_used,
            latency_ms=latency_ms,
        )

        await self.memory.update_after_turn(
            session_id=session_id,
            intent=nlu_result.intent.value,
            tool_used=tools_used[0] if tools_used else None,
            document_id=ctx.get("document_id"),
        )

        result = AgentResponse(
            content=response_text,
            session_id=session_id,
            dialog_id=dialog_id,
            intent=nlu_result.intent.value,
            tools_used=tools_used,
            model_used=model_used,
            latency_ms=latency_ms,
        )

        # Кэшируем только быстрые простые ответы
        if latency_ms < 5000 and not tools_used:
            await self.memory.medium.cache_set(
                cache_key, result.to_dict(), ttl=self.config.cache_ttl,
            )

        return result

    # ── Fast-path ─────────────────────────────────────────────────────────────

    async def _fast_path(
        self,
        nlu_result: NLUResult,
        user_id: str,
        session_id: str,
        start_time: float,
    ) -> AgentResponse:
        """Прямой вызов инструмента без LLM."""
        tool = nlu_result.suggested_tool
        args = nlu_result.suggested_args
        logger.info("Fast-path: tool=%s", tool)

        tool_result = await self.mcp.call_tool(tool, args)
        latency_ms = int((time.monotonic() - start_time) * 1000)

        if tool_result.get("success"):
            data = tool_result.get("data", {})
            response_text = self._format_tool_result_simple(tool, data)
        else:
            response_text = f"Не удалось выполнить операцию: {tool_result.get('error', 'Неизвестная ошибка')}"

        dialog_id = await self.memory.long.log_dialog(
            session_id=session_id,
            user_id=user_id,
            user_query=nlu_result.original,
            normalized_query=nlu_result.normalized,
            intent=nlu_result.intent.value,
            selected_tool=tool or "",
            final_response=response_text,
            latency_ms=latency_ms,
        )

        return AgentResponse(
            content=response_text,
            session_id=session_id,
            dialog_id=dialog_id,
            intent=nlu_result.intent.value,
            tools_used=[tool] if tool else [],
            model_used="fast_path",
            latency_ms=latency_ms,
        )

    # ── ReAct cycle ───────────────────────────────────────────────────────────

    async def _react_cycle(
        self,
        nlu_result: NLUResult,
        token: str,
        ctx: dict,
        complexity: float,
    ) -> tuple[str, list[str], str]:
        """
        ReAct цикл: Reasoning + Acting.

        Итерации:
        1. LLM анализирует запрос и выбирает инструмент (или отвечает напрямую)
        2. Если выбран инструмент — вызываем через MCP
        3. Результат добавляем в контекст, повторяем
        4. Когда LLM решит — финальный ответ пользователю
        """
        mcp_tools = await self.mcp.list_tools()
        tools_for_llm = self.mcp.get_tools_for_llm(mcp_tools)

        # Intent-based routing — передаём LLM только релевантные инструменты
        try:
            from tools import all_tools as _langchain_tools
            filtered_lc = get_tools_for_intent(
                nlu_result.intent,
                _langchain_tools,
                include_appeal=ctx.get("document_category") == "APPEAL",
            )
            filtered_names = {getattr(t, "name", None) for t in filtered_lc}
            tools_for_llm = [t for t in tools_for_llm if t.get("name") in filtered_names] or tools_for_llm
            logger.info("Router: %d/%d tools for intent=%s", len(tools_for_llm), len(mcp_tools), nlu_result.intent)
        except Exception as _re:
            logger.debug("Tool routing skipped: %s", _re)

        tools_used: list[str] = []
        model_used = ""

        for iteration in range(self.config.max_react_iterations):
            logger.debug("ReAct iteration %d", iteration + 1)
            messages = self.memory.short.get_messages()

            try:
                llm_response = await self.llm.chat(
                    messages=messages,
                    tools=tools_for_llm if iteration < self.config.max_react_iterations - 1 else None,
                    complexity_score=complexity,
                    temperature=0.1,
                )
                model_used = llm_response.get("model", "unknown")
            except Exception as exc:
                logger.error("LLM call failed in ReAct iteration %d: %s", iteration, exc)
                return (
                    f"Произошла ошибка при обработке запроса: {exc!s}. Попробуйте переформулировать.",
                    tools_used,
                    model_used,
                )

            # Нет tool_calls → финальный ответ
            if not llm_response.get("tool_calls"):
                content = llm_response.get("content", "")
                if content:
                    return content, tools_used, model_used
                # Пустой ответ — попробуем ещё раз без инструментов
                logger.warning("Empty LLM response at iteration %d", iteration)
                continue

            # Есть tool_calls — выполняем инструменты
            tool_calls = llm_response["tool_calls"]
            self.memory.short.add_assistant(
                content=llm_response.get("content", ""),
                tool_calls=tool_calls,
            )

            # Выполняем каждый tool_call
            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    raw_args = fn.get("arguments", "{}")
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}

                # Автоинъекция токена
                if "token" in [p for tool in mcp_tools if tool.get("name") == tool_name
                               for p in tool.get("inputSchema", {}).get("properties", {}).keys()]:
                    args.setdefault("token", token)

                logger.info("ReAct: calling tool '%s' args=%s", tool_name, str(args)[:100])
                tool_result = await self.mcp.call_tool(tool_name, args)
                tools_used.append(tool_name)

                result_text = json.dumps(
                    tool_result.get("data", {"error": tool_result.get("error", "no data")}),
                    ensure_ascii=False, default=str,
                )[:3000]

                self.memory.short.add_tool_result(result_text, tool_name)

        # Превысили максимум итераций
        return (
            "Я не смог полностью обработать ваш запрос за допустимое количество шагов. "
            "Пожалуйста, уточните запрос или разбейте его на более простые части.",
            tools_used,
            model_used,
        )

    # ── Plan and Execute ──────────────────────────────────────────────────────

    async def _plan_and_execute(
        self,
        nlu_result: NLUResult,
        user_message: str,
        token: str,
        ctx: dict,
        complexity: float,
    ) -> tuple[str, list[str], str]:
        """
        Plan and Execute для сложных многошаговых задач.

        Шаги:
        1. Planner агент строит план
        2. Executor выполняет каждый шаг
        3. Reviewer проверяет результат
        4. Explainer формирует финальный ответ
        """
        logger.info("Plan+Execute mode for intent: %s", nlu_result.intent.value)

        # ── 1. Planning ───────────────────────────────────────────────────────
        planner_messages = [
            {"role": "system", "content": self.prompts.get("planner_prompt")},
            {"role": "user", "content": f"Задача: {user_message}\n\nNLU анализ: {nlu_result.intent.value}"},
        ]
        try:
            plan_response = await self.llm.chat(
                messages=planner_messages,
                complexity_score=complexity,
                force_heavy=True,
                temperature=0.1,
            )
            plan_text = plan_response.get("content", "")
            model_used = plan_response.get("model", "")
        except Exception as exc:
            logger.error("Planner failed: %s", exc)
            # Fallback to ReAct
            return await self._react_cycle(nlu_result, token, ctx, complexity)

        # Парсим план (ожидаем JSON или нумерованный список)
        plan = self._parse_plan(plan_text, nlu_result.intent.value)
        if not plan or not plan.steps:
            logger.warning("Empty plan, falling back to ReAct")
            return await self._react_cycle(nlu_result, token, ctx, complexity)

        logger.info("Plan created: %d steps", len(plan.steps))

        # ── 2. Execution ──────────────────────────────────────────────────────
        tools_used: list[str] = []
        step_results: list[str] = []

        for step in plan.steps:
            step.status = "running"
            logger.info("Executing step %d: %s", step.step_number, step.description)

            if step.tool_name:
                # Инструментальный шаг
                args = step.tool_args.copy()
                args.setdefault("token", token)
                if ctx.get("document_id") and "document_id" in args:
                    args["document_id"] = args.get("document_id") or ctx["document_id"]

                tool_result = await self.mcp.call_tool(step.tool_name, args)
                tools_used.append(step.tool_name)

                if tool_result.get("success"):
                    step.status = "done"
                    step.result = tool_result.get("data", {})
                    step_results.append(
                        f"Шаг {step.step_number} ({step.description}): "
                        f"{json.dumps(step.result, ensure_ascii=False, default=str)[:300]}"
                    )
                else:
                    step.status = "failed"
                    step.error = tool_result.get("error", "")
                    step_results.append(
                        f"Шаг {step.step_number} ({step.description}): ОШИБКА — {step.error}"
                    )
            else:
                step.status = "skipped"
                step_results.append(f"Шаг {step.step_number}: {step.description} — пропущен")

        # ── 3. Review ─────────────────────────────────────────────────────────
        reviewer_messages = [
            {"role": "system", "content": self.prompts.get("reviewer_prompt")},
            {"role": "user", "content": (
                f"Исходная задача: {user_message}\n\n"
                f"Результаты выполнения:\n{chr(10).join(step_results)}"
            )},
        ]
        try:
            review_response = await self.llm.chat(
                messages=reviewer_messages,
                complexity_score=0.3,
                temperature=0.0,
            )
            review_text = review_response.get("content", "")
        except Exception as exc:
            logger.warning("Reviewer failed: %s", exc)
            review_text = ""

        # ── 4. Explainer ──────────────────────────────────────────────────────
        explainer_messages = [
            {"role": "system", "content": (
                "Ты — EDMS ассистент. Сформулируй понятный ответ пользователю "
                "на русском языке на основе результатов выполненных действий. "
                "Без технических деталей, конкретно и по делу."
            )},
            {"role": "user", "content": (
                f"Задача пользователя: {user_message}\n\n"
                f"Выполненные шаги:\n{chr(10).join(step_results)}\n\n"
                f"Оценка выполнения:\n{review_text}"
            )},
        ]
        try:
            final_response = await self.llm.chat(
                messages=explainer_messages,
                complexity_score=0.3,
                temperature=0.2,
            )
            final_text = final_response.get("content", "Задача выполнена.")
        except Exception as exc:
            logger.warning("Explainer failed: %s", exc)
            final_text = "Задача выполнена. " + " | ".join(step_results[:3])

        return final_text, tools_used, model_used

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _estimate_complexity(self, nlu_result: NLUResult, message: str) -> float:
        """Оценка сложности запроса (0.0 — 1.0)."""
        score = 0.0

        # Длина запроса
        words = len(message.split())
        if words > 30:
            score += 0.3
        elif words > 15:
            score += 0.15

        # Намерение
        complex_intents = {
            Intent.ANALYZE, Intent.COMPARE, Intent.CREATE_TASK,
            Intent.AGREE_DOCUMENT, Intent.START_ROUTING,
        }
        if nlu_result.intent in complex_intents:
            score += 0.3

        # Несколько сущностей
        entity_count = sum(len(v) for v in nlu_result.entities.values())
        if entity_count > 3:
            score += 0.2
        elif entity_count > 1:
            score += 0.1

        # Низкая уверенность NLU
        if nlu_result.confidence < 0.5:
            score += 0.2

        return min(score, 1.0)

    def _parse_plan(self, plan_text: str, intent: str) -> ExecutionPlan:
        """Парсинг плана из ответа LLM (JSON или нумерованный список)."""
        steps: list[PlanStep] = []

        # Попытка JSON парсинга
        try:
            json_start = plan_text.find("[")
            json_end = plan_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                raw_steps = json.loads(plan_text[json_start:json_end])
                for i, s in enumerate(raw_steps, 1):
                    role_map = {
                        "researcher": AgentRole.RESEARCHER,
                        "executor": AgentRole.EXECUTOR,
                        "reviewer": AgentRole.REVIEWER,
                    }
                    steps.append(PlanStep(
                        step_number=i,
                        description=s.get("description", f"Шаг {i}"),
                        agent_role=role_map.get(s.get("role", "executor"), AgentRole.EXECUTOR),
                        tool_name=s.get("tool"),
                        tool_args=s.get("args", {}),
                    ))
        except (json.JSONDecodeError, KeyError, TypeError):
            # Парсинг нумерованного списка
            lines = plan_text.strip().split("\n")
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    desc = line.lstrip("0123456789.-) ").strip()
                    if desc:
                        steps.append(PlanStep(
                            step_number=i,
                            description=desc,
                            agent_role=AgentRole.EXECUTOR,
                        ))

        return ExecutionPlan(
            goal=intent,
            steps=steps[:10],  # не более 10 шагов
        )

    def _format_tool_result_simple(self, tool_name: str, data: dict) -> str:
        """Простое форматирование результата инструмента для fast-path."""
        if not data:
            return "Операция выполнена успешно."

        if "error" in data:
            return f"Ошибка: {data['error']}"

        if tool_name == "get_document_statistics":
            stats = data.get("stats", {})
            parts = ["📊 Статистика документов:"]
            if stats.get("executor"):
                parts.append(f"• На исполнении: {stats['executor']}")
            if stats.get("control"):
                parts.append(f"• На контроле: {stats['control']}")
            if stats.get("author"):
                parts.append(f"• Я автор: {stats['author']}")
            return "\n".join(parts) if len(parts) > 1 else "Статистика получена."

        if tool_name == "search_employees":
            employees = data.get("employees", [])
            total = data.get("total", len(employees))
            if not employees:
                return "Сотрудники не найдены по заданным критериям."
            parts = [f"Найдено сотрудников: {total}"]
            for emp in employees[:5]:
                name = f"{emp.get('lastName', '')} {emp.get('firstName', '')}".strip()
                post = emp.get("post", {})
                post_name = post.get("postName", "") if isinstance(post, dict) else ""
                parts.append(f"• {name}" + (f" — {post_name}" if post_name else ""))
            return "\n".join(parts)

        if tool_name == "get_current_user":
            user = data.get("user", data)
            name = f"{user.get('lastName', '')} {user.get('firstName', '')}".strip()
            return f"Текущий пользователь: {name or 'Неизвестно'}"

        return json.dumps(data, ensure_ascii=False, default=str)[:500]

    async def save_feedback(
        self,
        dialog_id: str,
        rating: int,
        comment: str = "",
    ) -> bool:
        """Сохранить оценку пользователя и обновить RAG."""
        success = await self.memory.long.save_feedback(dialog_id, rating, comment)

        if success and rating == 1:
            # Добавляем успешный диалог в RAG
            async with self.memory.long._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM edms_ai.dialog_logs WHERE dialog_id = $1::uuid",
                    dialog_id,
                )
                if row:
                    from rag_module import RAGEntry
                    entry = RAGEntry(
                        dialog_id=dialog_id,
                        user_query=row["user_query"],
                        normalized_query=row["normalized_query"] or "",
                        intent=row["intent"] or "",
                        selected_tool=row["selected_tool"] or "",
                        tool_args=json.loads(row["tool_args"] or "{}"),
                        response_summary=(row["final_response"] or "")[:500],
                        full_response=row["final_response"] or "",
                        feedback_score=1,
                    )
                    await self.rag.add_entry(entry)

        return success

    async def get_health(self) -> dict[str, Any]:
        """Проверка состояния всех компонентов."""
        status: dict[str, Any] = {
            "orchestrator": "healthy",
            "timestamp": datetime.now().isoformat(),
        }

        # MCP
        try:
            tools = await self.mcp.list_tools()
            status["mcp_server"] = f"healthy ({len(tools)} tools)"
        except Exception as exc:
            status["mcp_server"] = f"unhealthy: {exc}"

        # Redis
        try:
            await self.memory.medium.cache_set("health_check", "ok", ttl=5)
            status["redis"] = "healthy"
        except Exception as exc:
            status["redis"] = f"unhealthy: {exc}"

        # PostgreSQL
        try:
            if self.memory.long._pool:
                async with self.memory.long._pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                status["postgresql"] = "healthy"
            else:
                status["postgresql"] = "not connected"
        except Exception as exc:
            status["postgresql"] = f"unhealthy: {exc}"

        # RAG
        rag_stats = await self.rag.get_stats()
        status["rag"] = rag_stats

        return status
