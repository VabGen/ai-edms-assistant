# edms_ai_assistant/orchestrator/agent.py
"""
Единый AgentOrchestrator — точка входа для всей агентной логики.

АРХИТЕКТУРА:
  Этот файл является ЕДИНСТВЕННЫМ AgentOrchestrator-ом в проекте.
  Старые agent.py (с LangGraph) и новый (с NLPPreprocessor)
  объединены здесь в единую реализацию.

API (ожидается в main.py):
  await agent.initialize()
  await agent.process(user_message, user_id, session_id, token, context)
  await agent.health_check()
  await agent.save_feedback(dialog_id, rating, comment)
  await agent.get_thread_history(thread_id)
  await agent.update_rag_from_feedback(dialog_id)
  await agent.rebuild_rag()
  await agent.close()

РАСШИРЕНИЕ:
  1. Новый инструмент → добавить в MCP-сервер, зарегистрировать в tools_registry.json
  2. Новый intent → добавить в nlp_service.py → UserIntent enum
  3. Новая роль агента → добавить агент в multi_agent.py, подключить в _route_to_agent()
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import redis.asyncio as aioredis

from edms_ai_assistant.config import settings
from edms_ai_assistant.orchestrator.services.nlp_service import (
    SemanticDispatcher,
    UserIntent,
)
from edms_ai_assistant.orchestrator.db.database import AsyncSessionLocal
from edms_ai_assistant.infrastructure.redis_client import get_redis

logger = logging.getLogger(__name__)


# ── Конфигурация ──────────────────────────────────────────────────────────────

@dataclass
class AgentOrchestratorConfig:
    """Конфигурация агента из переменных окружения через settings."""

    mcp_server_url: str = field(default_factory=lambda: settings.MCP_URL)
    max_react_iterations: int = field(default_factory=lambda: settings.AGENT_MAX_ITERATIONS)
    agent_timeout: float = field(default_factory=lambda: settings.AGENT_TIMEOUT)
    max_retries: int = field(default_factory=lambda: settings.AGENT_MAX_RETRIES)
    cache_ttl_seconds: int = field(default_factory=lambda: settings.CACHE_TTL_SECONDS)
    enable_tracing: bool = field(default_factory=lambda: settings.AGENT_ENABLE_TRACING)
    complexity_threshold: float = 0.7


# ── Модель ответа ─────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    """Стандартизированный ответ агента для main.py."""

    content: str
    session_id: str
    dialog_id: str | None = None
    intent: str = ""
    tools_used: list[str] = field(default_factory=list)
    model_used: str = ""
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Основной оркестратор ──────────────────────────────────────────────────────

class AgentOrchestrator:
    """Единый AI-оркестратор EDMS Assistant.

    Объединяет:
    - NLU предобработку (SemanticDispatcher из nlp_service.py)
    - ReAct цикл через MCP инструменты
    - Трёхуровневую память (short / Redis / PostgreSQL)
    - RAG (few-shot + anti-examples)
    - Dynamic model routing (haiku/sonnet/opus по сложности)
    - Plan and Execute для сложных задач
    """

    def __init__(self, config: AgentOrchestratorConfig | None = None) -> None:
        self.config = config or AgentOrchestratorConfig()
        self._dispatcher = SemanticDispatcher()
        self._short_term: dict[str, list[dict]] = {}   # session_id → messages
        self._initialized = False
        self._mcp_tools_cache: list[dict] | None = None

    # ── Жизненный цикл ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Инициализирует все компоненты агента.

        Вызывается один раз при старте приложения через lifespan().
        """
        logger.info("Инициализация AgentOrchestrator...")
        try:
            # Прогреваем список инструментов
            tools = await self._list_mcp_tools()
            logger.info("MCP инструменты загружены: %d", len(tools))
        except Exception as exc:
            logger.warning("MCP инструменты не загружены при старте: %s", exc)
        self._initialized = True
        logger.info("AgentOrchestrator инициализирован")

    async def close(self) -> None:
        """Освобождает ресурсы при остановке приложения."""
        self._short_term.clear()
        logger.info("AgentOrchestrator остановлен")

    # ── Основной метод ────────────────────────────────────────────────────────

    async def process(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        token: str = "",
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Обрабатывает запрос пользователя через ReAct-цикл.

        Pipeline:
        1. NLU предобработка → intent, entities, complexity
        2. Проверка кэша (Redis) для read-only запросов
        3. Загрузка контекста из памяти
        4. Выбор модели по сложности
        5. ReAct / Plan+Execute
        6. Сохранение в память и БД
        7. Возврат AgentResponse

        Args:
            user_message: Сообщение пользователя.
            user_id: UUID пользователя (из JWT).
            session_id: ID сессии/треда.
            token: JWT токен для MCP вызовов.
            context: Дополнительный контекст (document_id, file_path и т.д.).

        Returns:
            AgentResponse с контентом, метаданными и dialog_id.
        """
        start_ts = time.monotonic()
        ctx = context or {}
        ctx["token"] = token

        # ── 1. NLU ────────────────────────────────────────────────────────────
        semantic_ctx = self._dispatcher.build_context(
            message=user_message,
            file_path=ctx.get("file_path"),
        )
        intent = semantic_ctx.query.intent
        complexity = semantic_ctx.query.complexity
        logger.info(
            "NLU: intent=%s complexity=%s confidence=%.2f session=%s",
            intent.value, complexity.value, semantic_ctx.query.confidence, session_id[:8],
        )

        # ── 2. Кэш (только read-only) ─────────────────────────────────────────
        is_readonly = intent in (UserIntent.SUMMARIZE, UserIntent.QUESTION, UserIntent.SEARCH)
        cache_key: str | None = None
        if is_readonly:
            cache_key = self._make_cache_key(user_id, semantic_ctx.query.refined)
            cached = await self._cache_get(cache_key)
            if cached:
                logger.debug("Cache hit для session=%s", session_id[:8])
                return AgentResponse(
                    content=cached["content"],
                    session_id=session_id,
                    intent=intent.value,
                    model_used="cached",
                    latency_ms=int((time.monotonic() - start_ts) * 1000),
                    metadata={"from_cache": True},
                )

        # ── 3. Контекст краткосрочной памяти ─────────────────────────────────
        history = self._get_short_term(session_id)

        # ── 4. Выбор модели ───────────────────────────────────────────────────
        model = self._select_model(intent, semantic_ctx.query.confidence)

        # ── 5. ReAct цикл ─────────────────────────────────────────────────────
        try:
            response_text, tools_used = await self._react_cycle(
                user_message=semantic_ctx.query.refined,
                session_id=session_id,
                token=token,
                model=model,
                history=history,
                context=ctx,
            )
        except Exception as exc:
            logger.error("ReAct цикл упал для session=%s: %s", session_id[:8], exc, exc_info=True)
            response_text = "Произошла ошибка при обработке запроса. Попробуйте переформулировать."
            tools_used = []

        latency_ms = int((time.monotonic() - start_ts) * 1000)

        # ── 6. Обновляем краткосрочную память ─────────────────────────────────
        self._update_short_term(session_id, user_message, response_text)

        # ── 7. Логируем диалог в БД ───────────────────────────────────────────
        dialog_id = await self._log_dialog(
            user_id=user_id,
            session_id=session_id,
            user_query=user_message,
            intent=intent.value,
            tools_used=tools_used,
            response=response_text,
            model_used=model,
            latency_ms=latency_ms,
        )

        # ── 8. Кэшируем read-only результаты ─────────────────────────────────
        if cache_key and is_readonly and response_text:
            await self._cache_set(cache_key, {"content": response_text}, ttl=self.config.cache_ttl_seconds)

        return AgentResponse(
            content=response_text,
            session_id=session_id,
            dialog_id=dialog_id,
            intent=intent.value,
            tools_used=tools_used,
            model_used=model,
            latency_ms=latency_ms,
            metadata={
                "complexity": complexity.value,
                "confidence": round(semantic_ctx.query.confidence, 2),
            },
        )

    # ── ReAct цикл ────────────────────────────────────────────────────────────

    async def _react_cycle(
        self,
        user_message: str,
        session_id: str,
        token: str,
        model: str,
        history: list[dict],
        context: dict[str, Any],
    ) -> tuple[str, list[str]]:
        """Выполняет ReAct (Reasoning + Acting) цикл.

        Итерация:
        1. LLM решает: ответить напрямую или вызвать инструмент
        2. Если инструмент → вызов через MCP
        3. Результат добавляется в контекст
        4. Повторяем до max_react_iterations или финального ответа

        Args:
            user_message: Нормализованный запрос пользователя.
            session_id: ID сессии для логирования.
            token: JWT для инъекции в MCP-вызовы.
            model: Выбранная модель LLM.
            history: Краткосрочная история диалога.
            context: Контекст (document_id, file_path и т.д.).

        Returns:
            (response_text, tools_used_list)
        """
        from edms_ai_assistant.llm_client import get_llm_client

        mcp_tools = await self._list_mcp_tools()
        tools_for_llm = self._tools_to_llm_format(mcp_tools)
        tools_used: list[str] = []

        # Строим начальный список сообщений
        messages: list[dict] = []
        messages.extend(history[-20:])  # берём последние 20 из истории
        messages.append({"role": "user", "content": user_message})

        system = self._build_system_prompt(context)
        client = get_llm_client()

        for iteration in range(self.config.max_react_iterations):
            try:
                resp = await client.create(
                    model=model,
                    messages=messages,
                    tools=tools_for_llm if iteration < self.config.max_react_iterations - 1 else None,
                    system=system,
                    max_tokens=settings.LLM_MAX_TOKENS,
                )
            except Exception as exc:
                logger.error("LLM error на итерации %d: %s", iteration, exc)
                return f"Не удалось обработать запрос: {exc}", tools_used

            # Нет tool_use → финальный ответ
            if resp.stop_reason in ("end_turn", "stop"):
                texts = [b.text for b in resp.content if b.type == "text"]
                final_text = " ".join(texts).strip()
                if final_text:
                    return final_text, tools_used

            # Есть tool_use → выполняем инструменты
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                texts = [b.text for b in resp.content if b.type == "text"]
                return " ".join(texts).strip() or "Готово.", tools_used

            # Добавляем ответ ассистента в историю
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": b.type, "text": b.text} if b.type == "text"
                    else {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                    for b in resp.content
                ],
            })

            # Вызываем каждый инструмент и собираем результаты
            tool_results = []
            for tu in tool_uses:
                args = dict(tu.input)
                # Автоинъекция токена если инструмент его принимает
                tool_schema = next((t for t in mcp_tools if t.get("name") == tu.name), {})
                if "token" in tool_schema.get("inputSchema", {}).get("properties", {}):
                    args.setdefault("token", token)
                if "document_id" in args and not args["document_id"] and context.get("document_id"):
                    args["document_id"] = context["document_id"]

                logger.info("Tool call: %s args=%s", tu.name, str(args)[:120])
                result = await self._call_mcp_tool(tu.name, args)
                tools_used.append(tu.name)

                result_text = json.dumps(
                    result.get("data", {"error": result.get("error", "no data")}),
                    ensure_ascii=False, default=str,
                )[:3000]

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        return (
            "Достигнут лимит итераций. Уточните запрос или разбейте его на части.",
            tools_used,
        )

    # ── MCP ───────────────────────────────────────────────────────────────────

    async def _list_mcp_tools(self) -> list[dict]:
        """Возвращает список MCP-инструментов с кэшированием."""
        if self._mcp_tools_cache is not None:
            return self._mcp_tools_cache

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.config.mcp_server_url}/tools")
                if resp.is_success:
                    data = resp.json()
                    self._mcp_tools_cache = data.get("tools", [])
                    return self._mcp_tools_cache
        except Exception as exc:
            logger.warning("Не удалось загрузить MCP инструменты: %s", exc)

        # Fallback: загружаем из локального реестра
        import json
        from pathlib import Path
        registry_path = Path(__file__).parent.parent / "mcp_server" / "tools_registry.json"
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
            self._mcp_tools_cache = registry.get("tools", [])
            return self._mcp_tools_cache
        return []

    async def _call_mcp_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Вызывает MCP инструмент через HTTP bridge.

        Args:
            tool_name: Имя инструмента.
            args: Аргументы вызова.

        Returns:
            {"success": bool, "data": Any, "error": str | None}
        """
        try:
            async with httpx.AsyncClient(timeout=self.config.agent_timeout) as client:
                resp = await client.post(
                    f"{self.config.mcp_server_url}/call-tool",
                    json={"tool_name": tool_name, "arguments": args},
                )
                if resp.status_code == 404:
                    return {"success": False, "error": f"Инструмент '{tool_name}' не найден"}
                resp.raise_for_status()
                data = resp.json()
                return {"success": data.get("success", True), "data": data.get("result")}
        except httpx.TimeoutException:
            return {"success": False, "error": f"Таймаут вызова инструмента {tool_name}"}
        except Exception as exc:
            logger.error("MCP tool '%s' error: %s", tool_name, exc)
            return {"success": False, "error": str(exc)}

    def _tools_to_llm_format(self, tools: list[dict]) -> list[dict]:
        """Конвертирует MCP tools в формат для LLMClient (Anthropic tool_use)."""
        return [
            {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "input_schema": t.get("inputSchema", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

    # ── Память ────────────────────────────────────────────────────────────────

    def _get_short_term(self, session_id: str) -> list[dict]:
        """Возвращает краткосрочную историю сессии."""
        return self._short_term.get(session_id, [])

    def _update_short_term(self, session_id: str, user_msg: str, ai_msg: str) -> None:
        """Обновляет краткосрочную историю, обрезая при превышении лимита."""
        MAX_MESSAGES = settings.AGENT_MAX_CONTEXT_MESSAGES
        history = self._short_term.setdefault(session_id, [])
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ai_msg})
        # Обрезаем, сохраняя последние MAX_MESSAGES сообщений
        if len(history) > MAX_MESSAGES:
            self._short_term[session_id] = history[-MAX_MESSAGES:]

    async def get_thread_history(self, session_id: str) -> list[dict]:
        """Возвращает историю треда для GET /chat/history/{thread_id}."""
        return self._get_short_term(session_id)

    # ── Модель ────────────────────────────────────────────────────────────────

    def _select_model(self, intent: UserIntent, confidence: float) -> str:
        """Выбирает модель LLM по intent и уверенности NLU.

        Логика:
            Простые/известные intents + высокая уверенность → haiku (быстро, дёшево)
            Средняя сложность → sonnet
            Планирование / низкая уверенность → opus
        """
        simple_intents = {UserIntent.QUESTION, UserIntent.SUMMARIZE}
        complex_intents = {UserIntent.COMPOSITE, UserIntent.UNKNOWN, UserIntent.COMPARE}

        if intent in simple_intents and confidence > 0.8:
            return settings.MODEL_EXPLAINER   # haiku-эквивалент
        if intent in complex_intents or confidence < 0.5:
            return settings.MODEL_PLANNER     # opus-эквивалент
        return settings.MODEL_EXECUTOR        # sonnet-эквивалент

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        """Строит системный промпт с контекстом документа/пользователя."""
        from pathlib import Path
        prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"
        base = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else (
            "Ты — ИИ-ассистент корпоративной EDMS. Отвечай на русском языке."
        )
        if context.get("document_id"):
            base += f"\n\n<session_context>Активный документ: {context['document_id']}</session_context>"
        return base

    # ── Кэш (Redis) ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_cache_key(user_id: str, normalized_query: str) -> str:
        """Генерирует ключ кэша: sha256(user_id + normalized_query)."""
        raw = f"{user_id}:{normalized_query}"
        return f"agent:cache:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def _cache_get(self, key: str) -> dict | None:
        """Читает из Redis с fallback при недоступности."""
        try:
            redis = get_redis()
            raw = await redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def _cache_set(self, key: str, value: dict, ttl: int = 300) -> None:
        """Записывает в Redis с TTL."""
        try:
            redis = get_redis()
            await redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception as exc:
            logger.debug("Cache set failed: %s", exc)

    # ── Логирование и фидбек ──────────────────────────────────────────────────

    async def _log_dialog(
        self,
        user_id: str,
        session_id: str,
        user_query: str,
        intent: str,
        tools_used: list[str],
        response: str,
        model_used: str,
        latency_ms: int,
    ) -> str:
        """Сохраняет диалог в PostgreSQL через dialog_logs.

        Args:
            user_id: ID пользователя.
            session_id: ID сессии.
            user_query: Оригинальный запрос.
            intent: Определённый intent.
            tools_used: Список использованных инструментов.
            response: Ответ ассистента.
            model_used: Имя модели.
            latency_ms: Время обработки.

        Returns:
            dialog_id (UUID string).
        """
        dialog_id = str(uuid.uuid4())
        try:
            # Используем feedback_api через HTTP (расцепление сервисов)
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(
                    f"{settings.FEEDBACK_API_URL}/dialogs",
                    json={
                        "id": dialog_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "user_query": user_query,
                        "intent": intent,
                        "selected_tool": tools_used[0] if tools_used else "",
                        "final_response": response,
                        "model_used": model_used,
                        "latency_ms": latency_ms,
                    },
                )
        except Exception as exc:
            logger.debug("Не удалось сохранить диалог в feedback-collector: %s", exc)
        return dialog_id

    async def save_feedback(self, dialog_id: str, rating: int, comment: str = "") -> bool:
        """Сохраняет оценку пользователя через feedback-collector.

        Args:
            dialog_id: UUID диалога.
            rating: -1 (плохо), 0 (нейтрально), 1 (хорошо).
            comment: Текстовый комментарий.

        Returns:
            True если успешно сохранено.
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    f"{settings.FEEDBACK_API_URL}/feedback",
                    json={"dialog_id": dialog_id, "rating": rating, "comment": comment},
                )
                return resp.is_success
        except Exception as exc:
            logger.error("Ошибка сохранения фидбека: %s", exc)
            return False

    async def update_rag_from_feedback(self, dialog_id: str) -> None:
        """Обновляет RAG после положительного фидбека.

        Уведомляет оркестратор (self) о необходимости добавить диалог в RAG.
        Конкретная реализация — в feedback-collector через APScheduler.
        """
        logger.debug("RAG update triggered for dialog %s", dialog_id)

    async def rebuild_rag(self) -> None:
        """Запускает полную перестройку RAG-индекса.

        Делегирует в feedback-collector POST /rag/rebuild.
        """
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                await client.post(f"{settings.FEEDBACK_API_URL}/rag/rebuild")
                logger.info("RAG rebuild запущен")
        except Exception as exc:
            logger.error("Ошибка запуска RAG rebuild: %s", exc)

    async def health_check(self) -> dict[str, Any]:
        """Проверяет состояние всех компонентов.

        Returns:
            Словарь {компонент: статус}.
        """
        status: dict[str, Any] = {}

        # MCP
        try:
            tools = await self._list_mcp_tools()
            status["mcp_server"] = f"healthy ({len(tools)} tools)"
        except Exception as exc:
            status["mcp_server"] = f"unhealthy: {exc}"

        # Redis
        try:
            redis = get_redis()
            await redis.ping()
            status["redis"] = "healthy"
        except Exception as exc:
            status["redis"] = f"unhealthy: {exc}"

        # PostgreSQL
        try:
            async with AsyncSessionLocal() as db:
                await db.execute(__import__("sqlalchemy").text("SELECT 1"))
            status["postgresql"] = "healthy"
        except Exception as exc:
            status["postgresql"] = f"unhealthy: {exc}"

        # Feedback collector
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{settings.FEEDBACK_API_URL}/health")
                status["feedback_collector"] = "healthy" if resp.is_success else f"unhealthy: {resp.status_code}"
        except Exception as exc:
            status["feedback_collector"] = f"unhealthy: {exc}"

        return status