# orchestrator/llm.py
"""
Единый LLM-адаптер EDMS AI Assistant.

Поддерживает два бэкенда:
    1. Ollama / OpenAI-compatible API  — через httpx (нативный JSON)
    2. Anthropic API                   — через anthropic.AsyncAnthropic

Выбор бэкенда:
    - Если ANTHROPIC_API_KEY задан → Anthropic
    - Иначе → Ollama (OLLAMA_BASE_URL + MODEL_NAME)

Экспортирует:
    LLMMessage        — унифицированное сообщение
    LLMResponse       — унифицированный ответ
    LLMClient         — основной класс
    get_llm_client()  — синглтон

Совместимость с multi_agent.py:
    MultiAgentCoordinator получает LLMClient и вызывает
    client.create(model, messages, tools, system, max_tokens)
    который возвращает объект, совместимый с anthropic.types.Message
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

# ── Конфигурация из .env ──────────────────────────────────────────────────

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Модели для ролей агентов
_MODEL_PLANNER   = os.getenv("MODEL_PLANNER",   os.getenv("MODEL_NAME", "gpt-oss:120b-cloud"))
_MODEL_RESEARCHER = os.getenv("MODEL_RESEARCHER", os.getenv("MODEL_NAME", "gpt-oss:120b-cloud"))
_MODEL_EXECUTOR  = os.getenv("MODEL_EXECUTOR",  os.getenv("MODEL_NAME", "gpt-oss:120b-cloud"))
_MODEL_REVIEWER  = os.getenv("MODEL_REVIEWER",  os.getenv("MODEL_NAME", "gpt-oss:120b-cloud"))
_MODEL_EXPLAINER = os.getenv("MODEL_EXPLAINER", os.getenv("MODEL_NAME", "gpt-oss:120b-cloud"))

# Маппинг claude-* → реальные модели из .env
# Используется когда multi_agent.py передаёт хардкодный claude-* идентификатор
_CLAUDE_TO_ENV_MODEL: dict[str, str] = {
    "claude-opus-4-5":   _MODEL_PLANNER,
    "claude-opus-4-6":   _MODEL_PLANNER,
    "claude-sonnet-4-6": _MODEL_EXECUTOR,
    "claude-sonnet-4-5": _MODEL_EXECUTOR,
    "claude-haiku-4-5":  _MODEL_EXPLAINER,
    "claude-haiku-4-6":  _MODEL_EXPLAINER,
    # Общие алиасы
    "claude-opus":       _MODEL_PLANNER,
    "claude-sonnet":     _MODEL_EXECUTOR,
    "claude-haiku":      _MODEL_EXPLAINER,
}


def resolve_model(model_name: str) -> str:
    """
    Резолвит имя модели.

    claude-* → переменная из .env
    Всё остальное → без изменений (уже реальное имя модели)
    """
    if model_name.startswith("claude-"):
        resolved = _CLAUDE_TO_ENV_MODEL.get(model_name, _MODEL_EXECUTOR)
        logger.debug("Model resolved: %s → %s", model_name, resolved)
        return resolved
    return model_name


# ── Унифицированные типы ──────────────────────────────────────────────────


@dataclass
class LLMMessage:
    """Унифицированное сообщение для LLM."""
    role: str  # user | assistant | system
    content: str | list[dict[str, Any]]


@dataclass
class ContentBlock:
    """Блок контента в ответе LLM (совместим с anthropic.types)."""
    type: str               # text | tool_use
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """
    Унифицированный ответ LLM.

    Совместим с anthropic.types.Message по атрибутам:
        .content     — list[ContentBlock]
        .stop_reason — "end_turn" | "tool_use"
        .model       — имя модели
    """
    content: list[ContentBlock]
    stop_reason: str = "end_turn"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        # Совместимость: если stop_reason = "stop" (OpenAI) → нормализуем
        if self.stop_reason in ("stop", "length"):
            self.stop_reason = "end_turn"


# ── Ollama HTTP клиент ────────────────────────────────────────────────────


class OllamaClient:
    """
    Клиент для Ollama / OpenAI-compatible API.

    Использует эндпоинт: POST {OLLAMA_BASE_URL}/v1/chat/completions
    (Ollama совместим с OpenAI Chat Completions API начиная с v0.1.14)

    Tool use через OpenAI tool_calls формат.
    """

    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.6,
    ) -> LLMResponse:
        """
        Вызывает Ollama /v1/chat/completions.

        Конвертирует Anthropic-формат tools → OpenAI-формат functions.
        Конвертирует ответ OpenAI → LLMResponse.
        """
        # Строим список сообщений (добавляем system если нужен)
        chat_messages: list[dict[str, Any]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        # Конвертируем Anthropic tools → OpenAI tools
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]
            payload["tool_choice"] = "auto"

        url = f"{self._base_url}/v1/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"Ollama timeout after {self._timeout}s: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Ollama HTTP {exc.response.status_code}: {exc.response.text[:500]}") from exc

        return self._parse_response(data, model)

    def _parse_response(self, data: dict[str, Any], model: str) -> LLMResponse:
        """Парсит OpenAI-формат ответа → LLMResponse."""
        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(content=[], stop_reason="end_turn", model=model)

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        content_blocks: list[ContentBlock] = []

        # Текстовый контент
        text_content = message.get("content") or ""
        if text_content:
            content_blocks.append(ContentBlock(type="text", text=text_content))

        # Tool calls
        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}

            content_blocks.append(ContentBlock(
                type="tool_use",
                id=tc.get("id", f"call_{len(content_blocks)}"),
                name=fn.get("name", ""),
                input=args,
            ))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        if finish_reason == "length":
            stop_reason = "end_turn"

        usage = data.get("usage", {})
        return LLMResponse(
            content=content_blocks,
            stop_reason=stop_reason,
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


# ── Anthropic обёртка ─────────────────────────────────────────────────────


class AnthropicAdapter:
    """
    Обёртка над anthropic.AsyncAnthropic с унификацией вывода → LLMResponse.
    """

    def __init__(self, api_key: str) -> None:
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._available = True
        except ImportError:
            logger.warning("anthropic SDK not installed")
            self._client = None
            self._available = False

    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.6,
    ) -> LLMResponse:
        if not self._available or self._client is None:
            raise RuntimeError("Anthropic SDK not available")

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        content_blocks: list[ContentBlock] = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                content_blocks.append(ContentBlock(type="text", text=block.text))
            elif hasattr(block, "type") and block.type == "tool_use":
                content_blocks.append(ContentBlock(
                    type="tool_use",
                    id=block.id,
                    name=block.name,
                    input=block.input or {},
                ))

        return LLMResponse(
            content=content_blocks,
            stop_reason=response.stop_reason or "end_turn",
            model=response.model,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )


# ── Главный LLM клиент ────────────────────────────────────────────────────


class LLMClient:
    """
    Единый LLM-клиент с автовыбором бэкенда.

    Приоритет:
        1. Anthropic API (если ANTHROPIC_API_KEY задан и не пустой)
        2. Ollama (OLLAMA_BASE_URL)

    Используется в agent.py и multi_agent.py вместо прямых вызовов anthropic.
    """

    def __init__(self) -> None:
        self._backend: str
        self._ollama: OllamaClient | None = None
        self._anthropic: AnthropicAdapter | None = None

        if _ANTHROPIC_API_KEY:
            self._anthropic = AnthropicAdapter(_ANTHROPIC_API_KEY)
            self._backend = "anthropic"
            logger.info("LLMClient: using Anthropic backend")
        else:
            self._ollama = OllamaClient(_OLLAMA_BASE_URL)
            self._backend = "ollama"
            logger.info(
                "LLMClient: using Ollama backend at %s (model=%s)",
                _OLLAMA_BASE_URL, _MODEL_EXECUTOR,
            )

    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        """
        Создаёт LLM completion.

        Args:
            model:      Имя модели. claude-* автоматически резолвится через resolve_model().
            messages:   История сообщений в формате Anthropic/OpenAI.
            tools:      Инструменты в формате Anthropic (конвертируются при необходимости).
            system:     Системный промпт.
            max_tokens: Максимальное количество токенов в ответе.
            temperature: Температура (None = значение из .env).

        Returns:
            LLMResponse — унифицированный ответ.
        """
        resolved_model = resolve_model(model)
        temp = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.6"))
        start_ts = time.monotonic()

        try:
            if self._backend == "anthropic" and self._anthropic:
                result = await self._anthropic.create(
                    model=resolved_model,
                    messages=messages,
                    tools=tools,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temp,
                )
            else:
                assert self._ollama is not None
                result = await self._ollama.create(
                    model=resolved_model,
                    messages=messages,
                    tools=tools,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temp,
                )

            elapsed = round((time.monotonic() - start_ts) * 1000)
            logger.debug(
                "LLM call: model=%s backend=%s stop=%s tokens_out=%d latency_ms=%d",
                resolved_model, self._backend,
                result.stop_reason, result.output_tokens, elapsed,
            )
            return result

        except Exception as exc:
            elapsed = round((time.monotonic() - start_ts) * 1000)
            logger.error(
                "LLM error: model=%s backend=%s error=%s latency_ms=%d",
                resolved_model, self._backend, exc, elapsed,
            )
            raise

    @property
    def backend(self) -> str:
        return self._backend

    def model_for_role(self, role: str) -> str:
        """
        Возвращает имя модели для роли агента из .env.

        Роли: planner, researcher, executor, reviewer, explainer
        """
        role_map = {
            "planner":    _MODEL_PLANNER,
            "researcher": _MODEL_RESEARCHER,
            "executor":   _MODEL_EXECUTOR,
            "reviewer":   _MODEL_REVIEWER,
            "explainer":  _MODEL_EXPLAINER,
        }
        return role_map.get(role, os.getenv("MODEL_NAME", "gpt-oss:120b-cloud"))

    # mcp-server/llm.py
    """
    LLM-хелпер для MCP-сервера.
    Используется в reference_client (find_best_subject) и appeal_extraction_service.
    """


    async def get_llm_response(prompt: str, system: str | None = None) -> str:
        """
        Получить ответ от LLM через OpenAI-compatible API.

        Args:
            prompt: Пользовательское сообщение.
            system: Системный промпт (опционально).

        Returns:
            Текстовый ответ модели.
        """
        base_url = settings.LLM_GENERATIVE_URL.rstrip("/")
        model = settings.LLM_GENERATIVE_MODEL

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        return message.get("content", "").strip()


# ── Синглтон ──────────────────────────────────────────────────────────────

_llm_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Возвращает синглтон LLMClient."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance