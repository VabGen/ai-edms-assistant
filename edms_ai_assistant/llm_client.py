# edms_ai_assistant/llm_client.py
"""
Единый LLM-адаптер для всего EDMS AI Assistant.

Поддерживает два бэкенда:
    1. Ollama / OpenAI-compatible API  — через httpx (нативный JSON)
    2. Anthropic API                   — через anthropic.AsyncAnthropic

Ключевое различие форматов tool_result:
    Anthropic:  role=user, content=[{type:tool_result, tool_use_id, content}]
    OpenAI:     role=tool, tool_call_id, content (строка)

OllamaClient.create() конвертирует историю сообщений из Anthropic-формата
в OpenAI-формат перед отправкой, поэтому весь остальной код (agent.py,
ReAct-цикл) использует единственный Anthropic-формат внутри.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


def resolve_model(model_name: str) -> str:
    """Резолвит claude-* алиасы в модели из .env."""
    if not model_name.startswith("claude-"):
        return model_name
    mapping = {
        "claude-opus-4-5": settings.MODEL_PLANNER,
        "claude-opus-4-6": settings.MODEL_PLANNER,
        "claude-sonnet-4-6": settings.MODEL_EXECUTOR,
        "claude-sonnet-4-5": settings.MODEL_EXECUTOR,
        "claude-haiku-4-5": settings.MODEL_EXPLAINER,
        "claude-haiku-4-6": settings.MODEL_EXPLAINER,
        "claude-opus": settings.MODEL_PLANNER,
        "claude-sonnet": settings.MODEL_EXECUTOR,
        "claude-haiku": settings.MODEL_EXPLAINER,
    }
    resolved = mapping.get(model_name, settings.MODEL_EXECUTOR)
    logger.debug("Model resolved: %s → %s", model_name, resolved)
    return resolved


@dataclass
class ContentBlock:
    """Блок контента в ответе LLM."""

    type: str  # text | tool_use
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Унифицированный ответ LLM (совместим с anthropic.types.Message)."""

    content: list[ContentBlock]
    stop_reason: str = "end_turn"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.stop_reason in ("stop", "length"):
            self.stop_reason = "end_turn"


def _convert_messages_to_openai(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Конвертирует историю из Anthropic-формата в OpenAI-формат.

    Anthropic tool_result (внутренний формат агента):
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "id", "content": "..."}]}

    OpenAI tool message (что принимает Ollama):
        {"role": "tool", "tool_call_id": "id", "content": "..."}

    Anthropic tool_use в ответе ассистента:
        {"role": "assistant", "content": [{"type": "tool_use", "id": "id", "name": "fn", "input": {...}},
                                           {"type": "text", "text": "..."}]}

    OpenAI tool_calls в ответе ассистента:
        {"role": "assistant", "content": "...", "tool_calls": [{"id": "id", "type": "function",
          "function": {"name": "fn", "arguments": "{...}"}}]}
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # ── Обычное текстовое сообщение ────────────────────────────────────
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        # ── content — список блоков (Anthropic формат) ─────────────────────
        if isinstance(content, list):
            # Случай 1: user сообщение с tool_result блоками
            tool_results = [
                b
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if tool_results and role == "user":
                # Каждый tool_result → отдельное tool-сообщение в OpenAI формате
                for tr in tool_results:
                    tool_content = tr.get("content", "")
                    # content может быть списком [{type:text, text:...}]
                    if isinstance(tool_content, list):
                        tool_content = " ".join(
                            b.get("text", "")
                            for b in tool_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", ""),
                            "content": str(tool_content),
                        }
                    )
                continue

            # Случай 2: assistant сообщение с text и/или tool_use блоками
            if role == "assistant":
                text_parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                tool_uses = [
                    b
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                ]

                openai_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": " ".join(text_parts).strip() or None,
                }

                if tool_uses:
                    openai_msg["tool_calls"] = [
                        {
                            "id": tu.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tu.get("name", ""),
                                "arguments": json.dumps(
                                    tu.get("input", {}), ensure_ascii=False
                                ),
                            },
                        }
                        for i, tu in enumerate(tool_uses)
                    ]

                result.append(openai_msg)
                continue

            # Случай 3: user сообщение с текстовыми блоками (не tool_result)
            text_content = " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if text_content:
                result.append({"role": role, "content": text_content})

    return result


class OllamaClient:
    """Клиент для Ollama / OpenAI-compatible API."""

    def __init__(self) -> None:
        self._base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self._timeout = settings.LLM_TIMEOUT

    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.6,
    ) -> LLMResponse:
        """Вызывает Ollama /v1/chat/completions с конвертацией формата сообщений."""
        # Системное сообщение
        chat_messages: list[dict[str, Any]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})

        # Конвертируем историю из Anthropic → OpenAI формата
        chat_messages.extend(_convert_messages_to_openai(messages))

        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get(
                            "input_schema", {"type": "object", "properties": {}}
                        ),
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
            raise TimeoutError(f"Ollama timeout after {self._timeout}s") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama HTTP {exc.response.status_code}: {exc.response.text[:500]}"
            ) from exc

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

        text_content = message.get("content") or ""
        if text_content:
            content_blocks.append(ContentBlock(type="text", text=text_content))

        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            content_blocks.append(
                ContentBlock(
                    type="tool_use",
                    id=tc.get("id", f"call_{len(content_blocks)}"),
                    name=fn.get("name", ""),
                    input=args,
                )
            )

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


class AnthropicAdapter:
    """Обёртка над anthropic.AsyncAnthropic."""

    def __init__(self) -> None:
        self._available = False
        self._client: Any = None
        if settings.ANTHROPIC_API_KEY:
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic(
                    api_key=settings.ANTHROPIC_API_KEY.get_secret_value()
                )
                self._available = True
            except ImportError:
                logger.warning("anthropic SDK not installed")

    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.6,
    ) -> LLMResponse:
        """Вызывает Anthropic API (уже принимает Anthropic-формат напрямую)."""
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
                content_blocks.append(
                    ContentBlock(
                        type="tool_use",
                        id=block.id,
                        name=block.name,
                        input=block.input or {},
                    )
                )

        return LLMResponse(
            content=content_blocks,
            stop_reason=response.stop_reason or "end_turn",
            model=response.model,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )


class LLMClient:
    """Единый LLM-клиент с автовыбором бэкенда (Anthropic → Ollama)."""

    def __init__(self) -> None:
        self._anthropic = AnthropicAdapter()
        self._ollama = OllamaClient()
        self._backend = "anthropic" if self._anthropic._available else "ollama"
        logger.info(
            "LLMClient: backend=%s model=%s", self._backend, settings.MODEL_NAME
        )

    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Создаёт LLM completion, автоматически конвертируя формат для Ollama."""
        resolved_model = resolve_model(model)
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        start_ts = time.monotonic()

        try:
            if self._backend == "anthropic":
                result = await self._anthropic.create(
                    model=resolved_model,
                    messages=messages,
                    tools=tools,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temp,
                )
            else:
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
                "LLM: model=%s backend=%s stop=%s tokens_out=%d latency_ms=%d",
                resolved_model,
                self._backend,
                result.stop_reason,
                result.output_tokens,
                elapsed,
            )
            return result

        except Exception as exc:
            elapsed = round((time.monotonic() - start_ts) * 1000)
            logger.error(
                "LLM error: model=%s backend=%s error=%s latency_ms=%d",
                resolved_model,
                self._backend,
                exc,
                elapsed,
            )
            raise

    def model_for_role(self, role: str) -> str:
        """Возвращает имя модели для роли агента из .env."""
        role_map = {
            "planner": settings.MODEL_PLANNER,
            "researcher": settings.MODEL_RESEARCHER,
            "executor": settings.MODEL_EXECUTOR,
            "reviewer": settings.MODEL_REVIEWER,
            "explainer": settings.MODEL_EXPLAINER,
        }
        return role_map.get(role, settings.MODEL_NAME)

    @property
    def backend(self) -> str:
        return self._backend


async def get_llm_response(prompt: str, system: str | None = None) -> str:
    """Быстрый helper для простых LLM-вызовов без истории."""
    client = get_llm_client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    response = await client.create(
        model=settings.MODEL_NAME,
        messages=messages,
        system=system,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
    texts = [b.text for b in response.content if b.type == "text"]
    return " ".join(texts).strip()


_llm_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Возвращает синглтон LLMClient."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance