# edms_ai_assistant/orchestrator/llm.py
"""
Тонкая обёртка для получения LLM-клиента в контексте оркестратора.

Единственный правильный способ получить LLM-экземпляр внутри пакета orchestrator:

    from edms_ai_assistant.orchestrator.llm import get_chat_model, get_llm_client

get_chat_model() → LangChain-совместимый объект (нужен для цепочек .pipe / |)
get_llm_client() → наш нативный LLMClient (быстрее, меньше зависимостей)

Оба делегируют в edms_ai_assistant.llm_client — единственный источник истины.
"""

from __future__ import annotations

from typing import Any

# Единственный источник истины — корневой llm_client.py
from edms_ai_assistant.llm_client import (  # noqa: F401 — re-export
    LLMClient,
    LLMResponse,
    ContentBlock,
    get_llm_client,
    get_llm_response,
    resolve_model,
)
from edms_ai_assistant.config import settings


def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> "_LangChainCompatWrapper":
    """Возвращает LangChain-совместимый объект для использования в цепочках.

    Используется там, где нужен .ainvoke() / .with_config() / .pipe() —
    например в AppealExtractionService вместе с LangChain OutputParser.

    Args:
        model: Имя модели. None → settings.MODEL_NAME.
        temperature: Температура. None → settings.LLM_TEMPERATURE.
        max_tokens: Максимум токенов. None → settings.LLM_MAX_TOKENS.

    Returns:
        Обёртка с методами .ainvoke(), .with_config(), .pipe().
    """
    return _LangChainCompatWrapper(
        model=model or settings.MODEL_NAME,
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
    )


class _LangChainCompatWrapper:
    """LangChain-совместимая обёртка над нашим LLMClient.

    Реализует минимальный интерфейс Runnable, необходимый для цепочек:
        prompt | llm | output_parser

    with_config() создаёт новый экземпляр с переопределёнными параметрами —
    это идиома LangChain для создания производных объектов без мутации оригинала.
    """

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._config: dict[str, Any] = config or {}

    def with_config(self, config: dict[str, Any]) -> "_LangChainCompatWrapper":
        """Создаёт копию с применёнными параметрами конфигурации.

        Поддерживаемые ключи в config:
            temperature (float), max_tokens (int), model (str)

        Args:
            config: Словарь с переопределяемыми параметрами.

        Returns:
            Новый экземпляр с применённой конфигурацией.
        """
        merged = {**self._config, **config}
        return _LangChainCompatWrapper(
            model=merged.get("model", self._model),
            temperature=merged.get("temperature", self._temperature),
            max_tokens=merged.get("max_tokens", self._max_tokens),
            config=merged,
        )

    def pipe(self, *others: Any) -> "_LangChainPipeline":
        """Создаёт цепочку: llm | parser | ...

        Args:
            *others: Последующие элементы цепочки (OutputParser и т.д.).

        Returns:
            _LangChainPipeline, поддерживающий .ainvoke().
        """
        return _LangChainPipeline(self, list(others))

    def __or__(self, other: Any) -> "_LangChainPipeline":
        """Поддержка синтаксиса: llm | parser."""
        return self.pipe(other)

    async def ainvoke(
        self,
        input_: Any,
        config: dict[str, Any] | None = None,
    ) -> "_FakeLLMMessage":
        """Вызов LLM.

        Args:
            input_: Строка промпта или список сообщений [{role, content}].
            config: Дополнительная конфигурация (игнорируется, для совместимости).

        Returns:
            Объект с атрибутом .content (str).
        """
        client = get_llm_client()

        if isinstance(input_, str):
            messages = [{"role": "user", "content": input_}]
        elif isinstance(input_, list):
            messages = input_
        else:
            # LangChain передаёт PromptValue — конвертируем
            messages = [{"role": "user", "content": str(input_)}]

        response = await client.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        texts = [b.text for b in response.content if b.type == "text"]
        return _FakeLLMMessage(" ".join(texts).strip())

    async def invoke(self, input_: Any, config: dict[str, Any] | None = None) -> "_FakeLLMMessage":
        """Синхронный вызов (делегирует в async через asyncio)."""
        import asyncio
        return await self.ainvoke(input_, config)


class _FakeLLMMessage:
    """Минимальный аналог LangChain AIMessage.

    Достаточно для работы с JsonOutputParser и StrOutputParser.
    """

    def __init__(self, content: str) -> None:
        self.content = content

    def __str__(self) -> str:
        return self.content


class _LangChainPipeline:
    """Простая цепочка: LLM → [parser, ...]

    Позволяет использовать:
        chain = prompt | llm | parser
        result = await chain.ainvoke({"text": "..."})
    """

    def __init__(self, llm: _LangChainCompatWrapper, steps: list[Any]) -> None:
        self._llm = llm
        self._steps = steps

    def __or__(self, other: Any) -> "_LangChainPipeline":
        return _LangChainPipeline(self._llm, self._steps + [other])

    async def ainvoke(self, input_: Any, config: dict[str, Any] | None = None) -> Any:
        """Выполняет цепочку: промпт → LLM → парсер → ...

        Args:
            input_: Входные данные (dict для PromptTemplate, str для прямого вызова).
            config: Конфигурация (для совместимости, не используется).

        Returns:
            Результат последнего шага цепочки.
        """
        # Первый шаг — PromptTemplate или строка
        current = input_
        llm_result = await self._llm.ainvoke(current)

        result: Any = llm_result
        for step in self._steps:
            if hasattr(step, "ainvoke"):
                result = await step.ainvoke(result)
            elif hasattr(step, "parse"):
                # OutputParser
                content = result.content if hasattr(result, "content") else str(result)
                result = step.parse(content)
            elif callable(step):
                result = step(result)

        return result