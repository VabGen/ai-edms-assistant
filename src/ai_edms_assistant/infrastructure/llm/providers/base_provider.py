# src/ai_edms_assistant/infrastructure/llm/providers/base_provider.py
"""Base LangChain-backed implementation of AbstractLLMProvider.

Subclasses only need to implement the ``_chat_model`` property.
All protocol methods (complete, stream, embed, model_name, supports_tools)
are provided here.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel

from ....application.ports import AbstractLLMProvider, LLMMessage, LLMResponse


class BaseLLMProvider(AbstractLLMProvider):
    """Base implementation of AbstractLLMProvider using LangChain.

    Provides common utilities for converting between application-layer
    ``LLMMessage``/``LLMResponse`` and LangChain's ``BaseMessage`` types.

    Subclasses must implement:
        ``_chat_model`` — returns the concrete LangChain model instance.
    """

    # ── Abstract contract ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def _chat_model(self) -> BaseChatModel:
        """Returns the underlying LangChain chat model.

        Must be implemented by subclasses to provide the concrete model
        instance (``SafeChatOpenAI``, ``ChatAnthropic``, etc.).

        Returns:
            Configured ``BaseChatModel`` instance.
        """
        ...

    # ── AbstractLLMProvider implementation ───────────────────────────────────

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate a single completion from the LangChain model.

        Converts ``LLMMessage`` list → LangChain messages, invokes the model,
        and converts the response back to ``LLMResponse``.

        Args:
            messages: Conversation history.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Max tokens to generate. ``None`` uses model default.
            tools: Optional tool definitions for function calling.

        Returns:
            ``LLMResponse`` with content, tool calls, and usage stats.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
        bind_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens

        bound_model = self._chat_model.bind(**bind_kwargs)

        if tools:
            bound_model = bound_model.bind_tools(tools)

        response = await bound_model.ainvoke(lc_messages)

        tool_calls: list[dict[str, Any]] = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = [
                {
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc.get("id", ""),
                }
                for tc in response.tool_calls
            ]

        usage: dict[str, int] | None = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                "completion_tokens": response.usage_metadata.get("output_tokens", 0),
                "total_tokens": response.usage_metadata.get("total_tokens", 0),
            }

        return LLMResponse(
            content=str(response.content),
            tool_calls=tool_calls,
            finish_reason=getattr(response, "finish_reason", None),
            usage=usage,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion tokens as they are generated.

        Args:
            messages: Conversation history.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Yields:
            Individual text chunks from the model stream.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))

        bind_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens

        bound_model = self._chat_model.bind(**bind_kwargs)

        async for chunk in bound_model.astream(lc_messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings.

        ``BaseLLMProvider`` does not provide embeddings by default — a
        separate embedding model is required. Override in subclasses that
        bundle both generation and embedding capabilities.

        Args:
            texts: List of strings to embed.

        Raises:
            NotImplementedError: Always — embeddings need a dedicated model.
        """
        raise NotImplementedError(
            "Embeddings are not supported by this provider. "
            "Use a dedicated embedding model (e.g. OpenAIEmbeddings)."
        )

    # ── Property implementations ─────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Returns the underlying model identifier.

        Returns:
            Model name string from the LangChain model instance,
            e.g. ``"gpt-4o-mini"``, ``"default-llm-model"``.
        """
        return getattr(self._chat_model, "model_name", "unknown")

    @property
    def supports_tools(self) -> bool:
        """Returns True when the underlying model supports tool calling.

        Returns:
            ``True`` for models that implement ``bind_tools`` (OpenAI, Anthropic).
        """
        return hasattr(self._chat_model, "bind_tools")
