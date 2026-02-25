# src/ai_edms_assistant/infrastructure/llm/providers/base_provider.py
from __future__ import annotations

from abc import ABC

from langchain_core.language_models import BaseChatModel

from ....application.ports import AbstractLLMProvider, LLMMessage, LLMResponse


class BaseLLMProvider(AbstractLLMProvider, ABC):
    """Base implementation of AbstractLLMProvider using LangChain.

    Provides common utilities for converting between application-layer
    LLMMessage/LLMResponse and LangChain's BaseMessage types.

    Subclasses only need to provide the `_chat_model` property.
    """

    @property
    def _chat_model(self) -> BaseChatModel:
        """Returns the underlying LangChain chat model.

        Subclasses must implement this property to provide the concrete
        LangChain model instance (ChatOpenAI, ChatAnthropic, etc.).
        """
        raise NotImplementedError

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Generate completion using LangChain model."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        # Convert application messages to LangChain format
        lc_messages = []
        for msg in messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))

        # Bind model with parameters
        bound_model = self._chat_model.bind(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Bind tools if provided
        if tools:
            bound_model = bound_model.bind_tools(tools)

        # Invoke
        response = await bound_model.ainvoke(lc_messages)

        # Extract tool calls
        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = [
                {
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc.get("id", ""),
                }
                for tc in response.tool_calls
            ]

        # Extract usage
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                "completion_tokens": response.usage_metadata.get("output_tokens", 0),
                "total_tokens": response.usage_metadata.get("total_tokens", 0),
            }

        return LLMResponse(
            content=response.content,
            tool_calls=tool_calls,
            finish_reason=getattr(response, "finish_reason", None),
            usage=usage,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        """Stream completion tokens."""
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))

        bound_model = self._chat_model.bind(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        async for chunk in bound_model.astream(lc_messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings.

        NOTE: This requires a separate embeddings model.
        Subclasses should override if they provide embeddings.
        """
        raise NotImplementedError("Embeddings not supported by this provider")

    @property
    def model_name(self) -> str:
        """Returns model identifier."""
        return getattr(self._chat_model, "model_name", "unknown")

    @property
    def supports_tools(self) -> bool:
        """Returns True for providers with tool support."""
        return hasattr(self._chat_model, "bind_tools")
