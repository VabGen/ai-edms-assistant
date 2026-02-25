# src/ai_edms_assistant/application/ports/llm_port.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """Single message in a conversation with an LLM.

    Maps to LangChain's ``BaseMessage`` abstraction but avoids direct
    LangChain dependency in the port definition.

    Attributes:
        role: Message role — ``"system"``, ``"user"``, ``"assistant"``,
            or ``"tool"`` (for tool invocation results).
        content: Text content of the message.
        name: Optional name identifier for the message sender.
        tool_calls: List of tool invocation requests from the assistant.
        tool_call_id: ID linking a tool message to its originating call.
    """

    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_id: str | None = None


class LLMResponse(BaseModel):
    """Response from an LLM completion request.

    Attributes:
        content: Generated text content.
        tool_calls: Tool invocations requested by the model.
        finish_reason: Reason for completion (``"stop"``, ``"length"``,
            ``"tool_calls"``).
        usage: Token usage statistics (prompt + completion).
    """

    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


class AbstractLLMProvider(ABC):
    """Port (interface) for Large Language Model providers.

    Defines the contract between the application layer and LLM infrastructure
    implementations (OpenAI, Anthropic, local models). The application layer
    depends only on this ABC — concrete providers live in
    ``infrastructure/llm/providers/``.

    Architecture:
        - Application layer receives ``AbstractLLMProvider`` via DI.
        - Infrastructure layer provides ``OpenAIProvider``, ``AnthropicProvider``,
          etc., all implementing this interface.
        - Switch providers by changing the DI binding — zero application code change.

    Example:
        >>> # In a use case
        >>> class SummarizeUseCase:
        ...     def __init__(self, llm: AbstractLLMProvider) -> None:
        ...         self._llm = llm
        ...
        ...     async def execute(self, text: str) -> str:
        ...         response = await self._llm.complete(
        ...             messages=[LLMMessage(role="user", content=f"Summarize: {text}")]
        ...         )
        ...         return response.content
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate a single completion from the LLM.

        Args:
            messages: Conversation history as a list of ``LLMMessage`` objects.
            temperature: Sampling temperature in [0.0, 2.0]. Lower = more
                deterministic. Default 0.0 for reasoning tasks.
            max_tokens: Maximum tokens to generate. ``None`` uses provider default.
            tools: Optional list of tool definitions (LangChain / OpenAI format).

        Returns:
            ``LLMResponse`` containing the generated content and metadata.
        """

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion tokens as they are generated.

        Used for real-time agent responses in the FastAPI endpoint.

        Args:
            messages: Conversation history.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Individual text chunks as they are produced by the model.
        """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text strings.

        Used by the vector store to embed documents for semantic search.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
            Each vector is a list of floats, typically 1536-dimensional for
            OpenAI ``text-embedding-ada-002``.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the identifier of the underlying model.

        Returns:
            Model name string, e.g. ``"gpt-4"``, ``"claude-sonnet-4"``.
        """

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Returns True when the provider supports tool / function calling.

        Returns:
            ``True`` for providers that support tool invocation (OpenAI,
            Anthropic), ``False`` otherwise.
        """
