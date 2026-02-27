# src/ai_edms_assistant/application/agents/agent_config.py
"""Agent configuration model."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


_GRAPH_HARD_LIMIT: int = 10


class AgentConfig(BaseModel):
    """Configuration for EDMS agent behavior.

    Centralises all agent-level settings. Frozen after construction
    (``frozen=True``) — shared safely across async tasks without copy.

    Attributes:
        max_iterations: Application-level recursion depth limit
            for ``_orchestrate()``. Must be <= ``_MAX_GRAPH_ITERATIONS``
            (the graph-level guard). Enforced by ``validate_iterations``.
        timeout_seconds: Per-invocation execution timeout (seconds).
        temperature: LLM sampling temperature.
        max_tokens: Maximum tokens per LLM completion.
        enable_streaming: Enable token-streaming responses.
        enable_tool_validation: Enable tool result validation node
            in the graph. When True, empty / error results trigger a
            system notification injected into the message chain.

    Invariant:
        ``max_iterations <= _GRAPH_HARD_LIMIT`` — application guard fires
        before graph guard. Violation raises ``ValueError`` at construction.

    Example:
        >>> config = AgentConfig(max_iterations=5, timeout_seconds=60.0)
        >>> config.execution_timeout
        60.0
    """

    max_iterations: int = Field(default=10, ge=1, le=50)
    timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=100)
    enable_streaming: bool = Field(default=False)
    enable_tool_validation: bool = Field(default=True)

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_iterations(self) -> "AgentConfig":
        """Ensure max_iterations does not exceed the graph-level hard limit.

        Prevents a configuration where the application-layer guard (iteration
        counter) is set higher than the graph-layer guard
        (``_MAX_GRAPH_ITERATIONS``), which would make the application guard
        unreachable and allow uncontrolled recursion.

        Returns:
            Validated ``AgentConfig`` instance.

        Raises:
            ValueError: When ``max_iterations > _GRAPH_HARD_LIMIT``.
        """
        if self.max_iterations > _GRAPH_HARD_LIMIT:
            raise ValueError(
                f"AgentConfig.max_iterations={self.max_iterations} exceeds "
                f"the graph-level hard limit _MAX_GRAPH_ITERATIONS="
                f"{_GRAPH_HARD_LIMIT}. "
                f"Set max_iterations <= {_GRAPH_HARD_LIMIT} or raise "
                f"_MAX_GRAPH_ITERATIONS in edms_agent.py first."
            )
        return self

    @property
    def execution_timeout(self) -> float:
        """Execution timeout in seconds.

        Alias for ``timeout_seconds`` — provides a descriptive name for
        callers that pass this to ``asyncio.wait_for()``.

        Returns:
            Execution timeout in seconds.
        """
        return self.timeout_seconds