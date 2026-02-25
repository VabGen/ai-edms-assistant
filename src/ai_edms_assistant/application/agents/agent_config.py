# src/ai_edms_assistant/application/agents/agent_config.py
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for EDMS agent behavior.

    Centralizes all agent-level settings to avoid hardcoded constants
    scattered across the codebase.

    Attributes:
        max_iterations: Maximum reasoning loop iterations before timeout.
        timeout_seconds: Global execution timeout for agent invocation.
        temperature: LLM sampling temperature (0.0 = deterministic).
        max_tokens: Maximum completion tokens per LLM call.
        enable_streaming: Whether to stream responses token-by-token.
        enable_tool_validation: Whether to run post-tool validation checks.
    """

    max_iterations: int = Field(default=10, ge=1, le=50)
    timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=100)
    enable_streaming: bool = Field(default=False)
    enable_tool_validation: bool = Field(default=True)

    class Config:
        frozen = True
