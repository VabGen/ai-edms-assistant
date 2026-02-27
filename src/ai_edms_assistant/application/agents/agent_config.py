# src/ai_edms_assistant/application/agents/agent_config.py
"""
Agent configuration model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """
    Configuration for EDMS agent behavior.
    Centralizes all agent-level settings.
    """

    max_iterations: int = Field(default=10, ge=1, le=50)
    timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=100)
    enable_streaming: bool = Field(default=False)
    enable_tool_validation: bool = Field(default=True)

    model_config = {"frozen": True}

    @property
    def execution_timeout(self) -> float:
        """
        Execution timeout in seconds.

        Returns:
            Execution timeout in seconds.
        """
        return self.timeout_seconds
