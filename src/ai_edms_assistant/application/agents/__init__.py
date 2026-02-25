# src/ai_edms_assistant/application/agents/__init__.py
"""Conversational agents for EDMS operations.

Agents orchestrate LLM reasoning, tool invocation, and use case execution
to respond to natural-language user queries. Built on LangGraph for
stateful multi-turn conversations.

Agents:
    AbstractAgent: Base class defining the agent contract.
    EdmsDocumentAgent: Main agent for document analysis, task creation,
        employee search, and appeal autofill operations.

Supporting types:
    AgentState: TypedDict for LangGraph state management.
    AgentConfig: Pydantic configuration for agent behavior.
"""

from .agent_config import AgentConfig
from .agent_state import AgentState
from .base_agent import AbstractAgent
from .edms_agent import EdmsDocumentAgent

__all__ = [
    "AbstractAgent",
    "AgentState",
    "AgentConfig",
    "EdmsDocumentAgent",
]
