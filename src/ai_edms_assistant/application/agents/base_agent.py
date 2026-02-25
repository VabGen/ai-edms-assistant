# src/ai_edms_assistant/application/agents/base_agent.py
from __future__ import annotations

from abc import ABC, abstractmethod

from ..dto import AgentRequest, AgentResponse


class AbstractAgent(ABC):
    """Base class for all conversational agents.

    Defines the contract for agent execution. Concrete agents (e.g.
    ``EdmsDocumentAgent``) implement the ``chat`` method to handle
    user requests and orchestrate tools.
    """

    @abstractmethod
    async def chat(self, request: AgentRequest) -> AgentResponse:
        """Process a user message and return the agent's response.

        Args:
            request: User input with message, token, context.

        Returns:
            Agent response with content, tool calls, metadata.
        """
