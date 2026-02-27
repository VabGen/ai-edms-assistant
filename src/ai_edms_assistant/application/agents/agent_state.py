# src/ai_edms_assistant/application/agents/agent_state.py
"""
LangGraph state schemas for EDMS agent workflow.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Base state schema for LangGraph agent workflow.

    Attributes:
        messages: Conversation history with add_messages reducer
            (auto-append, dedup by message.id).
    """

    messages: Annotated[list[BaseMessage], add_messages]


class AgentStateWithCounter(AgentState, total=False):
    """
    Extended state with iteration counter for infinite loop protection.
    """

    graph_iterations: Annotated[int, operator.add]
