# src/ai_edms_assistant/application/agents/agent_state.py
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State schema for LangGraph agent workflow.

    Uses LangGraph's ``add_messages`` reducer to automatically append new
    messages to the conversation history without duplicates.

    Attributes:
        messages: Conversation history. New messages are appended via
            ``add_messages`` reducer.
    """

    messages: Annotated[list[BaseMessage], add_messages]
