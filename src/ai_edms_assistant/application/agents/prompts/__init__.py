# src/ai_edms_assistant/application/agents/prompts/__init__.py
"""Prompt templates for EdmsDocumentAgent."""

from .core_prompt import CORE_TEMPLATE
from .intent_guides import INTENT_GUIDES

__all__ = ["CORE_TEMPLATE", "INTENT_GUIDES"]