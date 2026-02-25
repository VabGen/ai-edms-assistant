# src/ai_edms_assistant/application/services/__init__.py
"""Application-level services.

Services in this layer orchestrate domain operations and provide
additional capabilities that don't belong in domain (like semantic
analysis, NLP extraction, etc.).

Services:
    SemanticDispatcher: Analyzes user queries and detects intent.
    UserIntent: Enum of detectable user intents.
    QueryAnalysis: Analyzed query result.
    SemanticContext: Complete semantic context for agent execution.

Example:
    >>> from ai_edms_assistant.application.services import SemanticDispatcher
    >>>
    >>> dispatcher = SemanticDispatcher()
    >>> context = dispatcher.build_context("Создай поручение для Иванова")
    >>> context.query.intent
    <UserIntent.CREATE_TASK: 'create_task'>
"""

from .semantic_dispatcher import (
    QueryAnalysis,
    SemanticContext,
    SemanticDispatcher,
    UserIntent,
)

__all__ = [
    "SemanticDispatcher",
    "UserIntent",
    "QueryAnalysis",
    "SemanticContext",
]
