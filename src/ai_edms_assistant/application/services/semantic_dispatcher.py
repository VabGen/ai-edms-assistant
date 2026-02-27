# src/ai_edms_assistant/application/services/semantic_dispatcher.py
"""Semantic dispatcher for intent classification and query analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ...domain.entities.document import Document


class UserIntent(StrEnum):
    """User intent types."""

    CREATE_TASK = "create_task"
    CREATE_INTRODUCTION = "create_introduction"
    SUMMARIZE = "summarize"
    SEARCH = "search"
    COMPARE = "compare"
    ANALYZE = "analyze"
    QUESTION = "question"
    UNKNOWN = "unknown"


class QueryComplexity(StrEnum):
    """Query complexity levels."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class QueryAnalysis:
    """Analyzed user query."""

    original: str
    refined: str
    intent: UserIntent
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    confidence: float = 1.0


@dataclass
class SemanticContext:
    """Semantic context for agent execution."""

    query: QueryAnalysis
    document: Document | None = None


class SemanticDispatcher:
    """Heuristic-based semantic dispatcher (MVP version)."""

    INTENT_KEYWORDS = {
        UserIntent.CREATE_TASK: ["поручение", "поручи", "создай задач", "задание"],
        UserIntent.CREATE_INTRODUCTION: ["ознаком", "список ознаком"],
        UserIntent.SUMMARIZE: ["суммари", "резюм", "кратко", "опиши"],
        UserIntent.COMPARE: ["сравни", "отличия", "разница"],
        UserIntent.SEARCH: ["найди", "поиск", "покажи"],
    }

    def build_context(
        self, message: str, document: Document | None = None
    ) -> SemanticContext:
        """Build semantic context from user message."""
        message_lower = message.lower()

        # Detect intent
        intent = UserIntent.UNKNOWN
        for i, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                intent = i
                break

        word_count = len(message.split())
        complexity = (
            QueryComplexity.COMPLEX
            if word_count > 20
            else QueryComplexity.MEDIUM if word_count > 10 else QueryComplexity.SIMPLE
        )

        return SemanticContext(
            query=QueryAnalysis(
                original=message,
                refined=message,
                intent=intent,
                complexity=complexity,
            ),
            document=document,
        )
