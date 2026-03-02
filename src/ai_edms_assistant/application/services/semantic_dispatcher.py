# src/ai_edms_assistant/application/services/semantic_dispatcher.py
"""Semantic dispatcher for intent classification and query analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ...domain.entities.document import Document


class UserIntent(StrEnum):
    """User intent types detected from message keywords.

    Attributes:
        CREATE_TASK: Создание поручения / задания.
        CREATE_INTRODUCTION: Создание листа ознакомления.
        SUMMARIZE: Суммаризация текста / вложения.
        SEARCH: Поиск документов / сотрудников.
        COMPARE: Сравнение документов ИЛИ версий одного документа.
        ANALYZE: Анализ / аналитика по документу.
        QUESTION: Вопрос о конкретном поле / данных документа.
        UNKNOWN: Не удалось определить намерение.
    """

    CREATE_TASK = "create_task"
    CREATE_INTRODUCTION = "create_introduction"
    SUMMARIZE = "summarize"
    SEARCH = "search"
    COMPARE = "compare"
    ANALYZE = "analyze"
    QUESTION = "question"
    UNKNOWN = "unknown"


class QueryComplexity(StrEnum):
    """Query complexity levels based on word count."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class QueryAnalysis:
    """Analyzed user query with detected intent and complexity.

    Attributes:
        original: Original user message.
        refined: Preprocessed message for agent (same as original in MVP).
        intent: Detected user intent.
        complexity: Estimated query complexity.
        confidence: Detection confidence 0.0–1.0.
    """

    original: str
    refined: str
    intent: UserIntent
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    confidence: float = 1.0


@dataclass
class SemanticContext:
    """Semantic context assembled for agent execution.

    Attributes:
        query: Analyzed query with intent and complexity.
        document: Fetched domain document (or None if unavailable).
    """

    query: QueryAnalysis
    document: Document | None = None


class SemanticDispatcher:
    """Heuristic-based semantic dispatcher.

    Classifies user messages into ``UserIntent`` using keyword matching.
    Priority order matches the dict iteration order — more specific intents
    should be listed first.
    """

    INTENT_KEYWORDS: dict[UserIntent, list[str]] = {
        # ── Поручение / задание ───────────────────────────────────────────────
        UserIntent.CREATE_TASK: [
            "поручение", "поручи", "создай задач", "задание",
            "поставь задачу", "назначь", "исполнитель",
        ],

        # ── Ознакомление ──────────────────────────────────────────────────────
        UserIntent.CREATE_INTRODUCTION: [
            "ознаком", "список ознаком", "лист ознакомления",
            "отправь на ознакомление",
        ],

        # ── Суммаризация ──────────────────────────────────────────────────────
        UserIntent.SUMMARIZE: [
            "суммари", "резюм", "кратко", "опиши", "перескажи",
            "краткое содержание", "тезисы", "факты из документа",
        ],

        # ── Сравнение / версии ────────────────────────────────────────────────
        UserIntent.COMPARE: [
            # Прямые команды сравнения
            "сравни", "сравнение", "отличия", "разница", "чем отличается",
            "что изменилось", "изменения между", "найди отличия",
            # Версионные запросы
            "версий", "версии", "версия", "сколько версий",
            "покажи версии", "история версий", "сравни версию",
            "версия 1", "версия 2", "v1", "v2",
            "предыдущая версия", "текущая версия", "старая версия",
        ],

        # ── Поиск ─────────────────────────────────────────────────────────────
        UserIntent.SEARCH: [
            "найди", "поиск", "покажи", "найти документ",
            "поищи", "где документ", "найти сотрудника",
        ],

        # ── Анализ ────────────────────────────────────────────────────────────
        UserIntent.ANALYZE: [
            "проанализируй", "анализ", "аналитика", "статистика",
            "отчёт", "сводка",
        ],

        # ── Вопрос о поле документа ───────────────────────────────────────────
        UserIntent.QUESTION: [
            "сколько", "какой", "какая", "какое", "кто", "когда", "где",
            "что это", "расскажи о", "информация о", "данные о",
        ],
    }

    def build_context(
        self,
        message: str,
        document: Document | None = None,
    ) -> SemanticContext:
        """Build semantic context from user message.

        Applies keyword matching in priority order. First match wins.
        Falls back to ``UNKNOWN`` if no keywords match.

        Args:
            message: Raw user message string.
            document: Fetched domain Document (may be None).

        Returns:
            ``SemanticContext`` with detected intent and complexity.
        """
        message_lower = message.lower()

        # ── Intent detection ──────────────────────────────────────────────────
        intent = UserIntent.UNKNOWN
        for candidate_intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                intent = candidate_intent
                break

        # ── Complexity by word count ──────────────────────────────────────────
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