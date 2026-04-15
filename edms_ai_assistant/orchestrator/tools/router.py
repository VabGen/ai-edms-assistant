# edms_ai_assistant/tools/router.py
"""
EDMS AI Assistant — Intent-Based Tool Router.

Слой: Interface (Tools).

Отвечает за два вопроса:
  1. Какой минимальный набор инструментов нужно передать в bind_tools
     для данного интента?
  2. Сколько токенов тратит тот или иной набор инструментов?

Принцип маппинга:
  - CREATE_DOCUMENT → create_document_from_file (один инструмент, ничего лишнего)
  - SUMMARIZE        → чтение файлов + суммаризация
  - COMPARE          → сравнение файлов / версий
  - SEARCH           → поиск документов + сотрудники
  - ANALYZE          → полный набор для анализа документа
  - QUESTION         → doc_get_details + employee_search
  - FILE_ANALYSIS    → локальный файл + суммаризация
  - CREATE_TASK      → task_create_tool + employee_search
  - CREATE_INTRO     → introduction_create_tool + employee_search
  - NOTIFICATION     → employee_search + doc_send_notification
  - UNKNOWN / COMPOSITE → все инструменты (fallback)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from edms_ai_assistant.orchestrator.services import UserIntent

logger = logging.getLogger(__name__)

_AVG_TOKENS_PER_TOOL: int = 120


# ─── Tool name constants ──────────────────────────────────────────────────────

# Documents
_DOC_GET_DETAILS = "doc_get_details"
_DOC_GET_VERSIONS = "doc_get_versions"
_DOC_COMPARE = "doc_compare_documents"
_DOC_SEARCH = "doc_search_tool"

# Content
_DOC_GET_FILE = "doc_get_file_content"
_READ_LOCAL_FILE = "read_local_file_content"
_DOC_COMPARE_WITH_LOCAL = "doc_compare_attachment_with_local"

# Analysis
_DOC_SUMMARIZE = "doc_summarize_text"

# Workflow
_INTRODUCTION_CREATE = "introduction_create_tool"
_TASK_CREATE = "task_create_tool"
_APPEAL_AUTOFILL = "autofill_appeal_document"
_CREATE_DOCUMENT_FROM_FILE = "create_document_from_file"

# People
_EMPLOYEE_SEARCH = "employee_search_tool"

# Notifications
_DOC_SEND_NOTIFICATION = "doc_send_notification"


# ─── Intent → tool names mapping ─────────────────────────────────────────────

_INTENT_TOOL_NAMES: dict[UserIntent, list[str]] = {
    UserIntent.CREATE_DOCUMENT: [
        _CREATE_DOCUMENT_FROM_FILE,
    ],
    # Суммаризация: прочитать файл → суммаризировать
    UserIntent.SUMMARIZE: [
        _DOC_GET_FILE,
        _READ_LOCAL_FILE,
        _DOC_SUMMARIZE,
        _DOC_GET_DETAILS,
    ],
    # Сравнение: файл vs вложение, или версии между собой
    UserIntent.COMPARE: [
        _DOC_COMPARE_WITH_LOCAL,
        _DOC_GET_VERSIONS,
        _DOC_COMPARE,
        _DOC_GET_DETAILS,
    ],
    # Поиск документов и сотрудников
    UserIntent.SEARCH: [
        _DOC_SEARCH,
        _EMPLOYEE_SEARCH,
        _DOC_GET_DETAILS,
    ],
    # Глубокий анализ документа
    UserIntent.ANALYZE: [
        _DOC_GET_DETAILS,
        _DOC_GET_FILE,
        _READ_LOCAL_FILE,
        _DOC_SUMMARIZE,
        _DOC_SEARCH,
    ],
    # Вопрос о документе / сотруднике
    UserIntent.QUESTION: [
        _DOC_GET_DETAILS,
        _DOC_GET_FILE,
        _EMPLOYEE_SEARCH,
        _DOC_SEARCH,
    ],
    # Анализ загруженного файла (без создания документа)
    UserIntent.FILE_ANALYSIS: [
        _READ_LOCAL_FILE,
        _DOC_GET_FILE,
        _DOC_SUMMARIZE,
        _DOC_COMPARE_WITH_LOCAL,
    ],
    # Создание поручения
    UserIntent.CREATE_TASK: [
        _TASK_CREATE,
        _EMPLOYEE_SEARCH,
    ],
    # Создание листа ознакомления
    UserIntent.CREATE_INTRODUCTION: [
        _INTRODUCTION_CREATE,
        _EMPLOYEE_SEARCH,
    ],
    # Уведомления и напоминания
    UserIntent.NOTIFICATION: [
        _EMPLOYEE_SEARCH,
        _DOC_SEND_NOTIFICATION,
        _DOC_GET_DETAILS,
    ],
    # Автозаполнение обращения (отдельный сценарий)
    UserIntent.EXTRACT: [
        _APPEAL_AUTOFILL,
        _DOC_GET_DETAILS,
        _DOC_GET_FILE,
    ],
}

_FULL_TOOLSET_INTENTS: frozenset[UserIntent] = frozenset(
    {
        UserIntent.UNKNOWN,
        UserIntent.COMPOSITE,
        UserIntent.UPDATE,
        UserIntent.DELETE,
    }
)


def get_tools_for_intent(
    intent: UserIntent,
    all_tools: list[Any],
    *,
    include_appeal: bool = False,
) -> list[Any]:
    """Return the minimal tool subset for the given intent.

    Фильтрует ``all_tools`` по именам из ``_INTENT_TOOL_NAMES[intent]``.
    Если интент не в маппинге или входит в ``_FULL_TOOLSET_INTENTS`` —
    возвращает все инструменты.

    ``autofill_appeal_document`` включается только когда:
    - ``include_appeal=True`` (пользователь работает с документом-обращением), или
    - интент явно нуждается в нём (EXTRACT).

    Args:
        intent: Primary intent from SemanticDispatcher.
        all_tools: Complete list of LangChain tool objects registered in the agent.
        include_appeal: When True, include ``autofill_appeal_document`` in the result
            even for intents that don't normally need it.

    Returns:
        Filtered list of tool objects; never empty (falls back to all_tools).
    """
    if intent in _FULL_TOOLSET_INTENTS:
        logger.debug(
            "Router: full toolset for intent=%s (%d tools)",
            intent.value,
            len(all_tools),
        )
        return all_tools

    allowed_names = _INTENT_TOOL_NAMES.get(intent)
    if not allowed_names:
        logger.warning(
            "Router: no tool mapping for intent=%s — returning full toolset",
            intent.value,
        )
        return all_tools

    allowed_set = set(allowed_names)

    if include_appeal:
        allowed_set.add(_APPEAL_AUTOFILL)

    selected = [t for t in all_tools if getattr(t, "name", None) in allowed_set]

    if not selected:
        logger.error(
            "Router: tool mapping for intent=%s produced 0 matches "
            "(allowed=%s) — returning full toolset",
            intent.value,
            sorted(allowed_set),
        )
        return all_tools

    logger.info(
        "Router: intent=%s → %d tools: %s",
        intent.value,
        len(selected),
        [getattr(t, "name", "?") for t in selected],
    )
    return selected


def estimate_tools_tokens(tools: list[Any]) -> int:
    """Rough estimate of tokens consumed by bind_tools schemas.

    Uses a simple heuristic: serialize each tool's schema to JSON and
    count characters / 4 (average chars-per-token for English/Russian mix).

    Falls back to ``_AVG_TOKENS_PER_TOOL * len(tools)`` if schema access fails.

    Args:
        tools: List of LangChain tool objects.

    Returns:
        Estimated token count (integer).
    """
    total_chars = 0
    for t in tools:
        try:
            schema = getattr(t, "args_schema", None)
            if schema is not None:
                total_chars += len(json.dumps(schema.model_json_schema()))
            else:
                total_chars += _AVG_TOKENS_PER_TOOL * 4  # ~120 tokens
        except Exception:
            total_chars += _AVG_TOKENS_PER_TOOL * 4

    return max(total_chars // 4, len(tools) * _AVG_TOKENS_PER_TOOL)
