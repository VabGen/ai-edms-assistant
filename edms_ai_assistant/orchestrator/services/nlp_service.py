# orchestrator/services/nlp_service.py
"""
NLP-сервис: SemanticDispatcher — обёртка над NLPPreprocessor.

Используется в agent.py для NLU-анализа перед вызовом LLM.
Разделён от nlp_preprocessor.py чтобы не импортировать FastAPI-зависимости в NLU.

Экспортирует:
    SemanticContext   — результат диспетчеризации
    SemanticQuery     — NLU-данные запроса
    SemanticDispatcher — фасад для agent.py
    EDMSNaturalLanguageService — совместимость с устаревшим кодом
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..nlp_preprocessor import NLPPreprocessor, NLUResult, get_preprocessor

logger = logging.getLogger(__name__)


class UserIntent(str, Enum):
    """Намерения пользователя, распознаваемые NLU."""

    GET_DOCUMENT = "get_document"
    SEARCH_DOCUMENTS = "search_documents"
    CREATE_DOCUMENT = "create_document"
    UPDATE_STATUS = "update_status"
    GET_HISTORY = "get_history"
    ASSIGN_DOCUMENT = "assign_document"
    GET_ANALYTICS = "get_analytics"
    GET_WORKFLOW_STATUS = "get_workflow_status"
    UNKNOWN = "unknown"


@dataclass
class SemanticQuery:
    """NLU-данные запроса в удобном формате для агента."""

    raw_text: str
    intent: UserIntent
    confidence: float
    normalized_query: str
    entities: dict[str, Any] = field(default_factory=dict)
    bypass_llm: bool = False
    required_tool: str | None = None
    tool_args: dict[str, Any] | None = None


@dataclass
class SemanticContext:
    """Полный семантический контекст запроса."""

    query: SemanticQuery
    file_path: str | None = None
    has_file: bool = False
    has_document_ids: bool = False


class SemanticDispatcher:
    """
    Фасад над NLPPreprocessor для agent.py.

    Конвертирует NLUResult → SemanticContext.
    Не несёт бизнес-логики — только адаптирует типы.
    """

    def __init__(self) -> None:
        self._preprocessor: NLPPreprocessor = get_preprocessor()

    def build_context(
        self,
        message: str,
        file_path: str | None = None,
    ) -> SemanticContext:
        """
        Анализирует сообщение и строит SemanticContext.

        Args:
            message:   Текст запроса пользователя.
            file_path: Путь к файлу или UUID вложения (если есть).

        Returns:
            SemanticContext с NLU-данными.
        """
        try:
            nlu: NLUResult = self._preprocessor.preprocess(message)

            try:
                intent = UserIntent(nlu.intent)
            except ValueError:
                intent = UserIntent.UNKNOWN

            entities_dict: dict[str, Any] = {
                "document_ids": nlu.entities.document_ids,
                "statuses": nlu.entities.statuses,
                "document_types": nlu.entities.document_types,
                "user_names": nlu.entities.user_names,
                "departments": nlu.entities.departments,
                "date_range": (
                    [
                        nlu.entities.date_range[0].isoformat(),
                        nlu.entities.date_range[1].isoformat(),
                    ]
                    if nlu.entities.date_range
                    else None
                ),
                "limit": nlu.entities.limit,
                "page_number": nlu.entities.page_number,
            }

            sq = SemanticQuery(
                raw_text=message,
                intent=intent,
                confidence=nlu.confidence,
                normalized_query=nlu.normalized_query,
                entities=entities_dict,
                bypass_llm=nlu.bypass_llm,
                required_tool=nlu.required_tool,
                tool_args=nlu.tool_args,
            )

            return SemanticContext(
                query=sq,
                file_path=file_path,
                has_file=bool(file_path),
                has_document_ids=bool(nlu.entities.document_ids),
            )

        except Exception as exc:
            logger.warning("SemanticDispatcher.build_context error: %s", exc)
            return SemanticContext(
                query=SemanticQuery(
                    raw_text=message,
                    intent=UserIntent.UNKNOWN,
                    confidence=0.0,
                    normalized_query=message,
                ),
                file_path=file_path,
                has_file=bool(file_path),
            )


# Совместимость с устаревшим кодом
EDMSNaturalLanguageService = SemanticDispatcher
