# edms_ai_assistant/orchestrator/services/__init__.py
"""
Публичный API пакета services.

Импортируй через этот модуль:
    from edms_ai_assistant.orchestrator.services import UserIntent, SemanticDispatcher
    from edms_ai_assistant.orchestrator.services import FileProcessorService

Расширение: добавь сервис в __all__ и импорт.
"""

from edms_ai_assistant.orchestrator.services.file_processor import FileProcessorService
from edms_ai_assistant.orchestrator.services.nlp_service import (
    EDMSNaturalLanguageService,
    EntityExtractor,
    EntityType,
    QueryComplexity,
    QueryRefiner,
    SemanticContext,
    SemanticDispatcher,
    UserIntent,
    UserQuery,
)
from edms_ai_assistant.orchestrator.services.document_enricher import DocumentEnricher
from edms_ai_assistant.orchestrator.services.introduction_service import IntroductionService
from edms_ai_assistant.orchestrator.services.task_service import TaskService
from edms_ai_assistant.orchestrator.services.appeal_extraction_service import (
    AppealExtractionService,
)

__all__ = [
    # NLP / Intent
    "UserIntent",
    "UserQuery",
    "QueryComplexity",
    "EntityType",
    "EntityExtractor",
    "QueryRefiner",
    "SemanticContext",
    "SemanticDispatcher",
    "EDMSNaturalLanguageService",
    # File
    "FileProcessorService",
    # Document
    "DocumentEnricher",
    # Workflow
    "IntroductionService",
    "TaskService",
    # Appeal
    "AppealExtractionService",
]