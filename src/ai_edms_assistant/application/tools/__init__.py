# src/ai_edms_assistant/application/tools/__init__.py
"""
LangChain tools for EDMS operations.
"""

from __future__ import annotations

from .attachment_tool import AttachmentFetchInput, AttachmentTool
from .autofill_tool import AppealAutofillInput, AppealAutofillTool
from .base_tool import AbstractEdmsTool
from .comparison_tool import ComparisonTool, DocumentComparisonInput
from .document_tool import DocumentAnalysisInput, DocumentAnalysisTool
from .employee_tool import EmployeeSearchInput, EmployeeSearchTool
from .introduction_tool import IntroductionInput, IntroductionTool
from .local_file_tool import LocalFileInput, LocalFileTool
from .summarization_tool import SummarizationTool, SummarizeInput, SummaryType
from .task_tool import TaskCreateInput, TaskCreationTool


def create_all_tools(
    document_repository,
    employee_repository,
    task_repository,
    llm_provider,
    nlp_extractor,
    document_comparer,
    appeal_validator,
    task_assigner,
    # storage=None оставлен для AppealAutofillTool
    storage=None,
    # attachment_client=None — EdmsAttachmentClient для AttachmentTool
    attachment_client=None,
):
    """Factory: create all EDMS tools with dependency injection.

    Called by EdmsDocumentAgent._create_tools() after all dependencies
    are constructed.

    Args:
        document_repository: AbstractDocumentRepository instance.
        employee_repository: AbstractEmployeeRepository instance.
        task_repository: AbstractTaskRepository instance.
        llm_provider: AbstractLLMProvider instance.
        nlp_extractor: AbstractNLPExtractor instance (or None).
        document_comparer: DocumentComparer domain service.
        appeal_validator: AppealValidator domain service.
        task_assigner: TaskAssigner domain service.
        storage: AbstractStorage (optional, for AppealAutofillTool only).
        attachment_client: EdmsAttachmentClient (optional, for AttachmentTool).
            If None — AttachmentTool creates client lazily on first use.

    Returns:
        List of instantiated AbstractEdmsTool objects (9 tools).
    """
    return [
        # 1. Анализ метаданных документа
        DocumentAnalysisTool(
            document_repository=document_repository,
        ),
        # 2. Извлечение текста из файлов-вложений
        AttachmentTool(
            document_repository=document_repository,
            attachment_client=attachment_client,
        ),
        # 3. Сравнение документов
        ComparisonTool(
            document_repository=document_repository,
            document_comparer=document_comparer,
            llm_provider=llm_provider,
        ),
        # 4. Суммаризация текста
        SummarizationTool(
            llm_provider=llm_provider,
        ),
        # 5. Создание поручений
        TaskCreationTool(
            document_repository=document_repository,
            employee_repository=employee_repository,
            task_repository=task_repository,
            task_assigner=task_assigner,
        ),
        # 6. Поиск сотрудников
        EmployeeSearchTool(
            employee_repository=employee_repository,
        ),
        # 7. Автозаполнение обращений (NLP)
        AppealAutofillTool(
            document_repository=document_repository,
            nlp_extractor=nlp_extractor,
            storage=storage,
            appeal_validator=appeal_validator,
        ),
        # 8. Создание листа ознакомления
        IntroductionTool(
            document_repository=document_repository,
            employee_repository=employee_repository,
        ),
        # 9. Чтение локального файла (загруженного пользователем)
        LocalFileTool(),
    ]


__all__ = [
    "AbstractEdmsTool",
    "DocumentAnalysisTool",
    "DocumentAnalysisInput",
    "AttachmentTool",
    "AttachmentFetchInput",
    "ComparisonTool",
    "DocumentComparisonInput",
    "SummarizationTool",
    "SummarizeInput",
    "SummaryType",
    "TaskCreationTool",
    "TaskCreateInput",
    "EmployeeSearchTool",
    "EmployeeSearchInput",
    "AppealAutofillTool",
    "AppealAutofillInput",
    "IntroductionTool",
    "IntroductionInput",
    "LocalFileTool",
    "LocalFileInput",
    "create_all_tools",
]
