# src/ai_edms_assistant/application/tools/__init__.py
"""LangChain tools for EDMS operations."""

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
        storage,
        document_comparer,
        appeal_validator,
        task_assigner,
):
    """Factory function to create all tools with injected dependencies.

    Called by EdmsDocumentAgent.__init__() after all dependencies are available.

    Args:
        document_repository: AbstractDocumentRepository instance
        employee_repository: AbstractEmployeeRepository instance
        task_repository: AbstractTaskRepository instance
        llm_provider: AbstractLLMProvider instance
        nlp_extractor: AbstractNLPExtractor instance
        storage: AbstractStorage instance
        document_comparer: DocumentComparer domain service
        appeal_validator: AppealValidator domain service
        task_assigner: TaskAssigner domain service

    Returns:
        List of instantiated tool objects
    """
    return [
        DocumentAnalysisTool(document_repository=document_repository),
        AttachmentTool(
            document_repository=document_repository,
            storage=storage,
        ),
        ComparisonTool(
            document_repository=document_repository,
            document_comparer=document_comparer,
            llm_provider=llm_provider,
        ),
        SummarizationTool(llm_provider=llm_provider),
        TaskCreationTool(
            document_repository=document_repository,
            employee_repository=employee_repository,
            task_repository=task_repository,
            task_assigner=task_assigner,
        ),
        EmployeeSearchTool(employee_repository=employee_repository),
        AppealAutofillTool(
            document_repository=document_repository,
            nlp_extractor=nlp_extractor,
            storage=storage,
            appeal_validator=appeal_validator,
        ),
        IntroductionTool(
            document_repository=document_repository,
            employee_repository=employee_repository,
        ),
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