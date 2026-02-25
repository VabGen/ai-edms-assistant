# src/ai_edms_assistant/application/tools/document_tool.py
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.repositories import AbstractDocumentRepository
from .base_tool import AbstractEdmsTool


class DocumentAnalysisInput(BaseModel):
    """Input schema for document analysis tool.

    Attributes:
        document_id: UUID of the document to analyze.
        token: JWT bearer token (auto-injected by agent).
    """

    document_id: UUID = Field(..., description="UUID документа")
    token: str = Field(..., description="JWT токен авторизации")


class DocumentAnalysisTool(AbstractEdmsTool):
    """Tool for analyzing EDMS document structure and metadata.

    Fetches the document from the repository and returns a structured
    summary of its metadata, attachments, tasks, and appeal data.

    Dependencies:
        - ``AbstractDocumentRepository``: Fetch document by ID.
    """

    name: str = "doc_get_details"
    description: str = (
        "Анализирует документ СЭД и все его вложенные сущности "
        "(поручения, процессы, обращения, договоры). "
        "Возвращает семантически структурированный контекст."
    )
    args_schema: type[BaseModel] = DocumentAnalysisInput

    def __init__(self, document_repository: AbstractDocumentRepository, **kwargs):
        """Initialize with injected document repository.

        Args:
            document_repository: Repository for fetching documents.
            **kwargs: Additional BaseTool init arguments.
        """
        super().__init__(**kwargs)
        self._doc_repo = document_repository

    async def _arun(self, document_id: UUID, token: str) -> dict[str, Any]:
        """Execute document analysis.

        Args:
            document_id: Document UUID.
            token: Auth token.

        Returns:
            Structured document analytics dict.
        """
        try:
            document = await self._doc_repo.get_by_id(
                entity_id=document_id,
                token=token,
            )

            if document is None:
                return self._handle_error(
                    ValueError(f"Документ {document_id} не найден")
                )

            # Build structured context
            analytics = {
                "основные_реквизиты": {
                    "рег_номер": document.reg_number,
                    "дата_регистрации": (
                        document.reg_date.isoformat() if document.reg_date else None
                    ),
                    "статус": document.status.value if document.status else None,
                    "категория": (
                        document.document_category.value
                        if document.document_category
                        else None
                    ),
                },
                "краткое_содержание": document.short_summary,
                "количество_вложений": len(document.attachments),
                "количество_поручений": len(document.tasks),
            }

            # Add appeal data if present
            if document.appeal:
                analytics["обращение"] = {
                    "заявитель": document.appeal.applicant_name,
                    "тип_заявителя": (
                        document.appeal.declarant_type.value
                        if document.appeal.declarant_type
                        else None
                    ),
                    "телефон": document.appeal.phone,
                    "email": document.appeal.email,
                }

            return self._success_response(
                data=analytics, message="Документ успешно проанализирован"
            )

        except Exception as e:
            return self._handle_error(e)

    def _run(self, *args, **kwargs) -> dict[str, Any]:
        """Sync execution not supported."""
        raise NotImplementedError("Use _arun for async execution")
