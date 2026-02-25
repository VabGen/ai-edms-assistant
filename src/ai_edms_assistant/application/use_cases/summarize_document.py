# src/ai_edms_assistant/application/use_cases/summarize_document.py
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.exceptions import DocumentNotFoundError
from ...domain.repositories import AbstractDocumentRepository
from ..ports import AbstractLLMProvider, LLMMessage
from .base import AbstractUseCase


class SummarizeDocumentRequest(BaseModel):
    """Request DTO for document summarization.

    Attributes:
        document_id: UUID of the document to summarize.
        token: JWT bearer token for EDMS API authentication.
        max_length: Optional maximum summary length in words.
            ``None`` uses default (150 words).
    """

    document_id: UUID
    token: str
    max_length: int | None = Field(default=None, ge=50, le=500)


class SummarizeDocumentUseCase(AbstractUseCase[SummarizeDocumentRequest, str]):
    """Use case: Generate a concise summary of a document's content.

    Fetches the document from the repository, extracts its text content,
    and uses the LLM to generate a natural-language summary.

    Dependencies:
        - ``AbstractDocumentRepository``: Fetch document by ID.
        - ``AbstractLLMProvider``: Generate summary via LLM.

    Business rules:
        - Document must exist (raises ``DocumentNotFoundError`` otherwise).
        - Summary length is bounded by ``max_length`` parameter.
        - If document has no text content (``summary`` and ``short_summary``
          are both empty), returns a notice instead of calling the LLM.
    """

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        llm_provider: AbstractLLMProvider,
    ) -> None:
        """Initialize with injected dependencies.

        Args:
            document_repository: Repository for fetching documents.
            llm_provider: LLM provider for generating summaries.
        """
        self._doc_repo = document_repository
        self._llm = llm_provider

    async def execute(self, request: SummarizeDocumentRequest) -> str:
        """Execute the summarization use case.

        Args:
            request: Summarization request with document ID and token.

        Returns:
            Generated summary text as a string.

        Raises:
            DocumentNotFoundError: When the document does not exist.
        """
        # 1. Fetch document
        document = await self._doc_repo.get_by_id(
            entity_id=request.document_id,
            token=request.token,
        )
        if document is None:
            raise DocumentNotFoundError(document_id=request.document_id)

        # 2. Extract text content
        full_text = document.get_full_text()
        if not full_text.strip():
            return (
                f"Документ '{document.reg_number or document.id}' не содержит "
                "текстового контента для суммаризации."
            )

        # 3. Build LLM prompt
        max_words = request.max_length or 150
        prompt = (
            f"Прочитай текст документа и создай краткое резюме на русском языке "
            f"(не более {max_words} слов). Сосредоточься на ключевых фактах и выводах.\n\n"
            f"ТЕКСТ ДОКУМЕНТА:\n{full_text}\n\n"
            f"РЕЗЮМЕ:"
        )

        messages = [LLMMessage(role="user", content=prompt)]

        # 4. Call LLM
        response = await self._llm.complete(
            messages=messages,
            temperature=0.3,
            max_tokens=max_words * 2,
        )

        return response.content.strip()
