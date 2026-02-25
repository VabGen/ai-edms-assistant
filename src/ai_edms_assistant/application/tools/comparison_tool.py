# src/ai_edms_assistant/application/tools/comparison_tool.py
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.repositories import AbstractDocumentRepository
from ...domain.services import DocumentComparer
from ..dto import ComparisonResultDto
from ..ports import AbstractLLMProvider, LLMMessage
from .base_tool import AbstractEdmsTool


class DocumentComparisonInput(BaseModel):
    """Input schema for document comparison."""

    token: str = Field(..., description="JWT токен")
    document_id_1: UUID = Field(..., description="UUID первого документа")
    document_id_2: UUID = Field(..., description="UUID второго документа")
    comparison_focus: str | None = Field(
        default="all", description="Аспект сравнения (metadata/attachments/all)"
    )


class ComparisonTool(AbstractEdmsTool):
    """Tool for comparing two document versions."""

    name: str = "doc_compare"
    description: str = "Сравнивает два документа или версии документа"
    args_schema: type[BaseModel] = DocumentComparisonInput

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        document_comparer: DocumentComparer,
        llm_provider: AbstractLLMProvider,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._doc_repo = document_repository
        self._comparer = document_comparer
        self._llm = llm_provider

    async def _arun(
        self,
        token: str,
        document_id_1: UUID,
        document_id_2: UUID,
        comparison_focus: str | None = "all",
    ) -> dict[str, Any]:
        try:
            doc1 = await self._doc_repo.get_by_id(document_id_1, token)
            doc2 = await self._doc_repo.get_by_id(document_id_2, token)

            if not doc1 or not doc2:
                return self._handle_error(ValueError("Документ(ы) не найден(ы)"))

            # Use domain service
            result = self._comparer.compare(base=doc1, new=doc2)
            dto = ComparisonResultDto.from_comparison_result(result)

            # Generate LLM summary
            prompt = f"Различия между документами:\n{result.summary}\n\nСоставь отчёт:"
            llm_resp = await self._llm.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
            )

            return self._success_response(
                data={
                    "summary": dto.summary,
                    "changed_fields": dto.changed_fields,
                    "llm_analysis": llm_resp.content.strip(),
                },
                message="Документы сравнены",
            )

        except Exception as e:
            return self._handle_error(e)

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use _arun")
