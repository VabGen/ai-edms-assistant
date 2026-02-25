# src/ai_edms_assistant/application/tools/summarization_tool.py
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ..ports import AbstractLLMProvider, LLMMessage
from .base_tool import AbstractEdmsTool


class SummaryType(StrEnum):
    """Summary format options."""

    EXTRACTIVE = "extractive"  # Ключевые факты списком
    ABSTRACTIVE = "abstractive"  # Краткий пересказ своими словами
    THESIS = "thesis"  # Тезисный план


class SummarizeInput(BaseModel):
    """Input schema for text summarization."""

    text: str = Field(..., description="Текст для суммаризации")
    summary_type: SummaryType | None = Field(
        default=None,
        description="Формат анализа. Если None, система предложит выбор.",
    )


class SummarizationTool(AbstractEdmsTool):
    """Tool for intelligent text summarization with format selection."""

    name: str = "doc_summarize_text"
    description: str = (
        "Выполняет интеллектуальный анализ и сжатие текста. "
        "Если summary_type не указан, возвращает requires_choice с рекомендацией."
    )
    args_schema: type[BaseModel] = SummarizeInput

    def __init__(self, llm_provider: AbstractLLMProvider, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_provider

    async def _arun(
        self, text: str, summary_type: SummaryType | None = None
    ) -> dict[str, Any]:
        try:
            # Clean text
            clean_text = text.strip()
            if len(clean_text) < 50:
                return self._success_response(
                    data={"content": "Текст слишком мал для анализа"},
                    message="Текст короткий",
                )

            # MODE 1: Suggest format
            if summary_type is None:
                return {
                    "status": "requires_choice",
                    "message": "Выберите формат анализа документа:",
                    "data": {
                        "options": [
                            {
                                "id": "extractive",
                                "label": "Ключевые факты",
                                "description": "Даты, суммы, конкретные обязательства",
                            },
                            {
                                "id": "abstractive",
                                "label": "Краткий пересказ",
                                "description": "Связный текст своими словами",
                            },
                            {
                                "id": "thesis",
                                "label": "Тезисный план",
                                "description": "Структурированный план с главными мыслями",
                            },
                        ]
                    },
                }

            # MODE 2: Generate summary
            instructions = {
                SummaryType.EXTRACTIVE: (
                    "Выдели ключевые факты, даты, суммы и конкретные обязательства. "
                    "Оформи списком."
                ),
                SummaryType.ABSTRACTIVE: (
                    "Напиши связный краткий пересказ сути документа своими словами "
                    "(1-2 абзаца)."
                ),
                SummaryType.THESIS: (
                    "Сформируй структурированный тезисный план документа "
                    "с выделением главных мыслей."
                ),
            }

            # Truncate long text
            processing_text = (
                clean_text[:12000] if len(clean_text) > 12000 else clean_text
            )

            prompt = f"{instructions[summary_type]}\n\nТЕКСТ:\n{processing_text}\n\nРЕЗУЛЬТАТ:"

            response = await self._llm.complete(
                messages=[
                    LLMMessage(
                        role="system",
                        content="Ты — ведущий аналитик СЭД. Пиши строго по делу, на русском.",
                    ),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=0.3,
            )

            return self._success_response(
                data={
                    "content": response.content.strip(),
                    "meta": {
                        "format_used": summary_type.value,
                        "text_length": len(clean_text),
                    },
                },
                message="Анализ завершён",
            )

        except Exception as e:
            return self._handle_error(e)

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use _arun")
