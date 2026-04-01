# edms_ai_assistant/tools/document_comparison.py
import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.llm import get_chat_model

logger = logging.getLogger(__name__)


class DocumentComparisonInput(BaseModel):
    """Схема входных данных для сравнения документов."""

    document_id_1: str = Field(..., description="UUID первого документа/версии")
    document_id_2: str = Field(..., description="UUID второго документа/версии")
    token: str = Field(..., description="Токен авторизации")
    comparison_focus: str | None = Field(
        None,
        description="Конкретный аспект сравнения (metadata, attachments, content, all)",
    )


@tool("doc_compare_documents", args_schema=DocumentComparisonInput)
async def doc_compare_documents(
    document_id_1: str,
    document_id_2: str,
    token: str,
    comparison_focus: str | None = "all",
) -> dict[str, Any]:
    """
    Сравнивает два документа или версии документа.

    Используй когда:
    - Пользователь спрашивает о различиях между версиями
    - Нужно найти изменения в документе
    - Требуется сравнить два документа

    comparison_focus может быть:
    - "metadata": только метаданные (рег.номер, дата, статус и т.д.)
    - "attachments": только вложения
    - "content": только текстовое содержимое вложений
    - "all": полное сравнение
    """
    try:
        async with DocumentClient() as client:
            doc1 = await client.get_document_metadata(token, document_id_1)
            doc2 = await client.get_document_metadata(token, document_id_2)

            if not doc1 or not doc2:
                return {
                    "status": "error",
                    "message": "Один или оба документа не найдены",
                }

            comparison_result = {
                "status": "success",
                "document_1_id": document_id_1,
                "document_2_id": document_id_2,
                "differences": {},
            }

            # Сравнение метаданных
            if comparison_focus in ["metadata", "all"]:
                metadata_diff = _compare_metadata(doc1, doc2)
                comparison_result["differences"]["metadata"] = metadata_diff

            # Сравнение вложений
            if comparison_focus in ["attachments", "all"]:
                attachments_diff = _compare_attachments(doc1, doc2)
                comparison_result["differences"]["attachments"] = attachments_diff

            llm = get_chat_model()
            summary_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Ты — аналитик СЭД. Проанализируй различия между двумя документами и составь краткий отчет на русском языке.",
                    ),
                    (
                        "user",
                        "Различия между документами:\n{differences}\n\nСоставь структурированный отчет об основных изменениях:",
                    ),
                ]
            )

            chain = summary_prompt | llm | StrOutputParser()
            summary = await chain.ainvoke(
                {"differences": str(comparison_result["differences"])}
            )

            comparison_result["summary"] = summary.strip()

            return comparison_result

    except Exception as e:
        logger.error(f"[DOC-COMPARE-TOOL] Error: {e}", exc_info=True)
        return {"status": "error", "message": f"Ошибка сравнения: {e!s}"}


def _compare_metadata(doc1: dict, doc2: dict) -> dict[str, Any]:
    """Сравнивает метаданные двух документов."""
    fields_to_compare = [
        "regNumber",
        "regDate",
        "status",
        "shortSummary",
        "author",
        "correspondentName",
        "outRegNumber",
        "outRegDate",
    ]

    differences = {}
    for field in fields_to_compare:
        val1 = doc1.get(field)
        val2 = doc2.get(field)

        if val1 != val2:
            differences[field] = {"document_1": val1, "document_2": val2}

    return differences


def _compare_attachments(doc1: dict, doc2: dict) -> dict[str, Any]:
    """Сравнивает списки вложений."""
    att1 = doc1.get("attachmentDocument") or []
    att2 = doc2.get("attachmentDocument") or []

    def _get_att_name(a: dict) -> str:
        return a.get("name") or a.get("originalName") or a.get("fileName") or ""

    att1_names = {_get_att_name(a) for a in att1 if _get_att_name(a)}
    att2_names = {_get_att_name(a) for a in att2 if _get_att_name(a)}

    return {
        "added_in_doc2": list(att2_names - att1_names),
        "removed_from_doc1": list(att1_names - att2_names),
        "common": list(att1_names & att2_names),
        "total_count_doc1": len(att1),
        "total_count_doc2": len(att2),
    }
