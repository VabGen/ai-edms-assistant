# src/ai_edms_assistant/application/tools/document_tool.py
"""Document analysis tool — semantic context builder for EDMS documents."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...infrastructure.edms_api.clients.document_client import EdmsDocumentClient
from ...infrastructure.nlp.processors.document_nlp_service import DocumentNLPService
from .base_tool import AbstractEdmsTool

logger = logging.getLogger(__name__)


class DocumentAnalysisInput(BaseModel):
    """Input schema for document analysis tool.

    Agent injects ``token`` and ``document_id`` automatically via
    the parameter injection loop in ``EdmsDocumentAgent._orchestrate()``.

    Attributes:
        document_id: UUID string of the document (context_ui_id from UI).
        token: JWT bearer token for EDMS API auth (auto-injected).
    """

    document_id: str = Field(..., description="UUID документа (context_ui_id)")
    token: str = Field(..., description="JWT токен авторизации пользователя")


class DocumentAnalysisTool(AbstractEdmsTool):
    """Semantic document analysis tool.

    Fetches raw document data from EDMS API and processes it through
    ``DocumentNLPService`` to produce a structured, LLM-ready context.

    Attributes:
        name: Tool name used by LangChain / LangGraph.
        description: Natural language description for LLM tool selection.
        args_schema: Pydantic input validation schema.
    """

    name: str = "doc_get_details"
    description: str = (
        "Анализирует документ СЭД и все его вложенные сущности "
        "(поручения, процессы, обращения, договоры, вложения). "
        "Возвращает семантически структурированный контекст документа."
    )
    args_schema: type[BaseModel] = DocumentAnalysisInput

    async def _arun(self, document_id: str, token: str) -> dict[str, Any]:
        """Execute document semantic analysis.

        Steps:
            1. Fetch raw dict from EDMS API (no domain mapping — avoids KeyError)
            2. Process through null-safe DocumentNLPService
            3. Return structured analytics for LLM consumption

        Args:
            document_id: Document UUID string.
            token: JWT bearer token.

        Returns:
            Dict with ``status`` and ``document_analytics`` keys on success,
            or ``error`` key on failure.
        """
        logger.info(
            "document_tool_start",
            extra={"document_id": document_id},
        )

        try:
            # ── Шаг 1: Получить RAW dict из API ──────────────────────────────
            raw_data = await self._fetch_raw(document_id, token)

            if not raw_data:
                logger.warning(
                    "document_tool_not_found",
                    extra={"document_id": document_id},
                )
                return {
                    "error": f"Документ {document_id} не найден или недоступен.",
                    "document_id": document_id,
                }

            # ── Шаг 2: NLP обработка (null-safe) ─────────────────────────────
            nlp = DocumentNLPService()
            context = nlp.process_document(raw_data)

            logger.info(
                "document_tool_success",
                extra={
                    "document_id": document_id,
                    "context_keys": list(context.keys()),
                },
            )

            return {
                "status": "success",
                "document_analytics": context,
            }

        except Exception as exc:
            logger.error(
                "document_tool_error",
                exc_info=True,
                extra={"document_id": document_id, "error": str(exc)},
            )
            return {
                "error": f"Ошибка обработки документа: {str(exc)}",
                "document_id": document_id,
            }

    @staticmethod
    async def _fetch_raw(document_id: str, token: str) -> dict[str, Any] | None:
        """Fetch raw document dict from EDMS API.

        Uses ``EdmsDocumentClient`` directly — bypasses ``EdmsDocumentRepository``
        and ``DocumentMapper`` to avoid null-field crashes.

        Args:
            document_id: Document UUID string.
            token: JWT bearer token.

        Returns:
            Raw API response dict, or None if not found / request failed.
        """
        try:
            doc_uuid = UUID(str(document_id))
        except ValueError:
            logger.error(
                "document_tool_invalid_uuid",
                extra={"document_id": document_id},
            )
            return None

        async with EdmsDocumentClient() as client:
            raw = await client.get_by_id(document_id=doc_uuid, token=token)

        if not raw:
            return None

        return raw

    def _run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Synchronous execution not supported.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "DocumentAnalysisTool supports only async execution via _arun()"
        )
