# edms_ai_assistant/tools/document.py
"""
EDMS AI Assistant — doc_get_details tool.

Слой: Interface (Tools).

Используем DocumentService.get_document_analysis(), который:
    1. GET /api/document/{id}?includes=FULL_DOC_INCLUDES
    2. DocumentEnricher.enrich() — дозапрос correspondentId, introductions и др.
    3. Redis-кэш (TTL 300s)
    4. EDMSNaturalLanguageService.process_document()
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.services.document_service import (
    DocumentNotFoundError,
    DocumentService,
    get_redis,
)

logger = logging.getLogger(__name__)


class DocDetailsInput(BaseModel):
    """Input schema for doc_get_details tool."""

    document_id: str = Field(..., description="UUID документа (context_ui_id)")
    token: str = Field(..., description="Токен авторизации пользователя")


def _clean(d: Any) -> Any:
    """Recursively remove None, empty lists and dicts from result.

    Args:
        d: Input data structure.

    Returns:
        Cleaned structure without empty values.
    """
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items() if v not in (None, [], {}, "")}
    if isinstance(d, list):
        return [_clean(i) for i in d if i not in (None, [], {}, "")]
    return d


@tool("doc_get_details", args_schema=DocDetailsInput)
async def doc_get_details(document_id: str, token: str) -> dict[str, Any]:
    """Анализирует документ СЭД и все его вложенные сущности.

    Возвращает полный семантически структурированный контекст:
    основные данные, регистрацию, участников, жизненный цикл, контроль,
    задачи, вложения, адресатов, контрагента, ознакомления и
    специализированные секции (договор / обращение / совещание / повестка).

    Использует Redis-кэш — повторный вызов для того же документа
    не делает запрос к Java API.

    Args:
        document_id: UUID документа в EDMS.
        token: JWT-токен авторизации пользователя.

    Returns:
        Dict со статусом и полным NLP-анализом документа.
    """
    try:
        svc = DocumentService(redis=get_redis())
        analysis = await svc.get_document_analysis(
            token=token,
            document_id=document_id,
        )
        return {"status": "success", "document_analytics": _clean(analysis)}

    except DocumentNotFoundError as exc:
        logger.warning(
            "Document not found in doc_get_details",
            extra={"document_id": document_id, "error": str(exc)},
        )
        return {"status": "error", "error": f"Документ {document_id} не найден."}

    except Exception as exc:
        logger.error(
            "doc_get_details failed",
            exc_info=True,
            extra={"document_id": document_id},
        )
        return {"status": "error", "error": f"Ошибка обработки документа: {exc}"}
