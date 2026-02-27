# src/ai_edms_assistant/interfaces/api/routes/document_routes.py
"""Document query endpoints — thin HTTP layer over DocumentRepository.

These endpoints are optional helpers for frontend direct calls.
The AI agent accesses documents via DocumentRepository through its tools.
"""

from __future__ import annotations

from uuid import UUID
from fastapi import APIRouter, HTTPException, Query

from ..dependencies import DocumentRepoDep
from ..schemas.document_schema import DocumentBriefResponse
from ai_edms_assistant.domain.value_objects.filters import DocumentFilter
from ....domain.repositories.base import PageRequest

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get(
    "/{document_id}",
    response_model=DocumentBriefResponse,
    summary="Получить документ по ID",
)
async def get_document(
    document_id: UUID,
    token: str = Query(..., description="JWT токен пользователя"),
    repo: DocumentRepoDep = None,
) -> DocumentBriefResponse:
    """Fetch a single document by UUID.

    Uses ``DocumentBriefResponse.from_entity()`` for consistent field mapping.

    Previously used a manual constructor that was missing ``reg_date`` and
    ``category`` fields added in the audit. ``from_entity()`` centralises all
    mapping logic — adding a new field only requires updating the schema, not
    every route that builds the response.

    Args:
        document_id: Document UUID path parameter.
        token: JWT bearer token (query param).
        repo: Injected AbstractDocumentRepository.

    Returns:
        ``DocumentBriefResponse`` for the requested document.

    Raises:
        HTTPException 404: When the document does not exist or is inaccessible.
    """
    doc = await repo.get_by_id(document_id, token)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")

    return DocumentBriefResponse.from_entity(doc)


@router.get("", summary="Поиск документов")
async def search_documents(
    token: str = Query(..., description="JWT токен пользователя"),
    reg_number: str | None = Query(None, description="Регистрационный номер"),
    short_summary: str | None = Query(None, description="Краткое содержание"),
    page: int = Query(0, ge=0),
    size: int = Query(20, ge=1, le=100),
    repo: DocumentRepoDep = None,
) -> dict:
    """Search documents with basic filters.

    Exposes a subset of ``DocumentFilter`` — for advanced filtering the AI
    agent uses ``DocumentRepository.search()`` directly via tools.

    Args:
        token: JWT bearer token.
        reg_number: Optional registration number filter (partial match).
        short_summary: Optional summary text filter (partial match).
        page: Zero-based page number (default 0).
        size: Page size between 1 and 100 (default 20).
        repo: Injected AbstractDocumentRepository.

    Returns:
        Dict with ``items``, ``page``, ``size``, ``hasNext``, ``total``.
    """
    result = await repo.search(
        DocumentFilter(reg_number=reg_number, short_summary=short_summary),
        token,
        PageRequest(page=page, size=size),
    )
    return {
        "items": [
            {
                "id": str(d.id),
                "regNumber": d.reg_number,
                "regDate": d.reg_date.isoformat() if d.reg_date else None,
                "shortSummary": d.short_summary,
                "status": d.status.value if d.status else None,
                "category": (
                    d.document_category.value if d.document_category else None
                ),
            }
            for d in result.items
        ],
        "page": result.page,
        "size": result.size,
        "hasNext": result.has_next,
        "total": result.total,
    }