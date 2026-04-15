# edms_ai_assistant/api/routes/cache.py
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError

from edms_ai_assistant.orchestrator.db.database import AsyncSessionLocal, SummarizationCache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cache", tags=["Cache"])

_VALID_SUMMARY_TYPES: frozenset[str] = frozenset(
    {"extractive", "abstractive", "thesis"}
)


@router.get("/summarization", summary="List all cached summarizations")
async def list_cache() -> dict[str, Any]:
    """Return all cached summarization entries ordered by creation date.

    Returns:
        Dict with total count and list of cache entries with previews.
    """
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SummarizationCache).order_by(
                    SummarizationCache.created_at.desc()
                )
            )
            rows = result.scalars().all()

        entries = [
            {
                "file_identifier": row.file_identifier,
                "summary_type": row.summary_type,
                "content_preview": (
                    row.content[:120] + "…" if len(row.content) > 120 else row.content
                ),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

        return {"status": "success", "total": len(entries), "entries": entries}

    except SQLAlchemyError as exc:
        logger.error("Failed to list cache: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка чтения кэша: {exc}",
        ) from exc


@router.delete(
    "/summarization",
    status_code=status.HTTP_200_OK,
    summary="Clear entire summarization cache",
)
async def clear_all_cache() -> dict[str, Any]:
    """Delete all summarization cache entries.

    Returns:
        Dict with count of deleted entries.
    """
    try:
        async with AsyncSessionLocal() as db, db.begin():
            result = await db.execute(delete(SummarizationCache))
            deleted = result.rowcount

        logger.info("Summarization cache cleared: %d entries deleted", deleted)
        return {
            "status": "success",
            "deleted": deleted,
            "message": f"Удалено {deleted} записей из кэша.",
        }

    except SQLAlchemyError as exc:
        logger.error("Failed to clear cache: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка очистки кэша: {exc}",
        ) from exc


@router.delete(
    "/summarization/{file_identifier}",
    status_code=status.HTTP_200_OK,
    summary="Delete all cached analyses for a specific file",
)
async def delete_file_cache(file_identifier: str) -> dict[str, Any]:
    """Delete all summary types cached for the given file_identifier.

    Args:
        file_identifier: UUID of EDMS attachment or SHA-256 hash of local file.

    Returns:
        Dict with count of deleted entries.

    Raises:
        HTTPException 404: If no cache entries found for this identifier.
    """
    try:
        async with AsyncSessionLocal() as db, db.begin():
            result = await db.execute(
                delete(SummarizationCache).where(
                    SummarizationCache.file_identifier == file_identifier
                )
            )
            deleted = result.rowcount

        if deleted == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Кэш для файла '{file_identifier}' не найден.",
            )

        logger.info(
            "Cache deleted for file_identifier=%s: %d entries",
            file_identifier[:16],
            deleted,
        )
        return {
            "status": "success",
            "file_identifier": file_identifier,
            "deleted": deleted,
            "message": f"Удалено {deleted} запис(ей) кэша для файла.",
        }

    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        logger.error("Failed to delete cache: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка удаления кэша: {exc}",
        ) from exc


@router.delete(
    "/summarization/{file_identifier}/{summary_type}",
    status_code=status.HTTP_200_OK,
    summary="Delete a specific summary type for a file",
)
async def delete_file_cache_by_type(
    file_identifier: str, summary_type: str
) -> dict[str, Any]:
    """Delete a single cache entry by file_identifier and summary_type.

    Args:
        file_identifier: UUID of EDMS attachment or SHA-256 hash of local file.
        summary_type: One of: extractive, abstractive, thesis.

    Returns:
        Dict confirming deletion.

    Raises:
        HTTPException 400: If summary_type is invalid.
        HTTPException 404: If the cache entry is not found.
    """
    if summary_type not in _VALID_SUMMARY_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Недопустимый тип: '{summary_type}'. Допустимые: {sorted(_VALID_SUMMARY_TYPES)}",
        )

    try:
        async with AsyncSessionLocal() as db, db.begin():
            result = await db.execute(
                delete(SummarizationCache).where(
                    SummarizationCache.file_identifier == file_identifier,
                    SummarizationCache.summary_type == summary_type,
                )
            )
            deleted = result.rowcount

        if deleted == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Кэш для файла '{file_identifier}' с типом '{summary_type}' не найден.",
            )

        logger.info(
            "Cache entry deleted: file=%s type=%s", file_identifier[:16], summary_type
        )
        return {
            "status": "success",
            "file_identifier": file_identifier,
            "summary_type": summary_type,
            "deleted": deleted,
            "message": f"Кэш типа '{summary_type}' для файла удалён.",
        }

    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        logger.error("Failed to delete cache entry: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка удаления записи кэша: {exc}",
        ) from exc
