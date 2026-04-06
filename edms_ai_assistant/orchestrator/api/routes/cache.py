# orchestrator/api/routes/cache.py
"""Управление кэшем суммаризаций через API."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, select

from edms_ai_assistant.db.database import SummarizationCache, get_session
from edms_ai_assistant.security import extract_user_id_from_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cache", tags=["Cache"])


@router.get("", summary="Статистика кэша суммаризаций")
async def get_cache_stats() -> dict:
    async with get_session() as session:
        result = await session.execute(
            select(
                SummarizationCache.summary_type,
                func.count(SummarizationCache.id).label("count"),
            ).group_by(SummarizationCache.summary_type)
        )
        rows = result.all()
        total_result = await session.execute(
            select(func.count(SummarizationCache.id))
        )
        total = total_result.scalar_one()
    return {
        "total": total,
        "by_type": {row.summary_type: row.count for row in rows},
    }


@router.delete("", summary="Очистить кэш суммаризаций")
async def clear_cache(
    file_identifier: str | None = None,
    summary_type: str | None = None,
) -> dict:
    async with get_session() as session:
        async with session.begin():
            stmt = delete(SummarizationCache)
            if file_identifier:
                stmt = stmt.where(SummarizationCache.file_identifier == file_identifier)
            if summary_type:
                stmt = stmt.where(SummarizationCache.summary_type == summary_type)
            result = await session.execute(stmt)
    return {"deleted": result.rowcount}
