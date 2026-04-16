# edms_ai_assistant/orchestrator/db/database.py
"""
SQLAlchemy async engine и модели.

ИСПРАВЛЕНИЕ: схема таблиц изменена с "edms_ai" → "edms"
чтобы соответствовать Alembic-миграции в alembic/versions/__init__.py,
которая создаёт `CREATE SCHEMA IF NOT EXISTS edms`.

Единое правило: ВСЕ таблицы находятся в схеме "edms".
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, AsyncGenerator

from sqlalchemy import DateTime, Text, UniqueConstraint, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)

# ── URL базы данных из переменных окружения ───────────────────────────────────
_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://"
    f"{os.environ.get('POSTGRES_USER', 'edms')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'edms_secret')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'edms_ai')}",
)

engine = create_async_engine(_DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""
    pass


# ИСПРАВЛЕНО: schema="edms" (было "edms_ai") — должно совпадать с Alembic-миграцией
_SCHEMA = "edms"


class SummarizationCache(Base):
    """Кэш результатов суммаризации файлов.

    Ключ: (file_identifier, summary_type) — уникальная пара.
    file_identifier: UUID вложения EDMS или SHA-256 хэш локального файла.
    summary_type: extractive | abstractive | thesis.
    """

    __tablename__ = "summarization_cache"
    __table_args__ = (
        UniqueConstraint("file_identifier", "summary_type", name="_file_summary_type_uc"),
        {"schema": _SCHEMA},
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    file_identifier: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    summary_type: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


async def get_db() -> AsyncGenerator[AsyncSession | Any, Any]:
    """FastAPI dependency: предоставляет async-сессию SQLAlchemy.

    Использование:
        @router.get("/...")
        async def handler(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        yield session


async def close_db_engine() -> None:
    """Закрывает пул соединений. Вызывается из lifespan при остановке."""
    await engine.dispose()
    logger.info("Database engine disposed")