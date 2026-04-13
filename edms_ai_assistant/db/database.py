# edms_ai_assistant/db/database.py
"""
SQLAlchemy async engine и модели для кэша суммаризаций.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


class SummarizationCache(Base):
    """Кэш результатов суммаризации вложений.

    Таблица создаётся через Alembic миграцию 001_init, схема edms.
    Уникальный ключ: (file_identifier, summary_type).
    """

    __tablename__ = "summarization_cache"
    __table_args__ = (
        UniqueConstraint(
            "file_identifier",
            "summary_type",
            name="_file_summary_uc",
        ),
        {"schema": "edms"},
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    file_identifier: Mapped[str] = mapped_column(
        String(255), index=True, nullable=False
    )
    summary_type: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


async def get_db() -> AsyncSession:
    """FastAPI dependency: предоставляет async-сессию SQLAlchemy."""
    async with AsyncSessionLocal() as session:
        yield session


async def close_db_engine() -> None:
    """Закрывает пул соединений. Вызывается из lifespan при остановке."""
    await engine.dispose()
    logger.info("Database engine disposed")
