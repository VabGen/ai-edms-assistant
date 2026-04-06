# orchestrator/db/database.py
"""
Инициализация базы данных и модель кэша суммаризаций.

Экспортирует:
    Base            — DeclarativeBase для всех моделей
    engine          — AsyncEngine
    get_session()   — async context manager для сессии
    init_db()       — создание таблиц (только для разработки, в prod — Alembic)
    SummarizationCache — модель кэша суммаризаций
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint, func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from edms_ai_assistant.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.DEBUG,
)

_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
)


@asynccontextmanager
async def get_session():
    """Async context manager для получения DB-сессии."""
    async with _session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Создаёт таблицы если не существуют.
    В production используй Alembic: `alembic upgrade head`.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── SummarizationCache ────────────────────────────────────────────────────


class SummarizationCache(Base):
    """
    Кэш суммаризаций документов/файлов.

    Ключ уникальности: (file_identifier, summary_type).
    file_identifier: UUID вложения или sha256 локального файла.
    summary_type:    short | detailed | key_points | action_items
    """

    __tablename__ = "summarization_cache"
    __table_args__ = (
        UniqueConstraint("file_identifier", "summary_type", name="_file_summary_uc"),
        {"schema": "edms"},
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    file_identifier: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    summary_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


async def get_cached_summary(
    file_identifier: str,
    summary_type: str,
) -> str | None:
    """Возвращает кэшированную суммаризацию или None."""
    async with get_session() as session:
        result = await session.execute(
            select(SummarizationCache).where(
                SummarizationCache.file_identifier == file_identifier,
                SummarizationCache.summary_type == summary_type,
            )
        )
        row = result.scalar_one_or_none()
        return row.content if row else None


async def save_summary_cache(
    file_identifier: str,
    summary_type: str,
    content: str,
) -> None:
    """Сохраняет суммаризацию в кэш (upsert)."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    async with get_session() as session:
        async with session.begin():
            stmt = pg_insert(SummarizationCache).values(
                id=str(uuid.uuid4()),
                file_identifier=file_identifier,
                summary_type=summary_type,
                content=content,
            )
            stmt = stmt.on_conflict_do_update(
                constraint="_file_summary_uc",
                set_={"content": content},
            )
            await session.execute(stmt)
