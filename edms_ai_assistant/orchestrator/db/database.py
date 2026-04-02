# edms_ai_assistant/db/database.py
import logging
from datetime import datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint, func, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(settings.DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


class Base(DeclarativeBase):
    pass


# class SummarizationCache(Base):
#     __tablename__ = "summarization_cache"  # type: ignore
#
#     __table_args__ = (
#         UniqueConstraint(
#             "file_identifier", "summary_type", name="_file_summary_type_uc"
#         ),
#         {"schema": "edms"}
#     )
#
#     id = Column(String, primary_key=True)
#     file_identifier = Column(String, index=True, nullable=False)
#     summary_type = Column(String, index=True, nullable=False)
#     content = Column(Text, nullable=False)
#     created_at = Column(DateTime, server_default=func.now())


class SummarizationCache(Base):
    __tablename__ = "summarization_cache"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    file_identifier: Mapped[str] = mapped_column(String, index=True, nullable=False)
    summary_type: Mapped[str] = mapped_column(String, index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "file_identifier", "summary_type", name="_file_summary_type_uc"
        ),
        {"schema": "edms"},
    )


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Инициализация БД: создание схемы edms и таблиц."""
    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE SCHEMA IF NOT EXISTS edms"))
            await conn.run_sync(Base.metadata.create_all)

            logger.info(
                "Database initialized: schema 'edms' and tables checked/created."
            )
        except Exception as e:
            logger.error(f"Error during database initialization: {e}")
            raise
