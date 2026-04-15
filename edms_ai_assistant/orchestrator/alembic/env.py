# orchestrator/alembic/env.py
"""
Alembic async environment для postgresql+asyncpg.

Запуск:
    alembic upgrade head
    alembic revision --autogenerate -m "description"
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

import sqlalchemy
from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

try:
    from edms_ai_assistant.orchestrator.db.database import Base

    target_metadata = Base.metadata
except ImportError:
    target_metadata = None

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://edms:change-me@localhost:5432/edms_ai",
)
config.set_main_option("sqlalchemy.url", DATABASE_URL)


def run_migrations_offline() -> None:
    """Offline-режим: генерирует SQL без подключения к БД."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema="edms",
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=True,
        version_table_schema="edms",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Async online-миграции через asyncpg."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.execute(sqlalchemy.text("CREATE SCHEMA IF NOT EXISTS edms"))
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()