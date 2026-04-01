"""
Database initializer — создаёт схему и таблицы при первом запуске.
Запускается из orchestrator при старте приложения.
"""
from __future__ import annotations

import asyncio
import logging
import os

import asyncpg

logger = logging.getLogger("db_init")

_DDL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS edms_ai;

CREATE TABLE IF NOT EXISTS edms_ai.user_profiles (
    user_id          TEXT PRIMARY KEY,
    first_name       TEXT DEFAULT '',
    last_name        TEXT DEFAULT '',
    department       TEXT DEFAULT '',
    role             TEXT DEFAULT '',
    preferred_language TEXT DEFAULT 'ru',
    frequent_categories TEXT[] DEFAULT '{}',
    frequent_statuses   TEXT[] DEFAULT '{}',
    favorite_filters    JSONB DEFAULT '{}',
    total_requests   INTEGER DEFAULT 0,
    last_active_at   TIMESTAMPTZ DEFAULT NOW(),
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    custom_settings  JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edms_ai.dialog_logs (
    id               BIGSERIAL PRIMARY KEY,
    dialog_id        UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id          TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    user_query       TEXT NOT NULL,
    normalized_query TEXT,
    intent           TEXT,
    entities         JSONB DEFAULT '{}',
    confidence       FLOAT DEFAULT 0.0,
    selected_tool    TEXT,
    tool_args        JSONB DEFAULT '{}',
    tool_results     JSONB DEFAULT '[]',
    agent_mode       TEXT DEFAULT 'react',
    model_used       TEXT DEFAULT '',
    final_response   TEXT,
    latency_ms       INTEGER DEFAULT 0,
    user_feedback    SMALLINT,
    feedback_comment TEXT,
    feedback_at      TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dialog_logs_user_id ON edms_ai.dialog_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_dialog_logs_session_id ON edms_ai.dialog_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_dialog_logs_intent ON edms_ai.dialog_logs(intent);
CREATE INDEX IF NOT EXISTS idx_dialog_logs_feedback ON edms_ai.dialog_logs(user_feedback)
    WHERE user_feedback IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_dialog_logs_created_at ON edms_ai.dialog_logs(created_at DESC);

CREATE TABLE IF NOT EXISTS edms_ai.rag_entries (
    id               BIGSERIAL PRIMARY KEY,
    entry_id         UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    dialog_id        TEXT,
    user_query       TEXT NOT NULL,
    normalized_query TEXT,
    intent           TEXT,
    selected_tool    TEXT,
    tool_args        JSONB DEFAULT '{}',
    response_summary TEXT,
    full_response    TEXT,
    feedback_score   SMALLINT DEFAULT 1,
    usage_count      INTEGER DEFAULT 0,
    embedding        vector(768),
    is_anti_example  BOOLEAN DEFAULT FALSE,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_intent ON edms_ai.rag_entries(intent);
CREATE INDEX IF NOT EXISTS idx_rag_feedback ON edms_ai.rag_entries(feedback_score);
CREATE INDEX IF NOT EXISTS idx_rag_anti ON edms_ai.rag_entries(is_anti_example);

CREATE TABLE IF NOT EXISTS edms_ai.session_states (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    state_data   JSONB DEFAULT '{}',
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    expires_at   TIMESTAMPTZ DEFAULT NOW() + INTERVAL '2 hours'
);

CREATE INDEX IF NOT EXISTS idx_session_user ON edms_ai.session_states(user_id);
CREATE INDEX IF NOT EXISTS idx_session_expires ON edms_ai.session_states(expires_at);

CREATE TABLE IF NOT EXISTS edms_ai.anti_patterns (
    id           BIGSERIAL PRIMARY KEY,
    intent       TEXT,
    tool_used    TEXT,
    pattern_desc TEXT,
    example_query TEXT,
    failure_count INTEGER DEFAULT 1,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);
"""


async def init_db(dsn: str) -> None:
    """
    Инициализирует схему БД.
    Вызывается один раз при старте приложения.
    Идемпотентна — повторный запуск безопасен (IF NOT EXISTS).
    """
    try:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(_DDL)
            logger.info("Database schema initialized successfully")
        finally:
            await conn.close()
    except Exception as exc:
        logger.error("Database initialization failed: %s", exc)
        raise


async def _try_create_vector_index(dsn: str) -> None:
    """Создаёт IVFFLAT индекс после загрузки данных (нужно >= 100 строк)."""
    try:
        conn = await asyncpg.connect(dsn)
        try:
            count = await conn.fetchval("SELECT COUNT(*) FROM edms_ai.rag_entries")
            if count and count >= 100:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rag_embedding_ivfflat
                    ON edms_ai.rag_entries
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 50)
                """)
                logger.info("IVFFLAT index created for %d RAG entries", count)
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("Could not create vector index: %s", exc)


if __name__ == "__main__":
    import os
    dsn = os.getenv("DATABASE_URL", "postgresql://edms:edms@localhost:5432/edms_ai")
    asyncio.run(init_db(dsn))
