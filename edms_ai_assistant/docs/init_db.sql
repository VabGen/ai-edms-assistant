-- EDMS AI Assistant — Database Initialization
-- Создаёт схему, таблицы и индексы для всех компонентов системы

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Schema ────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS edms_ai;

-- ── User profiles (долгосрочная память) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS edms_ai.user_profiles (
    user_id          TEXT PRIMARY KEY,
    first_name       TEXT DEFAULT '',
    last_name        TEXT DEFAULT '',
    department       TEXT DEFAULT '',
    role             TEXT DEFAULT '',
    -- Предпочтения
    preferred_language TEXT DEFAULT 'ru',
    frequent_categories TEXT[] DEFAULT '{}',
    frequent_statuses   TEXT[] DEFAULT '{}',
    favorite_filters    JSONB DEFAULT '{}',
    -- Статистика
    total_requests   INTEGER DEFAULT 0,
    last_active_at   TIMESTAMPTZ DEFAULT NOW(),
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    -- Кастомные настройки
    custom_settings  JSONB DEFAULT '{}'
);

-- ── Dialog logs ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS edms_ai.dialog_logs (
    id               BIGSERIAL PRIMARY KEY,
    dialog_id        UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id          TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    -- Запрос
    user_query       TEXT NOT NULL,
    normalized_query TEXT,
    intent           TEXT,
    entities         JSONB DEFAULT '{}',
    confidence       FLOAT DEFAULT 0.0,
    -- Выполнение
    selected_tool    TEXT,
    tool_args        JSONB DEFAULT '{}',
    tool_results     JSONB DEFAULT '[]',
    agent_mode       TEXT DEFAULT 'react',  -- react | plan_execute | fast_path
    model_used       TEXT DEFAULT '',
    -- Ответ
    final_response   TEXT,
    latency_ms       INTEGER DEFAULT 0,
    -- Оценка
    user_feedback    SMALLINT,  -- -1, 0, 1, NULL
    feedback_comment TEXT,
    feedback_at      TIMESTAMPTZ,
    -- Метаданные
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dialog_logs_user_id
    ON edms_ai.dialog_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_dialog_logs_session_id
    ON edms_ai.dialog_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_dialog_logs_intent
    ON edms_ai.dialog_logs(intent);
CREATE INDEX IF NOT EXISTS idx_dialog_logs_feedback
    ON edms_ai.dialog_logs(user_feedback)
    WHERE user_feedback IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_dialog_logs_created_at
    ON edms_ai.dialog_logs(created_at DESC);

-- ── RAG entries (векторный индекс успешных диалогов) ─────────────────────────
CREATE TABLE IF NOT EXISTS edms_ai.rag_index (
    id               BIGSERIAL PRIMARY KEY,
    entry_id         UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    dialog_id        TEXT,
    -- Содержимое
    user_query       TEXT NOT NULL,
    normalized_query TEXT,
    intent           TEXT,
    selected_tool    TEXT,
    tool_args        JSONB DEFAULT '{}',
    response_summary TEXT NOT NULL,
    full_response    TEXT,
    -- Качество
    feedback_score   SMALLINT DEFAULT 1,   -- 1 (success) | -1 (failure)
    usage_count      INTEGER DEFAULT 0,    -- сколько раз использовался как few-shot
    -- Эмбеддинг
    embedding        vector(768),          -- nomic-embed-text размерность
    -- Метаданные
    is_anti_example  BOOLEAN DEFAULT FALSE,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Индекс для IVFFLAT ANN-поиска (быстрый approx nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_rag_embedding
    ON edms_ai.rag_index
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_rag_intent
    ON edms_ai.rag_index(intent);
CREATE INDEX IF NOT EXISTS idx_rag_feedback
    ON edms_ai.rag_index(feedback_score);
CREATE INDEX IF NOT EXISTS idx_rag_is_anti
    ON edms_ai.rag_index(is_anti_example);

-- ── Session states (среднесрочная память — fallback если Redis недоступен) ───
CREATE TABLE IF NOT EXISTS edms_ai.session_states (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    state_data   JSONB DEFAULT '{}',
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    expires_at   TIMESTAMPTZ DEFAULT NOW() + INTERVAL '2 hours'
);

CREATE INDEX IF NOT EXISTS idx_session_user
    ON edms_ai.session_states(user_id);
CREATE INDEX IF NOT EXISTS idx_session_expires
    ON edms_ai.session_states(expires_at);

-- ── Anti-example patterns (паттерны из негативного фидбека) ─────────────────
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

-- ── Cleanup function (удаление устаревших сессий) ────────────────────────────
CREATE OR REPLACE FUNCTION edms_ai.cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM edms_ai.session_states
    WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ── View: статистика качества ─────────────────────────────────────────────────
CREATE OR REPLACE VIEW edms_ai.quality_stats AS
SELECT
    DATE_TRUNC('day', created_at) AS day,
    COUNT(*) AS total_dialogs,
    COUNT(*) FILTER (WHERE user_feedback = 1) AS positive,
    COUNT(*) FILTER (WHERE user_feedback = -1) AS negative,
    COUNT(*) FILTER (WHERE user_feedback IS NULL) AS unrated,
    ROUND(AVG(latency_ms)) AS avg_latency_ms,
    COUNT(DISTINCT user_id) AS unique_users
FROM edms_ai.dialog_logs
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY day DESC;

-- ── View: топ намерений ───────────────────────────────────────────────────────
CREATE OR REPLACE VIEW edms_ai.intent_stats AS
SELECT
    intent,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE user_feedback = 1) AS positive,
    COUNT(*) FILTER (WHERE user_feedback = -1) AS negative,
    ROUND(AVG(latency_ms)) AS avg_latency_ms,
    ROUND(100.0 * COUNT(*) FILTER (WHERE user_feedback = 1) /
          NULLIF(COUNT(*) FILTER (WHERE user_feedback IS NOT NULL), 0), 1) AS positive_rate_pct
FROM edms_ai.dialog_logs
WHERE intent IS NOT NULL
GROUP BY intent
ORDER BY total DESC;

-- ── Summarization cache (кэш суммаризаций из SQLAlchemy ORM) ─────────────────
CREATE TABLE IF NOT EXISTS edms_ai.summarization_cache (
    id               TEXT PRIMARY KEY,
    file_identifier  TEXT NOT NULL,
    summary_type     TEXT NOT NULL,
    content          TEXT NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT _file_summary_type_uc UNIQUE (file_identifier, summary_type)
);

CREATE INDEX IF NOT EXISTS idx_sum_cache_file
    ON edms_ai.summarization_cache(file_identifier);
