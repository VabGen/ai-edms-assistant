"""
Memory — Многоуровневая система памяти для EDMS AI Ассистента.

Уровни:
1. Short-term (краткосрочная): контекст текущего диалога, управление токенами
2. Medium-term (среднесрочная): состояние сессии в Redis с TTL
3. Long-term (долгосрочная): профиль пользователя в PostgreSQL

Все операции асинхронные. Падение Redis/DB не блокирует работу ассистента.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import asyncpg
import redis.asyncio as aioredis

logger = logging.getLogger("memory")


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class Message:
    """Одно сообщение в диалоге."""
    role: str           # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_calls: list[dict] | None = None
    tool_result: dict | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class SessionState:
    """Состояние текущей сессии пользователя."""
    session_id: str
    user_id: str
    document_id: str | None = None      # Активный документ в EDMS
    document_context: dict | None = None # Краткие данные о документе
    current_task: str | None = None     # Описание текущей задачи
    pending_clarification: dict | None = None  # Ожидаем уточнения от пользователя
    last_intent: str | None = None
    last_tool_used: str | None = None
    turn_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SessionState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class UserProfile:
    """Долгосрочный профиль пользователя."""
    user_id: str
    display_name: str = ""
    preferred_language: str = "ru"
    preferred_summary_format: str = "extractive"
    frequent_categories: list[str] = field(default_factory=list)
    frequent_departments: list[str] = field(default_factory=list)
    interaction_count: int = 0
    positive_ratings: int = 0
    negative_ratings: int = 0
    custom_instructions: str = ""       # Персональные инструкции пользователя
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> UserProfile:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "frequent_categories" in valid and isinstance(valid["frequent_categories"], str):
            valid["frequent_categories"] = json.loads(valid["frequent_categories"])
        if "frequent_departments" in valid and isinstance(valid["frequent_departments"], str):
            valid["frequent_departments"] = json.loads(valid["frequent_departments"])
        return cls(**valid)

    def to_system_prompt_block(self) -> str:
        """Формирует блок для системного промпта с данными профиля."""
        parts = [f"Пользователь: {self.display_name or self.user_id}"]
        if self.preferred_summary_format:
            parts.append(f"Предпочтительный формат суммаризации: {self.preferred_summary_format}")
        if self.frequent_categories:
            parts.append(f"Часто работает с категориями: {', '.join(self.frequent_categories)}")
        if self.custom_instructions:
            parts.append(f"Персональные инструкции: {self.custom_instructions}")
        return "\n".join(parts)


# ── Short-term memory ─────────────────────────────────────────────────────────


class ShortTermMemory:
    """
    Краткосрочная память: хранит сообщения текущего диалога.

    Автоматически обрезает контекст по максимальному числу токенов.
    """

    # Примерное число символов на токен для русского языка
    CHARS_PER_TOKEN = 3.5

    def __init__(self, max_tokens: int = 8000) -> None:
        self.max_tokens = max_tokens
        self._messages: list[Message] = []
        self._system_message: str = ""

    def set_system(self, system: str) -> None:
        """Установить системный промпт."""
        self._system_message = system

    def add(self, message: Message) -> None:
        """Добавить сообщение и автоматически обрезать если нужно."""
        self._messages.append(message)
        self._trim()

    def add_user(self, content: str, **meta: Any) -> None:
        self.add(Message(role="user", content=content, metadata=meta))

    def add_assistant(self, content: str, tool_calls: list | None = None, **meta: Any) -> None:
        self.add(Message(role="assistant", content=content, tool_calls=tool_calls, metadata=meta))

    def add_tool_result(self, content: str, tool_name: str) -> None:
        self.add(Message(role="tool", content=content, metadata={"tool": tool_name}))

    def get_messages(self, include_system: bool = True) -> list[dict]:
        """Возвращает список сообщений для передачи в LLM."""
        result = []
        if include_system and self._system_message:
            result.append({"role": "system", "content": self._system_message})
        for msg in self._messages:
            result.append(msg.to_dict())
        return result

    def get_last_n(self, n: int = 5) -> list[Message]:
        """Последние N сообщений."""
        return self._messages[-n:]

    def clear(self) -> None:
        """Очистить историю (новый диалог)."""
        self._messages = []

    def token_count(self) -> int:
        """Примерный подсчёт токенов."""
        total_chars = len(self._system_message)
        for m in self._messages:
            total_chars += len(m.content)
        return int(total_chars / self.CHARS_PER_TOKEN)

    def _trim(self) -> None:
        """Обрезает старые сообщения, сохраняя лимит токенов."""
        while self.token_count() > self.max_tokens and len(self._messages) > 2:
            # Удаляем самые старые, кроме первых системных
            if self._messages[0].role in ("system",):
                self._messages.pop(1)
            else:
                self._messages.pop(0)

    def summary(self) -> str:
        """Краткая сводка текущего контекста."""
        return (
            f"Сообщений: {len(self._messages)}, "
            f"токенов: ~{self.token_count()}/{self.max_tokens}"
        )


# ── Medium-term memory (Redis) ────────────────────────────────────────────────


class MediumTermMemory:
    """
    Среднесрочная память: состояние сессии в Redis.

    TTL: 2 часа по умолчанию. При каждом обновлении TTL сбрасывается.
    """

    DEFAULT_TTL = 7200  # 2 часа

    def __init__(self, redis_url: str, ttl: int = DEFAULT_TTL) -> None:
        self._redis_url = redis_url
        self._ttl = ttl
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis | None:
        if self._client is None:
            try:
                self._client = aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._client.ping()
            except Exception as exc:
                logger.warning("Redis unavailable: %s — session state won't persist", exc)
                self._client = None
        return self._client

    def _session_key(self, session_id: str) -> str:
        return f"edms:session:{session_id}"

    async def get_session(self, session_id: str) -> SessionState | None:
        """Загрузить состояние сессии из Redis."""
        client = await self._get_client()
        if not client:
            return None
        try:
            raw = await client.get(self._session_key(session_id))
            if raw:
                return SessionState.from_dict(json.loads(raw))
        except Exception as exc:
            logger.warning("MediumTermMemory.get_session error: %s", exc)
        return None

    async def save_session(self, state: SessionState) -> None:
        """Сохранить состояние сессии в Redis."""
        client = await self._get_client()
        if not client:
            return
        try:
            state.updated_at = datetime.now().isoformat()
            await client.setex(
                self._session_key(state.session_id),
                self._ttl,
                json.dumps(state.to_dict(), ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning("MediumTermMemory.save_session error: %s", exc)

    async def update_session(self, session_id: str, **updates: Any) -> SessionState:
        """Частичное обновление состояния сессии."""
        state = await self.get_session(session_id)
        if not state:
            state = SessionState(session_id=session_id, user_id=updates.get("user_id", ""))
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        state.turn_count += 1
        await self.save_session(state)
        return state

    async def delete_session(self, session_id: str) -> None:
        """Удалить сессию (по завершении диалога)."""
        client = await self._get_client()
        if client:
            try:
                await client.delete(self._session_key(session_id))
            except Exception:
                pass

    async def cache_get(self, key: str) -> Any | None:
        """Общее кэширование: получить значение."""
        client = await self._get_client()
        if not client:
            return None
        try:
            raw = await client.get(f"edms:cache:{key}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def cache_set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Общее кэширование: сохранить значение с TTL."""
        client = await self._get_client()
        if not client:
            return
        try:
            await client.setex(
                f"edms:cache:{key}", ttl,
                json.dumps(value, ensure_ascii=False, default=str),
            )
        except Exception as exc:
            logger.debug("cache_set error: %s", exc)

    async def cache_delete(self, key: str) -> None:
        """Инвалидировать кэш по ключу."""
        client = await self._get_client()
        if not client:
            return
        try:
            await client.delete(f"edms:cache:{key}")
        except Exception:
            pass


# ── Long-term memory (PostgreSQL) ─────────────────────────────────────────────


class LongTermMemory:
    """
    Долгосрочная память: профили пользователей и история в PostgreSQL.
    """

    SCHEMA = """
    CREATE SCHEMA IF NOT EXISTS edms_ai;

    CREATE TABLE IF NOT EXISTS edms_ai.user_profiles (
        user_id TEXT PRIMARY KEY,
        display_name TEXT DEFAULT '',
        preferred_language TEXT DEFAULT 'ru',
        preferred_summary_format TEXT DEFAULT 'extractive',
        frequent_categories JSONB DEFAULT '[]',
        frequent_departments JSONB DEFAULT '[]',
        interaction_count INTEGER DEFAULT 0,
        positive_ratings INTEGER DEFAULT 0,
        negative_ratings INTEGER DEFAULT 0,
        custom_instructions TEXT DEFAULT '',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS edms_ai.dialog_logs (
        dialog_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        user_query TEXT NOT NULL,
        normalized_query TEXT,
        intent TEXT,
        entities JSONB DEFAULT '{}',
        selected_tool TEXT,
        tool_args JSONB DEFAULT '{}',
        tool_result JSONB DEFAULT '{}',
        final_response TEXT,
        model_used TEXT,
        latency_ms INTEGER,
        user_feedback SMALLINT,  -- -1, 0, 1
        feedback_comment TEXT,
        error_occurred BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_dialog_logs_user_id ON edms_ai.dialog_logs(user_id);
    CREATE INDEX IF NOT EXISTS idx_dialog_logs_session_id ON edms_ai.dialog_logs(session_id);
    CREATE INDEX IF NOT EXISTS idx_dialog_logs_created_at ON edms_ai.dialog_logs(created_at);
    CREATE INDEX IF NOT EXISTS idx_dialog_logs_feedback ON edms_ai.dialog_logs(user_feedback)
        WHERE user_feedback IS NOT NULL;
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def init(self) -> None:
        """Инициализация: создание пула соединений и таблиц."""
        try:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(self.SCHEMA)
            logger.info("LongTermMemory initialized: PostgreSQL connected")
        except Exception as exc:
            logger.error("LongTermMemory init error: %s — long-term memory disabled", exc)
            self._pool = None

    async def close(self) -> None:
        """Закрыть пул соединений."""
        if self._pool:
            await self._pool.close()

    async def get_profile(self, user_id: str) -> UserProfile:
        """Получить профиль пользователя, создать если не существует."""
        if not self._pool:
            return UserProfile(user_id=user_id)
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM edms_ai.user_profiles WHERE user_id = $1",
                    user_id,
                )
                if row:
                    return UserProfile.from_dict(dict(row))
                # Создаём новый профиль
                await conn.execute(
                    "INSERT INTO edms_ai.user_profiles (user_id) VALUES ($1) ON CONFLICT DO NOTHING",
                    user_id,
                )
                return UserProfile(user_id=user_id)
        except Exception as exc:
            logger.warning("get_profile error: %s", exc)
            return UserProfile(user_id=user_id)

    async def update_profile(self, profile: UserProfile) -> None:
        """Обновить профиль пользователя."""
        if not self._pool:
            return
        try:
            profile.updated_at = datetime.now().isoformat()
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO edms_ai.user_profiles
                        (user_id, display_name, preferred_language, preferred_summary_format,
                         frequent_categories, frequent_departments, interaction_count,
                         positive_ratings, negative_ratings, custom_instructions, updated_at)
                    VALUES ($1,$2,$3,$4,$5::jsonb,$6::jsonb,$7,$8,$9,$10,$11)
                    ON CONFLICT (user_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        preferred_language = EXCLUDED.preferred_language,
                        preferred_summary_format = EXCLUDED.preferred_summary_format,
                        frequent_categories = EXCLUDED.frequent_categories,
                        frequent_departments = EXCLUDED.frequent_departments,
                        interaction_count = EXCLUDED.interaction_count,
                        positive_ratings = EXCLUDED.positive_ratings,
                        negative_ratings = EXCLUDED.negative_ratings,
                        custom_instructions = EXCLUDED.custom_instructions,
                        updated_at = EXCLUDED.updated_at
                """,
                    profile.user_id, profile.display_name, profile.preferred_language,
                    profile.preferred_summary_format,
                    json.dumps(profile.frequent_categories),
                    json.dumps(profile.frequent_departments),
                    profile.interaction_count, profile.positive_ratings,
                    profile.negative_ratings, profile.custom_instructions,
                    profile.updated_at,
                )
        except Exception as exc:
            logger.warning("update_profile error: %s", exc)

    async def log_dialog(
        self,
        session_id: str,
        user_id: str,
        user_query: str,
        normalized_query: str = "",
        intent: str = "",
        entities: dict | None = None,
        selected_tool: str = "",
        tool_args: dict | None = None,
        tool_result: dict | None = None,
        final_response: str = "",
        model_used: str = "",
        latency_ms: int = 0,
        error_occurred: bool = False,
    ) -> str | None:
        """Записать диалог в лог. Возвращает dialog_id."""
        if not self._pool:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO edms_ai.dialog_logs
                        (session_id, user_id, user_query, normalized_query, intent,
                         entities, selected_tool, tool_args, tool_result, final_response,
                         model_used, latency_ms, error_occurred)
                    VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8::jsonb,$9::jsonb,$10,$11,$12,$13)
                    RETURNING dialog_id::text
                """,
                    session_id, user_id, user_query, normalized_query, intent,
                    json.dumps(entities or {}, default=str),
                    selected_tool,
                    json.dumps(tool_args or {}, default=str),
                    json.dumps(tool_result or {}, default=str),
                    final_response, model_used, latency_ms, error_occurred,
                )
                return row["dialog_id"] if row else None
        except Exception as exc:
            logger.warning("log_dialog error: %s", exc)
            return None

    async def save_feedback(
        self,
        dialog_id: str,
        rating: int,
        comment: str = "",
    ) -> bool:
        """Сохранить оценку пользователя для диалога."""
        if not self._pool:
            return False
        if rating not in (-1, 0, 1):
            logger.warning("Invalid rating %d — must be -1, 0, or 1", rating)
            return False
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    """UPDATE edms_ai.dialog_logs
                       SET user_feedback = $1, feedback_comment = $2
                       WHERE dialog_id = $3::uuid""",
                    rating, comment, dialog_id,
                )
                return result == "UPDATE 1"
        except Exception as exc:
            logger.warning("save_feedback error: %s", exc)
            return False

    async def get_positive_dialogs(
        self,
        limit: int = 100,
        since_days: int = 30,
    ) -> list[dict]:
        """Получить успешные диалоги для обновления RAG базы."""
        if not self._pool:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT dialog_id::text, user_query, normalized_query, intent,
                           selected_tool, tool_args, final_response, user_feedback
                    FROM edms_ai.dialog_logs
                    WHERE user_feedback = 1
                      AND created_at > NOW() - INTERVAL '$1 days'
                      AND NOT error_occurred
                    ORDER BY created_at DESC
                    LIMIT $2
                """, since_days, limit)
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("get_positive_dialogs error: %s", exc)
            return []

    async def get_negative_dialogs(self, limit: int = 50) -> list[dict]:
        """Получить неуспешные диалоги для анализа и формирования анти-примеров."""
        if not self._pool:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT dialog_id::text, user_query, intent, selected_tool,
                           final_response, feedback_comment
                    FROM edms_ai.dialog_logs
                    WHERE user_feedback = -1
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit)
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("get_negative_dialogs error: %s", exc)
            return []

    async def update_user_stats(self, user_id: str, intent: str) -> None:
        """Обновить статистику взаимодействий пользователя."""
        if not self._pool:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    UPDATE edms_ai.user_profiles
                    SET interaction_count = interaction_count + 1,
                        updated_at = NOW()
                    WHERE user_id = $1
                """, user_id)
        except Exception as exc:
            logger.debug("update_user_stats error: %s", exc)


# ── Unified Memory Manager ────────────────────────────────────────────────────


class MemoryManager:
    """
    Единый менеджер памяти — объединяет все три уровня.

    Использование:
        mem = MemoryManager(redis_url="redis://...", pg_dsn="postgresql://...")
        await mem.init()

        # Получить/создать сессию
        session = await mem.get_or_create_session(session_id, user_id)

        # Профиль пользователя
        profile = await mem.get_user_profile(user_id)

        # Краткосрочная память
        mem.short.add_user("Привет!")
        mem.short.add_assistant("Здравствуйте!")
    """

    def __init__(self, redis_url: str, pg_dsn: str, max_context_tokens: int = 8000) -> None:
        self.short = ShortTermMemory(max_tokens=max_context_tokens)
        self.medium = MediumTermMemory(redis_url=redis_url)
        self.long = LongTermMemory(dsn=pg_dsn)
        self._current_session: SessionState | None = None
        self._current_profile: UserProfile | None = None

    async def init(self) -> None:
        """Инициализация всех компонентов памяти."""
        await self.long.init()
        logger.info("MemoryManager initialized")

    async def close(self) -> None:
        await self.long.close()

    async def get_or_create_session(
        self,
        session_id: str,
        user_id: str,
    ) -> SessionState:
        """Загрузить или создать сессию."""
        state = await self.medium.get_session(session_id)
        if not state:
            state = SessionState(session_id=session_id, user_id=user_id)
            await self.medium.save_session(state)
        self._current_session = state
        return state

    async def get_user_profile(self, user_id: str) -> UserProfile:
        """Загрузить профиль пользователя."""
        profile = await self.long.get_profile(user_id)
        self._current_profile = profile
        return profile

    async def build_system_context(
        self,
        base_system_prompt: str,
        nlu_context: str = "",
        rag_examples: str = "",
        anti_examples: str = "",
    ) -> str:
        """
        Собрать полный системный промпт с контекстом из памяти.

        Порядок блоков:
        1. Базовый системный промпт
        2. Профиль пользователя
        3. Состояние сессии
        4. RAG примеры
        5. Анти-примеры (из негативной обратной связи)
        6. NLU контекст
        """
        blocks = [base_system_prompt.strip()]

        if self._current_profile:
            profile_block = self._current_profile.to_system_prompt_block()
            if profile_block:
                blocks.append(f"\n<user_profile>\n{profile_block}\n</user_profile>")

        if self._current_session:
            session_parts = []
            if self._current_session.document_id:
                session_parts.append(f"Активный документ: {self._current_session.document_id}")
            if self._current_session.current_task:
                session_parts.append(f"Текущая задача: {self._current_session.current_task}")
            if self._current_session.last_intent:
                session_parts.append(f"Последнее намерение: {self._current_session.last_intent}")
            if session_parts:
                blocks.append(f"\n<session_context>\n{chr(10).join(session_parts)}\n</session_context>")

        if rag_examples:
            blocks.append(f"\n<few_shot_examples>\n{rag_examples}\n</few_shot_examples>")

        if anti_examples:
            blocks.append(f"\n<anti_examples>\n{anti_examples}\n</anti_examples>")

        if nlu_context:
            blocks.append(f"\n<nlu_analysis>\n{nlu_context}\n</nlu_analysis>")

        return "\n".join(blocks)

    async def update_after_turn(
        self,
        session_id: str,
        intent: str,
        tool_used: str | None = None,
        document_id: str | None = None,
    ) -> None:
        """Обновить состояние после каждого хода диалога."""
        updates: dict[str, Any] = {"last_intent": intent}
        if tool_used:
            updates["last_tool_used"] = tool_used
        if document_id:
            updates["document_id"] = document_id

        if self._current_session:
            await self.medium.update_session(session_id, **updates)

        if self._current_profile:
            await self.long.update_user_stats(self._current_profile.user_id, intent)
