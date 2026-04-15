"""
memory.py — Трёхуровневая система памяти EDMS AI Ассистента.

1. Short-term  (краткосрочная): контекст диалога в памяти, обрезка по токенам
2. Medium-term (среднесрочная): состояние сессии в Redis с TTL
3. Long-term   (долгосрочная): профиль пользователя в PostgreSQL (asyncpg)

Все операции асинхронные. Падение Redis/DB не блокирует работу ассистента.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import asyncpg
import redis.asyncio as aioredis

logger = logging.getLogger("memory")


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Message:
    """Одно сообщение в диалоге."""
    role: str        # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class UserProfile:
    """Профиль пользователя из долгосрочной памяти."""
    user_id: str
    first_name: str = ""
    last_name: str = ""
    department: str = ""
    role: str = ""
    preferred_language: str = "ru"
    frequent_categories: list[str] = field(default_factory=list)
    frequent_statuses: list[str] = field(default_factory=list)
    favorite_filters: dict[str, Any] = field(default_factory=dict)
    total_requests: int = 0
    last_active_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SessionState:
    """Состояние текущей сессии."""
    user_id: str
    session_id: str
    active_document_id: str | None = None
    active_document_title: str | None = None
    current_task: str | None = None
    pending_confirmation: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "active_document_id": self.active_document_id,
            "active_document_title": self.active_document_title,
            "current_task": self.current_task,
            "pending_confirmation": self.pending_confirmation,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Short-term memory ─────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Краткосрочная память: контекст текущего диалога.

    Хранит список сообщений в памяти процесса.
    Автоматически обрезает старые сообщения при превышении лимита токенов.
    """

    def __init__(self, max_tokens: int = 8000) -> None:
        self.max_tokens = max_tokens
        self._messages: list[Message] = []

    def add(self, role: str, content: str, name: str | None = None) -> None:
        """Добавить сообщение в контекст."""
        self._messages.append(Message(role=role, content=content, name=name))
        self._trim()
        logger.debug(
            "Short-term: added %s message, total=%d ~%d tokens",
            role, len(self._messages), self.token_count(),
        )

    def get_messages(self) -> list[dict[str, Any]]:
        """Получить все сообщения для передачи в LLM."""
        return [m.to_dict() for m in self._messages]

    def clear(self) -> None:
        """Очистить контекст диалога."""
        self._messages = []

    @property
    def token_count(self) -> int:
        """Приблизительный подсчёт токенов (4 символа ≈ 1 токен)."""
        total_chars = sum(len(m.content) for m in self._messages)
        return total_chars // 4

    def _trim(self) -> None:
        """Обрезать старые сообщения, сохраняя системное и последние сообщения."""
        while self.token_count > self.max_tokens and len(self._messages) > 2:
            # Находим первое не-системное сообщение и удаляем
            for i, msg in enumerate(self._messages):
                if msg.role != "system":
                    self._messages.pop(i)
                    break
            else:
                break
        logger.debug("Short-term trimmed: %d messages, ~%d tokens", len(self._messages), self.token_count)

    def __len__(self) -> int:
        return len(self._messages)


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
        self._fallback: dict[str, Any] = {}  # in-memory fallback при недоступности Redis

    async def _get_client(self) -> aioredis.Redis | None:
        if self._client is None:
            try:
                self._client = aioredis.from_url(
                    self._redis_url, encoding="utf-8", decode_responses=True
                )
                await self._client.ping()
            except Exception as exc:
                logger.warning("Redis unavailable: %s — using in-memory fallback", exc)
                self._client = None
        return self._client

    async def get_session(self, session_id: str) -> SessionState | None:
        """Получить состояние сессии."""
        client = await self._get_client()
        key = f"session:{session_id}"
        try:
            if client:
                raw = await client.get(key)
                if raw:
                    data = json.loads(raw)
                    return SessionState.from_dict(data)
            else:
                raw = self._fallback.get(key)
                if raw:
                    return SessionState.from_dict(raw)
        except Exception as exc:
            logger.error("Medium-term get_session error: %s", exc)
        return None

    async def save_session(self, state: SessionState) -> None:
        """Сохранить состояние сессии."""
        client = await self._get_client()
        key = f"session:{state.session_id}"
        try:
            data = state.to_dict()
            if client:
                await client.setex(key, self._ttl, json.dumps(data))
            else:
                self._fallback[key] = data
        except Exception as exc:
            logger.error("Medium-term save_session error: %s", exc)

    async def cache_get(self, cache_key: str) -> Any | None:
        """Получить кэшированный результат."""
        client = await self._get_client()
        key = f"cache:{cache_key}"
        try:
            if client:
                raw = await client.get(key)
                return json.loads(raw) if raw else None
            return self._fallback.get(key)
        except Exception as exc:
            logger.error("Medium-term cache_get error: %s", exc)
            return None

    async def cache_set(self, cache_key: str, value: Any, ttl: int = 300) -> None:
        """Кэшировать результат с TTL."""
        client = await self._get_client()
        key = f"cache:{cache_key}"
        try:
            if client:
                await client.setex(key, ttl, json.dumps(value, default=str))
            else:
                self._fallback[key] = value
        except Exception as exc:
            logger.error("Medium-term cache_set error: %s", exc)

    async def delete_session(self, session_id: str) -> None:
        """Удалить сессию."""
        client = await self._get_client()
        key = f"session:{session_id}"
        try:
            if client:
                await client.delete(key)
            else:
                self._fallback.pop(key, None)
        except Exception as exc:
            logger.error("Medium-term delete_session error: %s", exc)


# ── Long-term memory (PostgreSQL via asyncpg) ─────────────────────────────────

class LongTermMemory:
    """
    Долгосрочная память: профили пользователей в PostgreSQL.

    Использует asyncpg напрямую для максимальной производительности.
    Schema: edms_ai.user_profiles, edms_ai.dialog_logs
    """

    SCHEMA = """
    CREATE SCHEMA IF NOT EXISTS edms_ai;

    CREATE TABLE IF NOT EXISTS edms_ai.user_profiles (
        user_id TEXT PRIMARY KEY,
        first_name TEXT DEFAULT '',
        last_name TEXT DEFAULT '',
        department TEXT DEFAULT '',
        role TEXT DEFAULT '',
        preferred_language TEXT DEFAULT 'ru',
        frequent_categories TEXT[] DEFAULT '{}',
        frequent_statuses TEXT[] DEFAULT '{}',
        favorite_filters JSONB DEFAULT '{}',
        total_requests INTEGER DEFAULT 0,
        last_active_at TIMESTAMPTZ DEFAULT NOW(),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        custom_settings JSONB DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS edms_ai.dialog_logs (
        id BIGSERIAL PRIMARY KEY,
        dialog_id UUID DEFAULT gen_random_uuid() UNIQUE NOT NULL,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        user_query TEXT NOT NULL,
        normalized_query TEXT,
        intent TEXT,
        entities JSONB DEFAULT '{}',
        confidence FLOAT DEFAULT 0.0,
        selected_tool TEXT,
        tool_args JSONB DEFAULT '{}',
        tool_results JSONB DEFAULT '[]',
        agent_mode TEXT DEFAULT 'react',
        model_used TEXT DEFAULT '',
        final_response TEXT,
        latency_ms INTEGER DEFAULT 0,
        user_feedback SMALLINT,
        feedback_comment TEXT,
        feedback_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_dialog_logs_user_id ON edms_ai.dialog_logs(user_id);
    CREATE INDEX IF NOT EXISTS idx_dialog_logs_session_id ON edms_ai.dialog_logs(session_id);
    CREATE INDEX IF NOT EXISTS idx_dialog_logs_intent ON edms_ai.dialog_logs(intent);
    CREATE INDEX IF NOT EXISTS idx_dialog_logs_feedback
        ON edms_ai.dialog_logs(user_feedback) WHERE user_feedback IS NOT NULL;
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Инициализировать пул соединений и создать схему."""
        try:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(self.SCHEMA)
            logger.info("LongTermMemory: PostgreSQL pool initialized")
        except Exception as exc:
            logger.error("LongTermMemory init failed: %s", exc)
            self._pool = None

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def get_user_profile(self, user_id: str) -> UserProfile:
        """Получить профиль пользователя (создать если не существует)."""
        if not self._pool:
            return UserProfile(user_id=user_id)
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM edms_ai.user_profiles WHERE user_id = $1",
                    user_id,
                )
                if not row:
                    await conn.execute(
                        "INSERT INTO edms_ai.user_profiles (user_id) VALUES ($1) ON CONFLICT DO NOTHING",
                        user_id,
                    )
                    return UserProfile(user_id=user_id)
                return UserProfile(
                    user_id=row["user_id"],
                    first_name=row["first_name"] or "",
                    last_name=row["last_name"] or "",
                    department=row["department"] or "",
                    role=row["role"] or "",
                    preferred_language=row["preferred_language"] or "ru",
                    frequent_categories=list(row["frequent_categories"] or []),
                    frequent_statuses=list(row["frequent_statuses"] or []),
                    favorite_filters=dict(row["favorite_filters"] or {}),
                    total_requests=row["total_requests"] or 0,
                )
        except Exception as exc:
            logger.error("LongTermMemory get_user_profile error: %s", exc)
            return UserProfile(user_id=user_id)

    async def update_user_profile(self, profile: UserProfile) -> None:
        """Обновить профиль пользователя."""
        if not self._pool:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO edms_ai.user_profiles
                        (user_id, first_name, last_name, department, role,
                         preferred_language, frequent_categories, frequent_statuses,
                         favorite_filters, total_requests, last_active_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        first_name = EXCLUDED.first_name,
                        last_name = EXCLUDED.last_name,
                        department = EXCLUDED.department,
                        role = EXCLUDED.role,
                        preferred_language = EXCLUDED.preferred_language,
                        frequent_categories = EXCLUDED.frequent_categories,
                        frequent_statuses = EXCLUDED.frequent_statuses,
                        favorite_filters = EXCLUDED.favorite_filters,
                        total_requests = EXCLUDED.total_requests,
                        last_active_at = NOW()
                    """,
                    profile.user_id, profile.first_name, profile.last_name,
                    profile.department, profile.role, profile.preferred_language,
                    profile.frequent_categories, profile.frequent_statuses,
                    json.dumps(profile.favorite_filters), profile.total_requests,
                )
        except Exception as exc:
            logger.error("LongTermMemory update_user_profile error: %s", exc)

    async def log_dialog(
        self,
        *,
        user_id: str,
        session_id: str,
        user_query: str,
        normalized_query: str = "",
        intent: str = "",
        entities: dict | None = None,
        confidence: float = 0.0,
        selected_tool: str = "",
        tool_args: dict | None = None,
        tool_results: list | None = None,
        agent_mode: str = "react",
        model_used: str = "",
        final_response: str = "",
        latency_ms: int = 0,
    ) -> str:
        """Записать лог диалога, вернуть dialog_id."""
        if not self._pool:
            import uuid
            return str(uuid.uuid4())
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO edms_ai.dialog_logs
                        (user_id, session_id, user_query, normalized_query, intent,
                         entities, confidence, selected_tool, tool_args, tool_results,
                         agent_mode, model_used, final_response, latency_ms)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                    RETURNING dialog_id::text
                    """,
                    user_id, session_id, user_query, normalized_query, intent,
                    json.dumps(entities or {}), confidence, selected_tool,
                    json.dumps(tool_args or {}), json.dumps(tool_results or []),
                    agent_mode, model_used, final_response, latency_ms,
                )
                return row["dialog_id"] if row else ""
        except Exception as exc:
            logger.error("LongTermMemory log_dialog error: %s", exc)
            import uuid
            return str(uuid.uuid4())

    async def update_feedback(
        self, dialog_id: str, rating: int, comment: str = ""
    ) -> bool:
        """Обновить оценку пользователя для диалога."""
        if not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE edms_ai.dialog_logs
                    SET user_feedback = $1, feedback_comment = $2, feedback_at = NOW()
                    WHERE dialog_id = $3::uuid
                    """,
                    rating, comment, dialog_id,
                )
            return True
        except Exception as exc:
            logger.error("LongTermMemory update_feedback error: %s", exc)
            return False

    async def get_positive_dialogs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Получить диалоги с положительной оценкой для RAG."""
        if not self._pool:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT user_query, normalized_query, intent, selected_tool,
                           tool_args, final_response, user_feedback
                    FROM edms_ai.dialog_logs
                    WHERE user_feedback = 1
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("LongTermMemory get_positive_dialogs error: %s", exc)
            return []

    async def get_negative_dialogs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Получить диалоги с отрицательной оценкой для анти-примеров."""
        if not self._pool:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT user_query, intent, selected_tool, final_response,
                           feedback_comment, created_at
                    FROM edms_ai.dialog_logs
                    WHERE user_feedback = -1
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("LongTermMemory get_negative_dialogs error: %s", exc)
            return []


# ── MemoryManager — façade ────────────────────────────────────────────────────

class MemoryManager:
    """
    Фасад для работы с трёхуровневой памятью.

    Объединяет ShortTermMemory, MediumTermMemory, LongTermMemory.
    Подмешивает профиль пользователя и состояние сессии в системный промпт.
    """

    def __init__(
        self,
        redis_url: str,
        postgres_dsn: str,
        max_context_tokens: int = 8000,
        session_ttl: int = 7200,
    ) -> None:
        self.short = ShortTermMemory(max_tokens=max_context_tokens)
        self.medium = MediumTermMemory(redis_url=redis_url, ttl=session_ttl)
        self.long = LongTermMemory(dsn=postgres_dsn)

    async def initialize(self) -> None:
        await self.long.initialize()
        logger.info("MemoryManager initialized")

    async def close(self) -> None:
        await self.long.close()

    async def get_user_profile(self, user_id: str) -> UserProfile:
        return await self.long.get_user_profile(user_id)

    async def get_or_create_session(
        self, user_id: str, session_id: str
    ) -> SessionState:
        state = await self.medium.get_session(session_id)
        if not state:
            state = SessionState(user_id=user_id, session_id=session_id)
            await self.medium.save_session(state)
        return state

    async def build_system_context(
        self,
        user_id: str,
        session_id: str,
        base_prompt: str,
        rag_examples: str = "",
        anti_examples: str = "",
    ) -> str:
        """
        Формирует системный промпт с учётом профиля пользователя и сессии.
        """
        profile = await self.get_user_profile(user_id)
        session = await self.get_or_create_session(user_id, session_id)

        blocks: list[str] = [base_prompt]

        # Профиль пользователя
        profile_block = _build_profile_block(profile)
        if profile_block:
            blocks.append(profile_block)

        # Состояние сессии
        session_block = _build_session_block(session)
        if session_block:
            blocks.append(session_block)

        # RAG few-shot
        if rag_examples:
            blocks.append(f"\n<few_shot_examples>\n{rag_examples}\n</few_shot_examples>")

        # Анти-примеры
        if anti_examples:
            blocks.append(f"\n<anti_examples>\n{anti_examples}\n</anti_examples>")

        return "\n".join(blocks)

    async def log_dialog(self, **kwargs) -> str:
        return await self.long.log_dialog(**kwargs)

    async def update_feedback(self, dialog_id: str, rating: int, comment: str = "") -> bool:
        return await self.long.update_feedback(dialog_id, rating, comment)


def _build_profile_block(profile: UserProfile) -> str:
    lines: list[str] = []
    name_parts = [p for p in [profile.last_name, profile.first_name] if p]
    if name_parts:
        lines.append(f"Пользователь: {' '.join(name_parts)}")
    if profile.department:
        lines.append(f"Отдел: {profile.department}")
    if profile.role:
        lines.append(f"Роль: {profile.role}")
    if profile.frequent_categories:
        lines.append(f"Часто работает с категориями: {', '.join(profile.frequent_categories[:3])}")
    if profile.total_requests:
        lines.append(f"Опытный пользователь ({profile.total_requests} запросов)")
    if not lines:
        return ""
    return "\n<user_profile>\n" + "\n".join(lines) + "\n</user_profile>"


def _build_session_block(session: SessionState) -> str:
    lines: list[str] = []
    if session.active_document_id:
        title = session.active_document_title or session.active_document_id
        lines.append(f"Активный документ: {title}")
    if session.current_task:
        lines.append(f"Текущая задача: {session.current_task}")
    if not lines:
        return ""
    return "\n<session_context>\n" + "\n".join(lines) + "\n</session_context>"
