"""
orchestrator/memory.py — Трёхуровневая система памяти EDMS AI Assistant.

Уровни:
  - ShortTermMemory:  текущий диалог, управление токенами
  - MediumTermMemory: Redis TTL-сессии (текущий контекст, задача)
  - LongTermMemory:   PostgreSQL профиль, история действий, логи диалогов

MemoryManager объединяет все три уровня и предоставляет единый интерфейс.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://edms:edms@localhost:5432/edms_ai",
)
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_SHORT_TERM_TOKENS: int = int(os.getenv("MAX_SHORT_TERM_TOKENS", "8000"))


# ---------------------------------------------------------------------------
# SQLAlchemy: Base и движок
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""
    pass


# Создаём движок и фабрику сессий при импорте модуля
_engine = create_async_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10)
_async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_engine,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncSession:
    """Создать новую асинхронную сессию SQLAlchemy."""
    return _async_session_factory()


# ---------------------------------------------------------------------------
# SQLAlchemy: ORM-модели
# ---------------------------------------------------------------------------

class UserProfile(Base):
    """
    Долгосрочный профиль пользователя: предпочтения, настройки интерфейса.
    """
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(
        String(255), primary_key=True, comment="UUID пользователя из EDMS"
    )
    display_name: Mapped[str | None] = mapped_column(String(500), comment="Отображаемое имя")
    email: Mapped[str | None] = mapped_column(String(500), comment="Email пользователя")
    department: Mapped[str | None] = mapped_column(String(500), comment="Отдел")
    role_in_org: Mapped[str | None] = mapped_column(String(255), comment="Должность")
    preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default="{}",
        comment="Пользовательские настройки в формате JSONB",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="Дата создания профиля",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="Дата последнего обновления",
    )

    conversation_logs: Mapped[list[ConversationLog]] = relationship(
        "ConversationLog", back_populates="user", lazy="raise"
    )
    action_history: Mapped[list[ActionHistory]] = relationship(
        "ActionHistory", back_populates="user", lazy="raise"
    )


class ConversationLog(Base):
    """
    Лог сообщений текущего и прошлых диалогов.
    Используется для анализа и формирования контекста.
    """
    __tablename__ = "conversation_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("user_profiles.user_id", ondelete="CASCADE"),
        index=True,
        comment="UUID пользователя",
    )
    session_id: Mapped[str] = mapped_column(
        String(255), index=True, comment="UUID сессии"
    )
    role: Mapped[str] = mapped_column(
        String(50), comment="Роль: user, assistant, system, tool"
    )
    content: Mapped[str] = mapped_column(Text, comment="Текст сообщения")
    tokens: Mapped[int] = mapped_column(
        Integer, default=0, comment="Приблизительное количество токенов"
    )
    tool_name: Mapped[str | None] = mapped_column(
        String(255), comment="Имя вызванного инструмента (для role=tool)"
    )
    tool_result: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, comment="Результат инструмента"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        comment="Время создания сообщения",
    )

    user: Mapped[UserProfile] = relationship("UserProfile", back_populates="conversation_logs")


class ActionHistory(Base):
    """
    История действий пользователя в EDMS через ИИ-ассистента.
    Используется для персонализации и аудита.
    """
    __tablename__ = "action_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("user_profiles.user_id", ondelete="CASCADE"),
        index=True,
        comment="UUID пользователя",
    )
    action_type: Mapped[str] = mapped_column(
        String(255),
        index=True,
        comment="Тип действия: get_document, search_documents, create_document и т.д.",
    )
    entity_id: Mapped[str | None] = mapped_column(
        String(255), comment="ID сущности (например, UUID документа)"
    )
    action_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default="{}",
        comment="Параметры и результат действия",
    )
    success: Mapped[bool] = mapped_column(
        default=True, comment="Успешно ли выполнено действие"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        comment="Время выполнения действия",
    )

    user: Mapped[UserProfile] = relationship("UserProfile", back_populates="action_history")


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """
    Краткосрочная память: буфер сообщений текущего диалога.

    Автоматически обрезает старые сообщения при превышении лимита токенов,
    сохраняя системное сообщение и последние N обменов.
    """

    def __init__(self, max_tokens: int = MAX_SHORT_TERM_TOKENS) -> None:
        self.max_tokens = max_tokens
        self._messages: list[dict[str, Any]] = []
        self._total_tokens: int = 0

    def count_tokens(self, text: str) -> int:
        """
        Приблизительный подсчёт токенов.
        Формула: 1 токен ≈ 4 символа (работает для русского и английского).
        """
        return max(1, len(str(text)) // 4)

    def add_message(
        self,
        role: str,
        content: str,
        tool_name: str | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> None:
        """
        Добавить сообщение в буфер.

        Параметры:
            role: user | assistant | system | tool
            content: текст сообщения
            tool_name: имя инструмента (для role=tool)
            tool_result: результат инструмента в JSON
        """
        tokens = self.count_tokens(content)
        msg: dict[str, Any] = {
            "role": role,
            "content": content,
            "tokens": tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if tool_name:
            msg["tool_name"] = tool_name
        if tool_result is not None:
            msg["tool_result"] = tool_result

        self._messages.append(msg)
        self._total_tokens += tokens
        self._trim_if_needed()

    def _trim_if_needed(self) -> None:
        """
        Обрезать буфер если превышен лимит токенов.

        Алгоритм:
        1. Сохраняем системное сообщение (первое, если role=system)
        2. Удаляем старые сообщения из середины (не трогаем последние 4)
        3. Повторяем до выполнения лимита
        """
        if self._total_tokens <= self.max_tokens:
            return

        system_msgs = [m for m in self._messages if m["role"] == "system"]
        non_system = [m for m in self._messages if m["role"] != "system"]

        while self._total_tokens > self.max_tokens and len(non_system) > 2:
            # Удаляем самое старое не-системное сообщение
            removed = non_system.pop(0)
            self._total_tokens -= removed["tokens"]

        self._messages = system_msgs + non_system

    def get_context(self) -> list[dict[str, Any]]:
        """Получить текущий контекст диалога без внутренних полей."""
        return [
            {k: v for k, v in msg.items() if k not in ("tokens", "timestamp")}
            for msg in self._messages
        ]

    def get_messages_for_llm(self) -> list[dict[str, str]]:
        """Получить сообщения в формате для Anthropic/OpenAI API."""
        result = []
        for msg in self._messages:
            entry: dict[str, str] = {"role": msg["role"], "content": msg["content"]}
            result.append(entry)
        return result

    def clear(self) -> None:
        """Очистить буфер сообщений."""
        self._messages.clear()
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        """Текущее количество токенов в буфере."""
        return self._total_tokens

    @property
    def message_count(self) -> int:
        """Количество сообщений в буфере."""
        return len(self._messages)


# ---------------------------------------------------------------------------
# MediumTermMemory
# ---------------------------------------------------------------------------

class MediumTermMemory:
    """
    Среднесрочная память: Redis TTL-сессии.

    Хранит:
    - Текущий документ (document_id, document_title)
    - Текущую задачу (task_description, task_step)
    - Состояние диалога (pending_clarification, last_intent)
    - Контекстные данные пользователя
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        session_ttl: int = SESSION_TTL_SECONDS,
    ) -> None:
        self._redis_url = redis_url
        self.session_ttl = session_ttl
        self._client: redis.Redis | None = None

    async def _get_client(self) -> redis.Redis:
        """Получить Redis клиент (создать при первом вызове)."""
        if self._client is None:
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    def _session_key(self, session_id: str) -> str:
        """Ключ хранилища сессии."""
        return f"edms:session:{session_id}"

    def _key(self, session_id: str, key: str) -> str:
        """Полный ключ поля в сессии."""
        return f"edms:session:{session_id}:{key}"

    async def get(self, session_id: str, key: str) -> Any | None:
        """
        Получить значение из сессии.

        Параметры:
            session_id: идентификатор сессии
            key: ключ значения

        Возвращает:
            Значение или None если не найдено
        """
        client = await self._get_client()
        full_key = self._key(session_id, key)
        raw = await client.get(full_key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def set(
        self,
        session_id: str,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """
        Сохранить значение в сессии.

        Параметры:
            session_id: идентификатор сессии
            key: ключ значения
            value: сохраняемое значение (сериализуется в JSON)
            ttl: время жизни в секундах (по умолчанию session_ttl)
        """
        client = await self._get_client()
        full_key = self._key(session_id, key)
        effective_ttl = ttl if ttl is not None else self.session_ttl
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        await client.setex(full_key, effective_ttl, serialized)

    async def get_full_session(self, session_id: str) -> dict[str, Any]:
        """
        Получить все данные сессии.

        Возвращает:
            Словарь со всеми значениями сессии
        """
        client = await self._get_client()
        pattern = self._key(session_id, "*")
        keys = await client.keys(pattern)
        if not keys:
            return {}

        prefix = self._key(session_id, "")
        result: dict[str, Any] = {}
        for full_key in keys:
            short_key = full_key.removeprefix(prefix)
            raw = await client.get(full_key)
            if raw:
                try:
                    result[short_key] = json.loads(raw)
                except json.JSONDecodeError:
                    result[short_key] = raw
        return result

    async def clear(self, session_id: str) -> None:
        """
        Очистить все данные сессии.

        Параметры:
            session_id: идентификатор сессии для очистки
        """
        client = await self._get_client()
        pattern = self._key(session_id, "*")
        keys = await client.keys(pattern)
        if keys:
            await client.delete(*keys)

    async def extend_ttl(self, session_id: str) -> None:
        """Продлить TTL всех ключей сессии."""
        client = await self._get_client()
        pattern = self._key(session_id, "*")
        keys = await client.keys(pattern)
        for key in keys:
            await client.expire(key, self.session_ttl)

    async def close(self) -> None:
        """Закрыть соединение с Redis."""
        if self._client:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------

class LongTermMemory:
    """
    Долгосрочная память: PostgreSQL.

    Хранит профили пользователей, историю действий, логи диалогов.
    Использует SQLAlchemy AsyncSession.
    """

    async def get_profile(self, user_id: str) -> UserProfile | None:
        """
        Получить профиль пользователя.

        Параметры:
            user_id: UUID пользователя

        Возвращает:
            UserProfile или None если пользователь не найден
        """
        async with _async_session_factory() as session:
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def upsert_profile(
        self,
        user_id: str,
        preferences_update: dict[str, Any],
        display_name: str | None = None,
        email: str | None = None,
        department: str | None = None,
        role_in_org: str | None = None,
    ) -> UserProfile:
        """
        Создать или обновить профиль пользователя.

        Параметры:
            user_id: UUID пользователя
            preferences_update: обновления для поля preferences
            display_name: отображаемое имя
            email: email пользователя
            department: отдел
            role_in_org: должность

        Возвращает:
            Обновлённый UserProfile
        """
        async with _async_session_factory() as session:
            async with session.begin():
                stmt = select(UserProfile).where(UserProfile.user_id == user_id)
                result = await session.execute(stmt)
                profile = result.scalar_one_or_none()

                if profile is None:
                    profile = UserProfile(
                        user_id=user_id,
                        preferences=preferences_update,
                        display_name=display_name,
                        email=email,
                        department=department,
                        role_in_org=role_in_org,
                    )
                    session.add(profile)
                else:
                    # Мёрджим preferences (не перезаписываем полностью)
                    merged = {**(profile.preferences or {}), **preferences_update}
                    profile.preferences = merged
                    if display_name is not None:
                        profile.display_name = display_name
                    if email is not None:
                        profile.email = email
                    if department is not None:
                        profile.department = department
                    if role_in_org is not None:
                        profile.role_in_org = role_in_org

                await session.flush()
                await session.refresh(profile)
                return profile

    async def log_action(
        self,
        user_id: str,
        action_type: str,
        entity_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        success: bool = True,
    ) -> ActionHistory:
        """
        Записать действие пользователя в историю.

        Параметры:
            user_id: UUID пользователя
            action_type: тип действия (get_document, search_documents и т.д.)
            entity_id: ID сущности (например, UUID документа)
            metadata: дополнительные данные (параметры, результат)
            success: успешно ли выполнено действие

        Возвращает:
            Созданная запись ActionHistory
        """
        async with _async_session_factory() as session:
            async with session.begin():
                # Ensure user profile exists
                profile_stmt = select(UserProfile).where(UserProfile.user_id == user_id)
                profile_result = await session.execute(profile_stmt)
                if profile_result.scalar_one_or_none() is None:
                    session.add(UserProfile(user_id=user_id, preferences={}))
                    await session.flush()

                action = ActionHistory(
                    user_id=user_id,
                    action_type=action_type,
                    entity_id=entity_id,
                    action_metadata=metadata or {},
                    success=success,
                )
                session.add(action)
                await session.flush()
                await session.refresh(action)
                return action

    async def get_recent_actions(
        self,
        user_id: str,
        limit: int = 20,
        action_type: str | None = None,
    ) -> list[ActionHistory]:
        """
        Получить последние действия пользователя.

        Параметры:
            user_id: UUID пользователя
            limit: максимальное количество записей
            action_type: фильтр по типу действия

        Возвращает:
            Список ActionHistory, отсортированный по убыванию времени
        """
        async with _async_session_factory() as session:
            stmt = (
                select(ActionHistory)
                .where(ActionHistory.user_id == user_id)
                .order_by(ActionHistory.timestamp.desc())
                .limit(limit)
            )
            if action_type:
                stmt = stmt.where(ActionHistory.action_type == action_type)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def save_conversation_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        tokens: int = 0,
        tool_name: str | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> ConversationLog:
        """
        Сохранить сообщение диалога в долгосрочное хранилище.

        Параметры:
            user_id: UUID пользователя
            session_id: идентификатор сессии
            role: роль (user, assistant, system, tool)
            content: текст сообщения
            tokens: количество токенов
            tool_name: имя инструмента (для role=tool)
            tool_result: результат инструмента

        Возвращает:
            Сохранённая запись ConversationLog
        """
        async with _async_session_factory() as session:
            async with session.begin():
                # Ensure profile
                p_stmt = select(UserProfile).where(UserProfile.user_id == user_id)
                if not (await session.execute(p_stmt)).scalar_one_or_none():
                    session.add(UserProfile(user_id=user_id, preferences={}))
                    await session.flush()

                log_entry = ConversationLog(
                    user_id=user_id,
                    session_id=session_id,
                    role=role,
                    content=content,
                    tokens=tokens,
                    tool_name=tool_name,
                    tool_result=tool_result,
                )
                session.add(log_entry)
                await session.flush()
                await session.refresh(log_entry)
                return log_entry


# ---------------------------------------------------------------------------
# MemoryManager — объединяет все три уровня
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Единый менеджер памяти для EDMS AI Assistant.

    Объединяет ShortTermMemory, MediumTermMemory и LongTermMemory
    и предоставляет интерфейс для формирования контекста промпта.
    """

    def __init__(
        self,
        max_tokens: int = MAX_SHORT_TERM_TOKENS,
        redis_url: str = REDIS_URL,
        session_ttl: int = SESSION_TTL_SECONDS,
    ) -> None:
        self.short = ShortTermMemory(max_tokens=max_tokens)
        self.medium = MediumTermMemory(redis_url=redis_url, session_ttl=session_ttl)
        self.long = LongTermMemory()

    async def build_context_for_prompt(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Собрать контекст из всех уровней памяти для формирования промпта.

        Параметры:
            user_id: UUID пользователя
            session_id: идентификатор текущей сессии

        Возвращает:
            Словарь с полями:
            - user_profile: данные профиля (имя, роль, предпочтения)
            - session_state: текущее состояние сессии из Redis
            - short_term_messages: буфер текущего диалога
            - recent_actions: последние действия пользователя
        """
        # Загружаем данные параллельно
        import asyncio
        profile_task = asyncio.create_task(self.long.get_profile(user_id))
        session_task = asyncio.create_task(self.medium.get_full_session(session_id))
        actions_task = asyncio.create_task(self.long.get_recent_actions(user_id, limit=10))

        profile, session_state, recent_actions = await asyncio.gather(
            profile_task, session_task, actions_task,
        )

        profile_data: dict[str, Any] = {}
        if profile:
            profile_data = {
                "user_id": profile.user_id,
                "display_name": profile.display_name,
                "email": profile.email,
                "department": profile.department,
                "role_in_org": profile.role_in_org,
                "preferences": profile.preferences or {},
            }

        actions_data = [
            {
                "action_type": a.action_type,
                "entity_id": a.entity_id,
                "success": a.success,
                "timestamp": a.timestamp.isoformat() if a.timestamp else None,
            }
            for a in recent_actions
        ]

        return {
            "user_profile": profile_data,
            "session_state": session_state,
            "short_term_messages": self.short.get_context(),
            "recent_actions": actions_data,
        }

    async def record_exchange(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Записать обмен (запрос пользователя + ответ ассистента) во все уровни памяти.

        Параметры:
            user_id: UUID пользователя
            session_id: идентификатор сессии
            user_message: сообщение пользователя
            assistant_message: ответ ассистента
            tool_calls: список вызовов инструментов в этом обмене
        """
        import asyncio

        # Обновляем краткосрочную память
        self.short.add_message("user", user_message)
        self.short.add_message("assistant", assistant_message)

        # Сохраняем в долгосрочную память (не блокируем основной поток)
        user_tokens = self.short.count_tokens(user_message)
        assistant_tokens = self.short.count_tokens(assistant_message)

        save_tasks = [
            asyncio.create_task(
                self.long.save_conversation_message(
                    user_id=user_id,
                    session_id=session_id,
                    role="user",
                    content=user_message,
                    tokens=user_tokens,
                )
            ),
            asyncio.create_task(
                self.long.save_conversation_message(
                    user_id=user_id,
                    session_id=session_id,
                    role="assistant",
                    content=assistant_message,
                    tokens=assistant_tokens,
                )
            ),
        ]
        await asyncio.gather(*save_tasks, return_exceptions=True)

    async def close(self) -> None:
        """Освободить ресурсы (закрыть соединения)."""
        await self.medium.close()
        await _engine.dispose()
