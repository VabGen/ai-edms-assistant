"""
Cache module — простое in-memory кэширование + Redis кэш для оркестратора.

Используется для:
- Кэширования частых запросов (TTL-based)
- Дедупликации одинаковых вопросов в течение сессии
- Снижения нагрузки на LLM и MCP сервер
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger("cache")


class InMemoryCache:
    """
    Простой in-memory LRU-подобный кэш с TTL.

    Thread-safe для asyncio (одиночный event loop).
    Максимальный размер — 1000 записей (FIFO eviction при переполнении).
    """

    MAX_SIZE = 1000

    def __init__(self, default_ttl: int = 300) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expire_at)
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Вернуть значение из кэша или None если отсутствует/просрочен."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expire_at = entry
        if time.monotonic() > expire_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Сохранить значение с TTL."""
        if len(self._store) >= self.MAX_SIZE:
            # Удаляем самые старые записи (первые 10%)
            to_remove = list(self._store.keys())[: self.MAX_SIZE // 10]
            for k in to_remove:
                del self._store[k]

        expire_at = time.monotonic() + (ttl or self._default_ttl)
        self._store[key] = (value, expire_at)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def cleanup_expired(self) -> int:
        """Удалить просроченные записи. Возвращает количество удалённых."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
        return len(expired)

    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict[str, int]:
        return {"size": self.size(), "max_size": self.MAX_SIZE}


def make_cache_key(prefix: str, **kwargs: Any) -> str:
    """
    Создать детерминированный ключ кэша из произвольных аргументов.

    Args:
        prefix: Префикс ключа (например, "chat", "tool_result")
        **kwargs: Параметры для хэширования

    Returns:
        Строковый ключ вида "prefix:sha256_8chars"

    Example:
        make_cache_key("search", query="договор", status="IN_PROGRESS")
        → "search:a3f7c92b"
    """
    payload = json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{prefix}:{digest}"


def make_query_cache_key(user_id: str, message: str) -> str:
    """Ключ кэша для пользовательского запроса."""
    normalized = message.strip().lower()
    return make_cache_key("query", user_id=user_id, msg=normalized)
