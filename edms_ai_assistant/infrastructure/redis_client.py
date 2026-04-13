# edms_ai_assistant/infrastructure/redis_client.py
"""
Глобальный async Redis-клиент.

Предоставляет:
  - init_redis()   — инициализация при старте приложения (lifespan)
  - close_redis()  — закрытие при остановке приложения
  - get_redis()    — FastAPI dependency / прямой доступ к клиенту
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)

_redis_client: aioredis.Redis | None = None


async def init_redis() -> aioredis.Redis:
    """Инициализирует и проверяет глобальный Redis-клиент.

    Вызывается один раз из lifespan() при старте FastAPI-приложения.
    При недоступности Redis пишет WARNING, но не падает — кэш будет пропускаться.

    Returns:
        Настроенный экземпляр aioredis.Redis.
    """
    global _redis_client

    _redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )

    try:
        await _redis_client.ping()
        logger.info("Redis connected: %s", settings.REDIS_URL)
    except Exception as exc:
        logger.warning(
            "Redis unavailable at startup — caching disabled. Error: %s", exc
        )

    return _redis_client


async def close_redis() -> None:
    """Закрывает глобальный Redis-клиент.

    Вызывается из lifespan() при остановке приложения.
    """
    global _redis_client

    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis client closed")


def get_redis() -> aioredis.Redis:
    """FastAPI dependency: возвращает глобальный Redis-клиент.

    Returns:
        Экземпляр aioredis.Redis.

    Raises:
        RuntimeError: Если init_redis() не был вызван при старте.
    """
    if _redis_client is None:
        raise RuntimeError(
            "Redis не инициализирован. Убедитесь, что init_redis() "
            "вызывается в lifespan() приложения."
        )
    return _redis_client
