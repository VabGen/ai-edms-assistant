# orchestrator/services/document_service.py
"""
Сервис работы с документами: Redis-кэш, TTL-логика.

Экспортирует:
    init_redis()    — инициализация пула из lifespan
    close_redis()   — освобождение при остановке
    get_cached_response() / set_cached_response() — кэш ответов
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from ..config import settings

logger = logging.getLogger(__name__)

_redis_client: aioredis.Redis | None = None

_READ_ONLY_INTENTS = {
    "get_document", "search_documents", "get_document_history",
    "get_workflow_status", "get_analytics",
}


async def init_redis() -> None:
    """Инициализирует async Redis-пул. Вызывается из lifespan()."""
    global _redis_client
    try:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        await _redis_client.ping()
        logger.info("Redis connected: %s", settings.REDIS_URL.split("@")[-1])
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — caching disabled", exc)
        _redis_client = None


async def close_redis() -> None:
    """Закрывает Redis-соединение. Вызывается из lifespan() при остановке."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection closed")


def _cache_key(normalized_query: str, user_id: str) -> str:
    """Строит Redis-ключ для кэша ответов."""
    raw = f"{normalized_query}::{user_id}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"edms:response:{digest}"


async def get_cached_response(
    normalized_query: str,
    user_id: str,
    intent: str,
) -> dict[str, Any] | None:
    """
    Возвращает кэшированный ответ или None.

    Только для read-only намерений.
    """
    if _redis_client is None or intent not in _READ_ONLY_INTENTS:
        return None
    try:
        key = _cache_key(normalized_query, user_id)
        raw = await _redis_client.get(key)
        if raw:
            logger.debug("Cache hit: %s", key[:32])
            return json.loads(raw)
    except Exception as exc:
        logger.warning("Cache get error: %s", exc)
    return None


async def set_cached_response(
    normalized_query: str,
    user_id: str,
    intent: str,
    response: dict[str, Any],
    ttl: int | None = None,
) -> None:
    """
    Кэширует ответ для read-only запросов.

    TTL: настраивается через settings.CACHE_TTL_SECONDS (default 300s).
    Write-операции никогда не кэшируются.
    """
    if _redis_client is None or intent not in _READ_ONLY_INTENTS:
        return
    try:
        key = _cache_key(normalized_query, user_id)
        effective_ttl = ttl or settings.CACHE_TTL_SECONDS
        await _redis_client.setex(
            key,
            effective_ttl,
            json.dumps(response, ensure_ascii=False, default=str),
        )
        logger.debug("Cache set: %s TTL=%ds", key[:32], effective_ttl)
    except Exception as exc:
        logger.warning("Cache set error: %s", exc)
