# edms_ai_assistant/orchestrator/services/document_service.py
"""
Сервис документов — единая точка доступа к DocumentDto.

ИСПРАВЛЕНИЯ:
  - Импорты document_enricher и nlp_service теперь с полным пакетным путём
  - Убран зависший from document_enricher import DocumentEnricher (без пути)
  - Убран зависший from nlp_service import EDMSNaturalLanguageService
  - init_redis() / close_redis() / get_redis() перенесены в infrastructure.redis_client
    (здесь только используем get_redis из infrastructure)

РАСШИРЕНИЕ: добавляя новую операцию — создай метод в этом классе,
  используй клиенты через clients/__init__.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import redis.asyncio as aioredis
from pydantic import BaseModel, ConfigDict, Field

from edms_ai_assistant.config import settings

# ИСПРАВЛЕНО: полные пакетные пути
from edms_ai_assistant.orchestrator.clients import (
    FULL_DOC_INCLUDES,
    SEARCH_DOC_INCLUDES,
    DocumentClient,
)
from edms_ai_assistant.orchestrator.services.document_enricher import DocumentEnricher
from edms_ai_assistant.orchestrator.services.nlp_service import EDMSNaturalLanguageService

logger = logging.getLogger(__name__)

_CACHE_PREFIX: str = "edms:doc:"
_CACHE_PREFIX_ANALYSIS: str = "edms:doc_analysis:"
_DEFAULT_CACHE_TTL: int = 300


# ── FastAPI dependency (re-export из infrastructure) ──────────────────────────

def get_redis() -> aioredis.Redis:
    """FastAPI dependency: возвращает Redis-клиент из инфраструктуры.

    Это тонкая обёртка — настоящий синглтон в infrastructure/redis_client.py.
    """
    from edms_ai_assistant.infrastructure.redis_client import get_redis as _get_redis
    return _get_redis()


# ── Result models ─────────────────────────────────────────────────────────────

class DocumentSearchResult(BaseModel):
    """Постраничный результат поиска документов."""
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    items: list[dict[str, Any]] = Field(default_factory=list)
    total_elements: int = 0
    total_pages: int = 0
    current_page: int = 0
    page_size: int = 10


class DocumentStats(BaseModel):
    """Статистика документов текущего пользователя."""
    model_config = ConfigDict(extra="ignore")

    executor: dict[str, Any] | None = None
    control: dict[str, Any] | None = None
    author: dict[str, Any] | None = None


class DocumentServiceConfig(BaseModel):
    """Конфигурация DocumentService."""
    model_config = ConfigDict(extra="ignore")

    edms_base_url: str = ""
    cache_ttl_seconds: int = _DEFAULT_CACHE_TTL
    search_page_size: int = 20
    enrich_documents: bool = True

    def model_post_init(self, __context: Any) -> None:
        if not self.edms_base_url:
            self.edms_base_url = str(settings.EDMS_BASE_URL)
        if self.cache_ttl_seconds == _DEFAULT_CACHE_TTL:
            self.cache_ttl_seconds = settings.CACHE_TTL_SECONDS


# ── Exceptions ────────────────────────────────────────────────────────────────

class DocumentServiceError(Exception):
    def __init__(self, message: str, document_id: str | None = None) -> None:
        super().__init__(message)
        self.document_id = document_id


class DocumentNotFoundError(DocumentServiceError):
    pass


class DocumentOperationError(DocumentServiceError):
    pass


# ── Cache helper ──────────────────────────────────────────────────────────────

class _DocumentCache:
    """Async Redis-кэш документов."""

    def __init__(self, redis: aioredis.Redis, ttl: int) -> None:
        self._r = redis
        self._ttl = ttl

    async def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        try:
            raw = await self._r.get(f"{_CACHE_PREFIX}{doc_id}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("Redis get_doc error: %s", exc)
            return None

    async def set_doc(self, doc_id: str, doc: dict[str, Any]) -> None:
        try:
            await self._r.setex(f"{_CACHE_PREFIX}{doc_id}", self._ttl, json.dumps(doc, default=str))
        except Exception as exc:
            logger.warning("Redis set_doc error: %s", exc)

    async def get_analysis(self, doc_id: str) -> dict[str, Any] | None:
        try:
            raw = await self._r.get(f"{_CACHE_PREFIX_ANALYSIS}{doc_id}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("Redis get_analysis error: %s", exc)
            return None

    async def set_analysis(self, doc_id: str, analysis: dict[str, Any]) -> None:
        try:
            await self._r.setex(
                f"{_CACHE_PREFIX_ANALYSIS}{doc_id}", self._ttl, json.dumps(analysis, default=str)
            )
        except Exception as exc:
            logger.warning("Redis set_analysis error: %s", exc)

    async def invalidate(self, doc_id: str) -> None:
        try:
            await self._r.delete(f"{_CACHE_PREFIX}{doc_id}", f"{_CACHE_PREFIX_ANALYSIS}{doc_id}")
        except Exception as exc:
            logger.warning("Redis invalidate error: %s", exc)


# ── DocumentService ───────────────────────────────────────────────────────────

class DocumentService:
    """Единая точка входа для всех операций с документами.

    Использование в FastAPI:
        svc = DocumentService(redis=Depends(get_redis))
        doc = await svc.get_document(token, document_id)

    Расширение: добавь метод в этот класс, используй DocumentClient через
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
    """

    def __init__(
        self,
        redis: aioredis.Redis,
        config: DocumentServiceConfig | None = None,
    ) -> None:
        self._config = config or DocumentServiceConfig()
        self._cache = _DocumentCache(redis=redis, ttl=self._config.cache_ttl_seconds)
        self._enricher = DocumentEnricher(base_url=self._config.edms_base_url)
        self._nlp = EDMSNaturalLanguageService()

    # ── READ ──────────────────────────────────────────────────────────────────

    async def get_document(
        self, token: str, document_id: str, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Получить DocumentDto (dict) с обогащением через DocumentEnricher."""
        return await self._fetch_raw(token, document_id, force_refresh)

    async def get_document_analysis(
        self, token: str, document_id: str, force_refresh: bool = False
    ) -> dict[str, Any]:
        """NLP-анализ документа через EDMSNaturalLanguageService."""
        if not force_refresh:
            cached = await self._cache.get_analysis(document_id)
            if cached is not None:
                return cached

        raw = await self._fetch_raw(token, document_id, force_refresh)
        analysis = self._nlp.process_document(raw)
        await self._cache.set_analysis(document_id, analysis)
        return analysis

    async def get_document_history(self, token: str, document_id: str) -> list[dict[str, Any]]:
        """История движения документа."""
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            result = await client.get_document_history_v2(token=token, document_id=document_id)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("content") or result.get("items") or []
        return []

    async def get_document_versions(self, token: str, document_id: str) -> list[dict[str, Any]]:
        """Версии документа."""
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            result = await client.get_document_versions(token=token, document_id=document_id)
        return result if isinstance(result, list) else []

    async def get_document_stats(self, token: str) -> DocumentStats:
        """Статистика документов текущего пользователя (3 параллельных запроса)."""
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            results = await asyncio.gather(
                client.get_stat_user_executor(token),
                client.get_stat_user_control(token),
                client.get_stat_user_author(token),
                return_exceptions=True,
            )
        executor_stat, control_stat, author_stat = results
        return DocumentStats(
            executor=executor_stat if isinstance(executor_stat, dict) else None,
            control=control_stat if isinstance(control_stat, dict) else None,
            author=author_stat if isinstance(author_stat, dict) else None,
        )

    # ── SEARCH ────────────────────────────────────────────────────────────────

    async def search_documents(
        self,
        token: str,
        doc_filter: dict[str, Any] | None = None,
        page: int = 0,
        size: int | None = None,
        sort: str | None = None,
        includes: list[str] | None = None,
    ) -> DocumentSearchResult:
        """Поиск документов с фильтрами и пагинацией."""
        effective_size = size or self._config.search_page_size
        pageable: dict[str, Any] = {"page": page, "size": effective_size}
        if sort:
            pageable["sort"] = sort

        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            raw_page = await client._make_request(
                "GET", "api/document", token=token,
                params={**(doc_filter or {}), **pageable, "includes": includes or SEARCH_DOC_INCLUDES}
            )

        if not isinstance(raw_page, dict):
            return DocumentSearchResult(page_size=effective_size)

        content = raw_page.get("content") or []
        return DocumentSearchResult(
            items=content,
            total_elements=raw_page.get("totalElements") or 0,
            total_pages=raw_page.get("totalPages") or 0,
            current_page=raw_page.get("number") or page,
            page_size=effective_size,
        )

    # ── LIFECYCLE ─────────────────────────────────────────────────────────────

    async def start_document(self, token: str, document_id: str) -> bool:
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            success = await client.start_document(token=token, document_id=document_id)
        if not success:
            raise DocumentOperationError(f"Не удалось запустить документ {document_id}", document_id)
        await self._cache.invalidate(document_id)
        return True

    async def cancel_document(self, token: str, document_id: str, comment: str | None = None) -> bool:
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            success = await client.cancel_document(token=token, document_id=document_id, comment=comment)
        if not success:
            raise DocumentOperationError(f"Не удалось аннулировать документ {document_id}", document_id)
        await self._cache.invalidate(document_id)
        return True

    async def execute_operations(
        self, token: str, document_id: str, operations: list[dict[str, Any]]
    ) -> bool:
        if not operations:
            raise ValueError("operations list cannot be empty")
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            success = await client.execute_document_operations(token, document_id, operations)
        if not success:
            raise DocumentOperationError(f"Не удалось выполнить операции над {document_id}", document_id)
        await self._cache.invalidate(document_id)
        return True

    # ── CONTROL ───────────────────────────────────────────────────────────────

    async def set_control(
        self, token: str, document_id: str, control_type_id: str,
        date_control_end: str, control_employee_id: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"controlTypeId": control_type_id, "dateControlEnd": date_control_end}
        if control_employee_id:
            payload["controlEmployeeId"] = control_employee_id
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            result = await client.set_document_control(token, document_id, payload)
        if not result:
            raise DocumentOperationError(f"Не удалось поставить {document_id} на контроль", document_id)
        await self._cache.invalidate(document_id)
        return result

    async def remove_control(self, token: str, document_id: str) -> bool:
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            success = await client.remove_document_control(token, document_id)
        if not success:
            raise DocumentOperationError(f"Не удалось снять контроль {document_id}", document_id)
        await self._cache.invalidate(document_id)
        return True

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _fetch_raw(
        self, token: str, document_id: str, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Основной pipeline: кэш → API → обогащение → кэш."""
        if not force_refresh:
            cached = await self._cache.get_doc(document_id)
            if cached is not None:
                return cached

        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            raw = await client.get_document_metadata(token=token, document_id=document_id, includes=FULL_DOC_INCLUDES)

        if not raw:
            raise DocumentNotFoundError(f"Документ {document_id} не найден", document_id)

        if self._config.enrich_documents:
            raw = await self._enricher.enrich(raw, token=token)

        await self._cache.set_doc(document_id, raw)
        return raw