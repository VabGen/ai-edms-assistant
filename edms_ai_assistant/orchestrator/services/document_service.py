# edms_ai_assistant/services/document_service.py
"""
EDMS AI Assistant — Document Service.

Ответственность:
    Единственная точка входа для всей работы с документами в системе.
    Оркестрирует:

        DocumentClient              — HTTP I/O к Java EDMS API  (Infrastructure)
        DocumentEnricher            — дозапрос вложенных объектов по UUID  (Service)
        EDMSNaturalLanguageService  — NLP-анализ документа  (Service/Domain)
        Redis (через Depends)       — кэш с TTL  (Infrastructure)

Операции:
    READ    get_document()          — DocumentDto (generated OpenAPI model)
            get_document_analysis() — NLP-анализ (dict)
            get_document_history()  — история движения (list)
            get_document_versions() — версии (list)
            get_document_stats()    — статистика пользователя

    SEARCH  search_documents()      — полный фильтр + пагинация
            search_for_agent()      — плоские параметры для LangGraph @tool

    LIFECYCLE
            start_document()        — запуск маршрута
            cancel_document()       — аннулирование
            execute_operations()    — пакетное выполнение (AGREE, SIGN, REJECT…)

    CONTROL
            set_control()           — поставить на контроль
            remove_control()        — снять с контроля
            delete_control()        — удалить контроль

Pydantic-модели:
    Сервис использует DocumentDto из edms_ai_assistant/generated/resources_openapi.py
    (сгенерирован datamodel-codegen из OpenAPI-спецификации Java EDMS API).
    Это единственный источник истины — самописных дублирующих моделей нет.

    Конвертация Dict → DocumentDto: DocumentDto.model_validate(raw_dict).
    Поля в DocumentDto — camelCase (как в JSON API), без alias.

    DocumentSearchResult и DocumentStats — тонкие контейнеры результата,
    определены здесь как часть интерфейса сервисного слоя.

Redis / кэш:
    Сервис принимает redis.asyncio.Redis через __init__.
    В FastAPI-слое клиент передаётся через Depends(get_redis).
    Инвалидация при любой write-операции (start/cancel/execute/control).
    TTL конфигурируется через DocumentServiceConfig.cache_ttl_seconds.

Права:
    Java API сам возвращает 403.
    DocumentService только логирует ошибки — не блокирует на Python-стороне.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import redis.asyncio as aioredis
from document_enricher import DocumentEnricher
from nlp_service import EDMSNaturalLanguageService
from pydantic import BaseModel, ConfigDict, Field

from edms_ai_assistant.config import settings
from edms_ai_assistant.generated.resources_openapi import DocumentDto

from edms_ai_assistant.orchestrator.clients import (
    FULL_DOC_INCLUDES,
    SEARCH_DOC_INCLUDES,
    DocumentClient,
)

logger = logging.getLogger(__name__)

# ── Cache key prefixes ────────────────────────────────────────────────────────
_CACHE_PREFIX: str = "edms:doc:"
_CACHE_PREFIX_ANALYSIS: str = "edms:doc_analysis:"
_DEFAULT_CACHE_TTL: int = 300


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI dependency
# ══════════════════════════════════════════════════════════════════════════════


# Синглтон-клиент Redis — создаётся один раз при старте приложения.
_redis_client: aioredis.Redis | None = None


async def init_redis() -> aioredis.Redis:
    """Initialize and verify the shared Redis client from settings.

    Вызывается из lifespan() при старте приложения через await.
    Использует settings.REDIS_URL (property из config.py).
    После создания клиента выполняет PING — если Redis недоступен,
    пишет WARNING, но не падает: кэш просто будет пропускаться.

    Returns:
        Configured aioredis.Redis instance.
    """
    global _redis_client
    _redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    try:
        await _redis_client.ping()
        logger.info("Redis connected", extra={"url": settings.REDIS_URL})
    except Exception as exc:
        logger.warning(
            "Redis unavailable at startup — caching disabled",
            extra={"url": settings.REDIS_URL, "error": str(exc)},
        )
    return _redis_client


async def close_redis() -> None:
    """Close the shared Redis client.

    Вызывается из lifespan() при остановке приложения.
    """
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis client closed")


def get_redis() -> aioredis.Redis:
    """FastAPI dependency: provides the shared Redis client.

    Использование в роутере::

        @router.get("/documents/{doc_id}")
        async def get_doc(
            doc_id: str,
            token: str = Depends(get_current_token),
            redis: aioredis.Redis = Depends(get_redis),
        ) -> DocumentDto:
            service = DocumentService(redis=redis)
            return await service.get_document(token, doc_id)

    Returns:
        Shared aioredis.Redis instance.

    Raises:
        RuntimeError: If init_redis() was not called during startup.
    """
    if _redis_client is None:
        raise RuntimeError(
            "Redis не инициализирован. Вызовите init_redis() в lifespan()."
        )
    return _redis_client


# ══════════════════════════════════════════════════════════════════════════════
# Result / config models
# ══════════════════════════════════════════════════════════════════════════════


class DocumentSearchResult(BaseModel):
    """Paginated document search result (service layer contract).

    Обёртка над Spring Page<DocumentDto>:
    сохраняет метаданные пагинации, нужные агенту и UI.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    items: list[DocumentDto] = Field(default_factory=list)
    total_elements: int = 0
    total_pages: int = 0
    current_page: int = 0
    page_size: int = 10


class DocumentStats(BaseModel):
    """Document statistics for the current user."""

    model_config = ConfigDict(extra="ignore")

    executor: dict[str, Any] | None = None
    control: dict[str, Any] | None = None
    author: dict[str, Any] | None = None


class DocumentServiceConfig(BaseModel):
    """Configuration for DocumentService.

    Args:
        edms_base_url: Base URL of the Java EDMS API.
            Defaults to settings.EDMS_BASE_URL.
        cache_ttl_seconds: TTL for cached documents in seconds.
        search_page_size: Default page size for search queries.
        enrich_documents: Whether to run DocumentEnricher after fetch.
    """

    model_config = ConfigDict(extra="ignore")

    edms_base_url: str = ""
    cache_ttl_seconds: int = _DEFAULT_CACHE_TTL
    search_page_size: int = 20
    enrich_documents: bool = True

    def model_post_init(self, __context: Any) -> None:
        """Fill defaults from settings if not provided explicitly."""
        if not self.edms_base_url:
            self.edms_base_url = str(settings.EDMS_BASE_URL)
        if self.cache_ttl_seconds == _DEFAULT_CACHE_TTL:
            self.cache_ttl_seconds = settings.CACHE_TTL_SECONDS


# ══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ══════════════════════════════════════════════════════════════════════════════


class DocumentServiceError(Exception):
    """Base error for DocumentService operations."""

    def __init__(self, message: str, document_id: str | None = None) -> None:
        super().__init__(message)
        self.document_id = document_id


class DocumentNotFoundError(DocumentServiceError):
    """Document not found — API returned empty or None."""


class DocumentOperationError(DocumentServiceError):
    """Write operation failed (start / cancel / execute / control)."""


# ══════════════════════════════════════════════════════════════════════════════
# Redis cache helper
# ══════════════════════════════════════════════════════════════════════════════


class _DocumentCache:
    """Async Redis cache for DocumentService.

    Хранит JSON-строки. Ключи:
        edms:doc:{id}           → сырой DocumentDto dict
        edms:doc_analysis:{id}  → NLP-анализ dict

    Все операции catch exceptions — падение Redis не ломает основной флоу.

    Args:
        redis: Shared aioredis.Redis instance (decode_responses=True).
        ttl: Cache TTL in seconds.
    """

    def __init__(self, redis: aioredis.Redis, ttl: int) -> None:
        self._r = redis
        self._ttl = ttl

    async def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        """Get raw document dict from cache.

        Args:
            doc_id: Document UUID string.

        Returns:
            Parsed dict or None on cache miss or error.
        """
        try:
            raw = await self._r.get(f"{_CACHE_PREFIX}{doc_id}")
            if raw:
                logger.debug("Cache hit: doc", extra={"document_id": doc_id})
                return json.loads(raw)
        except Exception as exc:
            logger.warning(
                "Redis get_doc error",
                extra={"document_id": doc_id, "error": str(exc)},
            )
        return None

    async def set_doc(self, doc_id: str, doc: dict[str, Any]) -> None:
        """Store raw document dict in cache.

        Args:
            doc_id: Document UUID string.
            doc: Raw API response dict.
        """
        try:
            await self._r.setex(
                f"{_CACHE_PREFIX}{doc_id}",
                self._ttl,
                json.dumps(doc, default=str),
            )
        except Exception as exc:
            logger.warning(
                "Redis set_doc error",
                extra={"document_id": doc_id, "error": str(exc)},
            )

    async def get_analysis(self, doc_id: str) -> dict[str, Any] | None:
        """Get cached NLP analysis dict.

        Args:
            doc_id: Document UUID string.

        Returns:
            Analysis dict or None.
        """
        try:
            raw = await self._r.get(f"{_CACHE_PREFIX_ANALYSIS}{doc_id}")
            if raw:
                logger.debug("Cache hit: analysis", extra={"document_id": doc_id})
                return json.loads(raw)
        except Exception as exc:
            logger.warning(
                "Redis get_analysis error",
                extra={"document_id": doc_id, "error": str(exc)},
            )
        return None

    async def set_analysis(self, doc_id: str, analysis: dict[str, Any]) -> None:
        """Store NLP analysis dict in cache.

        Args:
            doc_id: Document UUID string.
            analysis: NLP analysis result dict.
        """
        try:
            await self._r.setex(
                f"{_CACHE_PREFIX_ANALYSIS}{doc_id}",
                self._ttl,
                json.dumps(analysis, default=str),
            )
        except Exception as exc:
            logger.warning(
                "Redis set_analysis error",
                extra={"document_id": doc_id, "error": str(exc)},
            )

    async def invalidate(self, doc_id: str) -> None:
        """Invalidate all cache entries for a document.

        Вызывается после любой write-операции (start/cancel/execute/control).

        Args:
            doc_id: Document UUID string.
        """
        try:
            await self._r.delete(
                f"{_CACHE_PREFIX}{doc_id}",
                f"{_CACHE_PREFIX_ANALYSIS}{doc_id}",
            )
            logger.debug("Cache invalidated", extra={"document_id": doc_id})
        except Exception as exc:
            logger.warning(
                "Redis invalidate error",
                extra={"document_id": doc_id, "error": str(exc)},
            )


# ══════════════════════════════════════════════════════════════════════════════
# DocumentService
# ══════════════════════════════════════════════════════════════════════════════


class DocumentService:
    """Main application service for all document operations.

    Единственная точка входа для работы с документами.
    Используется LangGraph-агентами, FastAPI-роутерами и любыми
    другими компонентами системы.

    Паттерн использования в FastAPI::

        # В роутере:
        @router.get("/documents/{doc_id}/analysis")
        async def analyze(
            doc_id: str,
            token: str = Depends(get_current_token),
            redis: aioredis.Redis = Depends(get_redis),
        ) -> Dict[str, Any]:
            svc = DocumentService(redis=redis)
            return await svc.get_document_analysis(token, doc_id)

        # В агенте (без FastAPI):
        redis = get_redis()
        svc = DocumentService(redis=redis)
        doc = await svc.get_document(token=jwt, document_id="uuid...")

    Args:
        redis: Shared aioredis.Redis client (из Depends(get_redis)).
        config: Optional DocumentServiceConfig. Defaults применяются из settings.
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

    # ──────────────────────────────────────────────────────────────────────────
    # READ
    # ──────────────────────────────────────────────────────────────────────────

    async def get_document(
        self,
        token: str,
        document_id: str,
        force_refresh: bool = False,
    ) -> DocumentDto:
        """Fetch, enrich and return a DocumentDto.

        Pipeline:
            Redis cache → GET /api/document/{id}?includes=FULL
            → DocumentEnricher.enrich() → Redis set → DocumentDto.model_validate()

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            force_refresh: Bypass cache and always hit the API.

        Returns:
            Parsed and enriched DocumentDto.

        Raises:
            DocumentNotFoundError: API returned empty response.
            DocumentServiceError: On unexpected failure.
        """
        raw = await self._fetch_raw(token, document_id, force_refresh)
        return DocumentDto.model_validate(raw)

    async def get_document_analysis(
        self,
        token: str,
        document_id: str,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch document and run NLP analysis via EDMSNaturalLanguageService.

        NLP-анализ кэшируется отдельным ключом (edms:doc_analysis:{id}).
        Можно инвалидировать анализ отдельно от документа или наоборот.

        Возвращаемый dict содержит секции:
            базовая_информация, регистрация, участники, жизненный_цикл,
            контроль, задачи, связи_и_вложения
            + опциональные: адресаты, корреспондент_орг, ознакомления,
              предвыбранные_дела, договор / обращение / совещание / повестка

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            force_refresh: Recalculate analysis, ignore cache.

        Returns:
            NLP analysis dict from EDMSNaturalLanguageService.process_document().

        Raises:
            DocumentNotFoundError: Document not found in API.
        """
        if not force_refresh:
            cached = await self._cache.get_analysis(document_id)
            if cached is not None:
                return cached

        # NLP-сервис принимает DocumentDto — конвертируем raw
        raw = await self._fetch_raw(token, document_id, force_refresh)
        doc = DocumentDto.model_validate(raw)
        analysis = self._nlp.process_document(doc)

        await self._cache.set_analysis(document_id, analysis)

        logger.info(
            "Document NLP analysis complete",
            extra={
                "document_id": document_id,
                "sections": list(analysis.keys()),
            },
        )
        return analysis

    async def get_document_history(
        self,
        token: str,
        document_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch document movement history (v2 format).

        История v2 включает этапы БП, исполнителей и результаты каждого шага.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            List of history record dicts. Empty list if unavailable.
        """
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            result = await client.get_document_history_v2(
                token=token, document_id=document_id
            )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("content") or result.get("items") or []
        return []

    async def get_document_versions(
        self,
        token: str,
        document_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch document version list.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            List of DocumentVersionDto dicts (camelCase keys).
        """
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            result = await client.get_document_versions(
                token=token, document_id=document_id
            )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("content") or []
        return []

    async def get_document_stats(self, token: str) -> DocumentStats:
        """Fetch document statistics for the current user (3 calls in parallel).

        Args:
            token: JWT bearer token.

        Returns:
            DocumentStats with executor / control / author dicts.
        """
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

    # ──────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ──────────────────────────────────────────────────────────────────────────

    async def search_documents(
        self,
        token: str,
        doc_filter: dict[str, Any] | None = None,
        page: int = 0,
        size: int | None = None,
        sort: str | None = None,
        includes: list[str] | None = None,
    ) -> DocumentSearchResult:
        """Search documents with full filter and pagination.

        Args:
            token: JWT bearer token.
            doc_filter: DocumentFilter fields as flat dict.
                        Поддерживаемые ключи (camelCase):
                        status, docCategoryConstant, regNumber, shortSummary,
                        authorId, responsibleExecutorId, regDateFrom, regDateTo,
                        controlFlag, journalId, documentTypeId и др.
                        None → без фильтра.
            page: Zero-based page index.
            size: Page size. None → config.search_page_size.
            sort: Sort expression "field,direction", e.g. "regDate,desc".
            includes: Custom includes list. None → SEARCH_DOC_INCLUDES.

        Returns:
            DocumentSearchResult с items (List[DocumentDto]) и метаданными.
        """
        effective_size = size or self._config.search_page_size
        pageable: dict[str, Any] = {"page": page, "size": effective_size}
        if sort:
            pageable["sort"] = sort

        effective_includes = includes if includes is not None else SEARCH_DOC_INCLUDES

        params: dict[str, Any] = {
            **(doc_filter or {}),
            **pageable,
            "includes": effective_includes,
        }

        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            raw_page = await client._make_request(
                "GET", "api/document", token=token, params=params
            )

        if not isinstance(raw_page, dict):
            logger.warning(
                "search_documents: unexpected response type",
                extra={"type": type(raw_page).__name__},
            )
            return DocumentSearchResult(page_size=effective_size)

        content: list[dict[str, Any]] = raw_page.get("content") or []
        items = [DocumentDto.model_validate(d) for d in content]

        logger.debug(
            "Document search complete",
            extra={
                "page": page,
                "size": effective_size,
                "total_elements": raw_page.get("totalElements"),
                "filter_keys": list((doc_filter or {}).keys()),
            },
        )

        return DocumentSearchResult(
            items=items,
            total_elements=raw_page.get("totalElements") or 0,
            total_pages=raw_page.get("totalPages") or 0,
            current_page=raw_page.get("number") or page,
            page_size=effective_size,
        )

    async def search_for_agent(
        self,
        token: str,
        status: str | None = None,
        category: str | None = None,
        reg_number: str | None = None,
        short_summary: str | None = None,
        author_id: str | None = None,
        responsible_executor_id: str | None = None,
        reg_date_from: str | None = None,
        reg_date_to: str | None = None,
        on_control: bool | None = None,
        page: int = 0,
        size: int = 10,
        sort: str = "regDate,desc",
    ) -> DocumentSearchResult:
        """Simplified search interface for LangGraph agent @tool decorators.

        Принимает плоские именованные параметры вместо сырого doc_filter.
        Избавляет код инструментов агента от ручной сборки фильтра.

        Args:
            token: JWT bearer token.
            status: DocumentStatus name, e.g. "REGISTERED", "EXECUTION".
            category: Category name, e.g. "INCOMING", "CONTRACT", "APPEAL".
            reg_number: Registration number substring match.
            short_summary: shortSummary substring search.
            author_id: Author employee UUID string.
            responsible_executor_id: Responsible executor UUID string.
            reg_date_from: Filter start date "YYYY-MM-DD".
            reg_date_to: Filter end date "YYYY-MM-DD".
            on_control: Filter by control flag (controlFlag=true/false).
            page: Zero-based page index.
            size: Page size (recommended max: 50).
            sort: Sort expression. Default: "regDate,desc".

        Returns:
            DocumentSearchResult.
        """
        doc_filter: dict[str, Any] = {}
        if status:
            doc_filter["status"] = status
        if category:
            doc_filter["docCategoryConstant"] = category
        if reg_number:
            doc_filter["regNumber"] = reg_number
        if short_summary:
            doc_filter["shortSummary"] = short_summary
        if author_id:
            doc_filter["authorId"] = author_id
        if responsible_executor_id:
            doc_filter["responsibleExecutorId"] = responsible_executor_id
        if reg_date_from:
            doc_filter["regDateFrom"] = reg_date_from
        if reg_date_to:
            doc_filter["regDateTo"] = reg_date_to
        if on_control is not None:
            doc_filter["controlFlag"] = on_control

        return await self.search_documents(
            token=token,
            doc_filter=doc_filter or None,
            page=page,
            size=size,
            sort=sort,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────────────────────────

    async def start_document(self, token: str, document_id: str) -> bool:
        """Start document routing process.

        После успеха инвалидирует кэш документа.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            True on success.

        Raises:
            DocumentOperationError: Java API returned error or timeout.
        """
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            success = await client.start_document(token=token, document_id=document_id)

        if not success:
            raise DocumentOperationError(
                f"Не удалось запустить документ {document_id}",
                document_id=document_id,
            )

        await self._cache.invalidate(document_id)
        logger.info("Document started", extra={"document_id": document_id})
        return True

    async def cancel_document(
        self,
        token: str,
        document_id: str,
        comment: str | None = None,
    ) -> bool:
        """Annul (cancel) a document.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            comment: Cancellation reason for audit trail.

        Returns:
            True on success.

        Raises:
            DocumentOperationError: Java API returned error.
        """
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            success = await client.cancel_document(
                token=token, document_id=document_id, comment=comment
            )

        if not success:
            raise DocumentOperationError(
                f"Не удалось аннулировать документ {document_id}",
                document_id=document_id,
            )

        await self._cache.invalidate(document_id)
        logger.info(
            "Document cancelled",
            extra={"document_id": document_id, "has_comment": bool(comment)},
        )
        return True

    async def execute_operations(
        self,
        token: str,
        document_id: str,
        operations: list[dict[str, Any]],
    ) -> bool:
        """Execute a batch of operations on a document.

        Поддерживаемые operationType (зависит от профиля Java):
            AGREE, SIGN, REJECT, INTRODUCE, REVIEW, REGISTER,
            DISPATCH, APPROVE, ACCEPT, EXECUTE, CANCEL и др.

        Пример::

            await svc.execute_operations(
                token=jwt,
                document_id="uuid-...",
                operations=[{"operationType": "AGREE", "comment": "Без замечаний"}],
            )

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            operations: List of dicts with "operationType" + optional "comment".

        Returns:
            True on success.

        Raises:
            ValueError: Empty operations list.
            DocumentOperationError: Java API returned error.
        """
        if not operations:
            raise ValueError("operations list cannot be empty")

        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            success = await client.execute_document_operations(
                token=token,
                document_id=document_id,
                operations=operations,
            )

        if not success:
            raise DocumentOperationError(
                f"Не удалось выполнить операции над документом {document_id}",
                document_id=document_id,
            )

        await self._cache.invalidate(document_id)
        logger.info(
            "Document operations executed",
            extra={
                "document_id": document_id,
                "operation_types": [op.get("operationType") for op in operations],
            },
        )
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # CONTROL
    # ──────────────────────────────────────────────────────────────────────────

    async def set_control(
        self,
        token: str,
        document_id: str,
        control_type_id: str,
        date_control_end: str,
        control_employee_id: str | None = None,
    ) -> dict[str, Any]:
        """Put document on control.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            control_type_id: ControlTypeDto UUID string.
            date_control_end: ISO datetime "YYYY-MM-DDTHH:MM:SS".
            control_employee_id: Optional controller employee UUID.

        Returns:
            Created ControlDto as dict (camelCase keys).

        Raises:
            DocumentOperationError: Control was not set.
        """
        payload: dict[str, Any] = {
            "controlTypeId": control_type_id,
            "dateControlEnd": date_control_end,
        }
        if control_employee_id:
            payload["controlEmployeeId"] = control_employee_id

        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            result = await client.set_document_control(
                token=token,
                document_id=document_id,
                control_request=payload,
            )

        if not result:
            raise DocumentOperationError(
                f"Не удалось поставить документ {document_id} на контроль",
                document_id=document_id,
            )

        await self._cache.invalidate(document_id)
        logger.info(
            "Document control set",
            extra={"document_id": document_id, "date_end": date_control_end},
        )
        return result

    async def remove_control(self, token: str, document_id: str) -> bool:
        """Remove control mark from document (снять с контроля).

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            True on success.

        Raises:
            DocumentOperationError: Removal failed.
        """
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            success = await client.remove_document_control(
                token=token, document_id=document_id
            )

        if not success:
            raise DocumentOperationError(
                f"Не удалось снять документ {document_id} с контроля",
                document_id=document_id,
            )

        await self._cache.invalidate(document_id)
        logger.info("Control removed", extra={"document_id": document_id})
        return True

    async def delete_control(self, token: str, document_id: str) -> bool:
        """Delete control record for a document.

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.

        Returns:
            True on success.

        Raises:
            DocumentOperationError: Deletion failed.
        """
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            success = await client.delete_document_control(
                token=token, document_id=document_id
            )

        if not success:
            raise DocumentOperationError(
                f"Не удалось удалить контроль документа {document_id}",
                document_id=document_id,
            )

        await self._cache.invalidate(document_id)
        logger.info("Control deleted", extra={"document_id": document_id})
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _fetch_raw(
        self,
        token: str,
        document_id: str,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Core document fetch pipeline: cache → API → enrich → cache.

        Центральный метод получения сырого DocumentDto dict.
        Вызывается из get_document() и get_document_analysis().

        Args:
            token: JWT bearer token.
            document_id: Document UUID string.
            force_refresh: Bypass cache lookup.

        Returns:
            Enriched raw document dict (camelCase keys).

        Raises:
            DocumentNotFoundError: API returned empty or None.
        """
        # 1. Кэш ──────────────────────────────────────────────────────────────
        if not force_refresh:
            cached = await self._cache.get_doc(document_id)
            if cached is not None:
                return cached

        # 2. Java API: GET /api/document/{id}?includes=... ─────────────────────
        async with DocumentClient(base_url=self._config.edms_base_url) as client:
            raw = await client.get_document_metadata(
                token=token,
                document_id=document_id,
                includes=FULL_DOC_INCLUDES,
            )

        if not raw:
            raise DocumentNotFoundError(
                f"Документ {document_id} не найден",
                document_id=document_id,
            )

        # 3. Enrich — дозапросы вложенных объектов по UUID ────────────────────
        if self._config.enrich_documents:
            raw = await self._enricher.enrich(raw, token=token)

        # 4. Сохраняем обогащённый dict в кэш ─────────────────────────────────
        await self._cache.set_doc(document_id, raw)

        logger.debug(
            "Document fetched and enriched",
            extra={"document_id": document_id},
        )
        return raw
