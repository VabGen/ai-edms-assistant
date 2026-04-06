# mcp-server/clients/base_client.py
"""
Базовый асинхронный HTTP-клиент для обращений к Java EDMS API.

Реализует:
    - Retry с экспоненциальным backoff
    - Стандартные заголовки авторизации
    - Обработку ошибок 4xx/5xx
    - Context manager (async with)
    - Логирование запросов

Перенесён из edms_ai_assistant/clients/base_client.py
и адаптирован для работы в mcp-server без зависимости от FastAPI/LangChain.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ..config import settings
from ..utils.retry_utils import async_retry

logger = logging.getLogger(__name__)

_NO_RETRY_STATUS_CODES: frozenset[int] = frozenset(
    {
        400,  # Bad Request — ошибка данных
        401,  # Unauthorized — токен недействителен
        403,  # Forbidden — нет прав
        404,  # Not Found — ресурс не существует
        405,  # Method Not Allowed
        409,  # Conflict
        410,  # Gone
        422,  # Unprocessable Entity — ошибка валидации
    }
)


class EdmsBaseClient:
    """Абстрактный базовый класс для всех клиентов EDMS API."""


class EdmsHttpClient(EdmsBaseClient):
    """
    Универсальный асинхронный HTTP-клиент для EDMS API.

    Все методы async/await.
    Используется как base class для доменных клиентов.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        resolved_base_url = base_url or settings.EDMS_BASE_URL
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout or settings.EDMS_TIMEOUT
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "EdmsHttpClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Закрывает HTTP-клиент."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Возвращает или создаёт async HTTP-клиент."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    @async_retry(
        max_attempts=3,
        delay=1.0,
        backoff=2.0,
        exceptions=(httpx.RequestError, httpx.HTTPStatusError),
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        token: str,
        is_json_response: bool = True,
        long_timeout: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]] | bytes | None:
        """
        Выполняет HTTP-запрос с авторизацией, обработкой ошибок и retry.

        Args:
            method: HTTP-метод (GET, POST, PUT, DELETE, PATCH).
            endpoint: Путь относительно base_url.
            token: JWT-токен авторизации.
            is_json_response: True → десериализовать JSON; False → вернуть bytes/None.
            long_timeout: True → увеличить таймаут на 30s.
            **kwargs: Дополнительные параметры httpx.request().

        Returns:
            Десериализованный ответ или None для 204.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        current_timeout = kwargs.pop("timeout", self.timeout)
        if long_timeout:
            current_timeout = self.timeout + 30

        current_headers = kwargs.get("headers", {})
        current_headers.update(headers)
        kwargs["headers"] = current_headers
        kwargs["timeout"] = current_timeout

        client = await self._get_client()
        response = await client.request(method, url, **kwargs)

        if response.is_error:
            status = response.status_code
            try:
                error_details = response.json()
            except Exception:
                error_details = {"text": response.text[:200]}

            logger.error(
                "API Error [%d] for %s %s. Details: %s",
                status, method, endpoint, error_details,
            )

            if status not in _NO_RETRY_STATUS_CODES:
                response.raise_for_status()
            else:
                logger.debug("Non-retriable HTTP %d for %s", status, endpoint)
                response.raise_for_status()

        if response.status_code == 204 or not response.content:
            return {} if is_json_response else None

        if not is_json_response:
            return response.content

        try:
            return response.json()
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from %s %s", method, endpoint)
            return {} if is_json_response else None