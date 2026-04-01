# edms_ai_assistant\clients\base_client.py
import json
import logging
from typing import Any

import httpx

from edms_ai_assistant.config import settings
from edms_ai_assistant.utils.api_utils import handle_api_error, prepare_auth_headers
from edms_ai_assistant.utils.retry_utils import async_retry

logger = logging.getLogger(__name__)


class EdmsBaseClient:
    """Абстрактный базовый класс для всех клиентов EDMS API."""


class EdmsHttpClient(EdmsBaseClient):
    """Универсальный асинхронный клиент для API EDMS, реализующий HTTP-запросы."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        resolved_base_url = base_url or settings.CHANCELLOR_NEXT_BASE_URL
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout or settings.EDMS_TIMEOUT
        self._client = None

    async def __aenter__(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Закрывает HTTP-клиент."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Получает или создает асинхронный клиент."""
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
        **kwargs,
    ) -> dict[str, Any] | list[dict[str, Any]] | bytes | None:
        """
        Выполняет HTTP-запрос с авторизацией, обработкой ошибок и повторными попытками.
        Возвращает десериализованный JSON или сырые байты.
        """
        response = await self._make_request_response_object(
            method, endpoint, token, long_timeout, **kwargs
        )

        if response.status_code == 204 or not response.content:
            return {} if is_json_response else None

        if not is_json_response:
            return response.content

        try:
            return response.json()
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response from {method} {response.url}")
            return {} if is_json_response else None

    @async_retry(
        max_attempts=3,
        delay=1.0,
        backoff=2.0,
        exceptions=(httpx.RequestError, httpx.HTTPStatusError),
    )
    async def _make_request_response_object(
        self,
        method: str,
        endpoint: str,
        token: str,
        long_timeout: bool = False,
        **kwargs,
    ) -> httpx.Response:
        """
        Выполняет HTTP-запрос и возвращает объект httpx.Response.
        Используется, когда необходимо получить заголовки ответа (например, для имени файла).
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = prepare_auth_headers(token)

        current_timeout = kwargs.pop("timeout", self.timeout)
        if long_timeout:
            current_timeout = self.timeout + 30

        current_headers = kwargs.get("headers", {})
        current_headers.update(headers)
        kwargs["headers"] = current_headers
        kwargs["timeout"] = current_timeout

        try:
            client = await self._get_client()
            response = await client.request(method, url, **kwargs)
            await handle_api_error(response, f"{method} {url}")
            return response
        except httpx.HTTPStatusError:
            raise
        except httpx.RequestError:
            raise
