# mcp-server/clients/base_client.py
"""
Базовый асинхронный HTTP-клиент для обращений к Java EDMS API.

Реализует:
    - Retry с экспоненциальным backoff (tenacity)
    - Стандартные заголовки авторизации
    - Обработку ошибок 4xx/5xx
    - Context manager (async with)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_NO_RETRY_STATUS = frozenset({400, 401, 403, 404, 405, 409, 410, 422})


class EdmsHttpClient:
    """
    Базовый async HTTP-клиент для Java EDMS API.

    Конфигурация через env-переменные:
        EDMS_BASE_URL  — базовый URL API
        EDMS_TIMEOUT   — таймаут запросов (секунды)
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (
            (base_url or os.getenv("EDMS_BASE_URL", "http://localhost:8098")).rstrip("/")
        )
        self.timeout = timeout or int(os.getenv("EDMS_TIMEOUT", "120"))
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "EdmsHttpClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _auth_headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        token: str,
        is_json_response: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Выполняет HTTP-запрос с retry (3 попытки, exp backoff).

        Args:
            method:           HTTP-метод (GET, POST, PUT, DELETE, PATCH).
            endpoint:         Путь относительно base_url.
            token:            JWT-токен авторизации.
            is_json_response: True → десериализовать JSON; False → вернуть bytes/None.
            **kwargs:         Дополнительные параметры httpx.request().

        Returns:
            Десериализованный ответ или None для 204.

        Raises:
            httpx.HTTPStatusError: На ошибках 4xx/5xx после всех попыток.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._auth_headers(token)
        current_headers = kwargs.pop("headers", {})
        current_headers.update(headers)

        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)

        attempt = 0
        async for att in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
            reraise=True,
        ):
            with att:
                attempt += 1
                response = await client.request(
                    method, url, headers=current_headers, **kwargs
                )

                # 4xx — не ретраить
                if response.status_code in _NO_RETRY_STATUS:
                    logger.debug(
                        "HTTP %d for %s %s", response.status_code, method, endpoint
                    )
                    response.raise_for_status()

                response.raise_for_status()

        if response.status_code == 204 or not response.content:
            return {} if is_json_response else None

        if not is_json_response:
            return response.content

        try:
            return response.json()
        except json.JSONDecodeError:
            logger.error("JSON decode error: %s %s", method, endpoint)
            return {} if is_json_response else None
