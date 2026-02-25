# src/ai_edms_assistant/infrastructure/edms_api/http_client.py
"""Base HTTP client for EDMS API with retry logic and structured error handling.

Production-ready implementation with:
- Automatic retries with exponential backoff
- Structured logging (structlog)
- Settings integration
- Response object access for headers
- Reusable utilities from shared.utils
"""

from __future__ import annotations

import json
import logging
from abc import ABC
from typing import Any

import httpx

from ...shared.config.settings import settings
from ...shared.utils.api_client import handle_api_error, prepare_auth_headers
from ...shared.utils.retry import async_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class EdmsBaseClient(ABC):
    """Abstract base class for all EDMS API clients.

    Provides common interface for document, employee, task, and other
    specialized clients.
    """

    pass


class EdmsHttpClient(EdmsBaseClient):
    """Universal async HTTP client for EDMS API.

    Features:
        - Automatic retries with exponential backoff (3 attempts)
        - Settings-based configuration (CHANCELLOR_NEXT_BASE_URL, EDMS_TIMEOUT)
        - Structured error handling and logging
        - Context manager support for connection pooling
        - Both parsed JSON and raw response access
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        """Initialize HTTP client with settings integration.

        Args:
            base_url: Override for EDMS backend URL.
                Defaults to settings.CHANCELLOR_NEXT_BASE_URL.
            timeout: Override for request timeout in seconds.
                Defaults to settings.EDMS_TIMEOUT.
        """
        resolved_base_url = base_url or settings.CHANCELLOR_NEXT_BASE_URL
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout or settings.EDMS_TIMEOUT
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "EdmsHttpClient":
        """Async context manager entry."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client (lazy initialization).

        Returns:
            Active httpx.AsyncClient instance.
        """
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
        """Execute HTTP request with automatic retries and error handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH).
            endpoint: API endpoint path (without leading slash).
            token: JWT bearer token for authorization.
            is_json_response: Whether to parse response as JSON.
                Set to False for binary downloads.
            long_timeout: Use extended timeout for large file operations.
            **kwargs: Additional arguments passed to httpx.request
                (params, json, data, headers, etc.).

        Returns:
            Parsed JSON dict/list, raw bytes, or None for 204 responses.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses after retries.
            httpx.RequestError: On network/connection errors after retries.
        """
        response = await self._make_request_response_object(
            method, endpoint, token, long_timeout, **kwargs
        )

        # Handle 204 No Content or empty responses
        if response.status_code == 204 or not response.content:
            return {} if is_json_response else None

        # Return raw bytes for binary content
        if not is_json_response:
            return response.content

        # Parse JSON response
        try:
            return response.json()
        except json.JSONDecodeError:
            logger.error(
                f"Failed to decode JSON response from {method} {response.url}",
                extra={"content": response.text[:500]},
            )
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
        """Execute HTTP request and return the raw Response object.

        Use this method when you need access to response headers
        (e.g. for Content-Disposition filename extraction).

        Args:
            method: HTTP method.
            endpoint: API endpoint path.
            token: JWT bearer token.
            long_timeout: Use extended timeout for large operations.
            **kwargs: Additional httpx.request arguments.

        Returns:
            httpx.Response object with headers and body.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses after retries.
            httpx.RequestError: On network/connection errors after retries.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = prepare_auth_headers(token)

        # Apply long timeout for large file operations
        current_timeout = kwargs.pop("timeout", self.timeout)
        if long_timeout:
            current_timeout = self.timeout + 30

        # Merge authorization headers with any user-provided headers
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
            # Re-raise after logging in handle_api_error
            raise
        except httpx.RequestError:
            # Re-raise network errors for retry decorator
            raise
