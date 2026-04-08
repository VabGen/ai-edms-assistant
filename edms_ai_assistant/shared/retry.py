# edms_ai_assistant/utils/retry_utils.py

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_NO_RETRY_STATUS_CODES: frozenset[int] = frozenset(
    {
        400,  # Bad Request — ошибка данных, повтор не поможет
        401,  # Unauthorized — токен недействителен
        403,  # Forbidden — нет прав
        404,  # Not Found — ресурс не существует
        405,  # Method Not Allowed
        409,  # Conflict
        410,  # Gone
        422,  # Unprocessable Entity — ошибка валидации
    }
)


def _should_retry(exc: Exception) -> bool:
    """Determine whether a request exception warrants a retry attempt.

    Business errors (4xx, except 408/429) must never be retried —
    the result will be identical. Only transient network errors and
    server-side 5xx failures are worth retrying.

    Args:
        exc: The caught exception.

    Returns:
        True if the caller should retry, False to raise immediately.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in _NO_RETRY_STATUS_CODES:
            return False
        return True
    if isinstance(exc, httpx.RequestError):
        return True
    return False


_EXPECTED_BUSINESS_STATUS_CODES: frozenset[int] = frozenset({400, 404, 422})


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Async retry decorator with exponential backoff.

    Skips retry for non-retriable HTTP errors (401, 403, 404, 422, etc.)
    to avoid wasting time and producing misleading log noise.

    For "business" 4xx responses (404 not found, 400 bad request, 422 validation)
    uses DEBUG level instead of ERROR — these are expected API outcomes, not faults.

    Args:
        max_attempts: Total attempts including the first call.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier applied to delay after each failed attempt.
        exceptions: Exception types that trigger the retry logic.

    Returns:
        Decorated async callable.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except exceptions as exc:
                    is_last = attempt == max_attempts - 1

                    if not _should_retry(exc):
                        status_code = (
                            exc.response.status_code
                            if isinstance(exc, httpx.HTTPStatusError)
                            else None
                        )
                        if status_code in _EXPECTED_BUSINESS_STATUS_CODES:
                            logger.debug(
                                "HTTP %s in %s: %s",
                                status_code,
                                func.__name__,
                                exc,
                            )
                        else:
                            logger.error(
                                "Non-retriable error in %s (attempt %d/%d): %s: %s",
                                func.__name__,
                                attempt + 1,
                                max_attempts,
                                type(exc).__name__,
                                exc,
                            )
                        raise

                    if is_last:
                        logger.error(
                            "Fatal error after %d attempts calling %s: %s: %s",
                            max_attempts,
                            func.__name__,
                            type(exc).__name__,
                            exc,
                        )
                        raise

                    logger.warning(
                        "Attempt %d/%d failed for %s. Retrying in %.2fs. Error: %s",
                        attempt + 1,
                        max_attempts,
                        func.__name__,
                        current_delay,
                        exc,
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            return None

        return wrapper

    return decorator
