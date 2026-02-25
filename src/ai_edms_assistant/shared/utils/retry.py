# src/ai_edms_assistant/shared/utils/retry.py
"""
Async retry decorator with exponential backoff.

Migrated and cleaned up from edms_ai_assistant/utils/retry_utils.py.
Removed the broken retry_utilss.py duplicate.
"""

from __future__ import annotations

import asyncio
import structlog
from functools import wraps
from typing import Any, Awaitable, Callable, Tuple, Type

logger = structlog.get_logger(__name__)

AsyncCallable = Callable[..., Awaitable[Any]]


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[AsyncCallable], AsyncCallable]:
    """
    Decorator: retry an async function with exponential backoff.

    Args:
        max_attempts: Total number of attempts (including the first).
        delay:        Initial delay between retries in seconds.
        backoff:      Multiplier applied to delay after each failure.
        exceptions:   Exception types that trigger a retry.

    Returns:
        Decorated coroutine function.

    Example::

        @async_retry(max_attempts=3, delay=1.0, exceptions=(httpx.RequestError,))
        async def fetch_data() -> dict:
            ...
    """

    def decorator(func: AsyncCallable) -> AsyncCallable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exc: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    is_last = attempt == max_attempts - 1

                    if is_last:
                        logger.error(
                            "async_retry_exhausted",
                            func=func.__name__,
                            attempts=max_attempts,
                            error=str(exc),
                        )
                        raise

                    logger.warning(
                        "async_retry_attempt",
                        func=func.__name__,
                        attempt=attempt + 1,
                        max=max_attempts,
                        retry_in=current_delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            # Unreachable but satisfies type checker
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"Retry exhausted for {func.__name__}")

        return wrapper

    return decorator
