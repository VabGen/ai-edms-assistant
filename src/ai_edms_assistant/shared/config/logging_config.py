# src/ai_edms_assistant/shared/config/logging_config.py
"""Logging configuration — structlog + stdlib bridge.

- development : colored ConsoleRenderer
- production  : JSON lines (structured, ELK-ready)

Call once at startup (FastAPI lifespan or main entrypoint).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from .settings import settings


def configure_logging() -> None:
    """Configure structlog and bridge stdlib logging through it.

    Args:
        None — reads from settings singleton.

    Side effects:
        - Replaces root logger handlers
        - Mutes noisy third-party loggers (httpx, httpcore, asyncio)

    Example:
        >>> # In FastAPI startup:
        >>> from ai_edms_assistant.shared.config.logging_config import configure_logging
        >>> configure_logging()
        >>> logger.info("app_started")
    """
    level = logging.getLevelName(settings.LOGGING_LEVEL.upper())

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.is_production()
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in ("httpx", "httpcore", "asyncio", "uvicorn.access", "hpack"):
        logging.getLogger(name).setLevel(logging.WARNING)

    structlog.get_logger(__name__).info(
        "logging_configured",
        level=settings.LOGGING_LEVEL,
        environment=settings.ENVIRONMENT,
        renderer="json" if settings.is_production() else "console",
    )
