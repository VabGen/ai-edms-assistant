# packages/core/logging/__init__.py
"""
Централизованная настройка логирования для всего проекта.

Поддерживает два режима:
    development → цветной console-вывод (structlog или стандартный logging)
    production  → JSON-формат для ELK/Grafana Loki

Использование (вызывать ОДИН РАЗ при старте приложения):
    from edms_ai_assistant.packages.core.logging import configure_logging
    configure_logging("production", "INFO")

Получение логгера в модуле:
    from edms_ai_assistant.packages.core.logging import get_logger
    logger = get_logger(__name__)
"""
from __future__ import annotations

import logging
import sys
from typing import Literal

Environment = Literal["development", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def configure_logging(
    environment: Environment = "development",
    log_level: LogLevel = "INFO",
    enable_json: bool | None = None,
) -> None:
    """
    Настраивает логирование для текущего сервиса.

    Args:
        environment: Среда запуска (влияет на формат вывода).
        log_level:   Минимальный уровень логирования.
        enable_json: Принудительно включить JSON (None = авто по environment).
    """
    use_json = enable_json if enable_json is not None else (environment != "development")
    level = getattr(logging, log_level, logging.INFO)

    try:
        import structlog

        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        if use_json:
            processors.extend([
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.JSONRenderer(sort_keys=True),
            ])
        else:
            processors.append(
                structlog.dev.ConsoleRenderer(colors=True, pad_event=40)
            )

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(sys.stdout),
            cache_logger_on_first_use=True,
        )
    except ImportError:
        pass

    fmt = (
        '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
        if use_json
        else "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)

    for noisy in ("httpx", "httpcore", "asyncpg", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Возвращает логгер. Использует structlog если доступен, иначе стандартный logging.

    Args:
        name: Имя логгера (обычно __name__ вызывающего модуля).
    """
    try:
        import structlog
        return structlog.get_logger(name)  # type: ignore[return-value]
    except ImportError:
        return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger", "Environment", "LogLevel"]
