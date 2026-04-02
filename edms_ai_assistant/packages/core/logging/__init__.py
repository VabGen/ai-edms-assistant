"""
edms_ai_assistant/packages/core/logging/__init__.py

Централизованная настройка логирования для всего проекта.
Использует structlog для структурированных логов с контекстом.

Features:
• Контекстные логи (user_id, thread_id, trace_id автоматически)
• Цветной вывод в dev / JSON в production
• Интеграция с OpenTelemetry для distributed tracing
• Фильтрация шумных библиотек
"""
from __future__ import annotations

import sys
import logging
from typing import Literal, Optional, Any

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
    get_contextvars,
)
from structlog.processors import CallsiteParameter, CallsiteParameterAdder

# Типы для аннотаций
Environment = Literal["development", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def configure_logging(
    environment: Environment = "development",
    log_level: LogLevel = "INFO",
    enable_json: Optional[bool] = None,
    include_trace_id: bool = True,
) -> None:
    """
    Настроить structlog для всего приложения.

    Вызывать ОДИН РАЗ при старте приложения, до создания FastAPI/Graph.

    Args:
        environment: Окружение запуска (development/staging/production)
        log_level: Минимальный уровень логирования
        enable_json: Принудительно включить JSON-вывод (переопределяет environment)
        include_trace_id: Добавлять trace_id из OpenTelemetry в логи
    """

    is_dev = environment == "development"
    use_json = enable_json if enable_json is not None else not is_dev

    # ── Базовые процессоры (работают всегда) ──────────────────────────────
    processors: list = [
        # Merge contextvars (user_id, thread_id и т.д. из asyncio context)
        merge_contextvars,

        # Добавлять уровень лога как поле
        structlog.processors.add_log_level,

        # Добавлять информацию о вызове (файл, строка, функция) — только в dev
        *(
            [CallsiteParameterAdder([
                CallsiteParameter.FILENAME,
                CallsiteParameter.FUNC_NAME,
                CallsiteParameter.LINENO,
            ])] if is_dev else []
        ),

        # Рендеринг stack trace для исключений
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,

        # Декодирование байтовых строк в unicode
        structlog.processors.UnicodeDecoder(),

        # Инжекция trace_id из OpenTelemetry (если включено)
        *(
            [structlog.processors.inject_trace_id] if include_trace_id else []
        ),
    ]

    # ── Формат вывода ────────────────────────────────────────────────────
    if is_dev and not use_json:
        # Development: цветной, человекочитаемый вывод в консоль
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
                pad_event=40,
                force_colors=True,
            )
        )
    else:
        # Production/Staging: JSON для сбора в ELK/Grafana Loki
        processors.extend([
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(sort_keys=True),
        ])

    # ── Конфигурация structlog ───────────────────────────────────────────
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # ── Настройка стандартного logging для сторонних библиотек ───────────
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(message)s" if (is_dev and not use_json) else "%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )

    # ── Подавление шумных логгеров ───────────────────────────────────────
    noisy_loggers = [
        "httpx", "httpcore", "asyncpg", "urllib3", "fakeredis",
        "sqlalchemy.pool", "sqlalchemy.engine", "apscheduler",
        "prometheus_client", "opentelemetry",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # ── Глобальный логгер для быстрого импорта ───────────────────────────
    global logger
    logger = structlog.get_logger("edms.core")

    logger.info(
        "Logging configured",
        environment=environment,
        log_level=log_level,
        output_format="json" if use_json else "console",
        trace_id_injection=include_trace_id,
    )


# ── Глобальный логгер (экспортируется) ───────────────────────────────────
logger: structlog.BoundLogger = structlog.get_logger("edms")


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Получить логгер с указанным именем.

    Args:
        name: Имя логгера (обычно __name__ модуля)

    Returns:
        BoundLogger с предустановленными процессорами
    """
    return structlog.get_logger(name)


# ── Convenience функции для работы с контекстом ─────────────────────────
def bind(**kwargs: Any) -> None:
    """
    Добавить ключи-значения в контекст текущего лога.

    Пример:
        bind(user_id="user_123", thread_id="abc")
    """
    bind_contextvars(**kwargs)


def unbind(*keys: str) -> None:
    """Удалить ключи из контекста."""
    for key in keys:
        structlog.contextvars.unbind_contextvars(key)


def get_context() -> dict[str, Any]:
    """Получить текущий контекст логов."""
    return get_contextvars()


def clear_context() -> None:
    """Очистить весь контекст (вызывать в конце запроса/задачи)."""
    clear_contextvars()


# ── Экспорт ─────────────────────────────────────────────────────────────
__all__ = [
    "configure_logging",
    "get_logger",
    "logger",
    "bind",
    "unbind",
    "get_context",
    "clear_context",
    "Environment",
    "LogLevel",
]