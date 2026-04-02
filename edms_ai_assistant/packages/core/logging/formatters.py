"""
edms_ai_assistant/packages/core/logging/formatters.py

Форматтеры для структурированного логирования.
Поддерживают консольный (dev) и JSON (production) вывод.

Features:
• Цветной вывод в development
• JSON для ELK/Grafana Loki в production
• Интеграция с OpenTelemetry (trace_id, span_id)
• Фильтрация чувствительных данных (пароли, токены)
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from structlog.types import EventDict, Processor


# ── Константы ───────────────────────────────────────────────────────────
SENSITIVE_FIELDS = {
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "credential",
    "private_key",
}

REDACTED_VALUE = "***REDACTED***"


# ── Процессоры structlog ────────────────────────────────────────────────
def redact_sensitive_fields(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Маскировать чувствительные поля в логах."""
    for key in list(event_dict.keys()):
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in SENSITIVE_FIELDS):
            event_dict[key] = REDACTED_VALUE
    return event_dict


def add_timestamp(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Добавить ISO-формат времени в лог."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_environment_info(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Добавить информацию об окружении."""
    from edms_ai_assistant.packages.core.settings import settings

    event_dict.setdefault("environment", settings.ENVIRONMENT)
    event_dict.setdefault("service", "edms_ai_assistant")
    event_dict.setdefault("version", "1.0.0")

    return event_dict


def format_exception(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Форматировать исключения в читаемый вид."""
    exc_info = event_dict.pop("exc_info", None)
    if exc_info:
        import traceback

        if isinstance(exc_info, BaseException):
            event_dict["exception"] = {
                "type": exc_info.__class__.__name__,
                "message": str(exc_info),
                "traceback": traceback.format_exception(
                    type(exc_info), exc_info, exc_info.__traceback__
                ),
            }
        elif isinstance(exc_info, tuple):
            event_dict["exception"] = {
                "type": exc_info[0].__name__ if exc_info[0] else "Unknown",
                "message": str(exc_info[1]) if exc_info[1] else "Unknown error",
                "traceback": traceback.format_exception(*exc_info),
            }

    return event_dict


def rename_event_key(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Переименовать 'event' в 'message' для совместимости с ELK."""
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


# ── Форматтеры для стандартного logging ─────────────────────────────────
class ConsoleFormatter(logging.Formatter):
    """Цветной форматтер для консольного вывода (development)."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if self.use_colors and levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        if hasattr(record, "trace_id") and record.trace_id:
            record.trace_id_formatted = f"[{record.trace_id[:8]}]"
        else:
            record.trace_id_formatted = ""

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON-форматтер для production (ELK/Grafana совместимость)."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
    ):
        super().__init__(fmt, datefmt)
        self._skip_fields = {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "levelname", "levelno", "lineno",
            "module", "msecs", "message", "msg", "name", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "thread", "threadName",
        }

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_data["span_id"] = record.span_id

        if record.exc_info:
            import traceback
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
                "traceback": "".join(traceback.format_exception(*record.exc_info)),
            }

        for key, value in record.__dict__.items():
            if key not in self._skip_fields and not key.startswith("_"):
                if any(sensitive in key.lower() for sensitive in SENSITIVE_FIELDS):
                    log_data[key] = REDACTED_VALUE
                else:
                    try:
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)

        return json.dumps(log_data, ensure_ascii=False, sort_keys=True)


# ── Фабрика форматтеров ─────────────────────────────────────────────────
def get_formatter(
    formatter_type: str = "console",
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    **kwargs: Any,
) -> logging.Formatter:
    """Получить форматтер по типу."""
    if formatter_type == "console":
        return ConsoleFormatter(
            fmt=fmt or "%(asctime)s %(trace_id_formatted)s %(levelname)s %(name)s: %(message)s",
            datefmt=datefmt or "%Y-%m-%d %H:%M:%S",
            use_colors=kwargs.get("use_colors", True),
        )
    elif formatter_type == "json":
        return JSONFormatter(fmt=fmt, datefmt=datefmt)
    elif formatter_type == "simple":
        return logging.Formatter(
            fmt=fmt or "%(levelname)s %(name)s: %(message)s",
            datefmt=datefmt,
        )
    else:
        raise ValueError(f"Unknown formatter type: {formatter_type}")


# ── Процессоры для structlog pipeline ───────────────────────────────────
def get_structlog_processors(
    environment: str = "development",
    include_trace_id: bool = True,
) -> List[Processor]:
    """Получить список процессоров для structlog.configure()."""
    processors: List[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        redact_sensitive_fields,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        format_exception,
        structlog.processors.UnicodeDecoder(),
    ]

    if include_trace_id:
        processors.append(structlog.processors.inject_trace_id)

    if environment == "development":
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
                pad_event=40,
            )
        )
    else:
        processors.extend([
            add_timestamp,
            add_environment_info,
            rename_event_key,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(sort_keys=True),
        ])

    return processors


# ── Экспорт ─────────────────────────────────────────────────────────────
__all__ = [
    "SENSITIVE_FIELDS",
    "REDACTED_VALUE",
    "redact_sensitive_fields",
    "add_timestamp",
    "add_environment_info",
    "format_exception",
    "rename_event_key",
    "ConsoleFormatter",
    "JSONFormatter",
    "get_formatter",
    "get_structlog_processors",
]