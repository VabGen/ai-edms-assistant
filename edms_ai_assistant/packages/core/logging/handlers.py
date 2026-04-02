"""
edms_ai_assistant/packages/core/logging/handlers.py

Обработчики (handlers) для логирования.
Поддерживают консоль, файл, HTTP и асинхронную отправку.

Features:
• Асинхронная запись (не блокирует приложение)
• RotatingFileHandler с архивацией
• HTTPHandler для отправки в ELK/Splunk
• Фильтрация по уровню и модулю
• Buffering для пакетной отправки
"""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import queue
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import httpx

from .formatters import get_formatter, JSONFormatter


# ── Типы ────────────────────────────────────────────────────────────────
LogLevel = Union[int, str]
PathLike = Union[str, Path]


# ── Асинхронный Queue Handler ───────────────────────────────────────────
class AsyncQueueHandler(logging.Handler):
    """
    Асинхронный обработчик логов через Queue.

    Преимущества:
    • Не блокирует основное приложение
    • Пакетная обработка логов
    • Graceful shutdown с очисткой очереди

    Использование:
        handler = AsyncQueueHandler(max_size=1000)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    """

    def __init__(
        self,
        queue: Optional[queue.Queue] = None,
        max_size: int = 1000,
        flush_interval: float = 1.0,
    ):
        super().__init__()
        self.queue = queue or queue.Queue(maxsize=max_size)
        self.max_size = max_size
        self.flush_interval = flush_interval
        self._shutdown = False
        self._worker_task: Optional[asyncio.Task] = None
        self._handlers: List[logging.Handler] = []

    def add_handler(self, handler: logging.Handler) -> None:
        """Добавить целевой обработчик (файл, HTTP, etc)."""
        self._handlers.append(handler)

    def emit(self, record: logging.LogRecord) -> None:
        """Поместить лог в очередь."""
        if self._shutdown:
            return

        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # Очередь переполнена — логируем предупреждение
            sys.stderr.write("Log queue full, dropping message\n")

    async def _worker(self) -> None:
        """Фоновая задача обработки очереди."""
        while not self._shutdown or not self.queue.empty():
            try:
                # Получаем с таймаутом для проверки shutdown
                record = self.queue.get(timeout=self.flush_interval)

                # Отправляем всем целевым обработчикам
                for handler in self._handlers:
                    try:
                        handler.emit(record)
                    except Exception:
                        self.handleError(record)

                self.queue.task_done()

            except queue.Empty:
                continue
            except Exception:
                self.handleError(None)

    def start(self) -> None:
        """Запустить фоновую задачу."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Остановить и очистить очередь."""
        self._shutdown = True

        if self._worker_task:
            # Ждём обработки оставшихся логов
            try:
                await asyncio.wait_for(self.queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

            self._worker_task = None

        # Закрываем целевые обработчики
        for handler in self._handlers:
            handler.close()

    def close(self) -> None:
        """Закрыть обработчик."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.stop())
        except RuntimeError:
            # Нет running loop — синхронная очистка
            for handler in self._handlers:
                handler.close()
        super().close()


# ── Rotating File Handler ───────────────────────────────────────────────
class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Файловый обработчик с ротацией.

    Автоматически создаёт директорию и архивирует старые логи.

    Args:
        filename: Путь к файлу лога
        max_bytes: Максимальный размер файла до ротации
        backup_count: Количество архивных файлов
        encoding: Кодировка файла
        format_type: 'json' или 'console'
    """

    def __init__(
        self,
        filename: PathLike,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        encoding: str = "utf-8",
        format_type: str = "json",
    ):
        # Создаём директорию если не существует
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename=str(filepath),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )

        self.setFormatter(get_formatter(format_type))


# ── Timed Rotating File Handler ─────────────────────────────────────────
class TimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Файловый обработчик с ротацией по времени.

    Ротирует логи ежедневно/еженедельно/ежемесячно.

    Args:
        filename: Путь к файлу лога
        when: Интервал ('midnight', 'D', 'W', 'M')
        interval: Количество интервалов
        backup_count: Количество архивных файлов
        format_type: 'json' или 'console'
    """

    def __init__(
        self,
        filename: PathLike,
        when: str = "midnight",
        interval: int = 1,
        backup_count: int = 7,
        encoding: str = "utf-8",
        format_type: str = "json",
    ):
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename=str(filepath),
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding=encoding,
        )

        self.setFormatter(get_formatter(format_type))


# ── HTTP Handler (для ELK/Splunk) ───────────────────────────────────────
class HTTPHandler(logging.Handler):
    """
    Отправка логов по HTTP (ELK, Splunk, Grafana Loki).

    Поддерживает:
    • Пакетную отправку (buffering)
    • Retry с exponential backoff
    • Асинхронный режим

    Пример для Grafana Loki:
        handler = HTTPHandler(
            endpoint="http://loki:3100/loki/api/v1/push",
            headers={"Content-Type": "application/json"},
            batch_size=100,
        )
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_retries: int = 3,
        timeout: float = 10.0,
        async_mode: bool = True,
    ):
        super().__init__()
        self.endpoint = endpoint
        self.headers = headers or {"Content-Type": "application/json"}
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.async_mode = async_mode

        self._buffer: List[Dict[str, Any]] = []
        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock() if async_mode else None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    async def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    def emit(self, record: logging.LogRecord) -> None:
        """Синхронная отправка лога."""
        if self.async_mode:
            # В асинхронном режиме добавляем в буфер
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._emit_async(record))
            except RuntimeError:
                pass
            return

        # Синхронная отправка
        log_entry = self._format_record(record)
        self._buffer.append(log_entry)

        if len(self._buffer) >= self.batch_size:
            self._flush()

    async def _emit_async(self, record: logging.LogRecord) -> None:
        """Асинхронная отправка лога."""
        if self._lock is None:
            return

        async with self._lock:
            log_entry = self._format_record(record)
            self._buffer.append(log_entry)

            if len(self._buffer) >= self.batch_size:
                await self._flush_async()

    def _format_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Форматировать лог для отправки."""
        return {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

    def _flush(self) -> None:
        """Синхронная отправка буфера."""
        if not self._buffer:
            return

        client = self._get_client()

        for attempt in range(self.max_retries):
            try:
                response = client.post(
                    self.endpoint,
                    headers=self.headers,
                    json={"logs": self._buffer},
                )
                response.raise_for_status()
                self._buffer.clear()
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    sys.stderr.write(f"Failed to send logs: {e}\n")
                else:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff

    async def _flush_async(self) -> None:
        """Асинхронная отправка буфера."""
        if not self._buffer:
            return

        client = await self._get_async_client()

        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    self.endpoint,
                    headers=self.headers,
                    json={"logs": self._buffer},
                )
                response.raise_for_status()
                self._buffer.clear()
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    sys.stderr.write(f"Failed to send logs: {e}\n")
                else:
                    await asyncio.sleep(2 ** attempt)

    def flush(self) -> None:
        """Принудительная отправка буфера."""
        if self.async_mode:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._flush_async())
            except RuntimeError:
                pass
        else:
            self._flush()

    def close(self) -> None:
        """Закрыть обработчик и отправить остатки."""
        self.flush()
        if self._client:
            self._client.close()
        if self._async_client:
            asyncio.create_task(self._async_client.aclose())
        super().close()


# ── Filter для чувствительных данных ────────────────────────────────────
class SensitiveDataFilter(logging.Filter):
    """
    Фильтр для удаления чувствительных данных из логов.

    Маскирует поля содержащие: password, secret, token, api_key, etc.
    """

    SENSITIVE_PATTERNS = [
        "password", "secret", "token", "api_key", "authorization",
        "credential", "private_key", "access_key", "secret_key",
    ]

    def __init__(self, name: str = "", redact_value: str = "***REDACTED***"):
        super().__init__(name)
        self.redact_value = redact_value

    def filter(self, record: logging.LogRecord) -> bool:
        """Маскировать чувствительные данные в сообщении."""
        if hasattr(record, "msg") and isinstance(record.msg, str):
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern in record.msg.lower():
                    record.msg = record.msg.replace(
                        pattern,
                        f"{pattern}={self.redact_value}"
                    )

        # Маскируем аргументы
        if hasattr(record, "args") and record.args:
            record.args = tuple(
                self.redact_value if isinstance(arg, str) and any(
                    p in arg.lower() for p in self.SENSITIVE_PATTERNS
                ) else arg
                for arg in record.args
            )

        return True


# ── Фабрика handlers ────────────────────────────────────────────────────
def get_handler(
    handler_type: str = "console",
    level: LogLevel = "INFO",
    **kwargs: Any,
) -> logging.Handler:
    """
    Получить обработчик по типу.

    Args:
        handler_type: 'console', 'file', 'rotating', 'http', 'async'
        level: Уровень логирования
        **kwargs: Дополнительные аргументы для обработчика

    Returns:
        Настроенный logging.Handler

    Examples:
        >>> get_handler("console", level="DEBUG")
        >>> get_handler("rotating", filename="./logs/app.log")
        >>> get_handler("http", endpoint="http://loki:3100/push")
    """
    # Конвертируем уровень
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if handler_type == "console":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(get_formatter("console"))

    elif handler_type == "file":
        filename = kwargs.get("filename", "./logs/app.log")
        handler = logging.FileHandler(filename)
        handler.setFormatter(get_formatter(kwargs.get("format_type", "json")))

    elif handler_type == "rotating":
        handler = RotatingFileHandler(
            filename=kwargs.get("filename", "./logs/app.log"),
            max_bytes=kwargs.get("max_bytes", 10 * 1024 * 1024),
            backup_count=kwargs.get("backup_count", 5),
            format_type=kwargs.get("format_type", "json"),
        )

    elif handler_type == "timed_rotating":
        handler = TimedRotatingFileHandler(
            filename=kwargs.get("filename", "./logs/app.log"),
            when=kwargs.get("when", "midnight"),
            interval=kwargs.get("interval", 1),
            backup_count=kwargs.get("backup_count", 7),
            format_type=kwargs.get("format_type", "json"),
        )

    elif handler_type == "http":
        handler = HTTPHandler(
            endpoint=kwargs["endpoint"],
            headers=kwargs.get("headers"),
            batch_size=kwargs.get("batch_size", 100),
            async_mode=kwargs.get("async_mode", True),
        )

    elif handler_type == "async":
        handler = AsyncQueueHandler(
            max_size=kwargs.get("max_size", 1000),
            flush_interval=kwargs.get("flush_interval", 1.0),
        )

    else:
        raise ValueError(f"Unknown handler type: {handler_type}")

    handler.setLevel(level)

    # Добавляем фильтр чувствительных данных
    if kwargs.get("filter_sensitive", True):
        handler.addFilter(SensitiveDataFilter())

    return handler


# ── Конфигурация logging через dictConfig ──────────────────────────────
def get_logging_config(
    environment: str = "development",
    log_level: str = "INFO",
    log_dir: Optional[PathLike] = None,
    enable_http: bool = False,
    http_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Получить конфигурацию для logging.dictConfig().

    Args:
        environment: 'development' или 'production'
        log_level: Уровень логирования
        log_dir: Директория для файловых логов
        enable_http: Включить HTTP-отправку
        http_endpoint: Endpoint для HTTP-отправки

    Returns:
        Dict для logging.dictConfig()

    Example:
        >>> config = get_logging_config("production", log_dir="./logs")
        >>> logging.dictConfig(config)
    """
    is_dev = environment == "development"

    # Форматтеры
    formatters = {
        "console": {
            "()": "edms_ai_assistant.packages.core.logging.formatters.ConsoleFormatter",
            "fmt": "%(asctime)s %(trace_id_formatted)s %(levelname)s %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": "edms_ai_assistant.packages.core.logging.formatters.JSONFormatter",
        },
    }

    # Handlers
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console" if is_dev else "json",
            "level": log_level,
            "stream": "ext://sys.stdout",
        },
    }

    # Файловые обработчики для production
    if not is_dev and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        handlers["file"] = {
            "class": "edms_ai_assistant.packages.core.logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "level": log_level,
            "filename": str(log_path / "app.log"),
            "max_bytes": 10 * 1024 * 1024,
            "backup_count": 5,
        }

        handlers["error_file"] = {
            "class": "edms_ai_assistant.packages.core.logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "level": "ERROR",
            "filename": str(log_path / "error.log"),
            "max_bytes": 10 * 1024 * 1024,
            "backup_count": 5,
        }

    # HTTP обработчик для ELK/Loki
    if enable_http and http_endpoint:
        handlers["http"] = {
            "class": "edms_ai_assistant.packages.core.logging.handlers.HTTPHandler",
            "formatter": "json",
            "level": log_level,
            "endpoint": http_endpoint,
            "async_mode": True,
        }

    # Root logger
    root_handlers = ["console"]
    if not is_dev and log_dir:
        root_handlers.extend(["file", "error_file"])
    if enable_http and http_endpoint:
        root_handlers.append("http")

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": log_level,
            "handlers": root_handlers,
        },
        "loggers": {
            "edms_ai_assistant": {
                "level": log_level,
                "handlers": root_handlers,
                "propagate": False,
            },
            # Тихие логгеры
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
            "asyncpg": {"level": "WARNING"},
            "sqlalchemy": {"level": "WARNING"},
        },
    }


# ── Экспорт ─────────────────────────────────────────────────────────────
__all__ = [
    # Handlers
    "AsyncQueueHandler",
    "RotatingFileHandler",
    "TimedRotatingFileHandler",
    "HTTPHandler",

    # Filters
    "SensitiveDataFilter",

    # Фабрики
    "get_handler",
    "get_logging_config",

    # Типы
    "LogLevel",
    "PathLike",
]