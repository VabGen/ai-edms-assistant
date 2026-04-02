"""
edms_ai_assistant/packages/core/logging/config.py

Централизованная конфигурация логирования для всего проекта.
Интегрируется с pydantic-settings для управления через .env

Features:
• Конфигурация через settings.py (единый источник истины)
• Поддержка development/production режимов
• Интеграция с OpenTelemetry для distributed tracing
• Автоматическая фильтрация чувствительных данных
• Готовые пресеты для разных окружений
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import structlog

# Локальные импорты
from .formatters import (
    ConsoleFormatter,
    JSONFormatter,
    get_formatter,
    get_structlog_processors,
    redact_sensitive_fields,
)
from .handlers import (
    AsyncQueueHandler,
    HTTPHandler,
    RotatingFileHandler,
    SensitiveDataFilter,
    TimedRotatingFileHandler,
    get_handler,
)


# ── Типы ────────────────────────────────────────────────────────────────
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["console", "json", "simple"]
LogHandler = Literal["console", "file", "rotating", "http", "async"]


# ── Конфигурация логирования ────────────────────────────────────────────
class LoggingConfig:
    """
    Конфигурация логирования для приложения.

    Использование:
        config = LoggingConfig.from_settings(settings)
        config.apply()
    """

    def __init__(
        self,
        environment: str = "development",
        log_level: LogLevel = "INFO",
        log_format: LogFormat = "console",
        log_dir: Optional[str] = None,
        enable_file_logging: bool = False,
        enable_http_logging: bool = False,
        http_endpoint: Optional[str] = None,
        include_trace_id: bool = True,
        filter_sensitive_data: bool = True,
        max_file_size_mb: int = 10,
        backup_count: int = 5,
    ):
        """
        Инициализировать конфигурацию логирования.

        Args:
            environment: 'development', 'staging', или 'production'
            log_level: Уровень логирования
            log_format: Формат вывода ('console', 'json', 'simple')
            log_dir: Директория для файловых логов
            enable_file_logging: Включить запись в файлы
            enable_http_logging: Включить отправку в HTTP (ELK/Loki)
            http_endpoint: URL для HTTP-отправки логов
            include_trace_id: Включить trace_id из OpenTelemetry
            filter_sensitive_data: Маскировать пароли, токены, etc.
            max_file_size_mb: Максимальный размер файла до ротации
            backup_count: Количество архивных файлов
        """
        self.environment = environment
        self.log_level = log_level
        self.log_format = log_format
        self.log_dir = log_dir
        self.enable_file_logging = enable_file_logging
        self.enable_http_logging = enable_http_logging
        self.http_endpoint = http_endpoint
        self.include_trace_id = include_trace_id
        self.filter_sensitive_data = filter_sensitive_data
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count

        # Вычисленные значения
        self.is_development = environment == "development"
        self.is_production = environment == "production"

        # Переопределяем формат для production
        if self.is_production and log_format == "console":
            self.log_format = "json"

    @classmethod
    def from_settings(cls, settings: Any) -> "LoggingConfig":
        """
        Создать конфигурацию из объекта settings (pydantic-settings).

        Args:
            settings: Объект настроек приложения

        Returns:
            Настроенный LoggingConfig

        Example:
            >>> from packages.core.settings import settings
            >>> config = LoggingConfig.from_settings(settings)
            >>> config.apply()
        """
        # Определяем включено ли файловое логирование
        enable_file = settings.IS_PRODUCTION if hasattr(settings, 'IS_PRODUCTION') else False

        # Определяем включено ли HTTP логирование
        enable_http = (
            settings.TELEMETRY_ENABLED
            and settings.TELEMETRY_ENDPOINT
            if hasattr(settings, 'TELEMETRY_ENABLED')
            else False
        )

        return cls(
            environment=settings.ENVIRONMENT,
            log_level=settings.LOG_LEVEL,
            log_format="json" if enable_file else "console",
            log_dir=getattr(settings, 'LOG_DIR', './logs'),
            enable_file_logging=enable_file,
            enable_http_logging=enable_http,
            http_endpoint=getattr(settings, 'TELEMETRY_ENDPOINT', None),
            include_trace_id=getattr(settings, 'LOGGING_INCLUDE_TRACE_ID', True),
            filter_sensitive_data=True,
            max_file_size_mb=getattr(settings, 'LOG_MAX_FILE_SIZE_MB', 10),
            backup_count=getattr(settings, 'LOG_BACKUP_COUNT', 5),
        )

    @classmethod
    def development(cls) -> "LoggingConfig":
        """Пресет для разработки."""
        return cls(
            environment="development",
            log_level="DEBUG",
            log_format="console",
            enable_file_logging=False,
            include_trace_id=True,
        )

    @classmethod
    def production(cls, log_dir: str = "./logs") -> "LoggingConfig":
        """Пресет для production."""
        return cls(
            environment="production",
            log_level="INFO",
            log_format="json",
            log_dir=log_dir,
            enable_file_logging=True,
            enable_http_logging=False,
            include_trace_id=True,
        )

    @classmethod
    def staging(cls, http_endpoint: str) -> "LoggingConfig":
        """Пресет для staging с отправкой в ELK."""
        return cls(
            environment="staging",
            log_level="INFO",
            log_format="json",
            enable_file_logging=True,
            enable_http_logging=True,
            http_endpoint=http_endpoint,
            include_trace_id=True,
        )

    def get_structlog_config(self) -> Dict[str, Any]:
        """
        Получить конфигурацию для structlog.configure().

        Returns:
            Dict для передачи в structlog.configure()
        """
        return {
            "processors": get_structlog_processors(
                environment=self.environment,
                include_trace_id=self.include_trace_id,
            ),
            "wrapper_class": structlog.make_filtering_bound_logger(
                getattr(logging, self.log_level, logging.INFO)
            ),
            "context_class": dict,
            "logger_factory": structlog.PrintLoggerFactory(sys.stdout),
            "cache_logger_on_first_use": True,
        }

    def get_logging_dict_config(self) -> Dict[str, Any]:
        """
        Получить конфигурацию для logging.dictConfig().

        Returns:
            Dict для передачи в logging.dictConfig()

        Example:
            >>> import logging.config
            >>> config = LoggingConfig.production()
            >>> logging.config.dictConfig(config.get_logging_dict_config())
        """
        # Форматтеры
        formatters = {
            "console": {
                "()": ConsoleFormatter,
                "fmt": "%(asctime)s %(trace_id_formatted)s %(levelname)s %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": self.is_development,
            },
            "json": {
                "()": JSONFormatter,
            },
            "simple": {
                "format": "%(levelname)s %(name)s: %(message)s",
            },
        }

        # Handlers
        handlers: Dict[str, Any] = {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console" if self.is_development else "json",
                "level": self.log_level,
                "stream": "ext://sys.stdout",
            },
        }

        # Файловые обработчики для production/staging
        if self.enable_file_logging and self.log_dir:
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            handlers["file"] = {
                "class": "edms_ai_assistant.packages.core.logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "level": self.log_level,
                "filename": str(log_path / "app.log"),
                "max_bytes": self.max_file_size_mb * 1024 * 1024,
                "backup_count": self.backup_count,
            }

            handlers["error_file"] = {
                "class": "edms_ai_assistant.packages.core.logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "level": "ERROR",
                "filename": str(log_path / "error.log"),
                "max_bytes": self.max_file_size_mb * 1024 * 1024,
                "backup_count": self.backup_count,
            }

        # HTTP обработчик для ELK/Loki
        if self.enable_http_logging and self.http_endpoint:
            handlers["http"] = {
                "class": "edms_ai_assistant.packages.core.logging.handlers.HTTPHandler",
                "formatter": "json",
                "level": self.log_level,
                "endpoint": self.http_endpoint,
                "async_mode": True,
                "batch_size": 100,
            }

        # Root logger handlers
        root_handlers = ["console"]
        if self.enable_file_logging and self.log_dir:
            root_handlers.extend(["file", "error_file"])
        if self.enable_http_logging and self.http_endpoint:
            root_handlers.append("http")

        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "root": {
                "level": self.log_level,
                "handlers": root_handlers,
            },
            "loggers": {
                "edms_ai_assistant": {
                    "level": self.log_level,
                    "handlers": root_handlers,
                    "propagate": False,
                },
                # Тихие логгеры
                "httpx": {"level": "WARNING"},
                "httpcore": {"level": "WARNING"},
                "asyncpg": {"level": "WARNING"},
                "sqlalchemy": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"},
                "prometheus_client": {"level": "WARNING"},
            },
        }

    def apply(self) -> None:
        """
        Применить конфигурацию логирования.

        Вызывать ОДИН РАЗ при старте приложения.

        Example:
            >>> config = LoggingConfig.from_settings(settings)
            >>> config.apply()
        """
        import logging.config

        # Настраиваем structlog
        structlog.configure(**self.get_structlog_config())

        # Настраиваем стандартный logging
        logging.config.dictConfig(self.get_logging_dict_config())

        # Логируем что конфигурация применена
        logger = structlog.get_logger("edms.logging")
        logger.info(
            "Logging configuration applied",
            environment=self.environment,
            log_level=self.log_level,
            log_format=self.log_format,
            file_logging=self.enable_file_logging,
            http_logging=self.enable_http_logging,
            trace_id_injection=self.include_trace_id,
            sensitive_data_filter=self.filter_sensitive_data,
        )

    def get_handlers(self) -> List[logging.Handler]:
        """
        Получить список настроенных handlers.

        Returns:
            Список logging.Handler объектов
        """
        handlers: List[logging.Handler] = []

        # Console handler
        console_handler = get_handler("console", level=self.log_level)
        handlers.append(console_handler)

        # File handlers
        if self.enable_file_logging and self.log_dir:
            file_handler = get_handler(
                "rotating",
                level=self.log_level,
                filename=Path(self.log_dir) / "app.log",
                max_bytes=self.max_file_size_mb * 1024 * 1024,
                backup_count=self.backup_count,
            )
            handlers.append(file_handler)

            error_handler = get_handler(
                "rotating",
                level="ERROR",
                filename=Path(self.log_dir) / "error.log",
                max_bytes=self.max_file_size_mb * 1024 * 1024,
                backup_count=self.backup_count,
            )
            handlers.append(error_handler)

        # HTTP handler
        if self.enable_http_logging and self.http_endpoint:
            http_handler = get_handler(
                "http",
                level=self.log_level,
                endpoint=self.http_endpoint,
                async_mode=True,
            )
            handlers.append(http_handler)

        return handlers


# ── Convenience функции ─────────────────────────────────────────────────
def setup_logging(
    environment: Optional[str] = None,
    log_level: Optional[LogLevel] = None,
    log_dir: Optional[str] = None,
    **kwargs: Any,
) -> LoggingConfig:
    """
    Быстрая настройка логирования.

    Args:
        environment: 'development', 'staging', 'production'
        log_level: Уровень логирования
        log_dir: Директория для логов
        **kwargs: Дополнительные аргументы для LoggingConfig

    Returns:
        Настроенный LoggingConfig

    Example:
        >>> config = setup_logging("production", log_dir="./logs")
    """
    config = LoggingConfig(
        environment=environment or "development",
        log_level=log_level or "INFO",
        log_dir=log_dir,
        **kwargs,
    )
    config.apply()
    return config


def setup_logging_from_settings(settings: Any) -> LoggingConfig:
    """
    Настроить логирование из объекта settings.

    Args:
        settings: Объект pydantic-settings

    Returns:
        Настроенный LoggingConfig

    Example:
        >>> from packages.core.settings import settings
        >>> config = setup_logging_from_settings(settings)
    """
    config = LoggingConfig.from_settings(settings)
    config.apply()
    return config


# ── Экспорт ─────────────────────────────────────────────────────────────
__all__ = [
    # Классы
    "LoggingConfig",

    # Типы
    "LogLevel",
    "LogFormat",
    "LogHandler",

    # Convenience функции
    "setup_logging",
    "setup_logging_from_settings",
]