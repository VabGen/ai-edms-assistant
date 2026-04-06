# orchestrator/config.py
"""
Конфигурация оркестратора EDMS AI Assistant.

Единственный источник истины для всех настроек.
Загружает переменные из .env через pydantic-settings.

Экспортирует глобальный синглтон: settings
"""
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Полная конфигурация оркестратора из переменных окружения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Приложение ─────────────────────────────────────────────────────────
    ENVIRONMENT: str = Field(default="development")
    API_PORT: int = Field(default=8002, ge=1, le=65535)
    DEBUG: bool = Field(default=False)
    LOGGING_LEVEL: str = Field(default="INFO")
    LOGGING_FORMAT: str = Field(
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"]
    )

    # ── Anthropic / LLM ────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: SecretStr | None = None
    # Fallback для Ollama/OpenAI-compatible
    LLM_GENERATIVE_URL: AnyHttpUrl = Field(default="http://127.0.0.1:11434")
    LLM_GENERATIVE_MODEL: str = Field(default="gpt-oss:120b-cloud")
    LLM_EMBEDDING_URL: AnyHttpUrl = Field(default="http://127.0.0.1:11434")
    LLM_EMBEDDING_MODEL: str = Field(default="nomic-embed-text")
    LLM_TEMPERATURE: float = Field(default=0.6, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=4096, ge=64)
    LLM_TIMEOUT: int = Field(default=120, ge=1)
    LLM_MAX_RETRIES: int = Field(default=3, ge=0)
    OPENAI_API_KEY: SecretStr | None = None
    LLM_API_KEY: SecretStr | None = None

    # ── MCP Server ─────────────────────────────────────────────────────────
    MCP_URL: AnyHttpUrl = Field(default="http://localhost:8001")
    MCP_HOST: str = Field(default="0.0.0.0")
    MCP_PORT: int = Field(default=8001, ge=1, le=65535)

    # ── EDMS API ───────────────────────────────────────────────────────────
    EDMS_BASE_URL: AnyHttpUrl = Field(default="http://127.0.0.1:8098")
    CHANCELLOR_NEXT_BASE_URL: AnyHttpUrl = Field(default="http://127.0.0.1:8098")
    EDMS_TIMEOUT: int = Field(default=120, ge=1)

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    POSTGRES_USER: str = Field(default="edms")
    POSTGRES_PASSWORD: SecretStr = Field(default=SecretStr("change-me"))
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432, ge=1, le=65535)
    POSTGRES_DB: str = Field(default="edms_ai")
    DATABASE_URL: str = Field(default="")
    # Отдельная БД для LangGraph checkpoints
    CHECKPOINT_DB_URL: str | None = None

    @model_validator(mode="after")
    def build_database_urls(self) -> "Settings":
        """Строит DATABASE_URL из компонентов если не задан явно."""
        if not self.DATABASE_URL:
            pw = self.POSTGRES_PASSWORD.get_secret_value()
            self.DATABASE_URL = (
                f"postgresql+asyncpg://{self.POSTGRES_USER}:{pw}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        return self

    # ── Redis ──────────────────────────────────────────────────────────────
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0, ge=0, le=15)
    REDIS_PASSWORD: SecretStr | None = None
    REDIS_URL: str = Field(default="")
    CACHE_TTL_SECONDS: int = Field(default=300, ge=1)

    @model_validator(mode="after")
    def build_redis_url(self) -> "Settings":
        """Строит REDIS_URL из компонентов если не задан явно."""
        if not self.REDIS_URL:
            if self.REDIS_PASSWORD:
                pw = self.REDIS_PASSWORD.get_secret_value()
                self.REDIS_URL = (
                    f"redis://:{pw}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
                )
            else:
                self.REDIS_URL = (
                    f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
                )
        return self

    # ── Агент ──────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = Field(default=10, ge=1, le=50)
    AGENT_MAX_CONTEXT_MESSAGES: int = Field(default=20, ge=5, le=100)
    AGENT_TIMEOUT: float = Field(default=120.0, ge=10.0)
    AGENT_MAX_RETRIES: int = Field(default=3, ge=0)
    AGENT_ENABLE_TRACING: bool = Field(default=False)
    AGENT_LOG_LEVEL: str = Field(default="INFO")

    # ── RAG ────────────────────────────────────────────────────────────────
    RAG_CHUNK_SIZE: int = Field(default=1200, ge=100)
    RAG_CHUNK_OVERLAP: int = Field(default=300, ge=0)
    RAG_BATCH_SIZE: int = Field(default=20, ge=1)
    RAG_EMBEDDING_BATCH_SIZE: int = Field(default=10, ge=1)
    RAG_UPDATE_HOUR: int = Field(default=3, ge=0, le=23)
    RAG_UPDATE_MINUTE: int = Field(default=0, ge=0, le=59)

    # ── Feedback collector ─────────────────────────────────────────────────
    FEEDBACK_PORT: int = Field(default=8003, ge=1, le=65535)
    FEEDBACK_API_URL: AnyHttpUrl = Field(default="http://localhost:8003")

    # ── Безопасность ───────────────────────────────────────────────────────
    JWT_SECRET_KEY: SecretStr = Field(
        default=SecretStr("change-me-in-production-min-32-chars!!")
    )

    # ── Файлы ──────────────────────────────────────────────────────────────
    UPLOAD_DIR: str = Field(default="/tmp/edms_ai_uploads")
    MAX_FILE_SIZE_MB: int = Field(default=50, ge=1)

    # ── UI ─────────────────────────────────────────────────────────────────
    SETTINGS_PANEL_SHOW_TECHNICAL: bool = Field(default=False)

    # ── Мониторинг ─────────────────────────────────────────────────────────
    PROMETHEUS_PORT: int = Field(default=9090)
    GRAFANA_PORT: int = Field(default=3000)

    @property
    def IS_PRODUCTION(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.ENVIRONMENT == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает синглтон настроек (кэшируется после первого вызова)."""
    s = Settings()
    logger.info(
        "Settings loaded: env=%s port=%d model=%s",
        s.ENVIRONMENT,
        s.API_PORT,
        s.LLM_GENERATIVE_MODEL,
    )
    return s


# Глобальный синглтон
settings: Settings = get_settings()
