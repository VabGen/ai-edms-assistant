# packages/core/settings.py
"""
Централизованная конфигурация EDMS AI Assistant.

Все значения читаются из переменных окружения / .env файла.
Экспортирует синглтон `settings`, доступный во всех сервисах:

    from edms_ai_assistant.config import settings

Группы настроек:
    APP         — окружение, порты, логирование
    ANTHROPIC   — Anthropic API
    LLM         — URL генерации, эмбеддингов
    EDMS        — базовый URL Java API
    POSTGRES    — параметры БД
    REDIS       — параметры кэша
    QDRANT      — векторная БД
    MCP         — MCP-сервер
    AGENT       — параметры агента
    CACHE       — TTL кэшей
"""
from __future__ import annotations

from typing import Any

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── APP ───────────────────────────────────────────────────────────────
    ENVIRONMENT: str = Field("development", pattern="^(development|staging|production)$")
    API_PORT: int = Field(8002, ge=1, le=65535)
    FEEDBACK_PORT: int = Field(8003, ge=1, le=65535)
    MCP_PORT: int = Field(8001, ge=1, le=65535)
    DEBUG: bool = Field(False)
    RELOAD: bool = Field(False)
    LOG_LEVEL: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOGGING_LEVEL: str = Field("INFO")
    LOGGING_FORMAT: str = Field("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ALLOWED_ORIGINS: str = Field("http://localhost:3000")
    UPLOAD_DIR: str = Field("/tmp/edms_uploads")
    SETTINGS_PANEL_SHOW_TECHNICAL: bool = Field(False)

    # ── ANTHROPIC ─────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: SecretStr = Field(...)

    # ── LLM ───────────────────────────────────────────────────────────────
    LLM_GENERATIVE_URL: HttpUrl = Field("http://localhost:11434")
    LLM_GENERATIVE_MODEL: str = Field("claude-sonnet-4-6")
    LLM_EMBEDDING_URL: HttpUrl = Field("http://localhost:11434")
    LLM_EMBEDDING_MODEL: str = Field("paraphrase-multilingual-MiniLM-L12-v2")
    LLM_TEMPERATURE: float = Field(0.6, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(4096, ge=100, le=32000)
    LLM_TIMEOUT: int = Field(120, ge=10, le=600)
    LLM_MAX_RETRIES: int = Field(3, ge=0, le=10)
    LLM_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None

    # ── EDMS ──────────────────────────────────────────────────────────────
    EDMS_BASE_URL: HttpUrl = Field("http://localhost:8098")
    CHANCELLOR_NEXT_BASE_URL: str = Field("http://localhost:8098")
    EDMS_TIMEOUT: int = Field(120, ge=10, le=600)

    # ── POSTGRES ──────────────────────────────────────────────────────────
    POSTGRES_USER: str = Field("edms")
    POSTGRES_PASSWORD: SecretStr = Field(...)
    POSTGRES_HOST: str = Field("localhost")
    POSTGRES_PORT: int = Field(5432, ge=1, le=65535)
    POSTGRES_DB: str = Field("edms_ai")
    DATABASE_URL: str = Field("")
    CHECKPOINT_DB_URL: str | None = None

    @model_validator(mode="after")
    def build_database_url(self) -> "Settings":
        if not self.DATABASE_URL and self.POSTGRES_PASSWORD:
            pw = self.POSTGRES_PASSWORD.get_secret_value()
            self.DATABASE_URL = (
                f"postgresql+asyncpg://{self.POSTGRES_USER}:{pw}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        if not self.CHECKPOINT_DB_URL and self.DATABASE_URL:
            self.CHECKPOINT_DB_URL = self.DATABASE_URL
        if not self.CHANCELLOR_NEXT_BASE_URL:
            self.CHANCELLOR_NEXT_BASE_URL = str(self.EDMS_BASE_URL)
        return self

    # ── REDIS ─────────────────────────────────────────────────────────────
    REDIS_HOST: str = Field("localhost")
    REDIS_PORT: int = Field(6379, ge=1, le=65535)
    REDIS_DB: int = Field(0, ge=0, le=15)
    REDIS_PASSWORD: SecretStr | None = None
    REDIS_URL: str = Field("")
    CACHE_TTL_SECONDS: int = Field(300, ge=1)

    @model_validator(mode="after")
    def build_redis_url(self) -> "Settings":
        if not self.REDIS_URL:
            if self.REDIS_PASSWORD:
                pw = self.REDIS_PASSWORD.get_secret_value()
                self.REDIS_URL = (
                    f"redis://:{pw}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
                )
            else:
                self.REDIS_URL = f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return self

    # ── QDRANT ────────────────────────────────────────────────────────────
    QDRANT_URL: HttpUrl = Field("http://localhost:6333")
    QDRANT_API_KEY: SecretStr | None = None

    # ── MCP ───────────────────────────────────────────────────────────────
    MCP_URL: HttpUrl = Field("http://localhost:8001")
    MCP_WORKERS: int = Field(1, ge=1)

    # ── AGENT ─────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = Field(10, ge=1, le=50)
    AGENT_MAX_CONTEXT_MESSAGES: int = Field(20, ge=5, le=100)
    AGENT_TIMEOUT: float = Field(120.0, ge=10.0, le=600.0)
    AGENT_MAX_RETRIES: int = Field(3, ge=0, le=10)
    AGENT_ENABLE_TRACING: bool = Field(False)
    AGENT_LOG_LEVEL: str = Field("INFO")

    # ── RAG ───────────────────────────────────────────────────────────────
    RAG_CHUNK_SIZE: int = Field(1200, ge=100, le=8000)
    RAG_CHUNK_OVERLAP: int = Field(300, ge=0, le=2000)
    RAG_BATCH_SIZE: int = Field(20, ge=1, le=100)
    RAG_EMBEDDING_BATCH_SIZE: int = Field(10, ge=1, le=50)
    RAG_UPDATE_HOUR: int = Field(3, ge=0, le=23)
    RAG_UPDATE_MINUTE: int = Field(0, ge=0, le=59)

    @property
    def CORS_ORIGINS(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]


settings = Settings()
