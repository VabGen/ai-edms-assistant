# edms_ai_assistant/config.py
"""
Единственный источник конфигурации EDMS AI Assistant.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Полная конфигурация приложения из переменных окружения / .env файла."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Приложение ────────────────────────────────────────────────────────
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    LOGGING_LEVEL: str = Field(default="INFO")
    LOGGING_FORMAT: str = Field(
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    RELOAD: bool = Field(default=False)
    DEBUG: bool = Field(default=False)
    ALLOWED_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:8080")
    UPLOAD_DIR: str = Field(default="/tmp/edms_uploads")
    MAX_FILE_SIZE_MB: int = Field(default=50, ge=1)
    SETTINGS_PANEL_SHOW_TECHNICAL: bool = Field(default=False)

    # ── Порты сервисов ────────────────────────────────────────────────────
    API_PORT: int = Field(default=8000, ge=1, le=65535)
    MCP_PORT: int = Field(default=8001, ge=1, le=65535)
    FEEDBACK_PORT: int = Field(default=8002, ge=1, le=65535)
    PROMETHEUS_PORT: int = Field(default=9090)
    GRAFANA_PORT: int = Field(default=3000)

    # ── LLM (Ollama / OpenAI-compatible) ──────────────────────────────────
    OLLAMA_BASE_URL: str = Field(default="http://127.0.0.1:11434")
    MODEL_NAME: str = Field(default="gpt-oss:120b-cloud")
    MODEL_PLANNER: str = Field(default="gpt-oss:120b-cloud")
    MODEL_RESEARCHER: str = Field(default="gpt-oss:120b-cloud")
    MODEL_EXECUTOR: str = Field(default="gpt-oss:120b-cloud")
    MODEL_REVIEWER: str = Field(default="gpt-oss:120b-cloud")
    MODEL_EXPLAINER: str = Field(default="gpt-oss:120b-cloud")

    # LLM — также поддерживаем Anthropic если ключ задан
    ANTHROPIC_API_KEY: SecretStr | None = Field(default=None)
    OPENAI_API_KEY: SecretStr | None = Field(default=None)

    # Общие параметры LLM
    LLM_GENERATIVE_URL: str = Field(default="http://127.0.0.1:11434")
    LLM_GENERATIVE_MODEL: str = Field(default="gpt-oss:120b-cloud")
    LLM_TEMPERATURE: float = Field(default=0.6, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2048, ge=64)
    LLM_TIMEOUT: int = Field(default=120, ge=1)
    LLM_MAX_RETRIES: int = Field(default=3, ge=0)
    LLM_STREAM_USAGE: bool = Field(default=False)

    # ── EDMS Java API ─────────────────────────────────────────────────────
    EDMS_BASE_URL: str = Field(default="http://127.0.0.1:8098")
    EDMS_TIMEOUT: int = Field(default=120, ge=1)

    # ── PostgreSQL ────────────────────────────────────────────────────────
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: SecretStr = Field(default=SecretStr("1234"))
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432, ge=1, le=65535)
    POSTGRES_DB: str = Field(default="edms")
    DATABASE_URL: str = Field(default="")
    CHECKPOINT_DB_URL: str = Field(default="")
    SQL_DB_URL: str = Field(default="")

    @model_validator(mode="after")
    def build_database_urls(self) -> "Settings":
        """Строит DATABASE_URL из компонентов если не задан явно."""
        if not self.DATABASE_URL:
            pw = self.POSTGRES_PASSWORD.get_secret_value()
            base = (
                f"postgresql+asyncpg://{self.POSTGRES_USER}:{pw}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
            self.DATABASE_URL = base
        if not self.CHECKPOINT_DB_URL:
            self.CHECKPOINT_DB_URL = self.DATABASE_URL
        if not self.SQL_DB_URL:
            self.SQL_DB_URL = self.DATABASE_URL
        return self

    # ── Redis ─────────────────────────────────────────────────────────────
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0, ge=0, le=15)
    REDIS_PASSWORD: SecretStr | None = Field(default=None)
    REDIS_URL: str = Field(default="")
    CACHE_TTL_SECONDS: int = Field(default=3600, ge=1)
    CACHE_TTL_READ: int = Field(default=300, ge=1)

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

    # ── Qdrant ────────────────────────────────────────────────────────────
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_API_KEY: SecretStr | None = Field(default=None)

    # ── MCP ───────────────────────────────────────────────────────────────
    MCP_HOST: str = Field(default="0.0.0.0")
    MCP_URL: str = Field(default="http://localhost:8001")

    # ── Агент ─────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = Field(default=10, ge=1, le=50)
    AGENT_MAX_CONTEXT_MESSAGES: int = Field(default=20, ge=5, le=100)
    AGENT_TIMEOUT: float = Field(default=120.0, ge=10.0)
    AGENT_MAX_RETRIES: int = Field(default=3, ge=0)
    AGENT_ENABLE_TRACING: bool = Field(default=False)
    AGENT_LOG_LEVEL: str = Field(default="INFO")

    # ── RAG ───────────────────────────────────────────────────────────────
    RAG_BATCH_SIZE: int = Field(default=20, ge=1)
    RAG_CHUNK_SIZE: int = Field(default=1200, ge=100)
    RAG_CHUNK_OVERLAP: int = Field(default=300, ge=0)
    RAG_EMBEDDING_BATCH_SIZE: int = Field(default=10, ge=1)
    RAG_COLLECTION_SUCCESSFUL: str = Field(default="successful_dialogs")
    RAG_COLLECTION_ANTI: str = Field(default="anti_examples")
    RAG_SIMILARITY_TOP_K: int = Field(default=5, ge=1)
    RAG_UPDATE_HOUR: int = Field(default=3, ge=0, le=23)
    RAG_UPDATE_MINUTE: int = Field(default=0, ge=0, le=59)
    FAISS_INDEX_PATH: str = Field(default="/tmp/edms_rag_faiss.pkl")

    # ── Embedding ─────────────────────────────────────────────────────────
    EMBEDDING_URL: str = Field(default="http://127.0.0.1:11434/v1")
    EMBEDDING_MODEL: str = Field(default="embedding-model")
    EMBEDDING_DIM: int = Field(default=384)
    EMBEDDING_TIMEOUT: int = Field(default=120)

    # ── Feedback ──────────────────────────────────────────────────────────
    FEEDBACK_API_URL: str = Field(default="http://localhost:8002")
    ORCHESTRATOR_URL: str = Field(default="http://localhost:8000")

    # ── Мониторинг ────────────────────────────────────────────────────────
    TELEMETRY_ENABLED: bool = Field(default=False)
    HEALTH_CHECK_ENABLED: bool = Field(default=True)
    GRAFANA_ADMIN_USER: str = Field(default="admin")
    GRAFANA_ADMIN_PASSWORD: str = Field(default="change-me-in-production")

    # ── Безопасность ──────────────────────────────────────────────────────
    JWT_SECRET_KEY: SecretStr = Field(
        default=SecretStr("change-me-in-production-min-32-chars!!")
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_MINUTES: int = Field(default=60)
    SAFETY_INPUT_FILTER: bool = Field(default=True)
    SAFETY_OUTPUT_FILTER: bool = Field(default=True)

    # ── Rate limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_MAX_REQUESTS: int = Field(default=10)
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60)

    # ── LangSmith (опционально) ───────────────────────────────────────────
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_API_KEY: SecretStr | None = Field(default=None)
    LANGSMITH_PROJECT: str = Field(default="edms_ai_assistant")

    @property
    def CORS_ORIGINS(self) -> list[str]:
        """Список разрешённых CORS origins."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def IS_PRODUCTION(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def llm_base_url(self) -> str:
        """Базовый URL для LLM (Ollama или внешний)."""
        return self.OLLAMA_BASE_URL.rstrip("/")

    @property
    def chancellor_next_base_url(self) -> str:
        """Алиас для EDMS_BASE_URL (совместимость)."""
        return self.EDMS_BASE_URL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает синглтон настроек (кэшируется после первого вызова)."""
    s = Settings()
    logger.info(
        "Settings loaded: env=%s api_port=%d mcp_port=%d model=%s",
        s.ENVIRONMENT,
        s.API_PORT,
        s.MCP_PORT,
        s.MODEL_NAME,
    )
    return s


settings: Settings = get_settings()
