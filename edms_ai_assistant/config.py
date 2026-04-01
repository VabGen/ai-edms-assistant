# edms_ai_assistant/config.py
"""
Production-ready configuration with validation, security, and environment separation.
"""

import os

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration with validation.

    Security notes:
    - Secrets are stored as SecretStr (not printed in logs)
    - URLs are validated as HttpUrl (optional where applicable)
    - Environment-specific defaults available
    """

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    ENVIRONMENT: str = Field(
        default="development", pattern="^(development|staging|production)$"
    )
    API_PORT: int = Field(default=8000, ge=1, le=65535)
    DEBUG: bool = Field(default=False)
    ALLOWED_ORIGINS: str = Field(default="*")

    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def parse_origins(cls, v: str) -> list[str]:
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",")]

    # ── Security ─────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: SecretStr = Field(default="change-me-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60

    # ── LLM Configuration ────────────────────────────────────────────────────
    LLM_GENERATIVE_URL: HttpUrl = Field(
        default="http://model-generative.shared.du.iba/v1"
    )
    LLM_GENERATIVE_MODEL: str = Field(default="generative-model")

    LLM_EMBEDDING_URL: HttpUrl = Field(
        default="http://model-embedding.shared.du.iba/v1"
    )
    LLM_EMBEDDING_MODEL: str = "embedding-model"

    LLM_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None

    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int | None = Field(default=2048, ge=100, le=8192)
    LLM_TIMEOUT: int = Field(default=120, ge=10, le=600)
    LLM_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    LLM_REQUEST_TIMEOUT: int = Field(default=120, ge=10, le=600)
    LLM_STREAM_USAGE: bool = False

    # ── Embedding Configuration ──────────────────────────────────────────────
    EMBEDDING_TIMEOUT: int = Field(default=120, ge=10, le=600)
    EMBEDDING_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    EMBEDDING_REQUEST_TIMEOUT: int = Field(default=120, ge=10, le=600)
    EMBEDDING_CTX_LENGTH: int = 8191
    EMBEDDING_CHUNK_SIZE: int = 1000
    EMBEDDING_MAX_RETRIES_PER_REQUEST: int = 6

    # ── EDMS Configuration ───────────────────────────────────────────────────
    EDMS_BASE_URL: HttpUrl = Field(default="http://127.0.0.1:8098")
    EDMS_TIMEOUT: int = Field(default=120, ge=10, le=600)
    EDMS_API_VERSION: str = "v1"

    @property
    def CHANCELLOR_NEXT_BASE_URL(self) -> str:
        """Alias for backward compatibility with existing clients."""
        return str(self.EDMS_BASE_URL)

    # ── Database Configuration ───────────────────────────────────────────────
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: SecretStr = Field(default="password")
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ai_assistant"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    CHECKPOINT_DB_URL: str | None = None
    SQL_DB_URL: str | None = None

    # ── Redis Configuration ──────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: SecretStr | None = None
    CACHE_TTL_SECONDS: int = Field(default=300, ge=30, le=86400)

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return (
                f"redis://:{self.REDIS_PASSWORD.get_secret_value()}@"
                f"{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
            )
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ── File Upload Configuration ────────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = Field(default=50, ge=1, le=500)
    ALLOWED_FILE_EXTENSIONS: str = ".docx,.doc,.pdf,.txt,.rtf,.xlsx,.xls,.pptx"

    @property
    def ALLOWED_EXTENSIONS_LIST(self) -> set[str]:
        return {ext.strip().lower() for ext in self.ALLOWED_FILE_EXTENSIONS.split(",")}

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    # ── Agent Configuration ──────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = Field(default=10, ge=1, le=50)
    AGENT_MAX_CONTEXT_MESSAGES: int = Field(default=20, ge=5, le=100)
    AGENT_TIMEOUT: float = Field(default=120.0, ge=10.0, le=600.0)
    AGENT_ENABLE_TRACING: bool = False
    AGENT_LOG_LEVEL: str = "INFO"
    AGENT_MAX_RETRIES: int = 3

    SETTINGS_PANEL_SHOW_TECHNICAL: bool = Field(
        default=True,
        description=(
            "Показывать технический раздел настроек в Chrome-плагине. "
            "false = скрыт (production default). "
            "true = виден (dev/admin)."
        ),
    )

    # ── RAG Configuration ────────────────────────────────────────────────────
    RAG_BATCH_SIZE: int = 20
    RAG_CHUNK_SIZE: int = 1200
    RAG_CHUNK_OVERLAP: int = 300
    RAG_EMBEDDING_BATCH_SIZE: int = 10
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # ── Rate Limiting ────────────────────────────────────────────────────────
    RATE_LIMIT_MAX_REQUESTS: int = Field(default=10, ge=1, le=1000)
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, ge=10, le=3600)

    # ── Logging Configuration ────────────────────────────────────────────────
    LOGGING_LEVEL: str = Field(
        default="ERROR", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    LOGGING_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    LOGGING_INCLUDE_TRACE_ID: bool = True

    # ── Telemetry & Monitoring ───────────────────────────────────────────────
    TELEMETRY_ENABLED: bool = False
    TELEMETRY_ENDPOINT: str | None = Field(default=None)
    HEALTH_CHECK_ENABLED: bool = True

    # ── Environment-specific defaults ────────────────────────────────────────
    @field_validator("DEBUG", mode="before")
    @classmethod
    def set_debug_default(cls, v: bool | None, info) -> bool:
        if v is not None:
            return v
        env = info.data.get("ENVIRONMENT", "development")
        return env == "development"

    @field_validator("LOGGING_LEVEL", mode="before")
    @classmethod
    def set_log_level_default(cls, v: str | None, info) -> str:
        if v is not None:
            return v
        env = info.data.get("ENVIRONMENT", "development")
        return "DEBUG" if env == "development" else "INFO"

    @field_validator("TELEMETRY_ENDPOINT", mode="before")
    @classmethod
    def validate_telemetry_endpoint(cls, v: str | None) -> str | None:
        if not v or v.strip() == "":
            return None

        if not (v.startswith("http://") or v.startswith("https://")):
            return None
        return v.strip()


# ── Global settings instance ────────────────────────────────────────────────
model_config = SettingsConfigDict(
    env_file="../.env", env_file_encoding="utf-8", extra="ignore"
)

settings = Settings()


# ── Convenience properties (for backward compatibility) ─────────────────────
@property
def LLM_ENDPOINT(self) -> str:
    return str(settings.LLM_GENERATIVE_URL)


@property
def EMBEDDING_ENDPOINT(self) -> str:
    return str(settings.LLM_EMBEDDING_URL)


@property
def LLM_MODEL_NAME(self) -> str:
    return settings.LLM_GENERATIVE_MODEL


@property
def EMBEDDING_MODEL_NAME(self) -> str:
    return settings.LLM_EMBEDDING_MODEL
