#
from typing import List, Optional, Set
from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # APPLICATION
    # ═══════════════════════════════════════════════════════════════════════
    ENVIRONMENT: str = Field(
        default="development",
        pattern="^(development|staging|production)$",
        description="Deployment environment"
    )
    API_PORT: int = Field(default=8000, ge=1, le=65535)
    FEEDBACK_PORT: int = Field(default=8002, ge=1, le=65535)
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    RELOAD: bool = Field(default=False)
    TIMEZONE: str = Field(default="Europe/Moscow")
    ALLOWED_ORIGINS: str = Field(default="http://localhost:3000")

    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def parse_origins(cls, v: str) -> str:
        return v.strip()

    @property
    def ALLOWED_ORIGINS_LIST(self) -> List[str]:
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    # ═══════════════════════════════════════════════════════════════════════
    # SECURITY
    # ═══════════════════════════════════════════════════════════════════════
    JWT_SECRET_KEY: SecretStr = Field(..., min_length=32)
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_MINUTES: int = Field(default=60, ge=1)

    # ═══════════════════════════════════════════════════════════════════════
    # LLM CONFIGURATION (Ollama)
    # ═══════════════════════════════════════════════════════════════════════
    OLLAMA_BASE_URL: HttpUrl = Field(default="http://127.0.0.1:11434")
    MODEL_NAME: str = Field(default="llama3.2:latest")
    MODEL_PLANNER: str = Field(default="llama3.2:latest")
    MODEL_RESEARCHER: str = Field(default="llama3.2:latest")
    MODEL_EXECUTOR: str = Field(default="llama3.2:latest")
    MODEL_REVIEWER: str = Field(default="llama3.2:latest")
    MODEL_EXPLAINER: str = Field(default="llama3.2:latest")

    LLM_TEMPERATURE: float = Field(default=0.6, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2048, ge=1)
    LLM_TIMEOUT: int = Field(default=120, ge=1)
    LLM_MAX_RETRIES: int = Field(default=3, ge=0)
    LLM_REQUEST_TIMEOUT: int = Field(default=120, ge=1)
    LLM_STREAM_USAGE: bool = Field(default=False)

    # External API keys (optional)
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    OPENAI_API_KEY: Optional[SecretStr] = None
    LLM_API_KEY: Optional[SecretStr] = None

    # ═══════════════════════════════════════════════════════════════════════
    # EMBEDDING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    EMBEDDING_URL: HttpUrl = Field(default="http://model-embedding.shared.du.iba/v1")
    EMBEDDING_MODEL: str = Field(default="embedding-model")
    EMBEDDING_DIM: int = Field(default=1536, ge=1)
    EMBEDDING_TIMEOUT: int = Field(default=120, ge=1)
    EMBEDDING_MAX_RETRIES: int = Field(default=3, ge=0)
    EMBEDDING_REQUEST_TIMEOUT: int = Field(default=120, ge=1)
    EMBEDDING_CTX_LENGTH: int = Field(default=8191, ge=1)
    EMBEDDING_CHUNK_SIZE: int = Field(default=1000, ge=1)
    EMBEDDING_MAX_RETRIES_PER_REQUEST: int = Field(default=6, ge=0)
    FAISS_INDEX_PATH: str = Field(default="/tmp/edms_rag_faiss.pkl")

    # ═══════════════════════════════════════════════════════════════════════
    # EDMS CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    EDMS_BASE_URL: HttpUrl = Field(default="http://127.0.0.1:8098")
    EDMS_TIMEOUT: int = Field(default=120, ge=1)

    @property
    def CHANCELLOR_NEXT_BASE_URL(self) -> str:
        """Alias for backward compatibility with existing clients."""
        return str(self.EDMS_BASE_URL)

    # ═══════════════════════════════════════════════════════════════════════
    # DATABASE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: SecretStr = Field(..., min_length=8)
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432, ge=1, le=65535)
    POSTGRES_DB: str = Field(default="edms")

    DATABASE_URL: str = Field(default="")
    CHECKPOINT_DB_URL: Optional[str] = None
    SQL_DB_URL: Optional[str] = None

    @model_validator(mode="after")
    def build_database_urls(self) -> "Settings":
        """Build DATABASE_URL if not provided."""
        if not self.DATABASE_URL and self.POSTGRES_PASSWORD:
            self.DATABASE_URL = (
                f"postgresql+asyncpg://{self.POSTGRES_USER}:"
                f"{self.POSTGRES_PASSWORD.get_secret_value()}@"
                f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        return self

    @property
    def DATABASE_URL_SECURE(self) -> str:
        """DATABASE_URL with masked password for logging."""
        if not self.DATABASE_URL:
            return ""
        return self.DATABASE_URL.replace(
            self.POSTGRES_PASSWORD.get_secret_value(),
            "*****"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # REDIS CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535)
    REDIS_DB: int = Field(default=0, ge=0, le=15)
    REDIS_PASSWORD: Optional[SecretStr] = None
    REDIS_URL: str = Field(default="")
    CACHE_TTL_SECONDS: int = Field(default=3600, ge=1)
    CACHE_TTL_READ: int = Field(default=300, ge=1)

    @property
    def REDIS_URL_BUILD(self) -> str:
        """Build REDIS_URL if not provided."""
        if self.REDIS_URL:
            return self.REDIS_URL
        if self.REDIS_PASSWORD:
            return (
                f"redis://:{self.REDIS_PASSWORD.get_secret_value()}@"
                f"{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
            )
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ═══════════════════════════════════════════════════════════════════════
    # QDRANT (Vector DB)
    # ═══════════════════════════════════════════════════════════════════════
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_URL: HttpUrl = Field(default="http://localhost:6333")
    QDRANT_PORT: int = Field(default=6333, ge=1, le=65535)
    QDRANT_API_KEY: Optional[SecretStr] = None

    # ═══════════════════════════════════════════════════════════════════════
    # MCP (Model Context Protocol)
    # ═══════════════════════════════════════════════════════════════════════
    MCP_HOST: str = Field(default="0.0.0.0")
    MCP_PORT: int = Field(default=8001, ge=1, le=65535)
    MCP_URL: HttpUrl = Field(default="http://localhost:8001")
    MCP_WORKERS: int = Field(default=1, ge=1)

    # ═══════════════════════════════════════════════════════════════════════
    # ORCHESTRATOR
    # ═══════════════════════════════════════════════════════════════════════
    ORCHESTRATOR_PORT: int = Field(default=8000, ge=1, le=65535)
    ORCHESTRATOR_WORKERS: int = Field(default=2, ge=1)
    FEEDBACK_API_URL: HttpUrl = Field(default="http://localhost:8002")
    FEEDBACK_COLLECTOR_PORT: int = Field(default=8002, ge=1, le=65535)
    FEEDBACK_WORKERS: int = Field(default=1, ge=1)

    # ═══════════════════════════════════════════════════════════════════════
    # MEMORY
    # ═══════════════════════════════════════════════════════════════════════
    MEMORY_SHORT_TERM_MAX_TOKENS: int = Field(default=8000, ge=1)
    MEMORY_MEDIUM_TERM_TTL: int = Field(default=3600, ge=1)

    # ═══════════════════════════════════════════════════════════════════════
    # FILE UPLOAD
    # ═══════════════════════════════════════════════════════════════════════
    UPLOAD_DIR: str = Field(default="./uploads")
    MAX_FILE_SIZE_MB: int = Field(default=50, ge=1)
    ALLOWED_FILE_EXTENSIONS: str = Field(default=".docx,.doc,.pdf,.txt,.rtf,.xlsx,.xls,.pptx")

    @property
    def ALLOWED_EXTENSIONS_LIST(self) -> Set[str]:
        return {ext.strip().lower() for ext in self.ALLOWED_FILE_EXTENSIONS.split(",")}

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    # ═══════════════════════════════════════════════════════════════════════
    # AGENT CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    AGENT_MAX_ITERATIONS: int = Field(default=10, ge=1)
    AGENT_MAX_CONTEXT_MESSAGES: int = Field(default=20, ge=1)
    AGENT_TIMEOUT: float = Field(default=120.0, ge=1.0)
    AGENT_ENABLE_TRACING: bool = Field(default=False)
    AGENT_LOG_LEVEL: str = Field(default="INFO")
    AGENT_MAX_RETRIES: int = Field(default=3, ge=0)
    SETTINGS_PANEL_SHOW_TECHNICAL: bool = Field(default=False)

    # ═══════════════════════════════════════════════════════════════════════
    # RAG CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    RAG_BATCH_SIZE: int = Field(default=20, ge=1)
    RAG_CHUNK_SIZE: int = Field(default=1200, ge=1)
    RAG_CHUNK_OVERLAP: int = Field(default=300, ge=0)
    RAG_EMBEDDING_BATCH_SIZE: int = Field(default=10, ge=1)
    RAG_COLLECTION_SUCCESSFUL: str = Field(default="successful_dialogs")
    RAG_COLLECTION_ANTI: str = Field(default="anti_examples")
    RAG_SIMILARITY_TOP_K: int = Field(default=5, ge=1)
    RAG_UPDATE_HOUR: int = Field(default=3, ge=0, le=23)
    RAG_UPDATE_MINUTE: int = Field(default=0, ge=0, le=59)
    CHROMA_PERSIST_DIR: str = Field(default="./chroma_db")

    # ═══════════════════════════════════════════════════════════════════════
    # RATE LIMITING
    # ═══════════════════════════════════════════════════════════════════════
    RATE_LIMIT_MAX_REQUESTS: int = Field(default=10, ge=1)
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, ge=1)

    # ═══════════════════════════════════════════════════════════════════════
    # LOGGING
    # ═══════════════════════════════════════════════════════════════════════
    LOGGING_LEVEL: str = Field(default="INFO")
    LOGGING_FORMAT: str = Field(default="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    LOGGING_INCLUDE_TRACE_ID: bool = Field(default=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TELEMETRY & MONITORING
    # ═══════════════════════════════════════════════════════════════════════
    TELEMETRY_ENABLED: bool = Field(default=False)
    TELEMETRY_ENDPOINT: Optional[str] = None
    HEALTH_CHECK_ENABLED: bool = Field(default=True)
    PROMETHEUS_PORT: int = Field(default=9090, ge=1, le=65535)
    GRAFANA_PORT: int = Field(default=3000, ge=1, le=65535)
    GRAFANA_ADMIN_USER: str = Field(default="admin")
    GRAFANA_ADMIN_PASSWORD: SecretStr = Field(default="admin")

    # ═══════════════════════════════════════════════════════════════════════
    # LANGSMITH (Optional)
    # ═══════════════════════════════════════════════════════════════════════
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_ENDPOINT: Optional[HttpUrl] = None
    LANGSMITH_API_KEY: Optional[SecretStr] = None
    LANGSMITH_PROJECT: str = Field(default="edms_ai_assistant")

    # ═══════════════════════════════════════════════════════════════════════
    # SAFETY
    # ═══════════════════════════════════════════════════════════════════════
    SAFETY_INPUT_FILTER: bool = Field(default=True)
    SAFETY_OUTPUT_FILTER: bool = Field(default=True)
    SAFETY_CONSTITUTIONAL_CHECK: bool = Field(default=False)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT-SPECIFIC DEFAULTS
    # ═══════════════════════════════════════════════════════════════════════
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def DEBUG(self) -> bool:
        return self.IS_DEVELOPMENT

    # ═══════════════════════════════════════════════════════════════════════
    # HELPER PROPERTIES
    # ═══════════════════════════════════════════════════════════════════════
    @property
    def LLM_ENDPOINT(self) -> str:
        """Ollama endpoint для генерации."""
        return str(self.OLLAMA_BASE_URL)

    @property
    def LLM_MODEL_NAME(self) -> str:
        """Модель для генерации."""
        return self.MODEL_NAME


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL SETTINGS INSTANCE
# ═══════════════════════════════════════════════════════════════════════════
settings = Settings()