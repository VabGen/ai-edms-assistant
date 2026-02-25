# src/ai_edms_assistant/shared/config/settings.py
from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration via environment variables / .env file.

    All field names are lowercase snake_case.
    UPPERCASE @property aliases provide backward compatibility.

    Example:
        >>> from ai_edms_assistant.shared.config.settings import settings
        >>> settings.LLM_ENDPOINT
        'http://model-generative.shared.du.iba/v1'
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM — генерация ──────────────────────────────────────────────────────
    llm__generative: str = "http://model-generative.shared.du.iba/v1"
    llm__generative_model: str = "default-llm-model"
    llm__embedding: str = "http://model-embedding.shared.du.iba/v1"
    llm__embedding_model: str = "default-embedding-model"
    llm_api_key: str | None = None
    openai_api_key: str | None = None
    llm_temperature: float = 0.6
    llm_max_tokens: int | None = None
    llm_timeout: int = 120
    llm_max_retries: int = 3
    llm_request_timeout: int = 120
    llm_stream_usage: bool = True

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_timeout: int = 120
    embedding_max_retries: int = 3
    embedding_request_timeout: int = 120
    embedding_ctx_length: int = 8191
    embedding_chunk_size: int = 1000
    embedding_max_retries_per_request: int = 6

    # ── Application ───────────────────────────────────────────────────────────
    environment: str = "development"
    api_port: int = 8000
    debug: bool = True
    react_app_api_url: str = "http://localhost:8000"

    # ── EDMS Backend ──────────────────────────────────────────────────────────
    chancellor_next_base_url: str = "http://127.0.0.1:8098"
    edms_timeout: int = 120

    # ── Storage ───────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    checkpoint_db_url: str = "postgresql://postgres:1234@localhost:5432/postgres"
    sql_db_url: str = "postgresql://postgres:1234@localhost:5432/postgres"
    redis_url: str = "redis://127.0.0.1:6379/0"

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent_enable_tracing: bool = True
    agent_log_level: str = "INFO"
    agent_max_retries: int = 3

    # ── Logging ───────────────────────────────────────────────────────────────
    logging_level: str = "DEBUG"
    logging_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # ── Telemetry / OpenTelemetry ─────────────────────────────────────────────
    telemetry_enabled: bool = False
    telemetry_endpoint: str = "http://127.0.0.1:4318"

    # ── LangSmith Tracing (опционально) ───────────────────────────────────────
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str | None = None
    langsmith_project: str = "edms-ai-assistant"

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_batch_size: int = 20
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 300
    rag_embedding_batch_size: int = 10

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_production(self) -> bool:
        """True when ENVIRONMENT=production."""
        return self.environment.lower() == "production"

    def is_debug(self) -> bool:
        """True when DEBUG=true."""
        return self.debug

    # ── UPPERCASE @property aliases (backward compatibility) ──────────────────

    @property
    def LLM_ENDPOINT(self) -> str:
        return self.llm__generative

    @property
    def LLM_MODEL_NAME(self) -> str:
        return self.llm__generative_model

    @property
    def EMBEDDING_ENDPOINT(self) -> str:
        return self.llm__embedding

    @property
    def EMBEDDING_MODEL_NAME(self) -> str:
        return self.llm__embedding_model

    @property
    def LLM_API_KEY(self) -> str | None:
        return self.llm_api_key

    @property
    def OPENAI_API_KEY(self) -> str | None:
        return self.openai_api_key

    @property
    def LLM_TEMPERATURE(self) -> float:
        return self.llm_temperature

    @property
    def LLM_MAX_TOKENS(self) -> int | None:
        return self.llm_max_tokens

    @property
    def LLM_TIMEOUT(self) -> int:
        return self.llm_timeout

    @property
    def LLM_MAX_RETRIES(self) -> int:
        return self.llm_max_retries

    @property
    def LLM_REQUEST_TIMEOUT(self) -> int:
        return self.llm_request_timeout

    @property
    def LLM_STREAM_USAGE(self) -> bool:
        return self.llm_stream_usage

    @property
    def EMBEDDING_TIMEOUT(self) -> int:
        return self.embedding_timeout

    @property
    def EMBEDDING_MAX_RETRIES(self) -> int:
        return self.embedding_max_retries

    @property
    def EMBEDDING_REQUEST_TIMEOUT(self) -> int:
        return self.embedding_request_timeout

    @property
    def EMBEDDING_CTX_LENGTH(self) -> int:
        return self.embedding_ctx_length

    @property
    def EMBEDDING_CHUNK_SIZE(self) -> int:
        return self.embedding_chunk_size

    @property
    def EMBEDDING_MAX_RETRIES_PER_REQUEST(self) -> int:
        return self.embedding_max_retries_per_request

    @property
    def CHANCELLOR_NEXT_BASE_URL(self) -> str:
        return self.chancellor_next_base_url

    @property
    def EDMS_TIMEOUT(self) -> int:
        return self.edms_timeout

    @property
    def CHROMA_PERSIST_DIR(self) -> str:
        return self.chroma_persist_dir

    @property
    def CHECKPOINT_DB_URL(self) -> str:
        return self.checkpoint_db_url

    @property
    def SQL_DB_URL(self) -> str:
        return self.sql_db_url

    @property
    def API_PORT(self) -> int:
        return self.api_port

    @property
    def DEBUG(self) -> bool:
        return self.debug

    @property
    def REDIS_URL(self) -> str:
        return self.redis_url

    @property
    def AGENT_ENABLE_TRACING(self) -> bool:
        return self.agent_enable_tracing

    @property
    def AGENT_LOG_LEVEL(self) -> str:
        return self.agent_log_level

    @property
    def AGENT_MAX_RETRIES(self) -> int:
        return self.agent_max_retries

    @property
    def LOGGING_LEVEL(self) -> str:
        return self.logging_level

    @property
    def LOGGING_FORMAT(self) -> str:
        return self.logging_format

    @property
    def TELEMETRY_ENABLED(self) -> bool:
        return self.telemetry_enabled

    @property
    def TELEMETRY_ENDPOINT(self) -> str:
        return self.telemetry_endpoint

    @property
    def LANGSMITH_TRACING(self) -> bool:
        return self.langsmith_tracing

    @property
    def LANGSMITH_ENDPOINT(self) -> str:
        return self.langsmith_endpoint

    @property
    def LANGSMITH_API_KEY(self) -> str | None:
        return self.langsmith_api_key

    @property
    def LANGSMITH_PROJECT(self) -> str:
        return self.langsmith_project

    @property
    def RAG_BATCH_SIZE(self) -> int:
        return self.rag_batch_size

    @property
    def RAG_CHUNK_SIZE(self) -> int:
        return self.rag_chunk_size

    @property
    def RAG_CHUNK_OVERLAP(self) -> int:
        return self.rag_chunk_overlap

    @property
    def RAG_EMBEDDING_BATCH_SIZE(self) -> int:
        return self.rag_embedding_batch_size

    @property
    def REACT_APP_API_URL(self) -> str:
        return self.react_app_api_url

    @property
    def ENVIRONMENT(self) -> str:
        return self.environment


# Module-level singleton
settings = Settings()
