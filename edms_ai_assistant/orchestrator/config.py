"""config.py — Конфигурация оркестратора из переменных окружения."""
from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    # LLM
    LLM_LIGHT_URL: str = os.getenv("LLM_LIGHT_URL", "http://localhost:11434/v1")
    LLM_LIGHT_MODEL: str = os.getenv("LLM_LIGHT_MODEL", "llama3.2:3b")
    LLM_HEAVY_URL: str = os.getenv("LLM_HEAVY_URL", "http://localhost:11434/v1")
    LLM_HEAVY_MODEL: str = os.getenv("LLM_HEAVY_MODEL", "gpt-oss:120b-cloud")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "120"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # Embedding
    LLM_EMBEDDING_URL: str = os.getenv("LLM_EMBEDDING_URL", "http://localhost:11434")
    LLM_EMBEDDING_MODEL: str = os.getenv("LLM_EMBEDDING_MODEL", "nomic-embed-text")

    # MCP
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://mcp-server:8001")

    # Databases
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"postgresql+asyncpg://{os.getenv('POSTGRES_USER','edms')}:"
        f"{os.getenv('POSTGRES_PASSWORD','edms_secret')}@"
        f"{os.getenv('POSTGRES_HOST','localhost')}:"
        f"{os.getenv('POSTGRES_PORT','5432')}/"
        f"{os.getenv('POSTGRES_DB','edms_ai')}",
    )
    POSTGRES_DSN: str = os.getenv(
        "POSTGRES_DSN",
        f"postgresql://{os.getenv('POSTGRES_USER','edms')}:"
        f"{os.getenv('POSTGRES_PASSWORD','edms_secret')}@"
        f"{os.getenv('POSTGRES_HOST','localhost')}:"
        f"{os.getenv('POSTGRES_PORT','5432')}/"
        f"{os.getenv('POSTGRES_DB','edms_ai')}",
    )
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # App
    API_PORT: int = int(os.getenv("API_PORT", "8002"))
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # Agent
    COMPLEXITY_THRESHOLD: float = float(os.getenv("COMPLEXITY_THRESHOLD", "0.7"))
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    AGENT_TIMEOUT: float = float(os.getenv("AGENT_TIMEOUT", "120.0"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))

    # EDMS
    EDMS_BASE_URL: str = os.getenv("EDMS_API_URL", "http://localhost:8098")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
