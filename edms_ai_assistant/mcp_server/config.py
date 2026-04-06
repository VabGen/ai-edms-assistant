# mcp-server/config.py
"""
Конфигурация MCP-сервера из переменных окружения.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPSettings(BaseSettings):
    """Настройки MCP-сервера."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # EDMS API
    EDMS_BASE_URL: str = Field(default="http://localhost:8098")
    EDMS_TIMEOUT: int = Field(default=120, ge=10)

    # MCP Server
    MCP_HOST: str = Field(default="0.0.0.0")
    MCP_PORT: int = Field(default=8001, ge=1, le=65535)
    LOG_LEVEL: str = Field(default="INFO")

    # LLM (для find_best_subject и appeal_extraction)
    LLM_GENERATIVE_URL: str = Field(default="http://127.0.0.1:11434")
    LLM_GENERATIVE_MODEL: str = Field(default="gpt-oss:120b-cloud")
    LLM_TEMPERATURE: float = Field(default=0.0)
    LLM_MAX_TOKENS: int = Field(default=2048)
    LLM_TIMEOUT: int = Field(default=120)

    @property
    def CHANCELLOR_NEXT_BASE_URL(self) -> str:
        return self.EDMS_BASE_URL


@lru_cache(maxsize=1)
def get_settings() -> MCPSettings:
    return MCPSettings()


settings: MCPSettings = get_settings()