# src/ai_edms_assistant/shared/config/__init__.py
"""Configuration module for EDMS AI Assistant.

Provides:
    - settings: Global settings singleton loaded from .env
    - configure_logging: Function to initialize structlog

Example:
    >>> from ai_edms_assistant.shared.config import settings, configure_logging
    >>> configure_logging()
    >>> print(settings.LLM_ENDPOINT)
"""

from .logging_config import configure_logging
from .settings import Settings, settings

__all__ = [
    "settings",
    "Settings",
    "configure_logging",
]
