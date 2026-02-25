# src/ai_edms_assistant/shared/logging/logger.py
"""
Structured JSON logger factory.

Provides get_logger() as the single import point for all modules.
Uses structlog configured via shared/config/logging_config.py.

Usage::

    from ai_edms_assistant.shared.logging.logger import get_logger

    logger = get_logger(__name__)
    logger.info("document_loaded", doc_id=str(doc_id), status=status)
"""

from __future__ import annotations

import structlog


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a bound structlog logger for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        Bound structlog logger with module name context.
    """
    return structlog.get_logger(name)
