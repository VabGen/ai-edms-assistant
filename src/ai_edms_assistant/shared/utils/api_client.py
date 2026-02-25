# src/ai_edms_assistant/shared/utils/api_client.py
"""
HTTP utility functions for EDMS API clients.

Migrated and cleaned from edms_ai_assistant/utils/api_utils.py.
Removed FastAPI dependency (HTTPException) — pure httpx utilities.
"""

from __future__ import annotations

import json
import uuid
import structlog
import httpx
from typing import Dict

logger = structlog.get_logger(__name__)


def prepare_auth_headers(token: str) -> Dict[str, str]:
    """
    Build Authorization header dict for EDMS API requests.

    Args:
        token: JWT bearer token (with or without 'Bearer ' prefix).

    Returns:
        {'Authorization': 'Bearer <token>', 'Content-Type': 'application/json'}
    """
    clean = token.strip()
    if not clean.startswith("Bearer "):
        clean = f"Bearer {clean}"
    return {
        "Authorization": clean,
        "Content-Type": "application/json",
    }


async def handle_api_error(response: httpx.Response, request_info: str = "") -> None:
    """
    Log and raise on HTTP error responses (4xx / 5xx).

    Args:
        response:     httpx.Response to check.
        request_info: Human-readable request description for log context.

    Raises:
        httpx.HTTPStatusError: On any 4xx / 5xx status code.
    """
    if not response.is_error:
        return

    try:
        details = response.json()
    except (json.JSONDecodeError, Exception):
        details = response.text[:300]

    logger.error(
        "api_error",
        status=response.status_code,
        request=request_info,
        details=details,
    )
    response.raise_for_status()


def validate_uuid(value: str | None) -> uuid.UUID | None:
    """
    Parse a string into a UUID.

    Args:
        value: UUID string or None.

    Returns:
        uuid.UUID instance or None when value is falsy.

    Raises:
        ValueError: When value is a non-empty invalid UUID string.
    """
    if not value:
        return None
    return uuid.UUID(value)
