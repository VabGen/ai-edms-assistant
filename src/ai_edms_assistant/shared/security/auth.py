# src/ai_edms_assistant/shared/security/auth.py
"""
JWT authentication utilities.

Migrated from edms_ai_assistant/security.py.
decode_jwt_payload() is NOT validation — it only decodes the payload base64.
Full JWT signature validation must be done at API Gateway / via PyJWT.
"""

from __future__ import annotations

import base64
import json
import structlog

logger = structlog.get_logger(__name__)


class JWTDecodeError(ValueError):
    """Raised when JWT payload cannot be decoded."""


def extract_user_id_from_token(user_token: str) -> str:
    """
    Decode JWT payload and extract user ID ('id' or 'sub' claim).

    WARNING: This does NOT validate the JWT signature, expiry, or issuer.
    Use this only for identifying the user after the token has been
    validated upstream (API Gateway, OAuth2 middleware, etc.).

    Args:
        user_token: Raw JWT string, optionally prefixed with 'Bearer '.

    Returns:
        User ID string extracted from 'id' or 'sub' claim.

    Raises:
        JWTDecodeError: When token format is invalid or user ID is missing.
    """
    try:
        token = user_token.strip()
        if token.startswith("Bearer "):
            token = token[7:]

        parts = token.split(".")
        if len(parts) != 3:
            raise JWTDecodeError(
                "Неверный формат JWT: ожидается три части (Header.Payload.Signature)."
            )

        _, payload_b64, _ = parts

        # Base64url padding
        padding = 4 - (len(payload_b64) % 4)
        if padding < 4:
            payload_b64 += "=" * padding

        payload_bytes = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        payload: dict = json.loads(payload_bytes)

        user_id = str(payload.get("id") or payload.get("sub") or "")
        if not user_id or user_id == "None":
            raise JWTDecodeError("Claim 'id' или 'sub' не найдены в JWT payload.")

        logger.debug("jwt_user_id_extracted", user_id=user_id)
        return user_id

    except JWTDecodeError:
        raise
    except (ValueError, IndexError, json.JSONDecodeError) as exc:
        logger.error("jwt_decode_failed", error=str(exc))
        raise JWTDecodeError(f"Ошибка декодирования токена: {exc}") from exc
    except Exception as exc:
        logger.error("jwt_unexpected_error", error=str(exc))
        raise JWTDecodeError("Внутренняя ошибка при обработке токена.") from exc


def decode_jwt_payload(token: str) -> dict:
    """
    Decode and return the full JWT payload dict without validation.

    Args:
        token: Raw JWT string, optionally prefixed with 'Bearer '.

    Returns:
        Decoded payload dict.

    Raises:
        JWTDecodeError: On invalid format or decode failure.
    """
    token = token.strip()
    if token.startswith("Bearer "):
        token = token[7:]

    parts = token.split(".")
    if len(parts) != 3:
        raise JWTDecodeError("Invalid JWT format")

    _, payload_b64, _ = parts
    padding = 4 - (len(payload_b64) % 4)
    if padding < 4:
        payload_b64 += "=" * padding

    try:
        return json.loads(base64.urlsafe_b64decode(payload_b64.encode()))
    except Exception as exc:
        raise JWTDecodeError(f"JWT payload decode failed: {exc}") from exc
