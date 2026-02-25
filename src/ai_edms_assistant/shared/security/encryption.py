# src/ai_edms_assistant/shared/security/encryption.py
"""
Encryption / hashing utilities.

Currently, provides HMAC-SHA256 helpers for webhook signature validation.
Extend as needed with field-level encryption when required by compliance.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets


def hmac_sha256(secret: str, payload: str | bytes) -> str:
    """
    Compute HMAC-SHA256 signature.

    Args:
        secret:  Shared secret key (UTF-8 string).
        payload: Message to sign (str or bytes).

    Returns:
        Hex-encoded HMAC digest.
    """
    key = secret.encode("utf-8")
    msg = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_hmac_sha256(secret: str, payload: str | bytes, signature: str) -> bool:
    """
    Constant-time HMAC-SHA256 verification (timing-attack safe).

    Args:
        secret:    Shared secret key.
        payload:   Original message.
        signature: Expected hex digest to compare against.

    Returns:
        True when signature matches.
    """
    expected = hmac_sha256(secret, payload)
    return hmac.compare_digest(expected, signature)


def generate_secret(nbytes: int = 32) -> str:
    """
    Generate a cryptographically secure random hex secret.

    Args:
        nbytes: Number of random bytes (default 32 → 64-char hex string).

    Returns:
        Hex-encoded random string.
    """
    return secrets.token_hex(nbytes)
