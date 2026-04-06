# packages/core/security/encryption.py
"""
Криптографические утилиты: шифрование, хэширование паролей, генерация токенов.

Экспортирует:
    encrypt(data) → str          — Fernet-шифрование
    decrypt(token) → str         — Fernet-расшифровка
    hash_password(pwd) → str     — Argon2id (fallback: PBKDF2)
    verify_password(h, p) → bool — проверка пароля
    generate_token(n) → str      — криптографически безопасный токен
    generate_api_key(prefix) → str
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import logging

logger = logging.getLogger(__name__)


def _derive_fernet_key(secret: str) -> bytes:
    """Дерайвит 32-байтный ключ из секрета через PBKDF2."""
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"edms_fernet_v1_salt",
        iterations=100_000,
    )
    raw_key = kdf.derive(secret.encode())
    return base64.urlsafe_b64encode(raw_key)


def _get_fernet(secret: str | None = None):
    from cryptography.fernet import Fernet

    if secret is None:
        try:
            from edms_ai_assistant.config import settings
            secret = settings.ANTHROPIC_API_KEY.get_secret_value()
        except Exception:
            secret = os.getenv("ANTHROPIC_API_KEY", "insecure-dev-key-32-chars-minimum!")

    key = _derive_fernet_key(secret[:64])
    return Fernet(key)


def encrypt(data: str | bytes, secret: str | None = None) -> str:
    """
    Шифрует данные с использованием Fernet (AES-128-CBC + HMAC).

    Args:
        data:   Строка или байты для шифрования.
        secret: Секретный ключ (по умолчанию из ANTHROPIC_API_KEY).

    Returns:
        Base64-encoded зашифрованная строка.
    """
    f = _get_fernet(secret)
    if isinstance(data, str):
        data = data.encode()
    return f.encrypt(data).decode()


def decrypt(token: str | bytes, secret: str | None = None) -> str:
    """
    Расшифровывает Fernet-токен.

    Args:
        token:  Зашифрованный токен.
        secret: Секретный ключ.

    Returns:
        Расшифрованная строка.

    Raises:
        ValueError: Если токен невалиден.
    """
    from cryptography.fernet import InvalidToken

    try:
        f = _get_fernet(secret)
        if isinstance(token, str):
            token = token.encode()
        return f.decrypt(token).decode()
    except InvalidToken as exc:
        raise ValueError("Невалидный или просроченный токен") from exc


def hash_password(password: str) -> str:
    """
    Хэширует пароль с Argon2id (fallback на PBKDF2-SHA256).

    Args:
        password: Пароль в открытом виде.

    Returns:
        Строковый хэш, включающий алгоритм и параметры.
    """
    try:
        from argon2 import PasswordHasher  # type: ignore[import]
        ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)
        return ph.hash(password)
    except ImportError:
        logger.debug("argon2-cffi not installed — using PBKDF2")
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000, 32)
        return f"$pbkdf2${base64.b64encode(salt).decode()}${base64.b64encode(key).decode()}"


def verify_password(hashed: str, password: str) -> bool:
    """
    Проверяет пароль против хэша.

    Args:
        hashed:   Хэш из hash_password().
        password: Пароль для проверки.

    Returns:
        True если пароль верный, False иначе.
    """
    try:
        from argon2 import PasswordHasher
        from argon2.exceptions import VerifyMismatchError

        ph = PasswordHasher()
        try:
            ph.verify(hashed, password)
            return True
        except VerifyMismatchError:
            return False
    except ImportError:
        pass

    # PBKDF2 fallback
    try:
        if not hashed.startswith("$pbkdf2$"):
            return False
        parts = hashed.split("$")
        salt = base64.b64decode(parts[2])
        stored_key = base64.b64decode(parts[3])
        derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000, 32)
        return hmac.compare_digest(derived, stored_key)
    except Exception:
        return False


def generate_token(length: int = 32) -> str:
    """Генерирует криптографически безопасный URL-safe токен."""
    return secrets.token_urlsafe(length)


def generate_api_key(prefix: str = "sk") -> str:
    """
    Генерирует API-ключ формата {prefix}_{random}.

    Пример: sk_abc123xyz789...
    """
    return f"{prefix}_{secrets.token_urlsafe(24)}"
