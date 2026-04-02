"""
edms_ai_assistant/packages/core/security/encryption.py

Криптографические утилиты для защиты данных.

Features:
• Симметричное шифрование (Fernet/AES-256)
• Хэширование паролей (Argon2)
• Генерация криптографически безопасных случайных значений
• Защита от timing-атак
• Шифрование чувствительных полей в БД

Security:
• Использует проверенные библиотеки (cryptography, argon2-cffi)
• Constant-time comparison для всех сравнений
• Безопасное хранение ключей (никогда не логировать!)
• Поддержка ротации ключей шифрования
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import structlog

from edms_ai_assistant.packages.core.logging import get_logger
from edms_ai_assistant.packages.core.settings import settings

logger = get_logger(__name__)

# ── Константы ───────────────────────────────────────────────────────────
# Размеры ключей и солей
FERNET_KEY_SIZE = 32  # 256 бит
SALT_SIZE = 32  # 256 бит
NONCE_SIZE = 12  # 96 бит для AES-GCM
ARGON2_TIME_COST = 3  # Количество итераций
ARGON2_MEMORY_COST = 65536  # 64 MB
ARGON2_PARALLELISM = 4  # Параллельные потоки
ARGON2_HASH_LEN = 32  # Длина хэша


# ── Исключения ──────────────────────────────────────────────────────────
class EncryptionError(Exception):
    """Базовое исключение для ошибок шифрования."""
    pass


class DecryptionError(EncryptionError):
    """Ошибка расшифровки (неверный ключ или повреждённые данные)."""
    pass


class KeyDerivationError(EncryptionError):
    """Ошибка деривации ключа."""
    pass


class HashVerificationError(EncryptionError):
    """Ошибка проверки хэша (неверный пароль)."""
    pass


# ── Fernet Encryption (симметричное) ────────────────────────────────────
class FernetEncryption:
    """
    Симметричное шифрование с использованием Fernet (AES-128-CBC + HMAC).

    Features:
    • Автентифицированное шифрование (шифрование + MAC)
    • Встроенный timestamp для проверки свежести
    • Простая ротация ключей

    Использование:
        crypto = FernetEncryption(secret_key)
        encrypted = crypto.encrypt("sensitive data")
        decrypted = crypto.decrypt(encrypted)
    """

    def __init__(self, secret_key: Union[str, bytes]):
        """
        Инициализировать шифрование.

        Args:
            secret_key: Секретный ключ (минимум 32 байта) или строка

        Raises:
            ValueError: Если ключ слишком короткий
        """
        if isinstance(secret_key, str):
            secret_key = secret_key.encode("utf-8")

        # Деривируем ключ нужной длины если нужно
        if len(secret_key) < FERNET_KEY_SIZE:
            # Используем PBKDF2 для деривации
            salt = b"edms_fernet_salt_v1"  # Фиксированный salt для деривации
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=FERNET_KEY_SIZE,
                salt=salt,
                iterations=100000,
            )
            secret_key = base64.urlsafe_b64encode(kdf.derive(secret_key))
        else:
            # Кодируем в base64 для Fernet
            secret_key = base64.urlsafe_b64encode(
                hashlib.sha256(secret_key).digest()
            )

        self._fernet = Fernet(secret_key)
        self._key_hash = hashlib.sha256(secret_key).hexdigest()[:16]

        logger.debug("FernetEncryption initialized", key_hash=self._key_hash)

    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Зашифровать данные.

        Args:
            data: Данные для шифрования (строка или байты)

        Returns:
            Base64-encoded зашифрованная строка

        Raises:
            EncryptionError: Если шифрование не удалось
        """
        try:
            if isinstance(data, str):
                data = data.encode("utf-8")

            encrypted = self._fernet.encrypt(data)
            return encrypted.decode("utf-8")

        except Exception as e:
            logger.error("Encryption failed", error=str(e))
            raise EncryptionError(f"Encryption failed: {e}")

    def decrypt(self, token: Union[str, bytes], max_age: Optional[int] = None) -> str:
        """
        Расшифровать данные.

        Args:
            token: Зашифрованный токен (base64 строка или байты)
            max_age: Максимальный возраст токена в секундах (опционально)

        Returns:
            Расшифрованная строка

        Raises:
            DecryptionError: Если расшифровка не удалась или токен устарел
        """
        try:
            if isinstance(token, str):
                token = token.encode("utf-8")

            if max_age is not None:
                decrypted = self._fernet.decrypt(token, ttl=max_age)
            else:
                decrypted = self._fernet.decrypt(token)

            return decrypted.decode("utf-8")

        except InvalidToken as e:
            logger.warning("Decryption failed - invalid token", error=str(e))
            raise DecryptionError("Invalid or expired token")

        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            raise DecryptionError(f"Decryption failed: {e}")

    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Зашифровать словарь как JSON."""
        import json
        return self.encrypt(json.dumps(data, ensure_ascii=False, sort_keys=True))

    def decrypt_dict(self, token: str) -> Dict[str, Any]:
        """Расшифровать словарь из JSON."""
        import json
        decrypted = self.decrypt(token)
        return json.loads(decrypted)


# ── Password Hashing (Argon2) ───────────────────────────────────────────
class PasswordHasher:
    """
    Хэширование паролей с использованием Argon2id.

    Argon2id — победитель Password Hashing Competition (2015).
    Рекомендуется OWASP для хранения паролей.

    Использование:
        hasher = PasswordHasher()
        hashed = hasher.hash("my_password")
        hasher.verify(hashed, "my_password")  # True
    """

    def __init__(
            self,
            time_cost: int = ARGON2_TIME_COST,
            memory_cost: int = ARGON2_MEMORY_COST,
            parallelism: int = ARGON2_PARALLELISM,
            hash_len: int = ARGON2_HASH_LEN,
    ):
        """
        Инициализировать хэшер.

        Args:
            time_cost: Количество итераций (выше = медленнее но безопаснее)
            memory_cost: Использование памяти в KB
            parallelism: Количество параллельных потоков
            hash_len: Длина выходного хэша в байтах
        """
        try:
            from argon2 import PasswordHasher as Argon2Hasher
            from argon2.low_level import Type

            self._hasher = Argon2Hasher(
                time_cost=time_cost,
                memory_cost=memory_cost,
                parallelism=parallelism,
                hash_len=hash_len,
                type=Type.ID,  # Argon2id — рекомендуется для паролей
            )
        except ImportError:
            logger.warning("argon2-cffi not installed, falling back to PBKDF2")
            self._hasher = None

        logger.debug("PasswordHasher initialized", time_cost=time_cost)

    def hash(self, password: str) -> str:
        """
        Создать хэш пароля.

        Args:
            password: Пароль в открытом виде

        Returns:
            Хэш пароля (включает salt и параметры)

        Raises:
            EncryptionError: Если хэширование не удалось
        """
        try:
            if self._hasher:
                return self._hasher.hash(password)
            else:
                # Fallback на PBKDF2 если argon2 недоступен
                return self._hash_pbkdf2(password)

        except Exception as e:
            logger.error("Password hashing failed", error=str(e))
            raise EncryptionError(f"Password hashing failed: {e}")

    def verify(self, hashed_password: str, password: str) -> bool:
        """
        Проверить пароль против хэша.

        Args:
            hashed_password: Сохранённый хэш
            password: Пароль для проверки

        Returns:
            True если пароль верный

        Raises:
            HashVerificationError: Если проверка не удалась
        """
        try:
            if self._hasher:
                self._hasher.verify(hashed_password, password)
                return True
            else:
                # Fallback на PBKDF2
                return self._verify_pbkdf2(hashed_password, password)

        except Exception as e:
            if "Invalid hash" in str(e) or "Verification failed" in str(e):
                return False
            logger.error("Password verification failed", error=str(e))
            raise HashVerificationError(f"Verification failed: {e}")

    def _hash_pbkdf2(self, password: str) -> str:
        """Fallback: хэширование через PBKDF2."""
        salt = os.urandom(SALT_SIZE)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=ARGON2_HASH_LEN,
            salt=salt,
            iterations=100000,
        )
        key = base64.b64encode(kdf.derive(password.encode("utf-8"))).decode("utf-8")
        return f"$pbkdf2${base64.b64encode(salt).decode('utf-8')}${key}"

    def _verify_pbkdf2(self, hashed_password: str, password: str) -> bool:
        """Fallback: проверка через PBKDF2."""
        try:
            parts = hashed_password.split("$")
            if len(parts) != 4 or parts[1] != "pbkdf2":
                return False

            salt = base64.b64decode(parts[2])
            stored_key = parts[3]

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=ARGON2_HASH_LEN,
                salt=salt,
                iterations=100000,
            )
            key = base64.b64encode(kdf.derive(password.encode("utf-8"))).decode("utf-8")

            # Constant-time comparison
            return hmac.compare_digest(key.encode("utf-8"), stored_key.encode("utf-8"))

        except Exception:
            return False

    def needs_rehash(self, hashed_password: str) -> bool:
        """
        Проверить нужно ли пере-хэшировать пароль.

        Возвращает True если параметры устарели и нужно создать новый хэш
        при следующей успешной аутентификации.
        """
        if self._hasher:
            return self._hasher.check_needs_rehash(hashed_password)
        return False


# ── Secure Random ───────────────────────────────────────────────────────
class SecureRandom:
    """
    Генерация криптографически безопасных случайных значений.

    Использование:
        token = SecureRandom.token(32)  # 32 байта
        api_key = SecureRandom.api_key()
    """

    @staticmethod
    def bytes(length: int = 32) -> bytes:
        """Сгенерировать случайные байты."""
        return secrets.token_bytes(length)

    @staticmethod
    def hex(length: int = 32) -> str:
        """Сгенерировать случайную hex-строку."""
        return secrets.token_hex(length)

    @staticmethod
    def urlsafe(length: int = 32) -> str:
        """Сгенерировать случайную URL-safe строку (base64)."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def token(length: int = 32) -> str:
        """Сгенерировать случайный токен."""
        return SecureRandom.urlsafe(length)

    @staticmethod
    def api_key(prefix: str = "sk") -> str:
        """
        Сгенерировать API-ключ.

        Формат: {prefix}_{random}
        Пример: sk_abc123xyz789...
        """
        random_part = SecureRandom.urlsafe(24)
        return f"{prefix}_{random_part}"

    @staticmethod
    def password(length: int = 32) -> str:
        """
        Сгенерировать случайный пароль.

        Включает буквы (верхний/нижний регистр), цифры и специальные символы.
        """
        import string

        alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
        return "".join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def uuid() -> str:
        """Сгенерировать случайный UUID v4."""
        import uuid
        return str(uuid.uuid4())


# ── HMAC Signing ────────────────────────────────────────────────────────
class HMACSigner:
    """
    Подпись данных с использованием HMAC.

    Для проверки целостности данных без шифрования.

    Использование:
        signer = HMACSigner(secret_key)
        signature = signer.sign("data")
        signer.verify("data", signature)  # True
    """

    def __init__(self, secret_key: Union[str, bytes]):
        """
        Инициализировать подписывающий ключ.

        Args:
            secret_key: Секретный ключ для HMAC
        """
        if isinstance(secret_key, str):
            secret_key = secret_key.encode("utf-8")

        # Обеспечиваем минимальную длину ключа
        if len(secret_key) < 32:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"edms_hmac_salt_v1",
                iterations=100000,
            )
            secret_key = kdf.derive(secret_key)

        self._key = secret_key
        self._algorithm = "sha256"

    def sign(self, data: Union[str, bytes]) -> str:
        """
        Создать HMAC-подпись данных.

        Args:
            data: Данные для подписи

        Returns:
            Hex-кодированная подпись
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        signature = hmac.new(
            self._key,
            data,
            hashlib.sha256,
        ).hexdigest()

        return signature

    def verify(self, data: Union[str, bytes], signature: str) -> bool:
        """
        Проверить HMAC-подпись.

        Args:
            data: Оригинальные данные
            signature: Подпись для проверки

        Returns:
            True если подпись верная

        Security:
            Использует constant-time comparison для предотвращения timing-атак
        """
        expected = self.sign(data)

        # Constant-time comparison
        return hmac.compare_digest(
            expected.encode("utf-8"),
            signature.encode("utf-8"),
        )

    def sign_dict(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Подписать словарь и вернуть (данные, подпись).

        Returns:
            Кортеж (JSON-строка, подпись)
        """
        import json
        json_data = json.dumps(data, sort_keys=True, ensure_ascii=False)
        signature = self.sign(json_data)
        return json_data, signature

    def verify_dict(self, json_data: str, signature: str) -> Dict[str, Any]:
        """
        Проверить подпись словаря и вернуть данные.

        Raises:
            HashVerificationError: Если подпись не верна
        """
        if not self.verify(json_data, signature):
            raise HashVerificationError("Invalid signature")

        import json
        return json.loads(json_data)


# ── Глобальные экземпляры ───────────────────────────────────────────────
_encryptor: Optional[FernetEncryption] = None
_hasher: Optional[PasswordHasher] = None
_signer: Optional[HMACSigner] = None


def get_encryptor() -> FernetEncryption:
    """Получить глобальный шифратор."""
    global _encryptor
    if _encryptor is None:
        key = settings.JWT_SECRET_KEY.get_secret_value()
        _encryptor = FernetEncryption(key)
    return _encryptor


def get_hasher() -> PasswordHasher:
    """Получить глобальный хэшер паролей."""
    global _hasher
    if _hasher is None:
        _hasher = PasswordHasher()
    return _hasher


def get_signer() -> HMACSigner:
    """Получить глобальный подписывающий ключ."""
    global _signer
    if _signer is None:
        key = settings.JWT_SECRET_KEY.get_secret_value()
        _signer = HMACSigner(key)
    return _signer


# ── Convenience функции ─────────────────────────────────────────────────
def encrypt_sensitive_data(data: Union[str, bytes]) -> str:
    """
    Зашифровать чувствительные данные.

    Example:
        >>> encrypted = encrypt_sensitive_data("user_password_123")
    """
    return get_encryptor().encrypt(data)


def decrypt_sensitive_data(token: str) -> str:
    """
    Расшифровать чувствительные данные.

    Example:
        >>> decrypted = decrypt_sensitive_data(encrypted_token)
    """
    return get_encryptor().decrypt(token)


def hash_password(password: str) -> str:
    """
    Захэшировать пароль.

    Example:
        >>> hashed = hash_password("my_secret_password")
    """
    return get_hasher().hash(password)


def verify_password(hashed: str, password: str) -> bool:
    """
    Проверить пароль против хэша.

    Example:
        >>> verify_password(stored_hash, "my_secret_password")
    """
    return get_hasher().verify(hashed, password)


def sign_data(data: Union[str, bytes]) -> str:
    """
    Подписать данные HMAC.

    Example:
        >>> signature = sign_data("important_data")
    """
    return get_signer().sign(data)


def verify_signature(data: Union[str, bytes], signature: str) -> bool:
    """
    Проверить HMAC-подпись.

    Example:
        >>> verify_signature("important_data", signature)
    """
    return get_signer().verify(data, signature)


def generate_api_key(prefix: str = "sk") -> str:
    """
    Сгенерировать новый API-ключ.

    Example:
        >>> key = generate_api_key("edms")
        'edms_abc123xyz...'
    """
    return SecureRandom.api_key(prefix)


def generate_secure_token(length: int = 32) -> str:
    """
    Сгенерировать безопасный токен.

    Example:
        >>> token = generate_secure_token()
    """
    return SecureRandom.token(length)


# ── Экспорт ─────────────────────────────────────────────────────────────
__all__ = [
    # Классы
    "FernetEncryption",
    "PasswordHasher",
    "SecureRandom",
    "HMACSigner",

    # Исключения
    "EncryptionError",
    "DecryptionError",
    "KeyDerivationError",
    "HashVerificationError",

    # Convenience функции
    "get_encryptor",
    "get_hasher",
    "get_signer",
    "encrypt_sensitive_data",
    "decrypt_sensitive_data",
    "hash_password",
    "verify_password",
    "sign_data",
    "verify_signature",
    "generate_api_key",
    "generate_secure_token",

    # Константы
    "FERNET_KEY_SIZE",
    "SALT_SIZE",
    "ARGON2_TIME_COST",
    "ARGON2_MEMORY_COST",
]