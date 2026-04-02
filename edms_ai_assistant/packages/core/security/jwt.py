"""
edms_ai_assistant/packages/core/security/jwt.py

Утилиты для работы с JWT-токенами (авторизация/аутентификация).

Security features:
• Валидация подписи и expiration
• Извлечение payload с типизацией
• Защита от timing-атак (constant-time comparison)
• Поддержка refresh-токенов (опционально)

⚠️ Никогда не логируйте полный токен или секретный ключ!
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, TypedDict, Literal

import jwt
from pydantic import BaseModel, Field, field_validator

# ── Типы данных ─────────────────────────────────────────────────────────
TokenPurpose = Literal["access", "refresh", "api_key"]


class TokenPayload(TypedDict, total=False):
    """Структура полезной нагрузки токена."""
    sub: str  # user_id или идентификатор сущности
    role: str  # user, admin, service
    permissions: list[str]  # список прав
    purpose: TokenPurpose  # тип токена
    iat: int  # issued at (timestamp)
    exp: int  # expiration (timestamp)
    jti: str  # unique token ID (для revocation)
    session_id: Optional[str]  # привязка к сессии


class TokenData(BaseModel):
    """Типизированные данные из валидного токена."""
    user_id: str = Field(..., alias="sub", description="Идентификатор пользователя")
    role: str = Field(default="user", description="Роль пользователя")
    permissions: list[str] = Field(default_factory=list)
    purpose: TokenPurpose = Field(default="access")
    issued_at: datetime
    expires_at: datetime
    token_id: Optional[str] = Field(None, alias="jti")
    session_id: Optional[str] = None

    @field_validator("issued_at", "expires_at", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> datetime:
        """Конвертировать Unix timestamp в datetime."""
        if isinstance(v, datetime):
            return v
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        raise ValueError(f"Expected timestamp, got {type(v)}")

    @property
    def is_expired(self) -> bool:
        """Проверить истёк ли токен."""
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def remaining_seconds(self) -> float:
        """Секунд до истечения токена."""
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0.0, delta.total_seconds())


# ── Исключения ──────────────────────────────────────────────────────────
class JWTError(Exception):
    """Базовое исключение для JWT-ошибок."""
    pass


class TokenExpiredError(JWTError):
    """Токен истёк."""
    pass


class InvalidSignatureError(JWTError):
    """Неверная подпись токена."""
    pass


class InvalidTokenError(JWTError):
    """Токен невалиден (формат, структура)."""
    pass


# ── Основные функции ────────────────────────────────────────────────────
def create_jwt_token(
        user_id: str,
        secret_key: str,
        *,
        role: str = "user",
        permissions: Optional[list[str]] = None,
        purpose: TokenPurpose = "access",
        expires_minutes: int = 60,
        additional_claims: Optional[dict[str, Any]] = None,
        algorithm: str = "HS256",
) -> str:
    """
    Создать подписанный JWT-токен.

    Args:
        user_id: Идентификатор пользователя (будет в поле 'sub')
        secret_key: Секретный ключ для подписи (хранить в env!)
        role: Роль пользователя (user/admin/service)
        permissions: Список прав доступа
        purpose: Тип токена (access/refresh/api_key)
        expires_minutes: Время жизни токена в минутах
        additional_claims: Дополнительные кастомные поля
        algorithm: Алгоритм подписи (HS256/RS256)

    Returns:
        Подписанный JWT-токен как строка

    Raises:
        ValueError: Если secret_key пустой или невалидный
    """
    if not secret_key or len(secret_key) < 32:
        raise ValueError("secret_key must be at least 32 characters for security")

    now = datetime.now(timezone.utc)

    payload: dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "permissions": permissions or [],
        "purpose": purpose,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
        "jti": f"{user_id}:{now.timestamp()}:{purpose}",  # Уникальный ID
    }

    if additional_claims:
        # Защита от перезаписи системных полей
        reserved = {"sub", "role", "iat", "exp", "jti", "purpose"}
        for key, value in additional_claims.items():
            if key not in reserved:
                payload[key] = value

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def verify_jwt_token(
        token: str,
        secret_key: str,
        *,
        algorithm: str = "HS256",
        expected_purpose: Optional[TokenPurpose] = None,
        leeway_seconds: int = 0,
) -> TokenData:
    """
    Верифицировать и декодировать JWT-токен.

    Args:
        token: JWT-токен из заголовка Authorization
        secret_key: Секретный ключ для проверки подписи
        algorithm: Ожидаемый алгоритм подписи
        expected_purpose: Ожидаемый тип токена (если нужно проверить)
        leeway_seconds: Допустимый люфт времени (для рассинхронизации часов)

    Returns:
        TokenData с типизированными данными токена

    Raises:
        TokenExpiredError: Если токен истёк
        InvalidSignatureError: Если подпись не совпадает
        InvalidTokenError: Если токен невалиден по другим причинам
    """
    if not token or not secret_key:
        raise InvalidTokenError("Token and secret_key are required")

    try:
        # Decode с автоматической проверкой exp и подписи
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[algorithm],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "require": ["sub", "iat", "exp"],  # Обязательные поля
            },
            leeway=timedelta(seconds=leeway_seconds),
        )

        # Проверка purpose если указано
        if expected_purpose and payload.get("purpose") != expected_purpose:
            raise InvalidTokenError(
                f"Token purpose mismatch: expected {expected_purpose}, got {payload.get('purpose')}"
            )

        # Конвертация в типизированную модель
        return TokenData.model_validate(payload)

    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Token has expired")
    except jwt.InvalidSignatureError:
        raise InvalidSignatureError("Token signature is invalid")
    except jwt.DecodeError as e:
        raise InvalidTokenError(f"Failed to decode token: {e}")
    except jwt.InvalidTokenError as e:
        raise InvalidTokenError(f"Invalid token: {e}")


def refresh_jwt_token(
        refresh_token: str,
        secret_key: str,
        *,
        new_expires_minutes: int = 60,
        algorithm: str = "HS256",
) -> str:
    """
    Обновить access-токен используя refresh-токен.

    Безопасность:
    • Проверяет что токен именно типа 'refresh'
    • Сохраняет user_id и permissions из оригинального токена
    • Генерирует новый jti для отслеживания

    Args:
        refresh_token: Валидный refresh-токен
        secret_key: Секретный ключ
        new_expires_minutes: Время жизни нового access-токена
        algorithm: Алгоритм подписи

    Returns:
        Новый access-токен

    Raises:
        InvalidTokenError: Если токен не refresh-типа или невалиден
    """
    # Верифицируем refresh-токен
    token_data = verify_jwt_token(
        refresh_token,
        secret_key,
        algorithm=algorithm,
        expected_purpose="refresh",
    )

    # Создаём новый access-токен с теми же правами
    return create_jwt_token(
        user_id=token_data.user_id,
        secret_key=secret_key,
        role=token_data.role,
        permissions=token_data.permissions,
        purpose="access",
        expires_minutes=new_expires_minutes,
        additional_claims={
            "session_id": token_data.session_id,  # Сохраняем привязку к сессии
            "refreshed_from": token_data.token_id,  # Audit: откуда обновлён
        },
        algorithm=algorithm,
    )


# ── Утилиты для FastAPI / HTTP ─────────────────────────────────────────
def extract_token_from_header(authorization_header: Optional[str]) -> Optional[str]:
    """
    Извлечь JWT-токен из заголовка Authorization.

    Поддерживает форматы:
    • "Bearer <token>"
    • "<token>" (без префикса, не рекомендуется)

    Args:
        authorization_header: Значение заголовка Authorization

    Returns:
        Токен без префикса или None если не найден
    """
    if not authorization_header:
        return None

    # Убираем лишние пробелы
    header = authorization_header.strip()

    # Поддержка Bearer scheme
    if header.lower().startswith("bearer "):
        return header[7:].strip()  # len("Bearer ") == 7

    # Fallback: вернуть как есть (если токен без префикса)
    return header if header else None


# ── Глобальные константы ────────────────────────────────────────────────
# Рекомендуемые настройки для production
DEFAULT_ALGORITHM = "HS256"
DEFAULT_ACCESS_EXPIRES_MINUTES = 60
DEFAULT_REFRESH_EXPIRES_MINUTES = 1440  # 24 часа
MIN_SECRET_KEY_LENGTH = 32

# ── Экспорт ─────────────────────────────────────────────────────────────
__all__ = [
    # Типы
    "TokenPayload",
    "TokenData",
    "TokenPurpose",

    # Исключения
    "JWTError",
    "TokenExpiredError",
    "InvalidSignatureError",
    "InvalidTokenError",

    # Основные функции
    "create_jwt_token",
    "verify_jwt_token",
    "refresh_jwt_token",

    # Утилиты
    "extract_token_from_header",

    # Константы
    "DEFAULT_ALGORITHM",
    "DEFAULT_ACCESS_EXPIRES_MINUTES",
    "DEFAULT_REFRESH_EXPIRES_MINUTES",
    "MIN_SECRET_KEY_LENGTH",
]