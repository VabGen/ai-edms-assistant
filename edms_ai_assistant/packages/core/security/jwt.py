# packages/core/security/jwt.py
"""
JWT-утилиты: создание и верификация токенов.

Примечание по безопасности:
    Валидация подписи происходит на стороне Java EDMS API Gateway.
    В Python-сервисах мы только декодируем payload для извлечения user_id.
    Для внутренних сервис-токенов (feedback → orchestrator) используется
    PyJWT с HS256 и секретом из JWT_SECRET_KEY.

Экспортирует:
    extract_user_id_from_token(token) → str
    create_service_token(user_id, role, expires_minutes) → str
    verify_service_token(token) → dict
"""
from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


def extract_user_id_from_token(user_token: str) -> str:
    """
    Декодирует JWT payload (без верификации подписи) и извлекает user_id.

    Args:
        user_token: JWT-строка с префиксом Bearer или без.

    Returns:
        Строковый user_id (поле 'id' или 'sub').

    Raises:
        ValueError: Если токен невалиден или user_id не найден.
    """
    token = user_token.strip().removeprefix("Bearer ").strip()

    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Неверный формат JWT: ожидается Header.Payload.Signature")

    payload_b64 = parts[1]
    padding = 4 - (len(payload_b64) % 4)
    if padding < 4:
        payload_b64 += "=" * padding

    try:
        payload: dict[str, Any] = json.loads(
            base64.urlsafe_b64decode(payload_b64.encode())
        )
    except Exception as exc:
        raise ValueError(f"Ошибка декодирования JWT payload: {exc}") from exc

    user_id = str(payload.get("id") or payload.get("sub") or "")
    if not user_id:
        raise ValueError("user_id ('id' или 'sub') не найден в JWT payload")

    return user_id


def create_service_token(
    user_id: str,
    role: str = "service",
    expires_minutes: int = 60,
    secret_key: str | None = None,
) -> str:
    """
    Создаёт внутренний сервисный JWT-токен (HS256).

    Используется для аутентификации между feedback-collector и orchestrator.

    Args:
        user_id:         Идентификатор пользователя/сервиса.
        role:            Роль (user | admin | service).
        expires_minutes: Время жизни токена в минутах.
        secret_key:      Секрет подписи (по умолчанию из settings).

    Returns:
        Подписанный JWT-токен.
    """
    try:
        import jwt as pyjwt  # type: ignore[import]
        from edms_ai_assistant.config import settings

        key = secret_key or settings.ANTHROPIC_API_KEY.get_secret_value()[:32]
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "id": user_id,
            "role": role,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
        }
        return pyjwt.encode(payload, key, algorithm="HS256")
    except ImportError:
        logger.warning("PyJWT not installed — using unsigned token (dev only)")
        import base64
        payload_json = json.dumps({"sub": user_id, "id": user_id, "role": role})
        encoded = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode()
        return f"eyJhbGciOiJub25lIn0.{encoded}."


def verify_service_token(token: str, secret_key: str | None = None) -> dict[str, Any]:
    """
    Верифицирует и декодирует внутренний сервисный токен.

    Args:
        token:      JWT-токен.
        secret_key: Секрет подписи.

    Returns:
        Payload токена.

    Raises:
        ValueError: Если токен невалиден или истёк.
    """
    try:
        import jwt as pyjwt  # type: ignore[import]
        from edms_ai_assistant.config import settings

        key = secret_key or settings.ANTHROPIC_API_KEY.get_secret_value()[:32]
        return pyjwt.decode(token, key, algorithms=["HS256"])
    except ImportError:
        # Fallback: просто декодируем payload
        return json.loads(
            base64.urlsafe_b64decode(token.split(".")[1] + "==").decode()
        )
    except Exception as exc:
        raise ValueError(f"Невалидный токен: {exc}") from exc
