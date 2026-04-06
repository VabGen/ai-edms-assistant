# orchestrator/security.py
"""
Утилиты безопасности: декодирование JWT payload для извлечения user_id.

ВАЖНО: это НЕ валидация JWT-подписи.
Валидация подписи происходит на стороне Java EDMS API.
Здесь только извлечение user_id из payload для формирования thread_id.
"""
from __future__ import annotations

import base64
import json
import logging

logger = logging.getLogger(__name__)


def extract_user_id_from_token(user_token: str) -> str:
    """
    Декодирует JWT payload и извлекает ID пользователя ('id' или 'sub').

    Args:
        user_token: JWT-строка (с префиксом Bearer или без).

    Returns:
        Строковый user_id.

    Raises:
        ValueError: Если токен невалиден или user_id не найден.
    """
    try:
        token = user_token.strip()
        if token.startswith("Bearer "):
            token = token[7:]

        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError(
                "Неверный формат JWT: ожидается Header.Payload.Signature"
            )

        payload_encoded = parts[1]
        # Добавляем padding если нужно
        padding = 4 - (len(payload_encoded) % 4)
        if padding < 4:
            payload_encoded += "=" * padding

        payload_bytes = base64.urlsafe_b64decode(payload_encoded.encode("utf-8"))
        payload: dict = json.loads(payload_bytes)

        user_id = str(payload.get("id") or payload.get("sub") or "")
        if not user_id:
            raise ValueError("user_id ('id' или 'sub') не найден в JWT payload")

        return user_id

    except (ValueError, IndexError, json.JSONDecodeError) as exc:
        logger.error("JWT decode error: %s", exc)
        raise ValueError(f"Ошибка декодирования токена: {exc}") from exc
    except Exception as exc:
        logger.error("Unexpected JWT error: %s", exc)
        raise ValueError("Внутренняя ошибка обработки токена") from exc
