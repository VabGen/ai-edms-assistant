# edms_ai_assistant/orchestrator/security.py
import base64
import json
import logging

logger = logging.getLogger(__name__)


def extract_user_id_from_token(user_token: str) -> str:
    """
    Декодирует JWT payload для извлечения ID пользователя ('id' или 'sub').

    ВНИМАНИЕ: Это *НЕ* валидация JWT. Это только декодирование PAYLOAD.
    Валидация JWT (подпись, срок действия) должна выполняться через
    специализированные библиотеки (pyjwt) или прокси (API Gateway).
    """
    try:
        token = user_token.strip()
        if token.startswith("Bearer "):
            token = token[7:]
            logger.debug("Удален префикс 'Bearer ' из токена")

        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError(
                "Неверный формат JWT: ожидается три части (Header.Payload.Signature)."
            )

        _, payload_encoded, _ = parts

        padding_needed = 4 - (len(payload_encoded) % 4)
        if padding_needed < 4:
            payload_encoded += "=" * padding_needed

        payload_decoded = base64.urlsafe_b64decode(payload_encoded.encode("utf-8"))
        payload: dict = json.loads(payload_decoded)

        user_id_for_thread = str(payload.get("id") or payload.get("sub"))

        if not user_id_for_thread:
            raise ValueError(
                "User ID ('id' или 'sub') не найдены в полезной нагрузке JWT."
            )

        logger.debug(f"Успешно извлечен user_id: {user_id_for_thread}")
        return user_id_for_thread

    except (ValueError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Ошибка декодирования/парсинга JWT: {e}")
        raise ValueError(f"Ошибка декодирования токена: {e}")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка в декодировании JWT: {e}")
        raise ValueError("Внутренняя ошибка при обработке токена.")
