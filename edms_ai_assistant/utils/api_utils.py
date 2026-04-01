# edms_ai_assistant/utils/api_utils.py
import json
import logging

import httpx

logger = logging.getLogger(__name__)


def prepare_auth_headers(token: str) -> dict[str, str]:
    """Создает стандартные заголовки для EDMS API."""
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def handle_api_error(response: httpx.Response, request_info: str):
    """
    Проверяет статус ответа и вызывает исключение, если обнаружена ошибка (>= 400).
    Логгирует детали ошибки.
    """
    if response.is_error:
        try:
            error_details = response.json()
        except (json.JSONDecodeError, AttributeError):
            error_details = {"text": response.text[:200]}

        status_code = response.status_code

        logger.error(
            f"API Error [{status_code}] for {request_info}. Details: {error_details}"
        )
        response.raise_for_status()
