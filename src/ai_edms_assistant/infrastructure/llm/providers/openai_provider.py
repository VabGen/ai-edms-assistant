# src/ai_edms_assistant/infrastructure/llm/providers/openai_provider.py
"""OpenAI LLM provider with robust error handling for OpenAI-compatible APIs."""

from __future__ import annotations

import functools
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ....shared.config import settings
from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------


class LLMAPIError(RuntimeError):
    """Raised when an OpenAI-compatible API returns a non-success payload.

    Attributes:
        code: API error code (e.g. "content_filter", "rate_limit_exceeded").
        api_message: Human-readable message from the API.
        raw_response: Full raw response dict for debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        api_message: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.api_message = api_message
        self.raw_response = raw_response

    def user_friendly(self) -> str:
        """Return локализованное сообщение для отображения пользователю."""
        _CODES: dict[str, str] = {
            "content_filter": (
                "Запрос заблокирован фильтром контента. "
                "Пожалуйста, переформулируйте вопрос."
            ),
            "rate_limit_exceeded": (
                "Превышен лимит запросов к модели. Попробуйте через несколько секунд."
            ),
            "model_not_found": "Указанная модель недоступна. Проверьте настройки.",
            "invalid_api_key": "Неверный API-ключ. Проверьте конфигурацию.",
            "context_length_exceeded": (
                "Запрос слишком длинный. Сократите контекст или документ."
            ),
        }
        if self.code and self.code in _CODES:
            return _CODES[self.code]
        return f"Ошибка модели: {self.api_message or self.code or 'неизвестная ошибка'}"


# ---------------------------------------------------------------------------
# Safe ChatOpenAI wrapper
# ---------------------------------------------------------------------------


class SafeChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that intercepts null-choices error responses.

    Some OpenAI-compatible APIs (vLLM, LocalAI, proxy services) return:
        {"choices": null, "message": "...", "type": "error", "code": "..."}

    instead of raising an HTTP error. The base ChatOpenAI raises an opaque
    TypeError. This subclass catches that pattern and raises LLMAPIError
    with a structured, actionable message.
    """

    def _create_chat_result(self, response: Any, generation_info: dict | None = None):
        """Override to detect and handle null-choices error payload.

        Args:
            response: Raw API response object from httpx/openai client.
            generation_info: Optional generation metadata.

        Returns:
            ChatResult from base implementation.

        Raises:
            LLMAPIError: When response contains null ``choices`` and an
                         error payload (``type`` / ``code`` / ``message``).
        """
        raw: dict[str, Any] = {}
        if hasattr(response, "model_dump"):
            raw = response.model_dump()
        elif hasattr(response, "dict"):
            raw = response.dict()
        elif isinstance(response, dict):
            raw = response

        choices = raw.get("choices")
        if choices is None and (raw.get("type") or raw.get("code")):
            code = raw.get("code")
            api_message = raw.get("message", "")
            error_type = raw.get("type", "api_error")

            logger.error(
                "llm_api_error_response",
                extra={
                    "error_code": code,
                    "error_type": error_type,
                    "api_message": api_message,
                    "response_keys": list(raw.keys()),
                },
            )

            raise LLMAPIError(
                f"LLM API returned error: [{code}] {api_message}",
                code=code,
                api_message=api_message,
                raw_response=raw,
            )

        return super()._create_chat_result(response, generation_info)


# ---------------------------------------------------------------------------
# Internal factory (без кэша — используется провайдером и get_chat_model)
# ---------------------------------------------------------------------------


def _build_chat_model() -> SafeChatOpenAI:
    """Build SafeChatOpenAI instance from current settings.

    Логика выбора API-ключа (приоритет):
        1. OPENAI_API_KEY — если задан явно
        2. LLM_API_KEY    — для корпоративных/локальных эндпоинтов
        3. "placeholder-key" — для эндпоинтов без аутентификации

    Returns:
        Configured SafeChatOpenAI instance ready for inference.
    """
    api_key: str = settings.OPENAI_API_KEY or settings.LLM_API_KEY or "placeholder-key"

    kwargs: dict[str, Any] = {
        "model": settings.LLM_MODEL_NAME,
        "temperature": settings.LLM_TEMPERATURE,
        "openai_api_key": api_key,
        "timeout": settings.LLM_TIMEOUT,
        "max_retries": settings.LLM_MAX_RETRIES,
    }

    if settings.LLM_ENDPOINT:
        kwargs["openai_api_base"] = settings.LLM_ENDPOINT

    if settings.LLM_MAX_TOKENS:
        kwargs["max_tokens"] = settings.LLM_MAX_TOKENS

    logger.info(
        "chat_model_building",
        extra={
            "endpoint": settings.LLM_ENDPOINT,
            "model": settings.LLM_MODEL_NAME,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "has_api_key": bool(settings.OPENAI_API_KEY or settings.LLM_API_KEY),
        },
    )

    return SafeChatOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Provider (для DI / BaseLLMProvider интерфейса)
# ---------------------------------------------------------------------------


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self) -> None:
        """Initialize OpenAI provider with current settings."""
        self._model = _build_chat_model()
        logger.info(
            "openai_provider_initialized",
            extra={
                "model": self.model_name,
                "endpoint": settings.LLM_ENDPOINT,
            },
        )

    @property
    def _chat_model(self) -> BaseChatModel:
        """Returns the underlying SafeChatOpenAI instance."""
        return self._model


# ---------------------------------------------------------------------------
# Factory function (cached singleton)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Return cached SafeChatOpenAI instance built from current settings.

    Singleton — создаётся один раз при первом вызове.
    Для сброса кэша (например, в тестах): ``get_chat_model.cache_clear()``.

    Returns:
        Cached SafeChatOpenAI instance.
    """
    model = _build_chat_model()
    logger.info(
        "chat_model_initialized",
        extra={
            "model": settings.LLM_MODEL_NAME,
            "endpoint": settings.LLM_ENDPOINT,
        },
    )
    return model
