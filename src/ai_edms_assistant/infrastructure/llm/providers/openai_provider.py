# src/ai_edms_assistant/infrastructure/llm/providers/openai_provider.py
from __future__ import annotations

import functools
import logging

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ....shared.config import settings
from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation.

    Uses LangChain's ChatOpenAI under the hood. Supports both OpenAI API
    and OpenAI-compatible endpoints (vLLM, LocalAI, etc.).

    Configuration via settings:
        - LLM_ENDPOINT / llm__generative
        - LLM_MODEL_NAME / llm__generative_model
        - OPENAI_API_KEY / LLM_API_KEY
        - LLM_TEMPERATURE, LLM_MAX_TOKENS, etc.
    """

    def __init__(self):
        """Initialize OpenAI provider with settings."""
        self._model = self._create_chat_model()
        logger.info(
            "openai_provider_initialized",
            model=self.model_name,
            endpoint=settings.LLM_ENDPOINT,
        )

    @property
    def _chat_model(self) -> BaseChatModel:
        """Returns the underlying ChatOpenAI instance."""
        return self._model

    def _create_chat_model(self) -> ChatOpenAI:
        """Create ChatOpenAI instance from settings.

        Returns:
            Configured ChatOpenAI instance.
        """
        # Build kwargs from settings
        kwargs = {
            "model": settings.LLM_MODEL_NAME,
            "temperature": settings.LLM_TEMPERATURE,
            "timeout": settings.LLM_TIMEOUT,
            "max_retries": settings.LLM_MAX_RETRIES,
        }

        # Add optional parameters
        if settings.LLM_MAX_TOKENS:
            kwargs["max_tokens"] = settings.LLM_MAX_TOKENS

        # OpenAI API key
        if settings.OPENAI_API_KEY:
            kwargs["openai_api_key"] = settings.OPENAI_API_KEY
        elif settings.LLM_API_KEY:
            kwargs["openai_api_key"] = settings.LLM_API_KEY
        else:
            kwargs["openai_api_key"] = "placeholder-key"

        # Custom endpoint (for vLLM, LocalAI, etc.)
        if settings.LLM_ENDPOINT:
            kwargs["openai_api_base"] = settings.LLM_ENDPOINT

        # Streaming
        if hasattr(settings, "llm_stream_usage"):
            kwargs["streaming"] = settings.llm_stream_usage

        return ChatOpenAI(**kwargs)


@functools.lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Factory function for getting cached ChatOpenAI instance.

    This provides backward compatibility with old code that imports
    `get_chat_model` from llm.py.

    Returns:
        Cached ChatOpenAI instance.
    """
    logger.info(
        "initializing_chat_model",
        endpoint=settings.LLM_ENDPOINT,
        model=settings.LLM_MODEL_NAME,
    )

    # Use custom configuration if available
    if settings.OPENAI_API_KEY and "proxyapi.ru" in settings.LLM_ENDPOINT:
        # Special configuration for proxy
        kwargs = {
            "model": "gpt-4o-mini",
            "temperature": 0.6,
            "openai_api_base": "https://api.proxyapi.ru/openai/v1",
            "openai_api_key": settings.OPENAI_API_KEY,
            "max_retries": 5,
            "timeout": 90,
            "streaming": True,
            "max_tokens": 4096,
            "seed": 42,
            "top_p": 0.0000001,
        }
    else:
        # Standard configuration from settings
        kwargs = {
            "model": settings.LLM_MODEL_NAME,
            "temperature": settings.LLM_TEMPERATURE,
            "openai_api_base": settings.LLM_ENDPOINT,
            "openai_api_key": settings.OPENAI_API_KEY
            or settings.LLM_API_KEY
            or "placeholder-key",
            "timeout": settings.LLM_TIMEOUT,
            "max_retries": settings.LLM_MAX_RETRIES,
        }

        if settings.LLM_MAX_TOKENS:
            kwargs["max_tokens"] = settings.LLM_MAX_TOKENS

    model = ChatOpenAI(**kwargs)
    logger.info(f"chat_model_initialized", model=kwargs["model"])
    return model
