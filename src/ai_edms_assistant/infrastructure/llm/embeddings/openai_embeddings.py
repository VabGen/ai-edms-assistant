# src/ai_edms_assistant/infrastructure/llm/embeddings/openai_embeddings.py
from __future__ import annotations

import functools
import logging

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ....shared.config import settings

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Factory function for getting cached OpenAI embeddings model.

    Provides backward compatibility with old code that imports
    `get_embedding_model` from llm.py.

    Configuration via settings:
        - EMBEDDING_ENDPOINT / llm__embedding
        - EMBEDDING_MODEL_NAME / llm__embedding_model
        - LLM_API_KEY (shared with chat model)
        - EMBEDDING_CHUNK_SIZE, EMBEDDING_REQUEST_TIMEOUT, etc.

    Returns:
        Cached OpenAIEmbeddings instance.
    """
    logger.info(
        "initializing_embedding_model",
        endpoint=settings.EMBEDDING_ENDPOINT,
        model=settings.EMBEDDING_MODEL_NAME,
    )

    # Build kwargs from settings
    kwargs = {
        "model": settings.EMBEDDING_MODEL_NAME,
        "openai_api_base": settings.EMBEDDING_ENDPOINT,
        "openai_api_key": settings.LLM_API_KEY or "placeholder-key",
        "request_timeout": settings.EMBEDDING_REQUEST_TIMEOUT,
        "max_retries": settings.EMBEDDING_MAX_RETRIES,
    }

    # Optional chunk size (for batching)
    if hasattr(settings, "embedding_chunk_size"):
        kwargs["chunk_size"] = settings.embedding_chunk_size

    embedding_model = OpenAIEmbeddings(**kwargs)
    logger.info("embedding_model_initialized", model=kwargs["model"])
    return embedding_model
