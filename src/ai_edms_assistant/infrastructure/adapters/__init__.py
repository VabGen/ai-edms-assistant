# src/ai_edms_assistant/infrastructure/adapters/__init__.py
"""Infrastructure adapters for Application layer ports."""

from .document_repository_adapter import DocumentRepositoryAdapter
from .user_context_normalizer import UserContextNormalizer

__all__ = ["DocumentRepositoryAdapter", "UserContextNormalizer"]