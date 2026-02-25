# src/ai_edms_assistant/infrastructure/storage/__init__.py
"""Storage implementations for file management.

Provides implementations of AbstractStorage port for different backends:
- LocalStorage: Filesystem-based (dev/testing)
- S3Storage: S3-compatible object storage (production)
"""

from .base_storage import BaseStorage
from .local_storage import LocalStorage
from .s3_storage import S3Storage

__all__ = [
    "BaseStorage",
    "LocalStorage",
    "S3Storage",
]
