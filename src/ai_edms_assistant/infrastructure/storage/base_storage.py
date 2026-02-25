# src/ai_edms_assistant/infrastructure/storage/base_storage.py
from __future__ import annotations

from abc import ABC
from pathlib import Path

from ...application.ports import AbstractStorage


class BaseStorage(AbstractStorage, ABC):
    """Base implementation of AbstractStorage.

    Provides common utilities for storage implementations.
    Subclasses must implement the actual upload/download logic.
    """

    def _validate_file_path(self, file_path: Path) -> None:
        """Validate local file path exists and is readable.

        Args:
            file_path: Path to local file.

        Raises:
            FileNotFoundError: When file doesn't exist.
            PermissionError: When file is not readable.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        if not file_path.stat().st_size:
            raise ValueError(f"File is empty: {file_path}")

    def _sanitize_key(self, key: str) -> str:
        """Sanitize storage key to prevent path traversal.

        Args:
            key: Raw storage key.

        Returns:
            Sanitized key.
        """
        # Remove leading slashes and parent directory references
        key = key.lstrip("/")
        parts = []
        for part in key.split("/"):
            if part and part not in (".", ".."):
                parts.append(part)
        return "/".join(parts)

    @property
    def backend_name(self) -> str:
        """Returns storage backend identifier."""
        return self.__class__.__name__.replace("Storage", "").lower()
