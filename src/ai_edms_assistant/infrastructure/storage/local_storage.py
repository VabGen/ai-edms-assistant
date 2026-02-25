# src/ai_edms_assistant/infrastructure/storage/local_storage.py
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .base_storage import BaseStorage

logger = logging.getLogger(__name__)


class LocalStorage(BaseStorage):
    """Local filesystem storage implementation.

    Stores files in a local directory. Useful for development and testing.
    NOT recommended for production (no redundancy, no horizontal scaling).

    Args:
        base_path: Root directory for file storage.
        create_dirs: Auto-create directories when uploading. Default True.
    """

    def __init__(self, base_path: str | Path, create_dirs: bool = True):
        """Initialize local storage.

        Args:
            base_path: Root directory for storage.
            create_dirs: Auto-create directories on upload.
        """
        self._base_path = Path(base_path)
        self._create_dirs = create_dirs

        # Create base directory if needed
        if not self._base_path.exists():
            self._base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"created_storage_directory", path=str(self._base_path))

    async def upload(
        self,
        file_path: Path,
        destination_key: str,
        bucket: str | None = None,
    ) -> str:
        """Upload file to local storage.

        Args:
            file_path: Local source file path.
            destination_key: Destination path (relative to base_path).
            bucket: Ignored (local storage has no buckets).

        Returns:
            Full local path to uploaded file.
        """
        # Validate source
        self._validate_file_path(file_path)

        # Sanitize destination
        safe_key = self._sanitize_key(destination_key)
        dest_path = self._base_path / safe_key

        # Create parent directories
        if self._create_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(file_path, dest_path)

        logger.info(
            "file_uploaded",
            source=str(file_path),
            destination=str(dest_path),
            size_bytes=dest_path.stat().st_size,
        )

        return str(dest_path)

    async def download(
        self,
        storage_key: str,
        bucket: str | None = None,
    ) -> bytes:
        """Download file from local storage.

        Args:
            storage_key: File path (relative to base_path).
            bucket: Ignored.

        Returns:
            File contents as bytes.
        """
        safe_key = self._sanitize_key(storage_key)
        file_path = self._base_path / safe_key

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {storage_key}")

        data = file_path.read_bytes()
        logger.debug(
            "file_downloaded",
            key=storage_key,
            size_bytes=len(data),
        )
        return data

    async def download_to_file(
        self,
        storage_key: str,
        local_path: Path,
        bucket: str | None = None,
    ) -> None:
        """Download file directly to local filesystem.

        Args:
            storage_key: File path (relative to base_path).
            local_path: Destination local path.
            bucket: Ignored.
        """
        safe_key = self._sanitize_key(storage_key)
        source_path = self._base_path / safe_key

        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {storage_key}")

        # Create parent directory if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(source_path, local_path)
        logger.debug(
            "file_downloaded_to_file",
            source=storage_key,
            destination=str(local_path),
        )

    async def delete(
        self,
        storage_key: str,
        bucket: str | None = None,
    ) -> None:
        """Delete file from local storage.

        Args:
            storage_key: File path (relative to base_path).
            bucket: Ignored.
        """
        safe_key = self._sanitize_key(storage_key)
        file_path = self._base_path / safe_key

        if file_path.exists():
            file_path.unlink()
            logger.info("file_deleted", key=storage_key)
        else:
            logger.warning("file_not_found_on_delete", key=storage_key)

    async def exists(
        self,
        storage_key: str,
        bucket: str | None = None,
    ) -> bool:
        """Check if file exists in local storage.

        Args:
            storage_key: File path (relative to base_path).
            bucket: Ignored.

        Returns:
            True if file exists.
        """
        safe_key = self._sanitize_key(storage_key)
        file_path = self._base_path / safe_key
        return file_path.exists() and file_path.is_file()
