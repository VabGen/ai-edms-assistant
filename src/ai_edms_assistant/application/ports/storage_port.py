# src/ai_edms_assistant/application/ports/storage_port.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AbstractStorage(ABC):
    """Port (interface) for file storage backends.

    Defines the contract for uploading and downloading files to/from
    object storage (MinIO, S3, local filesystem). Used by attachment
    processing tools.

    Implementations:
        - ``LocalStorage`` (dev / testing)
        - ``S3Storage`` (prod)
        - ``MinIOStorage`` (self-hosted prod)

    Architecture:
        - Application layer receives ``AbstractStorage`` via DI.
        - Infrastructure provides concrete storage implementations.
        - Switch storage backends by changing DI config.

    Example:
        >>> # In a file processor
        >>> class FileProcessor:
        ...     def __init__(self, storage: AbstractStorage) -> None:
        ...         self._storage = storage
        ...
        ...     async def process(self, path: str) -> bytes:
        ...         return await self._storage.download(path)
    """

    @abstractmethod
    async def upload(
        self,
        file_path: Path,
        destination_key: str,
        bucket: str | None = None,
    ) -> str:
        """Upload a file to object storage.

        Args:
            file_path: Local filesystem path to the file to upload.
            destination_key: Object key / path in the storage bucket.
            bucket: Optional bucket name. Uses default bucket when ``None``.

        Returns:
            Full storage path / URL to the uploaded file.
        """

    @abstractmethod
    async def download(
        self,
        storage_key: str,
        bucket: str | None = None,
    ) -> bytes:
        """Download a file from object storage.

        Args:
            storage_key: Object key / path in the storage bucket.
            bucket: Optional bucket name. Uses default bucket when ``None``.

        Returns:
            Raw file bytes.
        """

    @abstractmethod
    async def download_to_file(
        self,
        storage_key: str,
        local_path: Path,
        bucket: str | None = None,
    ) -> None:
        """Download a file from storage directly to the local filesystem.

        Convenience method for large files to avoid holding bytes in memory.

        Args:
            storage_key: Object key / path in the storage bucket.
            local_path: Local filesystem path where the file will be saved.
            bucket: Optional bucket name.
        """

    @abstractmethod
    async def delete(
        self,
        storage_key: str,
        bucket: str | None = None,
    ) -> None:
        """Delete a file from object storage.

        Args:
            storage_key: Object key / path in the storage bucket.
            bucket: Optional bucket name.
        """

    @abstractmethod
    async def exists(
        self,
        storage_key: str,
        bucket: str | None = None,
    ) -> bool:
        """Check whether a file exists in object storage.

        Args:
            storage_key: Object key / path to check.
            bucket: Optional bucket name.

        Returns:
            ``True`` when the file exists, ``False`` otherwise.
        """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Returns the name of the storage backend.

        Returns:
            Backend identifier, e.g. ``"local"``, ``"s3"``, ``"minio"``.
        """
