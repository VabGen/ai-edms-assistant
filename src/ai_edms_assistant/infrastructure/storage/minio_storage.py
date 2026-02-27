# src/ai_edms_assistant/infrastructure/storage/minio_storage.py
"""MinIO object storage adapter.

Stub implementation that raises ``NotImplementedError`` on all operations.
Replace with real MinIO client when object storage is configured.

Architecture:
    Infrastructure Layer → Storage adapter
    Implements: AbstractStorage port (storage_port.py)
    Referenced by: EdmsDocumentAgent._try_create_storage()
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MinioStorage:
    """MinIO storage adapter stub.

    Currently a no-op placeholder. When MinIO is configured:
    1. Add ``minio`` to project dependencies (``pip install minio``)
    2. Add ``MINIO_ENDPOINT``, ``MINIO_ACCESS_KEY``, ``MINIO_SECRET_KEY``
       to ``shared/config/settings.py``
    3. Implement ``upload``, ``download``, ``delete`` methods.

    EdmsDocumentAgent calls ``_try_create_storage()`` which wraps this
    in try/except — so if MinIO is not configured, agent continues
    without storage support (local file tools still work).
    """

    def __init__(self) -> None:
        """Initialize MinIO stub. Logs warning that storage is not configured."""
        logger.warning(
            "minio_storage_stub_active",
            extra={
                "message": (
                    "MinioStorage is a stub. Configure MINIO_ENDPOINT "
                    "and credentials in settings to enable object storage."
                )
            },
        )

    async def upload(self, bucket: str, key: str, data: bytes) -> str:
        """Upload object to MinIO bucket.

        Args:
            bucket: Bucket name.
            key: Object key (path).
            data: Object bytes.

        Returns:
            Object URL.

        Raises:
            NotImplementedError: Always — stub not implemented.
        """
        raise NotImplementedError(
            "MinioStorage.upload is not implemented. Configure MinIO credentials."
        )

    async def download(self, bucket: str, key: str) -> bytes:
        """Download object from MinIO bucket.

        Args:
            bucket: Bucket name.
            key: Object key.

        Returns:
            Object bytes.

        Raises:
            NotImplementedError: Always — stub not implemented.
        """
        raise NotImplementedError(
            "MinioStorage.download is not implemented. Configure MinIO credentials."
        )

    async def delete(self, bucket: str, key: str) -> None:
        """Delete object from MinIO bucket.

        Args:
            bucket: Bucket name.
            key: Object key.

        Raises:
            NotImplementedError: Always — stub not implemented.
        """
        raise NotImplementedError(
            "MinioStorage.delete is not implemented. Configure MinIO credentials."
        )
