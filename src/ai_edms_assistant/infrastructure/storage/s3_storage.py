# src/ai_edms_assistant/infrastructure/storage/s3_storage.py
from __future__ import annotations

import logging
from pathlib import Path

from .base_storage import BaseStorage

logger = logging.getLogger(__name__)

try:
    import aioboto3

    AIOBOTO3_AVAILABLE = True
except ImportError:
    AIOBOTO3_AVAILABLE = False
    logger.warning(
        "aioboto3_not_available",
        message="Install aioboto3 for S3 storage: pip install aioboto3",
    )


class S3Storage(BaseStorage):
    """S3-compatible object storage implementation.

    Works with AWS S3, MinIO, DigitalOcean Spaces, and other S3-compatible
    providers.

    Args:
        bucket_name: Default S3 bucket name.
        aws_access_key_id: AWS access key (or compatible).
        aws_secret_access_key: AWS secret key.
        endpoint_url: Custom S3 endpoint (for MinIO, etc.). None for AWS.
        region_name: AWS region. Default 'us-east-1'.
    """

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
    ):
        """Initialize S3 storage.

        Raises:
            ImportError: When aioboto3 is not installed.
        """
        if not AIOBOTO3_AVAILABLE:
            raise ImportError(
                "S3Storage requires aioboto3. Install: pip install aioboto3"
            )

        self._bucket_name = bucket_name
        self._endpoint_url = endpoint_url
        self._session = aioboto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

        logger.info(
            "s3_storage_initialized",
            bucket=bucket_name,
            endpoint=endpoint_url or "AWS S3",
        )

    def _get_bucket(self, bucket: str | None) -> str:
        """Get bucket name (use default if not specified).

        Args:
            bucket: Optional bucket override.

        Returns:
            Bucket name to use.
        """
        return bucket or self._bucket_name

    async def upload(
        self,
        file_path: Path,
        destination_key: str,
        bucket: str | None = None,
    ) -> str:
        """Upload file to S3.

        Args:
            file_path: Local source file path.
            destination_key: S3 object key.
            bucket: Optional bucket override.

        Returns:
            S3 URI (s3://bucket/key).
        """
        self._validate_file_path(file_path)
        safe_key = self._sanitize_key(destination_key)
        target_bucket = self._get_bucket(bucket)

        async with self._session.client("s3", endpoint_url=self._endpoint_url) as s3:
            await s3.upload_file(
                Filename=str(file_path),
                Bucket=target_bucket,
                Key=safe_key,
            )

        s3_uri = f"s3://{target_bucket}/{safe_key}"
        logger.info(
            "file_uploaded_to_s3",
            source=str(file_path),
            s3_uri=s3_uri,
            size_bytes=file_path.stat().st_size,
        )
        return s3_uri

    async def download(
        self,
        storage_key: str,
        bucket: str | None = None,
    ) -> bytes:
        """Download file from S3.

        Args:
            storage_key: S3 object key.
            bucket: Optional bucket override.

        Returns:
            File contents as bytes.
        """
        safe_key = self._sanitize_key(storage_key)
        target_bucket = self._get_bucket(bucket)

        async with self._session.client("s3", endpoint_url=self._endpoint_url) as s3:
            response = await s3.get_object(Bucket=target_bucket, Key=safe_key)
            data = await response["Body"].read()

        logger.debug(
            "file_downloaded_from_s3",
            key=storage_key,
            bucket=target_bucket,
            size_bytes=len(data),
        )
        return data

    async def download_to_file(
        self,
        storage_key: str,
        local_path: Path,
        bucket: str | None = None,
    ) -> None:
        """Download file from S3 to local filesystem.

        Args:
            storage_key: S3 object key.
            local_path: Destination local path.
            bucket: Optional bucket override.
        """
        safe_key = self._sanitize_key(storage_key)
        target_bucket = self._get_bucket(bucket)

        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        async with self._session.client("s3", endpoint_url=self._endpoint_url) as s3:
            await s3.download_file(
                Bucket=target_bucket,
                Key=safe_key,
                Filename=str(local_path),
            )

        logger.debug(
            "file_downloaded_to_file_from_s3",
            key=storage_key,
            destination=str(local_path),
        )

    async def delete(
        self,
        storage_key: str,
        bucket: str | None = None,
    ) -> None:
        """Delete file from S3.

        Args:
            storage_key: S3 object key.
            bucket: Optional bucket override.
        """
        safe_key = self._sanitize_key(storage_key)
        target_bucket = self._get_bucket(bucket)

        async with self._session.client("s3", endpoint_url=self._endpoint_url) as s3:
            await s3.delete_object(Bucket=target_bucket, Key=safe_key)

        logger.info(
            "file_deleted_from_s3",
            key=storage_key,
            bucket=target_bucket,
        )

    async def exists(
        self,
        storage_key: str,
        bucket: str | None = None,
    ) -> bool:
        """Check if file exists in S3.

        Args:
            storage_key: S3 object key.
            bucket: Optional bucket override.

        Returns:
            True if object exists.
        """
        safe_key = self._sanitize_key(storage_key)
        target_bucket = self._get_bucket(bucket)

        try:
            async with self._session.client(
                "s3", endpoint_url=self._endpoint_url
            ) as s3:
                await s3.head_object(Bucket=target_bucket, Key=safe_key)
            return True
        except Exception:
            return False

    @property
    def backend_name(self) -> str:
        """Returns 's3' or 'minio' based on endpoint."""
        if self._endpoint_url and "minio" in self._endpoint_url.lower():
            return "minio"
        return "s3"
