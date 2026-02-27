# src/ai_edms_assistant/infrastructure/edms_api/mappers/attachment_mapper.py
"""AttachmentDto dict → domain Attachment mapper.

Maps EDMS API attachment representations to domain entities with complete
field coverage. Handles both AttachmentDocumentDto (wrapper) and standalone
AttachmentDto responses.

Architecture:
    Infrastructure Layer → Domain Layer
    Raw API dict → Immutable domain entity
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from ....domain.entities.attachment import (
    Attachment,
    AttachmentDocumentType,
    AttachmentSignature,
    AttachmentType,
    ContentType,
    Signature,
)

logger = logging.getLogger(__name__)


class AttachmentMapper:
    """Stateless mapper: EDMS AttachmentDto/AttachmentDocumentDto → domain Attachment.

    Handles field name variations across different API endpoints:
        fileName / name → file_name
        fileSize / size → file_size
        type / attachmentType → attachment_type
        storageUrl / storagePath → storage_path
        minioName / objectName → minio_name
        hashED → content_hash
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def from_dto(data: dict[str, Any] | None) -> Attachment:
        """Map a single AttachmentDto dict to domain Attachment.

        Args:
            data: Raw dict from EDMS API. May be:
                  - AttachmentDocumentDto (wrapper with "attachment" key)
                  - Standalone AttachmentDto
                  - MiniDocumentAttachmentDto variant
                  - None (raises ValueError immediately)

        Returns:
            Populated domain Attachment entity.

        Raises:
            ValueError: When ``data`` is None or not a dict.
            KeyError: When mandatory ``id`` field is absent.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"AttachmentMapper.from_dto expects dict, got {type(data).__name__}"
            )

        raw: dict[str, Any] = (
            AttachmentMapper._safe_dict(data.get("attachment")) or data
        )

        # ── Parse attachment type enum ────────────────────────────────────────
        type_raw = raw.get("type") or raw.get("attachmentType")
        attachment_type: AttachmentType | None = None
        if type_raw:
            try:
                attachment_type = AttachmentType(type_raw)
            except ValueError:
                logger.warning(
                    "unknown_attachment_type",
                    extra={
                        "type_value": type_raw,
                        "known_values": [e.value for e in AttachmentType],
                        "attachment_id": raw.get("id"),
                    },
                )

        # ── Parse attachment document type enum ───────────────────────────────
        doc_type_raw = (
            data.get("attachmentDocumentType")
            or raw.get("attachmentDocumentType")
            or (data.get("type") if "attachment" in data else None)
        )
        attachment_document_type: AttachmentDocumentType | None = None
        if doc_type_raw:
            try:
                attachment_document_type = AttachmentDocumentType(doc_type_raw)
            except ValueError:
                logger.warning(
                    f"unknown_attachment_document_type [{doc_type_raw}] "
                    f"known={[e.value for e in AttachmentDocumentType]}",
                    extra={
                        "type_value": doc_type_raw,
                        "attachment_id": raw.get("id"),
                    },
                )

        # ── Infer content type from MIME or filename ──────────────────────────
        content_type = AttachmentMapper._infer_content_type(
            mime_type=raw.get("mimeType"),
            file_name=raw.get("fileName") or raw.get("name", ""),
        )

        # ── Parse upload/modify dates ─────────────────────────────────────────
        upload_date = AttachmentMapper._parse_datetime(
            raw.get("uploadDate") or raw.get("createDate")
        )
        modify_date = AttachmentMapper._parse_datetime(raw.get("modifyDate"))

        # ── Parse signatures ──────────────────────────────────────────────────
        signs: list[AttachmentSignature] = []
        for sig_raw in AttachmentMapper._safe_list(raw.get("signs")):
            if not isinstance(sig_raw, dict):
                logger.debug(
                    "attachment_signature_null_item",
                    extra={"attachment_id": raw.get("id")},
                )
                continue
            try:
                signs.append(AttachmentMapper._map_signature(sig_raw))
            except (KeyError, ValueError) as exc:
                logger.debug(
                    "attachment_signature_skip",
                    extra={"error": str(exc), "sig_id": sig_raw.get("id")},
                )

        # ── Parse author/modifier IDs ─────────────────────────────────────────
        author_id = None
        if raw.get("authorId"):
            try:
                author_id = UUID(raw["authorId"])
            except (ValueError, TypeError):
                pass

        last_modify_user_id = None
        if raw.get("lastModifyUserId"):
            try:
                last_modify_user_id = UUID(raw["lastModifyUserId"])
            except (ValueError, TypeError):
                pass

        # ── Parse document reference ──────────────────────────────────────────
        document_id = None
        if raw.get("documentId") or data.get("documentId"):
            try:
                document_id = UUID(raw.get("documentId") or data["documentId"])
            except (ValueError, TypeError):
                pass

        return Attachment(
            # ── Identity ──────────────────────────────────────────────────────
            id=UUID(raw["id"]),
            document_id=document_id,
            # ── File info ─────────────────────────────────────────────────────
            file_name=raw.get("fileName") or raw.get("name", "unnamed"),
            file_size=raw.get("fileSize") or raw.get("size", 0),
            mime_type=raw.get("mimeType"),
            content_type=content_type,
            # ── Type classification ───────────────────────────────────────────
            attachment_type=attachment_type,
            attachment_document_type=attachment_document_type,
            # ── Storage location ──────────────────────────────────────────────
            storage_path=raw.get("storageUrl") or raw.get("storagePath"),
            bucket_name=raw.get("bucketName") or raw.get("sourceBucketName"),
            minio_name=(
                raw.get("minioName")
                or raw.get("objectName")
                or raw.get("sourceMinioName")
            ),
            # ── Original source info ──────────────────────────────────────────
            source_original_name=raw.get("sourceOriginalName"),
            # ── Integrity & versioning ────────────────────────────────────────
            content_hash=raw.get("hashED"),
            version_number=raw.get("versionNumber", 1),
            # ── Authorship ────────────────────────────────────────────────────
            author_id=author_id,
            last_modify_user_id=last_modify_user_id,
            # ── Timestamps ────────────────────────────────────────────────────
            upload_date=upload_date,
            modify_date=modify_date,
            # ── State flags ───────────────────────────────────────────────────
            is_deleted=raw.get("deleted", False),
            # ── Signatures ────────────────────────────────────────────────────
            signs=signs,
        )

    @staticmethod
    def from_dto_list(items: list[dict[str, Any]] | None) -> list[Attachment]:
        """Map a list of attachment DTO dicts, skipping malformed items.

        Args:
            items: List of attachment dicts from API response.
                   Accepts ``None`` (returns empty list — Java API compat).

        Returns:
            List of successfully mapped domain Attachment entities.
            Logs warnings for skipped items.
        """
        result: list[Attachment] = []

        for item in AttachmentMapper._safe_list(items):
            if not isinstance(item, dict):
                logger.debug(
                    "attachment_list_null_item",
                    extra={"item_type": type(item).__name__},
                )
                continue
            try:
                result.append(AttachmentMapper.from_dto(item))
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "attachment_mapper_skip",
                    extra={
                        "error": str(exc),
                        "item_id": item.get("id"),
                    },
                )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_list(value: Any) -> list:
        """Normalise any value to a list, treating None/non-list as empty.

        Designed specifically for Java API responses where collection fields
        may be ``null`` instead of ``[]``.

        Args:
            value: Any value from API response.

        Returns:
            Original list if value is a list, otherwise empty list.

        Examples:
            >>> AttachmentMapper._safe_list(None)
            []
            >>> AttachmentMapper._safe_list([1, 2])
            [1, 2]
            >>> AttachmentMapper._safe_list("oops")
            []
        """
        if isinstance(value, list):
            return value
        if value is not None:
            logger.debug(
                "safe_list_unexpected_type",
                extra={"type": type(value).__name__, "value_preview": str(value)[:50]},
            )
        return []

    @staticmethod
    def _safe_dict(value: Any) -> dict | None:
        """Return value if it is a non-empty dict, otherwise None.

        Args:
            value: Any value from API response.

        Returns:
            Original dict or None.
        """
        if isinstance(value, dict):
            return value
        return None

    @staticmethod
    def _parse_datetime(raw: str | int | float | None) -> datetime | None:
        """Parse datetime from ISO string or Java timestamp (milliseconds).

        Args:
            raw: ISO string, Unix timestamp (ms), or None.

        Returns:
            Parsed datetime or None if invalid/absent.
        """
        if not raw:
            return None

        try:
            if isinstance(raw, str):
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            elif isinstance(raw, (int, float)):
                return datetime.fromtimestamp(raw / 1000)
        except (ValueError, TypeError, OSError) as exc:
            logger.debug("datetime_parse_failed", extra={"raw": raw, "error": str(exc)})

        return None

    @staticmethod
    def _infer_content_type(mime_type: str | None, file_name: str) -> ContentType:
        """Infer ContentType enum from MIME type or filename extension.

        Priority:
            1. MIME type (most reliable)
            2. File extension (fallback)
            3. ContentType.OTHER (unknown)

        Args:
            mime_type: MIME type string (e.g. "application/pdf").
            file_name: File name with extension.

        Returns:
            ContentType enum value.
        """
        if mime_type:
            mime_lower = mime_type.lower()
            if "pdf" in mime_lower:
                return ContentType.PDF
            if "msword" in mime_lower or "wordprocessingml" in mime_lower:
                if "openxmlformats" in mime_lower or "wordprocessingml" in mime_lower:
                    return ContentType.DOCX
                return ContentType.DOC
            if "tiff" in mime_lower or "tif" in mime_lower:
                return ContentType.TIFF
            if "jpeg" in mime_lower or "jpg" in mime_lower:
                return ContentType.JPEG
            if "png" in mime_lower:
                return ContentType.PNG
            if "gif" in mime_lower:
                return ContentType.GIF
            if "bmp" in mime_lower:
                return ContentType.BMP

        if "." not in file_name:
            return ContentType.OTHER

        ext = file_name.lower().rsplit(".", 1)[-1]
        ext_map: dict[str, ContentType] = {
            "pdf": ContentType.PDF,
            "docx": ContentType.DOCX,
            "doc": ContentType.DOC,
            "tif": ContentType.TIF,
            "tiff": ContentType.TIFF,
            "jpg": ContentType.JPG,
            "jpeg": ContentType.JPEG,
            "png": ContentType.PNG,
            "gif": ContentType.GIF,
            "bmp": ContentType.BMP,
        }
        return ext_map.get(ext, ContentType.OTHER)

    @staticmethod
    def _map_signature(data: dict[str, Any]) -> AttachmentSignature:
        """Map signature DTO dict to domain AttachmentSignature.

        Args:
            data: Raw signature dict from API with fields:
                  {id, date, check, sign: {data, signer, ...}}

        Returns:
            AttachmentSignature value object.

        Raises:
            KeyError: When mandatory ``id`` is missing.
        """
        sig_date = AttachmentMapper._parse_datetime(data.get("date"))

        sign_raw = data.get("sign")
        signature: Signature | None = None
        if sign_raw and isinstance(sign_raw, dict):
            signature = AttachmentMapper._map_signature_data(sign_raw)

        return AttachmentSignature(
            id=UUID(data["id"]),
            date=sig_date,
            is_verified=data.get("check"),
            signature=signature,
        )

    @staticmethod
    def _map_signature_data(data: dict[str, Any]) -> Signature:
        """Map nested Signature DTO to domain Signature value object.

        Args:
            data: Raw Signature dict with fields like:
                  {data, signer, signerFio, certSerial, keyId, ...}

        Returns:
            Signature value object.
        """
        return Signature(
            data=data.get("data", ""),
            key_id=data.get("keyId"),
            signer=data.get("signer"),
            sign_time=AttachmentMapper._parse_datetime(data.get("signtime")),
            signer_date=AttachmentMapper._parse_datetime(data.get("signerDate")),
            start=AttachmentMapper._parse_datetime(data.get("start")),
            end=AttachmentMapper._parse_datetime(data.get("end")),
            cert_serial=data.get("certSerial"),
            issuer=data.get("issuer"),
            signer_fio=data.get("signerFio"),
            signer_post=data.get("signerPost"),
            signer_org=data.get("signerOrg"),
            personal_number=data.get("personalNumber"),
            operation_type=data.get("operationType"),
            orig_signature=data.get("origSignature"),
            sign_count=data.get("signCount"),
            attr_cert_issuer=data.get("attrCertIssuer"),
            attr_cert_issuer_id=data.get("attrCertIssuerId"),
            attr_organization_name=data.get("attrOrganizationName"),
            attr_post=data.get("attrPost"),
            attr_unp=data.get("attrUnp"),
            attr_unpf=data.get("attrUnpf"),
            attr_address=data.get("attrAddress"),
            attr_start=AttachmentMapper._parse_datetime(data.get("attrStart")),
            attr_end=AttachmentMapper._parse_datetime(data.get("attrEnd")),
        )
