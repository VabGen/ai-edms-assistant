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

    @staticmethod
    def from_dto(data: dict[str, Any]) -> Attachment:
        """Map a single AttachmentDto dict to domain Attachment.

        Args:
            data: Raw dict from EDMS API. May be:
                  - AttachmentDocumentDto (wrapper with "attachment" key)
                  - Standalone AttachmentDto
                  - MiniDocumentAttachmentDto variant

        Returns:
            Populated domain Attachment entity.

        Raises:
            KeyError: When mandatory ``id`` field is absent.
            ValueError: When enum parsing fails for critical fields.
        """
        # AttachmentDocumentDto wraps the real attachment under "attachment" key
        # MiniDocumentAttachmentDto also uses "attachment" wrapper
        raw = data.get("attachment") or data

        # ── Parse attachment type enum ────────────────────────────────────────
        type_raw = raw.get("type") or raw.get("attachmentType")
        attachment_type = AttachmentType.ATTACHMENT  # default
        if type_raw:
            try:
                attachment_type = AttachmentType(type_raw)
            except ValueError:
                logger.warning(
                    "unknown_attachment_type",
                    extra={
                        "type_value": type_raw,
                        "attachment_id": raw.get("id"),
                    },
                )

        # ── Parse attachment document type enum ───────────────────────────────
        doc_type_raw = data.get("attachmentDocumentType") or raw.get(
            "attachmentDocumentType"
        )
        attachment_document_type: AttachmentDocumentType | None = None
        if doc_type_raw:
            try:
                attachment_document_type = AttachmentDocumentType(doc_type_raw)
            except ValueError:
                logger.warning(
                    "unknown_attachment_document_type",
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
        for sig_raw in raw.get("signs", []):
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
            # ISO 8601 string
            if isinstance(raw, str):
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            # Java timestamp in milliseconds
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
        # ── Try MIME type first ───────────────────────────────────────────────
        if mime_type:
            mime_lower = mime_type.lower()

            # PDF
            if "pdf" in mime_lower:
                return ContentType.PDF

            # Microsoft Word
            if "msword" in mime_lower or "wordprocessingml" in mime_lower:
                if "openxmlformats" in mime_lower or "wordprocessingml" in mime_lower:
                    return ContentType.DOCX
                return ContentType.DOC

            # Images
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

        # ── Fallback to file extension ────────────────────────────────────────
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

        # Parse nested Signature object
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
            # ── Core signature data ───────────────────────────────────────────
            data=data.get("data", ""),  # base64 signature, required
            key_id=data.get("keyId"),
            signer=data.get("signer"),
            # ── Timestamps ────────────────────────────────────────────────────
            sign_time=AttachmentMapper._parse_datetime(data.get("signtime")),
            signer_date=AttachmentMapper._parse_datetime(data.get("signerDate")),
            start=AttachmentMapper._parse_datetime(data.get("start")),
            end=AttachmentMapper._parse_datetime(data.get("end")),
            # ── Certificate info ──────────────────────────────────────────────
            cert_serial=data.get("certSerial"),
            issuer=data.get("issuer"),
            # ── Signer info ───────────────────────────────────────────────────
            signer_fio=data.get("signerFio"),
            signer_post=data.get("signerPost"),
            signer_org=data.get("signerOrg"),
            personal_number=data.get("personalNumber"),
            # ── Operation metadata ────────────────────────────────────────────
            operation_type=data.get("operationType"),
            orig_signature=data.get("origSignature"),
            sign_count=data.get("signCount"),
            # ── Attribute certificate fields ──────────────────────────────────
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

    @staticmethod
    def from_dto_list(items: list[dict[str, Any]]) -> list[Attachment]:
        """Map a list of attachment DTO dicts, skipping malformed items.

        Args:
            items: List of attachment dicts from API response.

        Returns:
            List of successfully mapped domain Attachment entities.
            Logs warnings for skipped items.
        """
        result: list[Attachment] = []
        for item in items or []:
            try:
                result.append(AttachmentMapper.from_dto(item))
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "attachment_mapper_skip",
                    extra={
                        "error": str(exc),
                        "item_id": item.get("id") if isinstance(item, dict) else None,
                    },
                )
        return result
