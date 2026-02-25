# src/ai_edms_assistant/infrastructure/edms_api/mappers/document_mapper.py
"""DocumentDto dict → domain Document mapper."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from ....domain.entities.attachment import Attachment
from ....domain.entities.appeal import DocumentAppeal
from ....domain.entities.document import Document, DocumentCategory, DocumentStatus
from .attachment_mapper import AttachmentMapper
from .employee_mapper import EmployeeMapper

logger = logging.getLogger(__name__)


class DocumentMapper:
    """FULL mapper: EDMS DocumentDto → domain Document (120+ fields)."""

    @staticmethod
    def from_dto(data: dict[str, Any]) -> Document:
        """Map complete DocumentDto to domain Document."""

        # ── Parse enums ────────────────────────────────────────────────
        status = DocumentMapper._parse_enum(
            data.get("state") or data.get("status"), DocumentStatus
        )
        prev_status = DocumentMapper._parse_enum(data.get("prevStatus"), DocumentStatus)
        document_category = DocumentMapper._parse_enum(
            data.get("docCategoryConstant"), DocumentCategory
        )

        # ── Parse attachments ──────────────────────────────────────────
        att_list = data.get("attachmentDocument") or data.get("attachments") or []
        attachments: list[Attachment] = AttachmentMapper.from_dto_list(att_list)

        # ── Parse appeal ───────────────────────────────────────────────
        appeal_raw = data.get("documentAppeal")
        appeal: DocumentAppeal | None = None
        if appeal_raw and isinstance(appeal_raw, dict):
            appeal = DocumentMapper._map_appeal(appeal_raw)

        # ── Parse user references ──────────────────────────────────────
        author = EmployeeMapper.to_user_info(data.get("author"))
        responsible_executor = EmployeeMapper.to_user_info(
            data.get("responsibleExecutor")
        )
        initiator = EmployeeMapper.to_user_info(data.get("initiator"))
        chairperson = EmployeeMapper.to_user_info(data.get("chairperson"))
        secretary = EmployeeMapper.to_user_info(data.get("secretary"))

        # ── Parse who_addressed (signed by) ───────────────────────────
        who_addressed = []
        for item in data.get("whoAddressed", []):
            u = EmployeeMapper.to_user_info(item)
            if u:
                who_addressed.append(u)

        # ── Parse timestamps ───────────────────────────────────────────
        create_date = DocumentMapper._parse_datetime(data.get("createDate"))
        reg_date = DocumentMapper._parse_datetime(data.get("regDate"))
        out_reg_date = DocumentMapper._parse_datetime(data.get("outRegDate"))
        reserved_reg_date = DocumentMapper._parse_datetime(data.get("reservedRegDate"))
        date_meeting = DocumentMapper._parse_datetime(data.get("dateMeeting"))
        date_meeting_question = DocumentMapper._parse_datetime(
            data.get("dateMeetingQuestion")
        )
        start_meeting = DocumentMapper._parse_datetime(data.get("startMeeting"))
        end_meeting = DocumentMapper._parse_datetime(data.get("endMeeting"))
        date_question = DocumentMapper._parse_datetime(data.get("dateQuestion"))

        return Document(
            # ── Identity ───────────────────────────────────────────────
            id=UUID(data["id"]),
            organization_id=data.get("organizationId"),
            # ── Status & Category ──────────────────────────────────────
            status=status,
            prev_status=prev_status,
            document_category=document_category,
            doc_category_constant=data.get("docCategoryConstant"),
            # ── Registration ───────────────────────────────────────────
            reg_number=data.get("regNumber"),
            reserved_reg_number=data.get("reservedRegNumber"),
            out_reg_number=data.get("outRegNumber"),
            reg_date=reg_date,
            reserved_reg_date=reserved_reg_date,
            out_reg_date=out_reg_date,
            journal_id=DocumentMapper._safe_uuid(data.get("journalId")),
            journal_number=data.get("journalNumber"),
            # ── Content ────────────────────────────────────────────────
            short_summary=data.get("shortSummary"),
            summary=data.get("summary"),
            note=data.get("note"),
            # ── Document Type ──────────────────────────────────────────
            document_type_name=data.get("documentTypeName"),
            document_type_id=data.get("documentTypeId"),
            profile_name=data.get("profileName"),
            profile_id=DocumentMapper._safe_uuid(data.get("profileId")),
            # ── Pages & Physical Properties ────────────────────────────
            pages=data.get("pages"),
            additional_pages=data.get("additionalPages"),
            exemplar_count=data.get("exemplarCount"),
            exemplar_number=data.get("exemplarNumber"),
            # ── Flags ──────────────────────────────────────────────────
            control_flag=data.get("controlFlag", False),
            remove_control=data.get("removeControl", False),
            dsp_flag=data.get("dspFlag", False),
            skip_registration=data.get("skipRegistration", False),
            version_flag=data.get("versionFlag", False),
            recipients=data.get("recipients", False),
            has_responsible_executor=data.get("hasResponsibleExecutor", False),
            has_question=data.get("hasQuestion", False),
            addition=data.get("addition", False),
            # ── Timestamps ─────────────────────────────────────────────
            create_date=create_date,
            # ── Document Relations ─────────────────────────────────────
            ref_doc_id=DocumentMapper._safe_uuid(data.get("refDocId")),
            ref_doc_org_id=data.get("refDocOrgId"),
            received_doc_id=DocumentMapper._safe_uuid(data.get("receivedDocId")),
            answer_doc_id=DocumentMapper._safe_uuid(data.get("answerDocId")),
            process_id=DocumentMapper._safe_uuid(data.get("processId")),
            document_version_id=DocumentMapper._safe_uuid(
                data.get("documentVersionId")
            ),
            # ── Correspondent ──────────────────────────────────────────
            correspondent_name=data.get("correspondentName"),
            correspondent_id=DocumentMapper._safe_uuid(data.get("correspondentId")),
            # ── Delivery ───────────────────────────────────────────────
            delivery_method_id=data.get("deliveryMethodId"),
            # ── Investment Program ─────────────────────────────────────
            invest_program_id=data.get("investProgramId"),
            # ── Users ──────────────────────────────────────────────────
            author=author,
            responsible_executor=responsible_executor,
            initiator=initiator,
            # ── Collections ────────────────────────────────────────────
            attachments=attachments,
            who_addressed=who_addressed,
            # ── Appeal ─────────────────────────────────────────────────
            appeal=appeal,
            # ── Counters ───────────────────────────────────────────────
            count_task=data.get("countTask", 0),
            introduction_count=data.get("introductionCount", 0),
            introduction_complete_count=data.get("introductionCompleteCount", 0),
            responsible_executors_count=data.get("responsibleExecutorsCount", 0),
            document_links_count=data.get("documentLinksCount", 0),
            write_off_affair_count=data.get("writeOffAffairCount", 0),
            pre_affair_count=data.get("preAffairCount", 0),
            invitees_count=data.get("inviteesCount", 0),
            # ── Meeting Fields ─────────────────────────────────────────
            date_meeting=date_meeting,
            date_meeting_question=date_meeting_question,
            start_meeting=start_meeting,
            end_meeting=end_meeting,
            place_meeting=data.get("placeMeeting"),
            chairperson=chairperson,
            secretary=secretary,
            external_invitees=data.get("externalInvitees"),
            number_question=data.get("numberQuestion"),
            date_question=date_question,
            comment_question=data.get("commentQuestion"),
            # ── Country ────────────────────────────────────────────────
            country_name=data.get("countryName"),
            country_id=DocumentMapper._safe_uuid(data.get("countryId")),
            # ── Custom Fields ──────────────────────────────────────────
            custom_fields=data.get("customFields", {}),
        )

    @staticmethod
    def _map_appeal(data: dict[str, Any]) -> DocumentAppeal:
        """Map DocumentAppealDto to domain DocumentAppeal."""
        return DocumentAppeal(
            id=UUID(data["id"]) if data.get("id") else None,
            appeal_number=data.get("appealNumber"),
            applicant_name=data.get("fioApplicant"),
            description=data.get("description"),
            declarant_type=data.get("declarantType"),
            collective=data.get("collective"),
            anonymous=data.get("anonymous"),
            reasonably=data.get("reasonably"),
            organization_name=data.get("organizationName"),
            full_address=data.get("fullAddress"),
            phone=data.get("phone"),
            email=data.get("email"),
            receipt_date=data.get("receiptDate"),
            country_appeal_id=data.get("countryAppealId"),
            region_id=data.get("regionId"),
            district_id=data.get("districtId"),
            city_id=data.get("cityId"),
            citizen_type_id=data.get("citizenTypeId"),
            correspondent_appeal_id=data.get("correspondentAppealId"),
            review_progress=data.get("reviewProgress"),
        )

    @staticmethod
    def _parse_datetime(raw: str | int | float | None):
        """Parse ISO string or Java timestamp."""
        if not raw:
            return None
        from datetime import datetime

        try:
            if isinstance(raw, str):
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            elif isinstance(raw, (int, float)):
                return datetime.fromtimestamp(raw / 1000)
        except (ValueError, TypeError, OSError):
            return None

    @staticmethod
    def _safe_uuid(raw: Any) -> UUID | None:
        """Parse UUID with None fallback."""
        if not raw:
            return None
        try:
            return UUID(str(raw))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_enum(raw: str | None, enum_class: type):
        """Generic enum parser."""
        if not raw:
            return None
        try:
            return enum_class(raw)
        except ValueError:
            logger.warning(f"unknown_{enum_class.__name__}: {raw}")
            return None

    @staticmethod
    def _parse_status(raw: str | None):
        """Legacy status parser."""
        return DocumentMapper._parse_enum(raw, DocumentStatus)

    @staticmethod
    def from_dto_list(items: list[dict[str, Any]]) -> list[Document]:
        """Map list, skip malformed."""
        result: list[Document] = []
        for item in items or []:
            try:
                result.append(DocumentMapper.from_dto(item))
            except (KeyError, ValueError) as exc:
                logger.warning(
                    f"document_mapper_skip: {exc}, item_id: {item.get('id')}"
                )
        return result
