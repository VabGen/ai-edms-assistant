# src/ai_edms_assistant/infrastructure/edms_api/mappers/document_mapper.py
"""DocumentDto dict → domain Document mapper.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any
from uuid import UUID

from ....domain.entities.attachment import Attachment
from ....domain.entities.appeal import DocumentAppeal, GeoLocation
from ....domain.entities.document import (
    Document,
    DocumentCategory,
    DocumentStatus,
    DocumentUserColor,
    CurrencyInfo,
    RegistrationJournalInfo,
    DocumentUserProps,
    DocumentQuestion,
    DocumentSpeaker,
    DocumentRecipient,
    RecipientStatus,
    RecipientType,
    SpeakerType,
)
from .attachment_mapper import AttachmentMapper
from .employee_mapper import EmployeeMapper

logger = logging.getLogger(__name__)


class DocumentMapper:
    """EDMS DocumentDto → domain Document.

    All parsing is null-safe. Failures are logged, never raise.
    """

    @staticmethod
    def from_dto(data: dict[str, Any]) -> Document:
        """Map complete Java DocumentDto to domain Document.

        Covers ALL fields from Java DocumentDto verified (2026-02 final).

        Args:
            data: Raw dict from EDMS API — the inner ``"document"`` dict
                  when response is permission-wrapped.

        Returns:
            Populated domain ``Document`` entity.

        Raises:
            ValueError: When mandatory ``id`` field is absent.
        """
        # ── Parse enums ────────────────────────────────────────────────
        status = DocumentMapper._parse_enum(
            data.get("state") or data.get("status"), DocumentStatus
        )
        prev_status = DocumentMapper._parse_enum(
            data.get("prevStatus"), DocumentStatus
        )
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
            try:
                appeal = DocumentMapper._map_appeal(appeal_raw)
            except Exception as exc:
                logger.warning(
                    "document_appeal_map_failed",
                    extra={"error": str(exc), "document_id": data.get("id")},
                )

        # ── Parse users ────────────────────────────────────────────────
        author = EmployeeMapper.to_user_info(data.get("author"))
        responsible_executor = EmployeeMapper.to_user_info(
            data.get("responsibleExecutor")
        )
        initiator = EmployeeMapper.to_user_info(data.get("initiator"))
        chairperson = EmployeeMapper.to_user_info(data.get("chairperson"))
        secretary = EmployeeMapper.to_user_info(data.get("secretary"))
        who_signed = EmployeeMapper.to_user_info(data.get("whoSigned"))

        # ── Parse who_addressed (MiniUserInfoDto = same as UserInfoDto subset) ──
        who_addressed: list[Any] = []
        for item in data.get("whoAddressed", []) or []:
            u = EmployeeMapper.to_user_info(item)
            if u:
                who_addressed.append(u)

        # ── Parse responsible_executors ────────────────────────────────
        # Java: List<DocumentResponsibleExecutorDto> {id, documentId, executor}
        responsible_executors: list[Any] = []
        for item in data.get("responsibleExecutors", []) or []:
            if not isinstance(item, dict):
                continue
            emp_raw = item.get("executor") or item
            u = EmployeeMapper.to_user_info(emp_raw)
            if u:
                responsible_executors.append(u)

        # ── Parse formula (List<String>, NOT str) ──────────────────────
        formula_raw = data.get("formula") or []
        formula: list[str] = (
            [str(f) for f in formula_raw if f]
            if isinstance(formula_raw, list)
            else ([str(formula_raw)] if formula_raw else [])
        )

        # ── Parse required_field ───────────────────────────────────────
        required_field: list[str] = [
            str(f) for f in (data.get("requiredField") or []) if f
        ]

        # ── Parse color (DocumentUserColorDto) ────────────────────────
        color_raw = data.get("color")
        color: DocumentUserColor | None = None
        if color_raw and isinstance(color_raw, dict) and color_raw.get("color"):
            color = DocumentUserColor(
                id=DocumentMapper._safe_uuid(color_raw.get("id")),
                documentId=DocumentMapper._safe_uuid(color_raw.get("documentId")),
                color=color_raw.get("color"),
            )

        # ── Parse currency (CurrencyDto) ───────────────────────────────
        currency_raw = data.get("currency")
        currency: CurrencyInfo | None = None
        if currency_raw and isinstance(currency_raw, dict):
            currency = CurrencyInfo(
                id=DocumentMapper._safe_uuid(currency_raw.get("id")),
                name=currency_raw.get("name"),
                code=currency_raw.get("code"),
            )

        # ── Parse registrationJournal ──────────────────────────────────
        journal_raw = data.get("registrationJournal")
        registration_journal: RegistrationJournalInfo | None = None
        if journal_raw and isinstance(journal_raw, dict):
            registration_journal = RegistrationJournalInfo(
                id=DocumentMapper._safe_uuid(journal_raw.get("id")),
                journalName=journal_raw.get("journalName"),
                counterValue=journal_raw.get("counterValue"),
            )

        # ── Parse userProps ────────────────────────────────────────────
        user_props_raw = data.get("userProps")
        user_props: DocumentUserProps | None = None
        if user_props_raw and isinstance(user_props_raw, dict):
            user_props = DocumentUserProps(
                createTaskCount=user_props_raw.get("createTaskCount"),
                createTaskExecutedCount=user_props_raw.get("createTaskExecutedCount"),
            )

        # ── Parse documentQuestions ────────────────────────────────────
        document_questions: list[DocumentQuestion] = []
        for q_raw in data.get("documentQuestions", []) or []:
            if not isinstance(q_raw, dict):
                continue
            try:
                question = DocumentMapper._map_document_question(q_raw)
                document_questions.append(question)
            except Exception as exc:
                logger.debug(
                    "document_question_map_failed",
                    extra={"error": str(exc)},
                )

        # ── Parse correspondent (full DocumentRecipient) ───────────────
        correspondent = DocumentMapper._map_recipient(data.get("correspondent"))

        # ── Parse recipientList ────────────────────────────────────────
        recipient_list: list[DocumentRecipient] = []
        for r_raw in data.get("recipientList", []) or []:
            r = DocumentMapper._map_recipient(r_raw)
            if r:
                recipient_list.append(r)

        # ── Parse document type name from object ───────────────────────
        doc_type_raw = data.get("documentType") or {}
        document_type_name: str | None = (
            doc_type_raw.get("typeName")
            if isinstance(doc_type_raw, dict)
            else data.get("documentTypeName")
        )
        document_type_id: int | None = (
            doc_type_raw.get("id")
            if isinstance(doc_type_raw, dict) and doc_type_raw.get("id")
            else data.get("documentTypeId")
        )

        # ── Parse delivery method name ─────────────────────────────────
        delivery_raw = data.get("deliveryMethod") or {}
        delivery_method_name: str | None = (
            delivery_raw.get("deliveryName")
            if isinstance(delivery_raw, dict)
            else None
        )

        # ── Parse contract sum ─────────────────────────────────────────
        contract_sum: Decimal | None = None
        contract_sum_raw = data.get("contractSum")
        if contract_sum_raw is not None:
            try:
                contract_sum = Decimal(str(contract_sum_raw))
            except Exception:
                pass

        # ── Parse timestamps ───────────────────────────────────────────
        _dt = DocumentMapper._parse_datetime
        create_date = _dt(data.get("createDate"))
        reg_date = _dt(data.get("regDate"))
        out_reg_date = _dt(data.get("outRegDate"))
        reserved_reg_date = _dt(data.get("reservedRegDate"))

        # ── Validate mandatory ID ──────────────────────────────────────
        doc_id = DocumentMapper._safe_uuid(data.get("id"))
        if doc_id is None:
            raise ValueError(
                f"Document DTO missing required 'id'. "
                f"Keys present: {list(data.keys())[:10]}"
            )

        return Document(
            # ── Identity ───────────────────────────────────────────────
            id=doc_id,
            organization_id=data.get("organizationId"),
            # ── Category & Type ────────────────────────────────────────
            status=status,
            prev_status=prev_status,
            document_category=document_category,
            doc_category_constant=data.get("docCategoryConstant"),
            current_bpmn_task_name=data.get("currentBpmnTaskName"),
            # ── Registration ───────────────────────────────────────────
            reg_number=data.get("regNumber"),
            reserved_reg_number=data.get("reservedRegNumber"),
            out_reg_number=data.get("outRegNumber"),
            reg_date=reg_date,
            reserved_reg_date=reserved_reg_date,
            out_reg_date=out_reg_date,
            journal_id=DocumentMapper._safe_uuid(data.get("journalId")),
            journal_number=data.get("journalNumber"),
            registration_journal=registration_journal,
            create_date=create_date,
            create_type=DocumentMapper._parse_enum(
                data.get("createType"), __import__(
                    "ai_edms_assistant.domain.entities.document",
                    fromlist=["DocumentCreateType"]
                ).DocumentCreateType
            ) if data.get("createType") else None,
            # ── Content ────────────────────────────────────────────────
            short_summary=data.get("shortSummary"),
            summary=data.get("summary"),
            note=data.get("note"),
            formula=formula,
            required_field=required_field,
            # ── Document Type ──────────────────────────────────────────
            document_type_name=document_type_name,
            document_type_id=document_type_id,
            profile_name=data.get("profileName"),
            profile_id=DocumentMapper._safe_uuid(data.get("profileId")),
            days_execution=data.get("daysExecution"),
            # ── Physical ───────────────────────────────────────────────
            pages=data.get("pages"),
            additional_pages=data.get("additionalPages"),
            exemplar_count=data.get("exemplarCount"),
            exemplar_number=data.get("exemplarNumber"),
            # ── Color ──────────────────────────────────────────────────
            color=color,
            # ── Flags ──────────────────────────────────────────────────
            control_flag=data.get("controlFlag") or False,
            remove_control=data.get("removeControl") or False,
            dsp_flag=data.get("dspFlag") or False,
            skip_registration=data.get("skipRegistration") or False,
            version_flag=data.get("versionFlag") or False,
            recipients=data.get("recipients") or False,
            has_responsible_executor=data.get("hasResponsibleExecutor") or False,
            has_question=data.get("hasQuestion"),
            addition=data.get("addition") or False,
            enable_access_grief=data.get("enableAccessGrief") or False,
            access_grief_id=DocumentMapper._safe_uuid(data.get("accessGriefId")),
            auto_routing=data.get("autoRouting"),
            # ── Relations ──────────────────────────────────────────────
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
            correspondent=correspondent,
            recipient_list=recipient_list,
            # ── Delivery ───────────────────────────────────────────────
            delivery_method_id=data.get("deliveryMethodId"),
            delivery_method_name=delivery_method_name,
            invest_program_id=DocumentMapper._safe_uuid(data.get("investProgramId")),
            currency_id=DocumentMapper._safe_uuid(data.get("currencyId")),
            currency=currency,
            # ── Users ──────────────────────────────────────────────────
            author=author,
            responsible_executor=responsible_executor,
            initiator=initiator,
            who_signed=who_signed,
            who_addressed=who_addressed,
            in_doc_signers=data.get("inDocSigners"),
            chairperson=chairperson,
            secretary=secretary,
            responsible_executors=responsible_executors,
            # ── Files & Tasks ──────────────────────────────────────────
            attachments=attachments,
            # ── User props ─────────────────────────────────────────────
            user_props=user_props,
            # ── Appeal ─────────────────────────────────────────────────
            appeal=appeal,
            # ── Counters ───────────────────────────────────────────────
            count_task=data.get("countTask") or 0,
            task_project_count=data.get("taskProjectCount") or 0,
            completed_task_count=data.get("completedTaskCount") or 0,
            introduction_count=data.get("introductionCount"),
            introduction_complete_count=data.get("introductionCompleteCount"),
            responsible_executors_count=data.get("responsibleExecutorsCount") or 0,
            document_links_count=data.get("documentLinksCount"),
            write_off_affair_count=data.get("writeOffAffairCount") or 0,
            pre_affair_count=data.get("preAffairCount") or 0,
            invitees_count=data.get("inviteesCount"),
            meeting_question_notify_count=data.get("meetingQuestionNotifyCount") or 0,
            # ── Meeting ────────────────────────────────────────────────
            date_meeting=_dt(data.get("dateMeeting")),
            date_meeting_question=_dt(data.get("dateMeetingQuestion")),
            start_meeting=_dt(data.get("startMeeting")),
            end_meeting=_dt(data.get("endMeeting")),
            place_meeting=data.get("placeMeeting"),
            external_invitees=data.get("externalInvitees"),
            number_question=data.get("numberQuestion"),
            date_question=_dt(data.get("dateQuestion")),
            comment_question=data.get("commentQuestion"),
            document_questions=document_questions,
            document_meeting_question_id=DocumentMapper._safe_uuid(
                data.get("documentMeetingQuestionId")
            ),
            document_meeting_question_org_id=data.get("documentMeetingQuestionOrgId"),
            addition_meeting_question_id=DocumentMapper._safe_uuid(
                data.get("additionMeetingQuestionId")
            ),
            addition_meeting_question_org_id=data.get("additionMeetingQuestionOrgId"),
            meeting_form_type=DocumentMapper._parse_enum(
                data.get("formMeetingType"),
                __import__(
                    "ai_edms_assistant.domain.entities.document",
                    fromlist=["MeetingFormType"]
                ).MeetingFormType,
            ) if data.get("formMeetingType") else None,
            # ── Country ────────────────────────────────────────────────
            country_name=data.get("countryName"),
            country_id=DocumentMapper._safe_uuid(data.get("countryId")),
            # ── Contract ───────────────────────────────────────────────
            contract_sum=contract_sum,
            contract_number=data.get("contractNumber"),
            contract_date=_dt(data.get("contractDate")),
            contract_signing_date=_dt(data.get("contractSigningDate")),
            contract_start_date=_dt(data.get("contractStartDate")),
            contract_duration_start=_dt(data.get("contractDurationStart")),
            contract_duration_end=_dt(data.get("contractDurationEnd")),
            contract_agreement=data.get("contractAgreement"),
            contract_auto_prolongation=data.get("contractAutoProlongation"),
            contract_typical=data.get("contractTypical") or False,
            # ── Custom & Form ──────────────────────────────────────────
            document_form_id=DocumentMapper._safe_uuid(data.get("documentFormId")),
            custom_fields=data.get("customFields") or {},
        )

    # ── Sub-entity mappers ─────────────────────────────────────────────────

    @staticmethod
    def _map_appeal(data: dict[str, Any]) -> DocumentAppeal:
        """Map DocumentAppealDto → DocumentAppeal.

        Full field coverage including repeatIdenticalAppeals (NEW).
        """
        # ── Build GeoLocation from flat fields ─────────────────────────
        region_id = DocumentMapper._safe_uuid(data.get("regionId"))
        district_id = DocumentMapper._safe_uuid(data.get("districtId"))
        city_id = DocumentMapper._safe_uuid(data.get("cityId"))
        geo_location: GeoLocation | None = None
        if any([region_id, district_id, city_id]):
            geo_location = GeoLocation(
                countryId=DocumentMapper._safe_uuid(data.get("countryAppealId")),
                countryName=data.get("countryAppealName"),
                regionId=region_id,
                regionName=data.get("regionName"),
                districtId=district_id,
                districtName=data.get("districtName"),
                cityId=city_id,
                cityName=data.get("cityName"),
            )

        # ── citizenType.name ───────────────────────────────────────────
        ct_raw = data.get("citizenType") or {}
        citizen_type_name: str | None = (
            ct_raw.get("name") if isinstance(ct_raw, dict) else None
        )

        # ── subject.name + parentSubject.name ──────────────────────────
        subj_raw = data.get("subject") or {}
        subject_name: str | None = None
        subject_parent_name: str | None = None
        if isinstance(subj_raw, dict):
            subject_name = subj_raw.get("name")
            parent_raw = subj_raw.get("parentSubject") or {}
            if isinstance(parent_raw, dict):
                subject_parent_name = parent_raw.get("name")

        # ── signed: Java = String (may be "") ─────────────────────────
        signed_raw = data.get("signed")
        signed: bool | None = (
            False if signed_raw == ""
            else (bool(signed_raw) if signed_raw is not None else None)
        )

        return DocumentAppeal(
            id=DocumentMapper._safe_uuid(data.get("id")),
            appeal_number=data.get("appealNumber"),
            applicant_name=data.get("fioApplicant"),
            description=data.get("description"),
            declarant_type=data.get("declarantType"),
            collective=data.get("collective"),
            anonymous=data.get("anonymous"),
            signed=signed,
            reasonably=data.get("reasonably"),
            organization_name=data.get("organizationName"),
            full_address=data.get("fullAddress"),
            phone=data.get("phone"),
            email=data.get("email"),
            index=data.get("index"),
            receipt_date=DocumentMapper._parse_datetime(data.get("receiptDate")),
            date_doc_correspondent_org=DocumentMapper._parse_datetime(
                data.get("dateDocCorrespondentOrg")
            ),
            country_appeal_id=DocumentMapper._safe_uuid(data.get("countryAppealId")),
            country_appeal_name=data.get("countryAppealName"),
            geo_location=geo_location,
            citizen_type_id=DocumentMapper._safe_uuid(data.get("citizenTypeId")),
            citizen_type_name=citizen_type_name,
            subject_id=DocumentMapper._safe_uuid(data.get("subjectId")),
            subject_name=subject_name,
            subject_parent_name=subject_parent_name,
            correspondent_org_number=data.get("correspondentOrgNumber"),
            correspondent_appeal_id=DocumentMapper._safe_uuid(
                data.get("correspondentAppealId")
            ),
            correspondent_appeal=data.get("correspondentAppeal"),
            index_date_cover_letter=data.get("indexDateCoverLetter"),
            repeat_identical_appeals=data.get("repeatIdenticalAppeals"),
            review_progress=data.get("reviewProgress"),
            solution_result_id=DocumentMapper._safe_uuid(
                data.get("solutionResultId")
            ),
            nomenclature_affair_id=DocumentMapper._safe_uuid(
                data.get("nomenclatureAffairId")
            ),
        )

    @staticmethod
    def _map_recipient(data: dict[str, Any] | None) -> DocumentRecipient | None:
        """Map DocumentRecipientDto → DocumentRecipient (FULL).

        Previous version only mapped 5 fields. Now maps all 15+ fields
        including status, delivered, type, to_people, unp, sign_date.

        Args:
            data: Raw DocumentRecipientDto dict or None.

        Returns:
            Populated ``DocumentRecipient`` or None.
        """
        if not data or not isinstance(data, dict):
            return None

        # ── delivery method name from nested object ────────────────────
        dm_raw = data.get("deliveryMethod") or {}
        dm_name: str | None = (
            dm_raw.get("deliveryName") if isinstance(dm_raw, dict) else None
        )

        # ── Parse enums ────────────────────────────────────────────────
        status = DocumentMapper._parse_enum(data.get("status"), RecipientStatus)
        rtype = DocumentMapper._parse_enum(data.get("type"), RecipientType)

        return DocumentRecipient(
            id=DocumentMapper._safe_uuid(data.get("id")),
            documentId=DocumentMapper._safe_uuid(data.get("documentId")),
            name=data.get("name"),
            status=status,
            delivered=data.get("delivered"),
            system=data.get("system"),
            type=rtype,
            toPeople=data.get("toPeople"),
            dateSend=DocumentMapper._parse_datetime(data.get("dateSend")),
            deliveryMethodName=dm_name,
            subscriberId=DocumentMapper._safe_uuid(data.get("subscriberId")),
            lock=data.get("lock"),
            unp=data.get("unp"),
            signDate=DocumentMapper._parse_datetime(data.get("signDate")),
            contractNumber=data.get("contractNumber"),
        )

    @staticmethod
    def _map_document_question(
        data: dict[str, Any],
    ) -> DocumentQuestion:
        """Map DocumentQuestionDto → DocumentQuestion.

        Args:
            data: Raw DocumentQuestionDto dict.

        Returns:
            Populated ``DocumentQuestion``.
        """
        speakers: list[DocumentSpeaker] = []
        for sp_raw in data.get("speakers", []) or []:
            if not isinstance(sp_raw, dict):
                continue
            emp = EmployeeMapper.to_user_info(sp_raw.get("employee"))
            sp_type = DocumentMapper._parse_enum(sp_raw.get("type"), SpeakerType)
            speakers.append(
                DocumentSpeaker(
                    id=DocumentMapper._safe_uuid(sp_raw.get("id")),
                    employee=emp,
                    type=sp_type,
                )
            )
        return DocumentQuestion(
            id=DocumentMapper._safe_uuid(data.get("id")),
            questionNumber=data.get("questionNumber"),
            question=data.get("question"),
            speakers=speakers,
        )

    # ── Parsing helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_datetime(raw: str | int | float | None):
        """Parse ISO string or Java ms timestamp to datetime."""
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
        """Generic enum parser with None fallback and warning."""
        if not raw:
            return None
        try:
            return enum_class(raw)
        except ValueError:
            logger.warning(
                "unknown_enum_value",
                extra={"enum": enum_class.__name__, "value": raw},
            )
            return None