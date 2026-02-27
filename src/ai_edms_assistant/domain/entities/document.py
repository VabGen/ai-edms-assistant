# src/ai_edms_assistant/domain/entities/document.py
"""Central EDMS document domain aggregate."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator, field_validator

from .appeal import DocumentAppeal
from .attachment import Attachment, AttachmentType
from .base import DomainModel, MutableDomainModel
from .employee import UserInfo
from .task import Task

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocumentStatus(StrEnum):
    """Document lifecycle statuses."""

    DRAFT = "DRAFT"
    NEW = "NEW"
    STATEMENT = "STATEMENT"
    APPROVED = "APPROVED"
    SIGNING = "SIGNING"
    SIGNED = "SIGNED"
    AGREEMENT = "AGREEMENT"
    AGREED = "AGREED"
    REVIEW = "REVIEW"
    REVIEWED = "REVIEWED"
    REGISTRATION = "REGISTRATION"
    REGISTERED = "REGISTERED"
    EXECUTION = "EXECUTION"
    EXECUTED = "EXECUTED"
    DISPATCH = "DISPATCH"
    SENT = "SENT"
    REJECT = "REJECT"
    CANCEL = "CANCEL"
    PREPARATION = "PREPARATION"
    PAPERWORK = "PAPERWORK"
    FORMALIZED = "FORMALIZED"
    ACCEPTANCE = "ACCEPTANCE"
    ACCEPTED = "ACCEPTED"
    CONTRACT_EXECUTION = "CONTRACT_EXECUTION"
    CONTRACT_CLOSED = "CONTRACT_CLOSED"
    ARCHIVE = "ARCHIVE"
    DELETED = "DELETED"
    ALL = "ALL"


class DocumentCategory(StrEnum):
    """Document category determining available workflows and fields."""

    INTERN = "INTERN"
    INCOMING = "INCOMING"
    OUTGOING = "OUTGOING"
    MEETING = "MEETING"
    QUESTION = "QUESTION"
    MEETING_QUESTION = "MEETING_QUESTION"
    APPEAL = "APPEAL"
    CONTRACT = "CONTRACT"
    CUSTOM = "CUSTOM"


class DocumentCreateType(StrEnum):
    """How the document was created."""

    MANUAL = "MANUAL"
    AISMV = "AISMV"
    DIRECTUM = "DIRECTUM"


class MeetingFormType(StrEnum):
    """Meeting format type."""

    INTRAMURAL = "INTRAMURAL"
    POLLING_METHOD = "POLLING_METHOD"


# ---------------------------------------------------------------------------
# Value objects embedded in Document
# ---------------------------------------------------------------------------


class DocumentVersion(DomainModel):
    """Immutable snapshot of a document version reference."""

    id: UUID
    version_number: int = Field(alias="versionNumber")
    document_id: UUID | None = Field(default=None, alias="documentId")
    create_date: datetime | None = Field(default=None, alias="createDate")
    author: UserInfo | None = None


class ControlInfo(DomainModel):
    """Document / task control metadata."""

    control_type_id: UUID = Field(alias="controlTypeId")
    control_date_start: datetime = Field(alias="controlDateStart")
    control_plan_date_end: datetime = Field(alias="controlPlanDateEnd")
    control_term_days: int = Field(alias="controlTermDays")
    control_initiator_id: UUID | None = Field(default=None, alias="controlInitiatorId")
    control_employee_id: UUID | None = Field(default=None, alias="controlEmployeeId")
    control_real_date_end: datetime | None = Field(
        default=None, alias="controlRealDateEnd"
    )


class AutoControlSettings(DomainModel):
    """Automatic control settings for a document."""

    auto_control: bool = Field(default=False, alias="autoControl")
    control_days: int | None = Field(default=None, alias="controlDays")
    control_type_id: UUID | None = Field(default=None, alias="controlTypeId")


class DocumentRecipient(DomainModel):
    """Addressee / correspondent of a document."""

    id: UUID | None = None
    name: str | None = None
    full_name: str | None = Field(default=None, alias="fullName")
    smdo_code: str | None = Field(default=None, alias="smdoCode")
    delivery_method: str | None = Field(default=None, alias="deliveryMethod")
    date_send: datetime | None = Field(default=None, alias="dateSend")


# ---------------------------------------------------------------------------
# Null-default mapping for Java API null fields
# ---------------------------------------------------------------------------

_BOOL_DEFAULTS: dict[str, bool] = {
    "skip_registration": False,
    "control_flag": False,
    "remove_control": False,
    "dsp_flag": False,
    "version_flag": False,
    "recipients": False,
    "has_responsible_executor": False,
    "addition": False,
}

_INT_DEFAULTS: dict[str, int] = {
    "pages_count": 0,
    "count_task": 0,
    "task_project_count": 0,
    "completed_task_count": 0,
    "write_off_affair_count": 0,
    "pre_affair_count": 0,
    "responsible_executors_count": 0,
}


# ---------------------------------------------------------------------------
# Central domain aggregate
# ---------------------------------------------------------------------------


class Document(MutableDomainModel):
    """Central domain aggregate representing an EDMS document.

    Attributes:
        id: Document UUID — primary key across all EDMS operations.
        organization_id: Multi-tenant organization identifier.
        (see full docstring in original file for all attributes)
    """

    # ── Identity ───────────────────────────────────────────────────────────
    id: UUID
    organization_id: str = Field(alias="organizationId")

    # ── Category & Type ────────────────────────────────────────────────────
    document_category: DocumentCategory | None = Field(
        default=None, alias="docCategoryConstant"
    )
    doc_category_constant: str | None = Field(default=None, alias="docCategoryConstant")
    document_type_id: int | None = Field(default=None, alias="documentTypeId")
    document_type_name: str | None = Field(default=None, alias="documentTypeName")
    profile_id: UUID | None = Field(default=None, alias="profileId")
    profile_name: str | None = Field(default=None, alias="profileName")

    # ── Journal ────────────────────────────────────────────────────────────
    journal_id: UUID | None = Field(default=None, alias="journalId")
    journal_number: int | None = Field(default=None, alias="journalNumber")

    # ── Registration ───────────────────────────────────────────────────────
    reg_number: str | None = Field(default=None, alias="regNumber")
    reg_date: datetime | None = Field(default=None, alias="regDate")
    create_date: datetime | None = Field(default=None, alias="createDate")
    reserved_reg_number: str | None = Field(default=None, alias="reservedRegNumber")
    reserved_reg_date: datetime | None = Field(default=None, alias="reservedRegDate")
    out_reg_number: str | None = Field(default=None, alias="outRegNumber")
    out_reg_date: datetime | None = Field(default=None, alias="outRegDate")
    days_execution: int | None = Field(default=None, alias="daysExecution")
    skip_registration: bool = Field(default=False, alias="skipRegistration")

    # ── Content ────────────────────────────────────────────────────────────
    short_summary: str | None = Field(default=None, alias="shortSummary")
    summary: str | None = None
    note: str | None = None

    pages_count: int = Field(default=0, alias="pages")
    additional_pages: str | None = Field(default=None, alias="additionalPages")
    exemplar_count: int | None = Field(default=None, alias="exemplarCount")
    exemplar_number: int | None = Field(default=None, alias="exemplarNumber")

    # ── Status ─────────────────────────────────────────────────────────────
    status: DocumentStatus | None = None
    prev_status: DocumentStatus | None = Field(default=None, alias="prevStatus")
    create_type: DocumentCreateType | None = Field(default=None, alias="createType")

    # ── Participants ───────────────────────────────────────────────────────
    author: UserInfo | None = None
    responsible_executor: UserInfo | None = Field(
        default=None, alias="responsibleExecutor"
    )
    initiator: UserInfo | None = None
    who_signed: UserInfo | None = Field(default=None, alias="whoSigned")
    who_addressed: list[UserInfo] = Field(default_factory=list, alias="whoAddressed")
    in_doc_signers: str | None = Field(default=None, alias="inDocSigners")

    # ── Correspondent ──────────────────────────────────────────────────────
    correspondent_name: str | None = Field(default=None, alias="correspondentName")
    correspondent_id: UUID | None = Field(default=None, alias="correspondentId")
    correspondent: DocumentRecipient | None = None
    country_name: str | None = Field(default=None, alias="countryName")
    country_id: UUID | None = Field(default=None, alias="countryId")
    recipient_list: list[DocumentRecipient] = Field(
        default_factory=list, alias="recipientList"
    )

    delivery_method_id: int | None = Field(default=None, alias="deliveryMethodId")
    invest_program_id: UUID | None = Field(default=None, alias="investProgramId")

    # ── Control ────────────────────────────────────────────────────────────
    control_flag: bool = Field(default=False, alias="controlFlag")
    remove_control: bool = Field(default=False, alias="removeControl")
    control: ControlInfo | None = None
    auto_control: AutoControlSettings | None = Field(default=None, alias="autoControl")
    dsp_flag: bool = Field(default=False, alias="dspFlag")

    # ── Version ────────────────────────────────────────────────────────────
    version_flag: bool = Field(default=False, alias="versionFlag")
    version: DocumentVersion | None = None
    document_version_id: UUID | None = Field(default=None, alias="documentVersionId")

    # ── Files & Tasks ──────────────────────────────────────────────────────
    attachments: list[Attachment] = Field(
        default_factory=list, alias="attachmentDocument"
    )
    tasks: list[Task] = Field(default_factory=list, alias="taskList")

    # ── Relations ──────────────────────────────────────────────────────────
    received_doc_id: UUID | None = Field(default=None, alias="receivedDocId")
    answer_doc_id: UUID | None = Field(default=None, alias="answerDocId")
    ref_doc_id: UUID | None = Field(default=None, alias="refDocId")
    ref_doc_org_id: str | None = Field(default=None, alias="refDocOrgId")
    process_id: UUID | None = Field(default=None, alias="processId")

    # ── Counters ───────────────────────────────────────────────────────────
    introduction_count: int | None = Field(default=None, alias="introductionCount")
    introduction_complete_count: int | None = Field(
        default=None, alias="introductionCompleteCount"
    )
    document_links_count: int | None = Field(default=None, alias="documentLinksCount")
    count_task: int = 0
    task_project_count: int = 0
    completed_task_count: int = 0
    write_off_affair_count: int = 0
    pre_affair_count: int = 0
    responsible_executors_count: int = 0

    # ── Meeting ────────────────────────────────────────────────────────────
    date_meeting: datetime | None = Field(default=None, alias="dateMeeting")
    date_meeting_question: datetime | None = Field(
        default=None, alias="dateMeetingQuestion"
    )
    start_meeting: datetime | None = Field(default=None, alias="startMeeting")
    end_meeting: datetime | None = Field(default=None, alias="endMeeting")
    place_meeting: str | None = Field(default=None, alias="placeMeeting")
    chairperson: UserInfo | None = None
    secretary: UserInfo | None = None
    external_invitees: str | None = Field(default=None, alias="externalInvitees")
    invitees_count: int | None = Field(default=None, alias="inviteesCount")
    meeting_form_type: MeetingFormType | None = Field(
        default=None, alias="formMeetingType"
    )

    # ── Question ───────────────────────────────────────────────────────────
    number_question: int | None = Field(default=None, alias="numberQuestion")
    has_question: bool | None = Field(default=None, alias="hasQuestion")

    recipients: bool = False
    has_responsible_executor: bool = False

    date_question: datetime | None = Field(default=None, alias="dateQuestion")
    comment_question: str | None = Field(default=None, alias="commentQuestion")
    document_meeting_question_id: UUID | None = Field(
        default=None, alias="documentMeetingQuestionId"
    )
    document_meeting_question_org_id: str | None = Field(
        default=None, alias="documentMeetingQuestionOrgId"
    )
    addition_meeting_question_org_id: str | None = Field(
        default=None, alias="additionMeetingQuestionOrgId"
    )
    addition: bool = False
    addition_meeting_question_id: UUID | None = Field(
        default=None, alias="additionMeetingQuestionId"
    )

    # ── Appeal ─────────────────────────────────────────────────────────────
    appeal: DocumentAppeal | None = Field(default=None, alias="documentAppeal")

    custom_fields: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pages_count", mode="before")
    @classmethod
    def coerce_pages_none(cls, v: int | None) -> int:
        """Coerces null pages from Java API to 0.

        Java ``DocumentDto.pages`` is ``Long`` (nullable). When the backend
        returns ``null``, Pydantic receives ``None`` which fails ``int``
        validation. Treat ``null`` as 'unknown / zero pages'.

        Args:
            v: Raw value from API response.

        Returns:
            Integer page count; 0 when value is None.
        """
        return v if v is not None else 0

    # ------------------------------------------------------------------
    # Null-safety validator (Java API compat)
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _normalize_java_nulls(cls, values: Any) -> Any:
        """
        Normalize Java nulls to Python defaults.
        Args:
            values: Raw input dict from API mapper.

        Returns:
            Cleaned dict with Java nulls replaced by Python defaults.
        """
        if not isinstance(values, dict):
            return values

        bool_fields_defaults: dict[str, bool] = {
            "skip_registration": False,
            "control_flag": False,
            "remove_control": False,
            "dsp_flag": False,
            "version_flag": False,
            "recipients": False,
            "has_responsible_executor": False,
            "addition": False,
        }
        for field_name, default in bool_fields_defaults.items():
            if values.get(field_name) is None:
                values[field_name] = default

        int_fields_defaults: dict[str, int] = {
            "pages_count": 0,
            "count_task": 0,
            "task_project_count": 0,
            "completed_task_count": 0,
            "write_off_affair_count": 0,
            "pre_affair_count": 0,
            "responsible_executors_count": 0,
        }
        for field_name, default in int_fields_defaults.items():
            if values.get(field_name) is None:
                values[field_name] = default

        if values.get("custom_fields") is None:
            values["custom_fields"] = {}

        list_fields = ["who_addressed", "attachments", "tasks", "recipient_list"]
        for field_name in list_fields:
            if values.get(field_name) is None:
                values[field_name] = []

        return values

    # ------------------------------------------------------------------
    # Domain properties
    # ------------------------------------------------------------------

    @property
    def is_registered(self) -> bool:
        """Returns True when the document has been formally registered."""
        return bool(self.reg_number and self.reg_date)

    @property
    def is_appeal(self) -> bool:
        """Returns True for citizen-appeal documents."""
        return self.document_category == DocumentCategory.APPEAL

    @property
    def is_meeting(self) -> bool:
        """Returns True for meeting-related documents."""
        return self.document_category in (
            DocumentCategory.MEETING,
            DocumentCategory.MEETING_QUESTION,
        )

    @property
    def is_on_control(self) -> bool:
        """Returns True when the document is under active control."""
        return self.control_flag and not self.remove_control

    @property
    def has_attachments(self) -> bool:
        """Returns True when the document has at least one attachment."""
        return bool(self.attachments)

    @property
    def main_attachment(self) -> Attachment | None:
        """Returns the primary attachment of the document.

        Searches for the first attachment with ``AttachmentType.ATTACHMENT``
        role. Falls back to the first attachment in the list if none has
        the expected type.

        Returns:
            The main ``Attachment`` instance, or ``None`` when no attachments.
        """
        for att in self.attachments:
            if att.attachment_type == AttachmentType.ATTACHMENT:
                return att
        return self.attachments[0] if self.attachments else None

    def get_full_text(self) -> str:
        """Assembles the document text context for LLM prompts.

        Returns:
            Multi-line string suitable for LLM context injection.
        """
        parts: list[str] = []
        if self.short_summary:
            parts.append(f"Заголовок: {self.short_summary}")
        if self.summary:
            parts.append(f"Содержание: {self.summary}")
        if self.note:
            parts.append(f"Примечание: {self.note}")
        return "\n".join(parts)

    def get_participant_summary(self) -> str:
        """Returns a human-readable summary of document participants.

        Returns:
            Semicolon-separated list of participants with role labels.
        """
        parts: list[str] = []
        if self.author:
            parts.append(f"Автор: {self.author.name}")
        if self.responsible_executor:
            parts.append(f"Ответственный: {self.responsible_executor.name}")
        if self.initiator:
            parts.append(f"Инициатор: {self.initiator.name}")
        if self.who_signed:
            parts.append(f"Подписан: {self.who_signed.name}")
        return "; ".join(parts)
