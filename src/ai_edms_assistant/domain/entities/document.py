# src/ai_edms_assistant/domain/entities/document.py
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field

from .appeal import DocumentAppeal
from .attachment import Attachment, AttachmentType
from .base import DomainModel, MutableDomainModel
from .employee import UserInfo
from .task import Task

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocumentStatus(StrEnum):
    """Document lifecycle statuses.

    Follows the full EDMS document workflow from creation to archival.
    Each status corresponds to a workflow stage (процесс) in the Java backend.
    """

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
    """Document category determining available workflows and fields.

    The category drives which sections of ``Document`` are populated:
    - ``APPEAL`` — populates the ``appeal`` field (DocumentAppeal).
    - ``MEETING`` / ``MEETING_QUESTION`` — populates meeting-specific fields.
    - ``CONTRACT`` — enables contract execution workflow.
    """

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
    """How the document was created.

    Used to distinguish manual creation from external system integrations.
    """

    MANUAL = "MANUAL"
    AISMV = "AISMV"
    DIRECTUM = "DIRECTUM"


class MeetingFormType(StrEnum):
    """Meeting format type.

    Determines whether the meeting is held in person or by polling method.
    """

    INTRAMURAL = "INTRAMURAL"
    POLLING_METHOD = "POLLING_METHOD"


# ---------------------------------------------------------------------------
# Value objects embedded in Document
# ---------------------------------------------------------------------------


class DocumentVersion(DomainModel):
    """Immutable snapshot of a document version reference.

    Maps to ``DocumentVersionDto``. Frozen because version references are
    set once per document and treated as immutable within a request cycle.

    Attributes:
        id: Version record UUID.
        version_number: Sequential version counter.
        document_id: UUID of the parent document.
        create_date: Version creation timestamp.
        author: UserInfo of the version author.
    """

    id: UUID
    version_number: int = Field(alias="versionNumber")
    document_id: UUID | None = Field(default=None, alias="documentId")
    create_date: datetime | None = Field(default=None, alias="createDate")
    author: UserInfo | None = None


class ControlInfo(DomainModel):
    """Document / task control metadata.

    Maps to ``ControlDto``. Frozen because control parameters are set once
    and treated as immutable within a single request cycle.

    Attributes:
        control_type_id: UUID reference to the control type dictionary.
        control_date_start: Date when control was assigned.
        control_plan_date_end: Planned control end date.
        control_term_days: Control duration in calendar days.
        control_initiator_id: UUID of the employee who initiated control.
        control_employee_id: UUID of the controller (контролёр).
        control_real_date_end: Actual control completion date.
    """

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
    """Automatic control settings for a document (AutoControl DTO).

    Attributes:
        auto_control: Whether automatic control is enabled.
        control_days: Number of days for automatic control calculation.
        control_type_id: UUID of the control type to apply automatically.
    """

    auto_control: bool = Field(default=False, alias="autoControl")
    control_days: int | None = Field(default=None, alias="controlDays")
    control_type_id: UUID | None = Field(default=None, alias="controlTypeId")


class DocumentRecipient(DomainModel):
    """Addressee / correspondent of a document (DocumentRecipientDto).

    Used both for ``correspondent`` (single) and ``recipient_list`` (multiple).
    Frozen because recipient records represent a point-in-time snapshot of
    routing decisions.

    Attributes:
        id: Recipient UUID.
        name: Short recipient name.
        full_name: Full recipient name.
        smdo_code: SMDO integration code for inter-org document routing.
        delivery_method: Delivery method name (for LLM context).
        date_send: Date when the document was sent to this recipient.
    """

    id: UUID | None = None
    name: str | None = None
    full_name: str | None = Field(default=None, alias="fullName")
    smdo_code: str | None = Field(default=None, alias="smdoCode")
    delivery_method: str | None = Field(default=None, alias="deliveryMethod")
    date_send: datetime | None = Field(default=None, alias="dateSend")


# ---------------------------------------------------------------------------
# Central domain aggregate
# ---------------------------------------------------------------------------


class Document(MutableDomainModel):
    """Central domain aggregate representing an EDMS document.

    This is the primary entity of the entire system. It is a large aggregate
    that combines registration metadata, participants, attachments, tasks,
    and — conditionally — meeting or appeal data.

    Design decisions:
        - ``MutableDomainModel`` base: the document changes state throughout
          its lifecycle (DRAFT → REGISTERED → EXECUTION → ...).
        - Tasks are embedded as a list of ``Task`` entities, not as UUIDs,
          to allow the LLM agent to reason over task content without extra
          API calls.
        - ``appeal`` field is populated only when ``document_category == APPEAL``.
          Other category-specific fields (meeting, question) follow the same
          lazy population pattern.
        - ``custom_fields`` provides an escape hatch for organization-specific
          extensions without breaking the domain model.

    Attributes:
        id: Document UUID — primary key across all EDMS operations.
        organization_id: Multi-tenant organization identifier.
        document_category: Category driving available workflows and fields.
        document_type_id: Integer ID of the document type (from dictionary).
        document_type_name: Human-readable document type name (for LLM).
        profile_id: UUID of the document profile / template.
        profile_name: Human-readable profile name (for LLM).
        reg_number: Official registration number (for LLM context).
        reg_date: Official registration date.
        create_date: Document creation timestamp.
        reserved_reg_number: Pre-reserved registration number.
        reserved_reg_date: Date when the number was reserved.
        out_reg_number: Outgoing registration number (for correspondence).
        out_reg_date: Outgoing registration date.
        days_execution: Execution deadline in calendar days.
        skip_registration: Whether the document skips the registration step.
        short_summary: Document title / brief content (краткое содержание → LLM).
        summary: Full document body text (→ LLM).
        note: Internal note field.
        pages_count: Number of document pages.
        additional_pages: String description of attachment page count.
        exemplar_count: Total number of document copies.
        exemplar_number: Copy number of this instance.
        status: Current document status in the workflow.
        prev_status: Previous status (for transition analysis).
        create_type: How the document was created (manual / integration).
        author: UserInfo of the document author.
        responsible_executor: UserInfo of the primary responsible executor.
        initiator: UserInfo of the process initiator (for LLM).
        who_signed: UserInfo of the signer (for outgoing / internal docs).
        who_addressed: List of UserInfo who were addressed (подписанты).
        in_doc_signers: String list of signers (denormalized from API).
        correspondent_name: Correspondent name string (for LLM).
        correspondent_id: Correspondent UUID.
        correspondent: Structured correspondent object.
        recipient_list: List of document recipients.
        control_flag: Whether the document is under active control.
        remove_control: Flag indicating removal from control.
        control: Detailed control metadata.
        auto_control: Automatic control settings.
        dsp_flag: Whether the document has a DСП (limited distribution) stamp.
        version_flag: Whether this document record is a version.
        version: Current version metadata.
        document_version_id: UUID of the current document version.
        attachments: List of attached files (→ LLM for content analysis).
        tasks: List of associated tasks / assignments (→ LLM for context).
        received_doc_id: UUID of the document this was received in response to.
        answer_doc_id: UUID of the document created in response to this one.
        introduction_count: Total introduction list entries.
        introduction_complete_count: Completed introduction list entries.
        document_links_count: Number of linked documents.
        date_meeting: Meeting date (MEETING category only).
        start_meeting: Meeting start time.
        end_meeting: Meeting end time.
        place_meeting: Meeting location string (for LLM).
        chairperson: Meeting chairperson UserInfo.
        secretary: Meeting secretary UserInfo.
        external_invitees: Free-text string of external invitees.
        invitees_count: Count of internal invitees.
        meeting_form_type: Meeting format (in-person or polling).
        number_question: Question number in the meeting agenda.
        has_question: Whether the document contains agenda questions.
        date_question: Date of the agenda question.
        comment_question: Comment on the agenda question.
        document_meeting_question_id: UUID of the linked meeting-question document.
        addition: Whether this is an addition to the agenda.
        addition_meeting_question_id: UUID of the linked addition document.
        appeal: DocumentAppeal entity (APPEAL category only).
        custom_fields: Organization-specific extension fields.
    """

    id: UUID
    organization_id: str = Field(alias="organizationId")

    document_category: DocumentCategory | None = Field(
        default=None, alias="docCategoryConstant"
    )
    doc_category_constant: str | None = Field(default=None, alias="docCategoryConstant")
    document_type_id: int | None = Field(default=None, alias="documentTypeId")
    document_type_name: str | None = Field(default=None, alias="documentTypeName")
    profile_id: UUID | None = Field(default=None, alias="profileId")
    profile_name: str | None = Field(default=None, alias="profileName")

    journal_id: UUID | None = Field(default=None, alias="journalId")
    journal_number: int | None = Field(default=None, alias="journalNumber")

    reg_number: str | None = Field(default=None, alias="regNumber")
    reg_date: datetime | None = Field(default=None, alias="regDate")
    create_date: datetime | None = Field(default=None, alias="createDate")
    reserved_reg_number: str | None = Field(default=None, alias="reservedRegNumber")
    reserved_reg_date: datetime | None = Field(default=None, alias="reservedRegDate")
    out_reg_number: str | None = Field(default=None, alias="outRegNumber")
    out_reg_date: datetime | None = Field(default=None, alias="outRegDate")
    days_execution: int | None = Field(default=None, alias="daysExecution")
    skip_registration: bool = Field(default=False, alias="skipRegistration")

    short_summary: str | None = Field(default=None, alias="shortSummary")
    summary: str | None = None
    note: str | None = None
    pages_count: int = Field(default=0, alias="pages")
    additional_pages: str | None = Field(default=None, alias="additionalPages")
    exemplar_count: int | None = Field(default=None, alias="exemplarCount")
    exemplar_number: int | None = Field(default=None, alias="exemplarNumber")

    status: DocumentStatus | None = None
    prev_status: DocumentStatus | None = Field(default=None, alias="prevStatus")
    create_type: DocumentCreateType | None = Field(default=None, alias="createType")

    author: UserInfo | None = None
    responsible_executor: UserInfo | None = Field(
        default=None, alias="responsibleExecutor"
    )
    initiator: UserInfo | None = None
    who_signed: UserInfo | None = Field(default=None, alias="whoSigned")
    who_addressed: list[UserInfo] = Field(default_factory=list, alias="whoAddressed")
    in_doc_signers: str | None = Field(default=None, alias="inDocSigners")

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

    control_flag: bool = Field(default=False, alias="controlFlag")
    remove_control: bool = Field(default=False, alias="removeControl")
    control: ControlInfo | None = None
    auto_control: AutoControlSettings | None = Field(default=None, alias="autoControl")

    dsp_flag: bool = Field(default=False, alias="dspFlag")

    version_flag: bool = Field(default=False, alias="versionFlag")
    version: DocumentVersion | None = None
    document_version_id: UUID | None = Field(default=None, alias="documentVersionId")

    attachments: list[Attachment] = Field(
        default_factory=list, alias="attachmentDocument"
    )
    tasks: list[Task] = Field(default_factory=list, alias="taskList")

    received_doc_id: UUID | None = Field(default=None, alias="receivedDocId")
    answer_doc_id: UUID | None = Field(default=None, alias="answerDocId")

    ref_doc_id: UUID | None = Field(default=None, alias="refDocId")
    ref_doc_org_id: str | None = Field(default=None, alias="refDocOrgId")
    process_id: UUID | None = Field(default=None, alias="processId")

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

    appeal: DocumentAppeal | None = Field(default=None, alias="documentAppeal")

    custom_fields: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Domain properties
    # ------------------------------------------------------------------

    @property
    def is_registered(self) -> bool:
        """Returns True when the document has been formally registered.

        A document is considered registered when both ``reg_number`` and
        ``reg_date`` are present.

        Returns:
            ``True`` when the document has a registration number and date.
        """
        return bool(self.reg_number and self.reg_date)

    @property
    def is_appeal(self) -> bool:
        """Returns True for citizen-appeal documents.

        Returns:
            ``True`` when ``document_category == APPEAL``.
        """
        return self.document_category == DocumentCategory.APPEAL

    @property
    def is_meeting(self) -> bool:
        """Returns True for meeting-related documents.

        Returns:
            ``True`` for ``MEETING`` and ``MEETING_QUESTION`` categories.
        """
        return self.document_category in (
            DocumentCategory.MEETING,
            DocumentCategory.MEETING_QUESTION,
        )

    @property
    def is_on_control(self) -> bool:
        """Returns True when the document is under active control.

        Active control means ``control_flag=True`` and ``remove_control=False``.

        Returns:
            ``True`` when the document is actively controlled.
        """
        return self.control_flag and not self.remove_control

    @property
    def has_attachments(self) -> bool:
        """Returns True when the document has at least one attachment.

        Returns:
            ``True`` when the ``attachments`` list is non-empty.
        """
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

        Concatenates ``short_summary`` (title), ``summary`` (body), and
        ``note`` (annotations) with Russian-language section labels.

        Returns:
            Multi-line string suitable for LLM context injection.
            Empty string when none of the text fields are populated.

        Example:
            >>> doc.get_full_text()
            'Заголовок: О проведении совещания\\nСодержание: ...'
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

        Used for LLM context injection to provide agent with participant
        information without embedding the full entity objects.

        Returns:
            Semicolon-separated list of participants with role labels.
            Empty string when no participants are set.

        Example:
            >>> doc.get_participant_summary()
            'Автор: Иванов И.И.; Ответственный: Петров П.П.'
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
