# src/ai_edms_assistant/domain/entities/document.py
"""Central EDMS document domain aggregate.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
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
    """Document lifecycle statuses (28 values). Maps to Java DocumentStatus."""

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

    @property
    def label(self) -> str:
        """Returns Russian label for display."""
        _labels: dict[str, str] = {
            "DRAFT": "Черновик", "NEW": "Новый", "STATEMENT": "На утверждении",
            "APPROVED": "Утверждён", "SIGNING": "На подписании", "SIGNED": "Подписан",
            "AGREEMENT": "На согласовании", "AGREED": "Согласован",
            "REVIEW": "На рассмотрении", "REVIEWED": "Рассмотрен",
            "REGISTRATION": "На регистрации", "REGISTERED": "Зарегистрирован",
            "EXECUTION": "На исполнении", "EXECUTED": "Исполнен",
            "DISPATCH": "На отправке", "SENT": "Отправлен",
            "REJECT": "Отклонён", "CANCEL": "Аннулирован",
            "PREPARATION": "Подготовка", "PAPERWORK": "На оформлении",
            "FORMALIZED": "Оформлен", "ACCEPTANCE": "На одобрении",
            "ACCEPTED": "Одобрен", "CONTRACT_EXECUTION": "Исполнение договора",
            "CONTRACT_CLOSED": "Закрыт", "ARCHIVE": "Архив", "DELETED": "Удалён",
            "ALL": "Все",
        }
        return _labels.get(self.value, self.value)


class DocumentCategory(StrEnum):
    """Document category — determines available workflows and fields."""

    INTERN = "INTERN"
    INCOMING = "INCOMING"
    OUTGOING = "OUTGOING"
    MEETING = "MEETING"
    QUESTION = "QUESTION"
    MEETING_QUESTION = "MEETING_QUESTION"
    APPEAL = "APPEAL"
    CONTRACT = "CONTRACT"
    CUSTOM = "CUSTOM"

    @property
    def label(self) -> str:
        """Returns Russian label for display."""
        _labels: dict[str, str] = {
            "INTERN": "Внутренний", "INCOMING": "Входящий", "OUTGOING": "Исходящий",
            "MEETING": "Совещание", "QUESTION": "Вопрос повестки",
            "MEETING_QUESTION": "Повестка заседания", "APPEAL": "Обращение",
            "CONTRACT": "Договор", "CUSTOM": "Пользовательский",
        }
        return _labels.get(self.value, self.value)


class DocumentCreateType(StrEnum):
    """How the document was created. Maps to Java DocumentCreateType."""

    MANUAL = "MANUAL"
    AISMV = "AISMV"
    DIRECTUM = "DIRECTUM"


class MeetingFormType(StrEnum):
    """Meeting format. Maps to Java FormMeetingQuestionType."""

    INTRAMURAL = "INTRAMURAL"
    POLLING_METHOD = "POLLING_METHOD"

    @property
    def label(self) -> str:
        """Returns Russian label."""
        return {"INTRAMURAL": "Очно", "POLLING_METHOD": "Методом опроса"}.get(
            self.value, self.value
        )


class RecipientStatus(StrEnum):
    """Delivery status of a document recipient. Maps to Java RecipientStatus."""

    SENDING = "SENDING"
    SENDED = "SENDED"
    RECEIVED = "RECEIVED"
    AISMV_SENDED = "AISMV_SENDED"
    AISMV_RECEIVED = "AISMV_RECEIVED"
    CANCEL = "CANCEL"
    FAILED = "FAILED"

    @property
    def label(self) -> str:
        """Returns Russian label."""
        _labels: dict[str, str] = {
            "SENDING": "Отправляется", "SENDED": "Отправлен",
            "RECEIVED": "Получен", "AISMV_SENDED": "Отправлен (АИС МВ)",
            "AISMV_RECEIVED": "Получен (АИС МВ)", "CANCEL": "Отменён",
            "FAILED": "Ошибка",
        }
        return _labels.get(self.value, self.value)


class RecipientType(StrEnum):
    """Type of recipient. Maps to Java RecipientType."""

    NORMAL = "NORMAL"
    AISMV = "AISMV"
    DIRECTUM = "DIRECTUM"
    GTB_ORG = "GTB_ORG"


class SpeakerType(StrEnum):
    """Speaker role in a meeting question. Maps to Java SpeakerType."""

    SPEAKER = "SPEAKER"
    CO_REPORTER = "CO_REPORTER"

    @property
    def label(self) -> str:
        """Returns Russian label."""
        return {"SPEAKER": "Докладчик", "CO_REPORTER": "Содокладчик"}.get(
            self.value, self.value
        )


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


class DocumentUserColor(DomainModel):
    """User-specific color marking for a document.

    Maps to Java ``DocumentUserColorDto``.
    Used to highlight documents in the UI per-user (not global).

    Attributes:
        id: Color record UUID.
        document_id: Parent document UUID.
        color: Hex color string (e.g. '#FF5733') or color name.
    """

    id: UUID | None = None
    document_id: UUID | None = Field(default=None, alias="documentId")
    color: str | None = None


class CurrencyInfo(DomainModel):
    """Currency reference value object.

    Maps to Java ``CurrencyDto``.
    Lightweight subset needed for LLM context (id + name).

    Attributes:
        id: Currency UUID in EDMS dictionary.
        name: Currency name (e.g. 'Белорусский рубль').
        code: ISO currency code (e.g. 'BYR', 'USD').
    """

    id: UUID | None = None
    name: str | None = None
    code: str | None = None


class RegistrationJournalInfo(DomainModel):
    """Registration journal reference.

    Maps to Java ``RegistrationJournalDto`` (lightweight subset).
    Needed by LLM to display journal name instead of UUID.

    Attributes:
        id: Journal UUID.
        journal_name: Human-readable journal name.
        counter_value: Current counter value (last registered number).
    """

    id: UUID | None = None
    journal_name: str | None = Field(default=None, alias="journalName")
    counter_value: int | None = Field(default=None, alias="counterValue")


class DocumentUserProps(DomainModel):
    """Per-user task statistics for this document.

    Attributes:
        create_task_count: Total tasks created by user on this document.
        create_task_executed_count: Completed tasks created by user.
    """

    create_task_count: int | None = Field(default=None, alias="createTaskCount")
    create_task_executed_count: int | None = Field(
        default=None, alias="createTaskExecutedCount"
    )


class DocumentSpeaker(DomainModel):
    """Speaker record in a meeting question.

    Maps to Java ``SpeakerDto``.

    Attributes:
        id: Speaker record UUID.
        employee: Speaker UserInfo.
        speaker_type: SPEAKER or CO_REPORTER.
    """

    id: UUID | None = None
    employee: UserInfo | None = None
    speaker_type: SpeakerType | None = Field(default=None, alias="type")

    @property
    def type_label(self) -> str:
        """Returns Russian label for speaker type."""
        return self.speaker_type.label if self.speaker_type else "Докладчик"


class DocumentQuestion(DomainModel):
    """Question item in a meeting agenda.

    Maps to Java ``DocumentQuestionDto``.
    Used in MEETING_QUESTION documents.

    Attributes:
        id: Question UUID.
        question_number: Ordinal number in agenda.
        question: Question text / formulation.
        speakers: List of speakers for this question.
    """

    id: UUID | None = None
    question_number: int | None = Field(default=None, alias="questionNumber")
    question: str | None = None
    speakers: list[DocumentSpeaker] = Field(default_factory=list)


class DocumentRecipient(DomainModel):
    """Addressee / correspondent of a document.

    Attributes:
        id: Recipient record UUID.
        document_id: Parent document UUID.
        name: Recipient display name.
        status: Delivery status (SENDING → RECEIVED, etc.).
        delivered: True when physical delivery is confirmed.
        system: True when sent automatically by the system.
        type: Recipient type (NORMAL / AISMV / DIRECTUM / GTB_ORG).
        to_people: Free-text addressee name override.
        date_send: Timestamp when the document was sent.
        delivery_method_name: Human-readable delivery method (e.g. 'Курьер').
        subscriber_id: SMDO subscriber UUID.
        lock: True when the recipient record is locked.
        unp: UNP (taxpayer ID) for contract correspondents.
        sign_date: Date when correspondent signed the contract.
        contract_number: Correspondent's contract number.
    """

    id: UUID | None = None
    document_id: UUID | None = Field(default=None, alias="documentId")
    name: str | None = None

    # Delivery tracking
    status: RecipientStatus | None = None
    delivered: bool | None = None
    system: bool | None = None
    type: RecipientType | None = None
    to_people: str | None = Field(default=None, alias="toPeople")
    date_send: datetime | None = Field(default=None, alias="dateSend")
    delivery_method_name: str | None = Field(
        default=None, alias="deliveryMethodName"
    )
    subscriber_id: UUID | None = Field(default=None, alias="subscriberId")
    lock: bool | None = None

    # Contract correspondent fields
    unp: str | None = None
    sign_date: datetime | None = Field(default=None, alias="signDate")
    contract_number: str | None = Field(default=None, alias="contractNumber")

    @property
    def status_label(self) -> str:
        """Returns Russian label for delivery status."""
        return self.status.label if self.status else "Не указан"

    @property
    def is_delivered(self) -> bool:
        """True when delivery is confirmed."""
        return bool(self.delivered) or self.status == RecipientStatus.RECEIVED


# ---------------------------------------------------------------------------
# Central domain aggregate
# ---------------------------------------------------------------------------


class Document(MutableDomainModel):
    """Central domain aggregate for an EDMS document.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    id: UUID
    organization_id: str | None = Field(default=None, alias="organizationId")

    # ── Category & Type ───────────────────────────────────────────────────────
    document_category: DocumentCategory | None = Field(
        default=None, alias="docCategoryConstant"
    )
    doc_category_constant: str | None = Field(
        default=None, alias="docCategoryConstantStr"
    )
    profile_name: str | None = Field(default=None, alias="profileName")
    profile_id: UUID | None = Field(default=None, alias="profileId")
    document_type_name: str | None = Field(default=None, alias="documentTypeName")
    document_type_id: int | None = Field(default=None, alias="documentTypeId")
    days_execution: int | None = Field(default=None, alias="daysExecution")

    formula: list[str] = Field(default_factory=list)

    required_field: list[str] = Field(default_factory=list, alias="requiredField")

    # ── Color marking (per-user) ──────────────────────────────────────────────
    # NEW: color = DocumentUserColorDto
    color: DocumentUserColor | None = None

    # ── Registration ──────────────────────────────────────────────────────────
    reg_number: str | None = Field(default=None, alias="regNumber")
    reserved_reg_number: str | None = Field(
        default=None, alias="reservedRegNumber"
    )
    out_reg_number: str | None = Field(default=None, alias="outRegNumber")
    reg_date: datetime | None = Field(default=None, alias="regDate")
    reserved_reg_date: datetime | None = Field(
        default=None, alias="reservedRegDate"
    )
    out_reg_date: datetime | None = Field(default=None, alias="outRegDate")
    journal_id: UUID | None = Field(default=None, alias="journalId")
    journal_number: int | None = Field(default=None, alias="journalNumber")
    registration_journal: RegistrationJournalInfo | None = Field(
        default=None, alias="registrationJournal"
    )
    create_date: datetime | None = Field(default=None, alias="createDate")

    # ── Physical properties ───────────────────────────────────────────────────
    pages_count: int = Field(default=0, alias="pages")
    additional_pages: str | None = Field(default=None, alias="additionalPages")
    exemplar_count: int | None = Field(default=None, alias="exemplarCount")
    exemplar_number: int | None = Field(default=None, alias="exemplarNumber")

    # ── Status ────────────────────────────────────────────────────────────────
    status: DocumentStatus | None = None
    prev_status: DocumentStatus | None = Field(default=None, alias="prevStatus")
    create_type: DocumentCreateType | None = Field(default=None, alias="createType")
    current_bpmn_task_name: str | None = Field(
        default=None, alias="currentBpmnTaskName"
    )

    # ── Participants ──────────────────────────────────────────────────────────
    author: UserInfo | None = None
    responsible_executor: UserInfo | None = Field(
        default=None, alias="responsibleExecutor"
    )
    initiator: UserInfo | None = None
    who_signed: UserInfo | None = Field(default=None, alias="whoSigned")
    who_addressed: list[UserInfo] = Field(
        default_factory=list, alias="whoAddressed"
    )
    in_doc_signers: str | None = Field(default=None, alias="inDocSigners")
    # NEW: responsible_executors — список ответственных за подготовку материалов
    responsible_executors: list[UserInfo] = Field(
        default_factory=list, alias="responsibleExecutors"
    )

    # ── Correspondent ─────────────────────────────────────────────────────────
    correspondent_name: str | None = Field(default=None, alias="correspondentName")
    correspondent_id: UUID | None = Field(default=None, alias="correspondentId")
    correspondent: DocumentRecipient | None = None
    country_name: str | None = Field(default=None, alias="countryName")
    country_id: UUID | None = Field(default=None, alias="countryId")
    recipient_list: list[DocumentRecipient] = Field(
        default_factory=list, alias="recipientList"
    )
    delivery_method_id: int | None = Field(default=None, alias="deliveryMethodId")
    delivery_method_name: str | None = Field(
        default=None, alias="deliveryMethodName"
    )
    invest_program_id: UUID | None = Field(default=None, alias="investProgramId")

    # ── Control ───────────────────────────────────────────────────────────────
    control_flag: bool = Field(default=False, alias="controlFlag")
    remove_control: bool = Field(default=False, alias="removeControl")
    control: ControlInfo | None = None
    auto_control: AutoControlSettings | None = Field(
        default=None, alias="autoControl"
    )
    dsp_flag: bool = Field(default=False, alias="dspFlag")
    skip_registration: bool = Field(default=False, alias="skipRegistration")
    enable_access_grief: bool = Field(default=False, alias="enableAccessGrief")
    access_grief_id: UUID | None = Field(default=None, alias="accessGriefId")
    auto_routing: bool | None = Field(default=None, alias="autoRouting")

    # ── Version ───────────────────────────────────────────────────────────────
    version_flag: bool = Field(default=False, alias="versionFlag")
    version: DocumentVersion | None = None
    document_version_id: UUID | None = Field(
        default=None, alias="documentVersionId"
    )

    # ── Files & Tasks ──────────────────────────────────────────────────────────
    attachments: list[Attachment] = Field(
        default_factory=list, alias="attachmentDocument"
    )
    tasks: list[Task] = Field(default_factory=list, alias="taskList")

    # ── Relations ─────────────────────────────────────────────────────────────
    received_doc_id: UUID | None = Field(default=None, alias="receivedDocId")
    answer_doc_id: UUID | None = Field(default=None, alias="answerDocId")
    ref_doc_id: UUID | None = Field(default=None, alias="refDocId")
    ref_doc_org_id: str | None = Field(default=None, alias="refDocOrgId")
    process_id: UUID | None = Field(default=None, alias="processId")

    # ── Counters ───────────────────────────────────────────────────────────────
    introduction_count: int | None = Field(default=None, alias="introductionCount")
    introduction_complete_count: int | None = Field(
        default=None, alias="introductionCompleteCount"
    )
    document_links_count: int | None = Field(
        default=None, alias="documentLinksCount"
    )
    count_task: int = 0
    task_project_count: int = 0
    completed_task_count: int = 0
    write_off_affair_count: int = 0
    pre_affair_count: int = 0
    responsible_executors_count: int = 0
    meeting_question_notify_count: int = Field(
        default=0, alias="meetingQuestionNotifyCount"
    )

    # ── Per-user stats ────────────────────────────────────────────────────────
    user_props: DocumentUserProps | None = Field(
        default=None, alias="userProps"
    )

    # ── Meeting ────────────────────────────────────────────────────────────────
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
    document_questions: list[DocumentQuestion] = Field(
        default_factory=list, alias="documentQuestions"
    )

    # ── Question ───────────────────────────────────────────────────────────────
    number_question: int | None = Field(default=None, alias="numberQuestion")
    has_question: bool | None = Field(default=None, alias="hasQuestion")
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
    has_responsible_executor: bool | None = Field(
        default=None, alias="hasResponsibleExecutor"
    )
    recipients: bool = False

    # ── Content ────────────────────────────────────────────────────────────────
    short_summary: str | None = Field(default=None, alias="shortSummary")
    summary: str | None = None
    note: str | None = None

    # ── Contract ───────────────────────────────────────────────────────────────
    contract_sum: Decimal | None = Field(default=None, alias="contractSum")
    contract_number: str | None = Field(default=None, alias="contractNumber")
    contract_date: datetime | None = Field(default=None, alias="contractDate")
    contract_signing_date: datetime | None = Field(
        default=None, alias="contractSigningDate"
    )
    contract_start_date: datetime | None = Field(
        default=None, alias="contractStartDate"
    )
    contract_duration_start: datetime | None = Field(
        default=None, alias="contractDurationStart"
    )
    contract_duration_end: datetime | None = Field(
        default=None, alias="contractDurationEnd"
    )
    contract_agreement: bool | None = Field(
        default=None, alias="contractAgreement"
    )
    contract_auto_prolongation: bool | None = Field(
        default=None, alias="contractAutoProlongation"
    )
    contract_typical: bool = Field(default=False, alias="contractTypical")
    currency_id: UUID | None = Field(default=None, alias="currencyId")
    currency: CurrencyInfo | None = None

    # ── Appeal ─────────────────────────────────────────────────────────────────
    appeal: DocumentAppeal | None = Field(
        default=None, alias="documentAppeal"
    )

    # ── Custom & Form ──────────────────────────────────────────────────────────
    custom_fields: dict[str, Any] = Field(
        default_factory=dict, alias="customFields"
    )
    document_form_id: UUID | None = Field(
        default=None, alias="documentFormId"
    )

    # ── Computed properties ────────────────────────────────────────────────────

    @field_validator("pages_count", mode="before")
    @classmethod
    def coerce_pages_none(cls, v: int | None) -> int:
        """Coerces null pages from Java API to 0."""
        return v if v is not None else 0

    @property
    def is_registered(self) -> bool:
        """True when document has a registration number."""
        return bool(self.reg_number)

    @property
    def is_contract(self) -> bool:
        """True when document category is CONTRACT."""
        return self.document_category == DocumentCategory.CONTRACT

    @property
    def is_appeal(self) -> bool:
        """True when document category is APPEAL."""
        return self.document_category == DocumentCategory.APPEAL

    @property
    def is_meeting(self) -> bool:
        """True when document category is MEETING or MEETING_QUESTION."""
        return self.document_category in (
            DocumentCategory.MEETING,
            DocumentCategory.MEETING_QUESTION,
        )

    @property
    def author_full_name(self) -> str | None:
        """Returns formatted full name of the author."""
        if not self.author:
            return None
        parts = [
            getattr(self.author, "last_name", "") or "",
            getattr(self.author, "first_name", "") or "",
            getattr(self.author, "middle_name", "") or "",
        ]
        return " ".join(p for p in parts if p).strip() or None

    @property
    def status_label(self) -> str:
        """Returns Russian label for current status."""
        return self.status.label if self.status else "Не указан"

    @property
    def category_label(self) -> str:
        """Returns Russian label for document category."""
        return self.document_category.label if self.document_category else "Не указана"

    @property
    def formula_text(self) -> str | None:
        """Returns formula as joined string for display.

        Java formula is List<String>. This property joins parts for LLM context.

        Returns:
            Formula string like '01-01/{year}' or None if empty.
        """
        if not self.formula:
            return None
        return "/".join(self.formula)