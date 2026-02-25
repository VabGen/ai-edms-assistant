# src/ai_edms_assistant/domain/value_objects/filters.py
"""Domain filter value objects for EDMS API queries.

Design decisions:
    - Filters are Pydantic models (``DomainModel``) — validated on creation,
      not just on serialization. This catches invalid combinations early.
    - Enums that duplicate ``domain/entities/`` are intentionally NOT
      re-defined here. Import ``DocumentStatus``, ``DocumentCategory``,
      ``TaskType``, ``TaskStatus``, ``DeclarantType`` from their entity modules.
    - ``FilterValidationError`` lives in ``domain/exceptions/`` and is imported
      here for use in ``validate()`` methods.
    - ``as_api_params()`` / ``as_api_payload()`` methods live on the filter
      objects to keep serialization logic co-located with the filter definition.

Java sources:
    - ``DocumentFilter`` → ``DocumentFilter``
    - ``DocumentFilterValidator`` → ``DocumentFilter.validate()``
    - ``DocumentLinkFilter`` → ``DocumentLinkFilter``
    - ``EmployeeFilter`` → ``EmployeeFilter``
    - ``UserActionFilter`` → ``UserActionFilter``
    - ``TaskFilter`` → ``TaskFilter``
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from ..entities.base import DomainModel
from ..entities.document import DocumentCategory, DocumentCreateType, DocumentStatus
from ..entities.appeal import DeclarantType
from ..entities.task import TaskStatus
from ..exceptions.validation_exceptions import FilterValidationError

# ---------------------------------------------------------------------------
# DocumentFilter enums (filter-specific — not in domain entities)
# ---------------------------------------------------------------------------


class DocumentIoOption(StrEnum):
    """IO-filtering option for document queries.

    Java ``DocumentFilter.IoOption``. Controls whether substitutes (ИО)
    and subordinates are included in ``current_user_*`` flag queries.

    Attributes:
        SELF: Only the current user.
        IO: Only the user's IO (исполняющий обязанности).
        SUBORDINATES: All subordinates.
        SELF_AND_IO: Current user and their IO (default).
        SUBORDINATES_AND_SELF_AND_IO: Full transitive scope.
    """

    SELF = "SELF"
    IO = "IO"
    SUBORDINATES = "SUBORDINATES"
    SELF_AND_IO = "SELF_AND_IO"
    SUBORDINATES_AND_SELF_AND_IO = "SUBORDINATES_AND_SELF_AND_IO"


class DocumentFilterInclude(StrEnum):
    """Related models to eager-load alongside a document.

    Java ``DocumentFilter.Include``.
    """

    DOCUMENT_TYPE = "DOCUMENT_TYPE"
    DELIVERY_METHOD = "DELIVERY_METHOD"
    CORRESPONDENT = "CORRESPONDENT"
    RECIPIENT = "RECIPIENT"
    USER_COLOR = "USER_COLOR"
    PRE_NOMENCLATURE_AFFAIRS = "PRE_NOMENCLATURE_AFFAIRS"
    CITIZEN_TYPE = "CITIZEN_TYPE"
    CURRENCY = "CURRENCY"
    PARENT_SUBJECT = "PARENT_SUBJECT"
    REGISTRATION_JOURNAL = "REGISTRATION_JOURNAL"
    SOLUTION_RESULT = "SOLUTION_RESULT"
    ADDITIONAL_DOCUMENT_AND_TYPE = "ADDITIONAL_DOCUMENT_AND_TYPE"


class DocumentLinkType(StrEnum):
    """Type of cross-reference link between documents."""

    RECEIVED_IN_RESPONSE_TO = "RECEIVED_IN_RESPONSE_TO"
    CREATE_IN_RESPONSE_TO = "CREATE_IN_RESPONSE_TO"
    LINK_DOC = "LINK_DOC"


class DocumentLinkFilterInclude(StrEnum):
    """Related models for document link queries."""

    DOCUMENT = "DOCUMENT"
    DOCUMENT_LINK = "DOCUMENT_LINK"


class EmployeeFilterInclude(StrEnum):
    """Related models to eager-load alongside an employee.

    ``EmployeeFilter.Include``. Both values are included by default
    in all agent employee-search calls.
    """

    POST = "POST"
    DEPARTMENT = "DEPARTMENT"


class UserActionType(StrEnum):
    """User activity log action types."""

    DOCUMENT_CREATE = "DOCUMENT_CREATE"
    DOCUMENT_UPDATE = "DOCUMENT_UPDATE"
    DOCUMENT_DELETE = "DOCUMENT_DELETE"
    DOCUMENT_READ = "DOCUMENT_READ"
    TASK_CREATE = "TASK_CREATE"
    TASK_UPDATE = "TASK_UPDATE"
    TASK_EXECUTE = "TASK_EXECUTE"
    TASK_DELETE = "TASK_DELETE"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"


class TaskFilterInclude(StrEnum):
    """Related models to eager-load alongside a task."""

    DOCUMENT = "DOCUMENT"
    DOCUMENT_CORRESPONDENT = "DOCUMENT_CORRESPONDENT"


# ---------------------------------------------------------------------------
# DocumentFilter
# ---------------------------------------------------------------------------


class DocumentFilter(DomainModel):
    """Domain filter value object for document list queries.

    Direct Python port of Java ``DocumentFilter`` — all 70+ fields preserved.
    Validated on construction via Pydantic and on serialization via
    ``validate_business_rules()``.

    Enums reused from ``domain/entities/``:
        - ``DocumentStatus`` → from ``entities.document``
        - ``DocumentCategory`` → from ``entities.document``
        - ``DocumentCreateType`` → from ``entities.document``
        - ``DeclarantType`` → from ``entities.appeal``

    Attributes:
        includes: Related models to eager-load.
        category_constants: Document category filter (AND condition).
        io_option: IO-filtering scope. Defaults to ``SELF_AND_IO``.
        status: Allowed document statuses (AND).
        exclude_statuses: Excluded document statuses (AND NOT).

    Example:
        >>> f = DocumentFilter(
        ...     category_constants=[DocumentCategory.INCOMING],
        ...     status=[DocumentStatus.REGISTERED],
        ...     author_last_name="Иванов",
        ... )
        >>> params = f.as_api_params()
    """

    includes: list[DocumentFilterInclude] = Field(default_factory=list)
    category_constants: list[DocumentCategory] = Field(default_factory=list)
    document_type_ids: list[int] = Field(default_factory=list)
    document_type_names: list[str] = Field(default_factory=list)
    io_option: DocumentIoOption = DocumentIoOption.SELF_AND_IO
    create_type: DocumentCreateType | None = None

    reg_number: str | None = None
    reg_number_and: str | None = None
    date_reg_start: datetime | None = None
    date_reg_end: datetime | None = None
    out_reg_number: str | None = None
    out_reg_number_or: str | None = None
    has_reg_date: bool | None = None
    reg_number_is_not_null: bool = False
    journal_id: UUID | None = None

    short_summary: str | None = None
    short_summary_and: str | None = None

    author_id: UUID | None = None
    author_first_name: str | None = None
    author_last_name: str | None = None
    author_last_name_or: str | None = None
    author_middle_name: str | None = None
    author_current_user: bool | None = None

    executor_signing_stage_id: UUID | None = None
    executor_signing_stage_first_name: str | None = None
    executor_signing_stage_last_name: str | None = None
    executor_signing_stage_middle_name: str | None = None

    recipient_ids: list[UUID] = Field(default_factory=list)
    recipient_id: UUID | None = None
    recipient_name: str | None = None
    recipient_name_or: str | None = None
    correspondent_ids: list[UUID] = Field(default_factory=list)
    correspondent_name: str | None = None
    correspondent_name_or: str | None = None
    correspondent_recipient_id: UUID | None = None
    recipient_correspondent_id: UUID | None = None

    status: list[DocumentStatus] = Field(default_factory=list)
    exclude_statuses: list[DocumentStatus] = Field(default_factory=list)

    delivery_method_id: int | None = None

    date_control_start: datetime | None = None
    date_control_end: datetime | None = None

    task_executor_ids: list[UUID] = Field(default_factory=list)
    task_executor_first_name: str | None = None
    task_executor_last_name: str | None = None
    task_executor_middle_name: str | None = None
    date_task_start: datetime | None = None
    date_task_end: datetime | None = None

    process_executor_current_user: bool | None = None
    meeting_executor_current_user: bool | None = None
    meeting_question_executor_current_user: bool | None = None
    question_executor_current_user: bool | None = None
    contract_executor_current_user: bool | None = None
    control_user_current_user: bool | None = None
    introduction_current_user: bool | None = None
    additional_agreement_user: bool | None = None
    additional_agreement_all_user: bool | None = None
    signing_current_user: bool | None = None
    review_current_user: bool | None = None
    agreement_current_user: bool | None = None
    statement_current_user: bool | None = None
    task_executor_current_user: bool | None = None
    introduction_all_current_user: bool | None = None
    control_off_current_user: bool | None = None
    control_expire_current_user: bool | None = None
    control_expire_execution_current_user: bool | None = None
    execution_expire_current_user: bool | None = None
    execution_overdue_current_user: bool | None = None
    author_expire_execution_current_user: bool | None = None
    author_overdue_execution_current_user: bool | None = None
    smdo_wait_outgoing_user: bool | None = None
    smdo_fail_outgoing_user: bool | None = None

    user_color_is_null: bool | None = None
    user_color: str | None = None

    only_user_organization: bool | None = None
    process_completed: bool | None = None

    organization_name: str | None = None
    organization_name_or: str | None = None
    fio_applicant: str | None = None
    fio_applicant_or: str | None = None
    declarant_type: DeclarantType | None = None
    repeat_identical: str | None = None
    anonymous: bool | None = None
    collective: bool | None = None
    reasonably: bool | None = None
    citizen_type_id: UUID | None = None
    region_id: UUID | None = None
    country_appeal_id: UUID | None = None
    district_id: UUID | None = None
    city_id: UUID | None = None
    receipt_date_start: datetime | None = None
    receipt_date_end: datetime | None = None

    contract_agreement: bool | None = None
    contract_auto_prolongation: bool | None = None
    contract_typical: bool | None = None
    has_additional_document: bool | None = None
    recipient_signed_from: datetime | None = None
    recipient_signed_to: datetime | None = None
    sign_date_from: datetime | None = None
    sign_date_to: datetime | None = None
    currency_ids: list[UUID] = Field(default_factory=list)
    contract_sum_from: Decimal | None = None
    contract_sum_to: Decimal | None = None
    days_to_completion: int | None = None
    recipient_contract_number_or: str | None = None

    affair_exist: bool | None = None
    investment_program_ids: list[UUID] = Field(default_factory=list)
    create_year: int | None = None

    @model_validator(mode="after")
    def validate_business_rules(self) -> "DocumentFilter":
        """Validates mutual-exclusion rules from Java DocumentFilterValidator.

        Rules (direct port from Java):
            1. ``author_current_user`` cannot be combined with any explicit
               author field (``author_id``, names).
            2. ``author_id`` cannot be combined with author name fields.
            3. ``author_current_user`` cannot be ``False``.
            4. ``introduction_current_user`` cannot be ``False``.

        Returns:
            The validated filter instance (for chaining).

        Raises:
            FilterValidationError: On the first violated rule.
        """
        has_author_name = any(
            [
                self.author_first_name,
                self.author_last_name,
                self.author_middle_name,
            ]
        )
        has_author_explicit = bool(self.author_id) or has_author_name

        if has_author_explicit and self.author_current_user is not None:
            raise FilterValidationError(
                "author_current_user конфликтует с явными полями автора "
                "(author_id / author_first_name / author_last_name / author_middle_name)",
                field="author_current_user",
            )
        if self.author_id is not None and has_author_name:
            raise FilterValidationError(
                "author_id конфликтует с полями ФИО автора",
                field="author_id",
            )
        if not self.author_current_user:
            raise FilterValidationError(
                "author_current_user не может быть False — используйте None",
                field="author_current_user",
            )
        if self.introduction_current_user is False:
            raise FilterValidationError(
                "introduction_current_user не может быть False — используйте None",
                field="introduction_current_user",
            )
        return self

    def as_api_params(self) -> dict:
        """Serializes to a flat query-params dict for EDMS GET /api/document.

        Only non-None / non-empty values are included. UUIDs are converted
        to strings. Datetimes are serialized as ISO 8601.

        Returns:
            Dict suitable for ``params=`` in ``httpx.AsyncClient.get()``.
        """
        params: dict = {}

        _list_enums = {
            "includes": self.includes,
            "categoryConstants": self.category_constants,
            "status": self.status,
            "excludeStatuses": self.exclude_statuses,
        }
        for key, lst in _list_enums.items():
            if lst:
                params[key] = [v.value for v in lst]

        _list_ints = {"documentTypeIds": self.document_type_ids}
        for key, lst in _list_ints.items():
            if lst:
                params[key] = lst

        _list_strs = {"documentTypeNames": self.document_type_names}
        for key, lst in _list_strs.items():
            if lst:
                params[key] = lst

        _list_uuids: dict[str, list[UUID]] = {
            "recipientIds": self.recipient_ids,
            "correspondentIds": self.correspondent_ids,
            "taskExecutorIds": self.task_executor_ids,
            "currencyIds": self.currency_ids,
            "investmentProgramIds": self.investment_program_ids,
        }
        for key, lst in _list_uuids.items():
            if lst:
                params[key] = [str(u) for u in lst]

        _str_map = {
            "regNumber": self.reg_number,
            "regNumberAnd": self.reg_number_and,
            "outRegNumber": self.out_reg_number,
            "outRegNumberOr": self.out_reg_number_or,
            "shortSummary": self.short_summary,
            "shortSummaryAnd": self.short_summary_and,
            "authorFirstName": self.author_first_name,
            "authorLastName": self.author_last_name,
            "authorLastNameOr": self.author_last_name_or,
            "authorMiddleName": self.author_middle_name,
            "executorSigningStageFirstName": self.executor_signing_stage_first_name,
            "executorSigningStageLastName": self.executor_signing_stage_last_name,
            "executorSigningStageMiddleName": self.executor_signing_stage_middle_name,
            "recipientName": self.recipient_name,
            "recipientNameOr": self.recipient_name_or,
            "correspondentName": self.correspondent_name,
            "correspondentNameOr": self.correspondent_name_or,
            "taskExecutorFirstName": self.task_executor_first_name,
            "taskExecutorLastName": self.task_executor_last_name,
            "taskExecutorMiddleName": self.task_executor_middle_name,
            "userColor": self.user_color,
            "organizationName": self.organization_name,
            "organizationNameOr": self.organization_name_or,
            "fioApplicant": self.fio_applicant,
            "fioApplicantOr": self.fio_applicant_or,
            "repeatIdentical": self.repeat_identical,
            "recipientContractNumberOr": self.recipient_contract_number_or,
        }
        for key, val in _str_map.items():
            if val is not None:
                params[key] = val

        _uuid_map: dict[str, UUID | None] = {
            "authorId": self.author_id,
            "executorSigningStageId": self.executor_signing_stage_id,
            "recipientId": self.recipient_id,
            "correspondentRecipientId": self.correspondent_recipient_id,
            "recipientCorrespondentId": self.recipient_correspondent_id,
            "journalId": self.journal_id,
            "citizenTypeId": self.citizen_type_id,
            "regionId": self.region_id,
            "countryAppealId": self.country_appeal_id,
            "districtId": self.district_id,
            "cityId": self.city_id,
        }
        for key, val in _uuid_map.items():
            if val is not None:
                params[key] = str(val)

        _date_map: dict[str, datetime | None] = {
            "dateRegStart": self.date_reg_start,
            "dateRegEnd": self.date_reg_end,
            "dateControlStart": self.date_control_start,
            "dateControlEnd": self.date_control_end,
            "dateTaskStart": self.date_task_start,
            "dateTaskEnd": self.date_task_end,
            "receiptDateStart": self.receipt_date_start,
            "receiptDateEnd": self.receipt_date_end,
            "recipientSignedFrom": self.recipient_signed_from,
            "recipientSignedTo": self.recipient_signed_to,
            "signDateFrom": self.sign_date_from,
            "signDateTo": self.sign_date_to,
        }
        for key, val in _date_map.items():
            if val is not None:
                params[key] = val.isoformat()

        if self.create_type:
            params["createType"] = self.create_type.value
        if self.declarant_type:
            params["declarantType"] = self.declarant_type.value
        params["ioOption"] = self.io_option.value

        for key, val in {
            "deliveryMethodId": self.delivery_method_id,
            "daysToCompletion": self.days_to_completion,
            "createYear": self.create_year,
        }.items():
            if val is not None:
                params[key] = val

        if self.contract_sum_from is not None:
            params["contractSumFrom"] = str(self.contract_sum_from)
        if self.contract_sum_to is not None:
            params["contractSumTo"] = str(self.contract_sum_to)
        if self.reg_number_is_not_null:
            params["regNumberIsNotNull"] = True

        _bool_map: dict[str, bool | None] = {
            "authorCurrentUser": self.author_current_user,
            "processExecutorCurrentUser": self.process_executor_current_user,
            "meetingExecutorCurrentUser": self.meeting_executor_current_user,
            "meetingQuestionExecutorCurrentUser": self.meeting_question_executor_current_user,
            "questionExecutorCurrentUser": self.question_executor_current_user,
            "contractExecutorCurrentUser": self.contract_executor_current_user,
            "controlUserCurrentUser": self.control_user_current_user,
            "introductionCurrentUser": self.introduction_current_user,
            "additionalAgreementUser": self.additional_agreement_user,
            "additionalAgreementAllUser": self.additional_agreement_all_user,
            "signingCurrentUser": self.signing_current_user,
            "reviewCurrentUser": self.review_current_user,
            "agreementCurrentUser": self.agreement_current_user,
            "statementCurrentUser": self.statement_current_user,
            "taskExecutorCurrentUser": self.task_executor_current_user,
            "introductionAllCurrentUser": self.introduction_all_current_user,
            "controlOffCurrentUser": self.control_off_current_user,
            "controlExpireCurrentUser": self.control_expire_current_user,
            "controlExpireExecutionCurrentUser": self.control_expire_execution_current_user,
            "executionExpireCurrentUser": self.execution_expire_current_user,
            "executionOverdueCurrentUser": self.execution_overdue_current_user,
            "authorExpireExecutionCurrentUser": self.author_expire_execution_current_user,
            "authorOverdueExecutionCurrentUser": self.author_overdue_execution_current_user,
            "smdoWaitOutgoingUser": self.smdo_wait_outgoing_user,
            "smdoFailOutgoingUser": self.smdo_fail_outgoing_user,
            "userColorIsNull": self.user_color_is_null,
            "onlyUserOrganization": self.only_user_organization,
            "processCompleted": self.process_completed,
            "hasRegDate": self.has_reg_date,
            "anonymous": self.anonymous,
            "collective": self.collective,
            "reasonably": self.reasonably,
            "affairExist": self.affair_exist,
            "contractAgreement": self.contract_agreement,
            "contractAutoProlongation": self.contract_auto_prolongation,
            "contractTypical": self.contract_typical,
            "hasAdditionalDocument": self.has_additional_document,
        }
        for key, val in _bool_map.items():
            if val is not None:
                params[key] = val

        return params


# ---------------------------------------------------------------------------
# DocumentLinkFilter
# ---------------------------------------------------------------------------


class DocumentLinkFilter(DomainModel):
    """Filter for document cross-reference (связи) queries.

    Direct Python port of Java ``DocumentLinkFilter``.

    Attributes:
        doc_id: UUID of the source document.
        doc_org_id: Organization ID of the source document.
        doc_link_id: UUID of the target (linked) document.
        doc_link_org_id: Organization ID of the target document.
        document_link_type: Type of the cross-reference.
        deleted: Whether to include deleted links.
        includes: Related models to eager-load.
    """

    doc_id: UUID | None = None
    doc_org_id: str | None = None
    doc_link_id: UUID | None = None
    doc_link_org_id: str | None = None
    document_link_type: DocumentLinkType | None = None
    deleted: bool | None = None
    includes: list[DocumentLinkFilterInclude] = Field(default_factory=list)

    def as_api_params(self) -> dict:
        """Serializes to query params for the EDMS document-link API.

        Returns:
            Dict suitable for ``params=`` in ``httpx.AsyncClient.get()``.
        """
        params: dict = {}
        if self.doc_id:
            params["docId"] = str(self.doc_id)
        if self.doc_org_id:
            params["docOrgId"] = self.doc_org_id
        if self.doc_link_id:
            params["docLinkId"] = str(self.doc_link_id)
        if self.doc_link_org_id:
            params["docLinkOrgId"] = self.doc_link_org_id
        if self.document_link_type:
            params["documentLinkType"] = self.document_link_type.value
        if self.deleted is not None:
            params["deleted"] = self.deleted
        if self.includes:
            params["includes"] = [i.value for i in self.includes]
        return params


# ---------------------------------------------------------------------------
# EmployeeFilter
# ---------------------------------------------------------------------------


class EmployeeFilter(DomainModel):
    """Filter for employee search queries.

    Direct Python port of Java ``EmployeeFilter`` with all fields.

    By default, includes ``POST`` and ``DEPARTMENT`` related models —
    this mirrors the Java default and ensures all agent tools receive
    full employee records with department and position data.

    Attributes:
        first_name: Имя (LIKE filter).
        last_name: Фамилия (LIKE filter).
        middle_name: Отчество (LIKE filter).
        fired: Whether to include fired employees.
        active: Whether to filter for active employees only.
        ids: Specific list of employee UUIDs.
        department_id: List of department UUIDs.
        includes: Related models (defaults to POST + DEPARTMENT).
        child_departments: Whether to include child department employees.
    """

    first_name: str | None = None
    last_name: str | None = None
    middle_name: str | None = None
    fired: bool | None = None
    active: bool | None = None
    full_post_name: str | None = None
    post_id: int | None = None
    ids: list[UUID] = Field(default_factory=list)
    department_id: list[UUID] = Field(default_factory=list)
    employee_leader_department_id: UUID | None = None
    include_child_leaders_employee_leader_department_id: bool | None = None
    employee_leader_department_all_id: UUID | None = None
    only_leaders_employee_leader_department_all: bool | None = None
    includes: list[EmployeeFilterInclude] = Field(
        default_factory=lambda: [
            EmployeeFilterInclude.POST,
            EmployeeFilterInclude.DEPARTMENT,
        ]
    )
    org_id: str | None = None
    exclude_role_id: UUID | None = None
    exclude_group_id: UUID | None = None
    exclude_personal_group_id: UUID | None = None
    exclude_grief_id: UUID | None = None
    exclude_ids: list[UUID] = Field(default_factory=list)
    all: bool | None = None
    child_departments: bool = False

    def as_api_payload(self) -> dict:
        """Serializes to a JSON body for POST /api/employee/search.

        Returns:
            Dict suitable for ``json=`` in ``httpx.AsyncClient.post()``.
        """
        payload: dict = {}

        for key, val in {
            "firstName": self.first_name,
            "lastName": self.last_name,
            "middleName": self.middle_name,
            "fullPostName": self.full_post_name,
            "orgId": self.org_id,
        }.items():
            if val is not None:
                payload[key] = val

        for key, val in {
            "fired": self.fired,
            "active": self.active,
            "includeChildLeadersEmployeeLeaderDepartmentId": (
                self.include_child_leaders_employee_leader_department_id
            ),
            "onlyLeadersEmployeeLeaderDepartmentAll": (
                self.only_leaders_employee_leader_department_all
            ),
            "all": self.all,
        }.items():
            if val is not None:
                payload[key] = val

        if self.child_departments:
            payload["childDepartments"] = True
        if self.post_id is not None:
            payload["postId"] = self.post_id

        for key, val in {
            "employeeLeaderDepartmentId": self.employee_leader_department_id,
            "employeeLeaderDepartmentAllId": self.employee_leader_department_all_id,
            "excludeRoleId": self.exclude_role_id,
            "excludeGroupId": self.exclude_group_id,
            "excludePersonalGroupId": self.exclude_personal_group_id,
            "excludeGriefId": self.exclude_grief_id,
        }.items():
            if val is not None:
                payload[key] = str(val)

        if self.ids:
            payload["ids"] = [str(i) for i in self.ids]
        if self.department_id:
            payload["departmentId"] = [str(d) for d in self.department_id]
        if self.exclude_ids:
            payload["excludeIds"] = [str(e) for e in self.exclude_ids]
        if self.includes:
            payload["includes"] = [i.value for i in self.includes]

        return payload

    def as_api_params(self) -> dict:
        """Serializes to query params for GET /api/employee.

        For simple GET requests; use ``as_api_payload`` for POST /search.

        Returns:
            Same structure as ``as_api_payload`` (GET endpoint mirrors POST).
        """
        return self.as_api_payload()


# ---------------------------------------------------------------------------
# UserActionFilter
# ---------------------------------------------------------------------------


class UserActionFilter(DomainModel):
    """Filter for user activity log queries.

    Direct Python port of Java ``UserActionFilter``.
    ``start`` and ``end`` are required for Dashboard queries
    (Java ``@NotNull(groups = Dashboard.class)``).

    Attributes:
        types: Action types to filter by.
        group_ids: Filter by user groups.
        department_ids: Filter by departments.
        employee_ids: Filter by specific employees.
        start: Range start (required for Dashboard queries).
        end: Range end (required for Dashboard queries).
    """

    types: list[UserActionType] = Field(default_factory=list)
    group_ids: list[UUID] = Field(default_factory=list)
    department_ids: list[UUID] = Field(default_factory=list)
    employee_ids: list[UUID] = Field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None

    def validate_for_dashboard(self) -> None:
        """Validates that start and end are set for Dashboard queries.

        Raises:
            FilterValidationError: When ``start`` or ``end`` is missing.
        """
        if self.start is None:
            raise FilterValidationError(
                "start обязателен для Dashboard-запросов UserActionFilter",
                field="start",
            )
        if self.end is None:
            raise FilterValidationError(
                "end обязателен для Dashboard-запросов UserActionFilter",
                field="end",
            )

    def as_api_params(self) -> dict:
        """Serializes to query params for GET /api/employee/{id}/user-action.

        Returns:
            Dict suitable for ``params=`` in ``httpx.AsyncClient.get()``.
        """
        params: dict = {}
        if self.types:
            params["types"] = [t.value for t in self.types]
        if self.group_ids:
            params["groupIds"] = [str(g) for g in self.group_ids]
        if self.department_ids:
            params["departmentIds"] = [str(d) for d in self.department_ids]
        if self.employee_ids:
            params["employeeIds"] = [str(e) for e in self.employee_ids]
        if self.start:
            params["start"] = self.start.isoformat()
        if self.end:
            params["end"] = self.end.isoformat()
        return params


# ---------------------------------------------------------------------------
# TaskFilter
# ---------------------------------------------------------------------------


class TaskFilter(DomainModel):
    """Filter for task (поручение) list queries.

    Direct Python port of Java ``TaskFilter`` DTO extended with fields
    from ``TaskOnStatusReportFilter``. Reuses ``TaskStatus`` and
    ``TaskType`` from ``domain/entities/task.py``.

    Attributes:
        current_user_author: Current user is the task author.
        current_user_on_control: Current user is the controller.
        current_user_on_execution: Current user is an executor.
        current_user_expires_task: Deadline is approaching for current user.
        current_user_overdue: Deadline is overdue for current user.
        current_user_endless: Endless tasks for current user.
        task_text: Substring search in task text body.
        task_status: Filter by task status.
        document_reg_num: Parent document registration number filter.
        executor_last_name: Executor last name (LIKE).
        fetch_executors: Whether to load executor list. Default ``True``.
        includes: Related models to eager-load.
    """

    current_user_author: bool | None = None
    current_user_on_control: bool | None = None
    current_user_on_execution: bool | None = None
    current_user_expires_task: bool | None = None
    current_user_overdue: bool | None = None
    current_user_endless: bool | None = None
    task_text: str | None = None
    task_status: TaskStatus | None = None
    document_reg_num: str | None = None
    executor_first_name: str | None = None
    executor_last_name: str | None = None
    executor_middle_name: str | None = None
    fetch_executors: bool = True
    includes: list[TaskFilterInclude] = Field(default_factory=list)

    def as_api_params(self) -> dict:
        """Serializes to query params for GET /api/task.

        Returns:
            Dict suitable for ``params=`` in ``httpx.AsyncClient.get()``.
        """
        params: dict = {"fetchExecutors": self.fetch_executors}

        for key, val in {
            "currentUserAuthor": self.current_user_author,
            "currentUserOnControl": self.current_user_on_control,
            "currentUserOnExecution": self.current_user_on_execution,
            "currentUserExpiresTask": self.current_user_expires_task,
            "currentUserOverdue": self.current_user_overdue,
            "currentUserEndless": self.current_user_endless,
        }.items():
            if val is not None:
                params[key] = val

        for key, val in {
            "taskText": self.task_text,
            "documentRegNum": self.document_reg_num,
            "executorFirstName": self.executor_first_name,
            "executorLastName": self.executor_last_name,
            "executorMiddleName": self.executor_middle_name,
        }.items():
            if val is not None:
                params[key] = val

        if self.task_status:
            params["taskStatus"] = self.task_status.value
        if self.includes:
            params["includes"] = [i.value for i in self.includes]

        return params
