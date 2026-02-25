# src/ai_edms_assistant/domain/entities/task.py
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field, computed_field

from .base import DomainModel, MutableDomainModel
from .employee import UserInfo

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    """Task execution lifecycle statuses.

    Statuses follow the EDMS workflow progression:
    NEW → IN_PROGRESS → (ON_CHECK | REVISION) → COMPLETED | OVERDUE.

    Attributes:
        NEW: Поручение создано, исполнение не начато.
        IN_PROGRESS: Поручение принято в работу.
        COMPLETED: Поручение исполнено.
        OVERDUE: Срок исполнения истёк.
        ON_CHECK: На проверке у контролёра.
        REVISION: Отправлено на доработку.
        ENDLESS: Бессрочное поручение (без даты окончания).
    """

    NEW = "NEW"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    OVERDUE = "OVERDUE"
    ON_CHECK = "ON_CHECK"
    REVISION = "REVISION"
    ENDLESS = "ENDLESS"


class TaskType(StrEnum):
    """Task type determining workflow context.

    Maps to the process type in which the task was created (согласование,
    подписание, исполнение и т.д.).
    """

    NEW = "NEW"
    AGREEMENT = "AGREEMENT"
    SIGNING = "SIGNING"
    STATEMENT = "STATEMENT"
    REGISTRATION = "REGISTRATION"
    REVIEW = "REVIEW"
    EXECUTION = "EXECUTION"
    DISPATCH = "DISPATCH"
    PREPARATION = "PREPARATION"
    PAPERWORK = "PAPERWORK"
    ACCEPTANCE = "ACCEPTANCE"
    CONTRACT_EXECUTION = "CONTRACT_EXECUTION"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class TaskExecutor(DomainModel):
    """Single executor entry within a task (Исполнитель поручения).

    Immutable because an executor record represents a point-in-time snapshot
    of who was assigned and what their completion state is. Mutations (e.g.
    marking as done) happen at the ``Task`` aggregate level by replacing the
    executor list, not by mutating individual entries.

    Attributes:
        id: Executor record UUID.
        executor: Lightweight ``UserInfo`` reference to the employee.
        responsible: Whether this executor is the primary responsible person.
        executed_date: Timestamp when this executor completed the task.
        stamp_text: Execution stamp text (visible in document history).
        executed_for_all: Whether this executor completed on behalf of all.
        revision: Whether this executor sent the task back for revision.
        link_count: Count of linked documents (UI hint, not business logic).
        execution_doc_count: Count of execution documents (UI hint).
    """

    id: UUID
    executor: UserInfo

    responsible: bool = False
    executed_date: datetime | None = Field(default=None, alias="executedDate")
    stamp_text: str | None = Field(default=None, alias="stampText")

    executed_for_all: bool = Field(default=False, alias="executedForAll")
    revision: bool = False

    link_count: int | None = Field(default=None, alias="linkCount")
    execution_doc_count: int | None = Field(default=None, alias="executionDocCount")

    @property
    def is_done(self) -> bool:
        """Returns True when this executor has completed the task.

        Returns:
            ``True`` when ``executed_date`` is set.
        """
        return self.executed_date is not None


# ---------------------------------------------------------------------------
# Main entity
# ---------------------------------------------------------------------------


class Task(MutableDomainModel):
    """Domain entity representing an EDMS task / assignment (Поручение).

    Maps to ``TaskDto`` in the Java backend. Tasks are created within
    document workflows and assigned to one or more executors. A task can
    have sub-tasks (``parent_task_id`` reference).

    Business rules encoded as properties:
    - ``is_overdue`` checks the status enum, not the date, to respect
      the backend's authoritative calculation.
    - ``is_completed`` checks all executors when the list is non-empty;
      falls back to status comparison when executors are not loaded.
    - ``completion_ratio`` provides a 0.0–1.0 float for progress display.

    Attributes:
        id: Task UUID.
        text: Task instruction text (taskText → LLM context).
        status: Current execution status.
        task_number: Human-readable task number (for LLM and display).
        external_id: External system identifier (integration use).
        organization_id: Multi-tenant organization identifier.
        task_type: Workflow type that created this task.
        deadline: Planned completion date (planedDateEnd in Java DTO).
        real_date_end: Actual completion timestamp.
        create_date: Task creation timestamp.
        document_reg_date: Parent document registration date.
        on_control: Whether the task is under active control.
        remove_control: Flag indicating removal from control.
        revision: Whether the task is in revision state.
        endless: Whether the task has no deadline.
        period_task: Whether this is a recurring periodic task.
        period_task_count: Number of period occurrences.
        author: UserInfo of the task creator.
        executors: List of assigned executor records.
        document_id: UUID of the parent document.
        document_reg_num: Parent document registration number (for LLM).
        parent_task_id: UUID of the parent task (if this is a sub-task).
        parent_task_number: Parent task number (for LLM display).
        count_exec: Total executor count (denormalized from API).
        count_completed_exec: Completed executor count (denormalized).
        date_request_count: Number of extension requests.
    """

    id: UUID
    text: str = Field(alias="taskText")
    note: str | None = None
    status: TaskStatus

    task_number: str | None = Field(default=None, alias="taskNumber")
    external_id: str | None = Field(default=None, alias="externalId")
    organization_id: str | None = Field(default=None, alias="organizationId")
    task_type: TaskType | None = Field(default=None, alias="taskType")

    deadline: datetime | None = Field(default=None, alias="planedDateEnd")
    real_date_end: datetime | None = Field(default=None, alias="realDateEnd")
    create_date: datetime | None = Field(default=None, alias="createDate")
    modify_date: datetime | None = Field(default=None, alias="modifyDate")
    document_reg_date: datetime | None = Field(default=None, alias="documentRegDate")

    on_control: bool = Field(default=False, alias="onControl")
    remove_control: bool = Field(default=False, alias="removeControl")
    revision: bool = False
    endless: bool = False
    days_execution: int | None = Field(default=None, alias="daysExecution")
    can_change_date: bool = False
    period_task: bool = Field(default=False, alias="periodTask")
    period_days: int | None = Field(default=None, alias="periodDays")
    period_day_week: int | None = Field(default=None, alias="periodDayWeek")
    period_day_month: int | None = Field(default=None, alias="periodDayMonth")
    period_month_year: int | None = Field(default=None, alias="periodMonthYear")
    period_date_begin: datetime | None = Field(default=None, alias="periodDateBegin")
    period_date_end: datetime | None = Field(default=None, alias="periodDateEnd")
    period_task_count: int | None = Field(default=None, alias="periodTaskCount")

    author: UserInfo | None = None
    executors: list[TaskExecutor] = Field(default_factory=list)

    document_id: UUID | None = Field(default=None, alias="documentId")
    document_reg_num: str | None = Field(default=None, alias="documentRegNum")
    parent_task_id: UUID | None = Field(default=None, alias="parentTaskId")
    parent_task_number: str | None = Field(default=None, alias="parentTaskNumber")

    count_exec: int | None = Field(default=None, alias="countExec")
    count_completed_exec: int | None = Field(default=None, alias="countCompletedExec")
    date_request_count: int | None = Field(default=None, alias="dateRequestCount")

    @computed_field
    @property
    def responsible_executor(self) -> UserInfo | None:
        """Returns the UserInfo of the responsible executor, if assigned.

        Iterates the executors list and returns the first entry marked
        as ``responsible=True``.

        Returns:
            ``UserInfo`` of the responsible executor, or ``None`` when not set.
        """
        for ex in self.executors:
            if ex.responsible:
                return ex.executor
        return None

    @property
    def is_overdue(self) -> bool:
        """Returns True when the task has overdue status.

        Uses the authoritative backend status rather than comparing dates
        locally to avoid timezone inconsistencies.

        Returns:
            ``True`` when ``status == TaskStatus.OVERDUE``.
        """
        return self.status == TaskStatus.OVERDUE

    @property
    def is_completed(self) -> bool:
        """Returns True when all executors have completed the task.

        When the executors list is empty (not loaded from API), falls back
        to checking ``status == COMPLETED``.

        Returns:
            ``True`` when all executors have an ``executed_date``, or when
            status is ``COMPLETED`` and executors list is empty.
        """
        if not self.executors:
            return self.status == TaskStatus.COMPLETED
        return all(ex.is_done for ex in self.executors)

    @property
    def completion_ratio(self) -> float:
        """Fraction of executors who have completed the task (0.0–1.0).

        Used for progress bars and completion analytics in LLM context.

        Returns:
            Float in range [0.0, 1.0]. Returns 0.0 when there are no executors.

        Example:
            >>> from uuid import uuid4
            >>> task = Task(id=uuid4(), text="...", status=TaskStatus.IN_PROGRESS,
            ...             executors=[done_ex, done_ex, pending_ex])
            >>> task.completion_ratio
            0.6666666666666666
        """
        if not self.executors:
            return 0.0
        done = sum(1 for ex in self.executors if ex.is_done)
        return done / len(self.executors)

    def get_summary(self) -> str:
        """Compact task description for LLM context injection.

        Assembles a pipe-separated string with the most relevant fields
        for agent reasoning: number, status, deadline, responsible executor.

        Returns:
            Formatted string like:
            'Поручение №123-П | Статус: IN_PROGRESS | Срок: 15.03.2025 | Исполнитель: Иванов И.И.'

        Example:
            >>> task.get_summary()
            'Поручение №2025-001 | Статус: NEW | Срок: 01.12.2025'
        """
        parts: list[str] = []
        num = self.task_number or str(self.id)[:8]
        parts.append(f"Поручение №{num}")
        parts.append(f"Статус: {self.status.value}")
        if self.deadline:
            parts.append(f"Срок: {self.deadline.strftime('%d.%m.%Y')}")
        if self.responsible_executor:
            parts.append(f"Исполнитель: {self.responsible_executor.name}")
        return " | ".join(parts)
