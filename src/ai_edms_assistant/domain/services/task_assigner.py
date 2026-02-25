# src/ai_edms_assistant/domain/services/task_assigner.py
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from ..entities.employee import Employee
from ..entities.task import TaskStatus
from ..exceptions.document_exceptions import TaskCreationError
from ..exceptions.validation_exceptions import EmployeeResolutionError


@dataclass(frozen=True)
class ExecutorAssignment:
    """Immutable executor assignment record for task creation.

    Represents a resolved and validated executor before the task is
    submitted to the EDMS API.

    Attributes:
        employee_id: UUID of the resolved employee.
        display_name: Short name for logging and user-facing messages.
        is_responsible: Whether this executor is the primary responsible person.
    """

    employee_id: UUID
    display_name: str
    is_responsible: bool = False

    def to_api_tuple(self) -> tuple[UUID, bool]:
        """Converts to the tuple format expected by ``AbstractTaskRepository.create``.

        Returns:
            Tuple of ``(employee_id, is_responsible)``.
        """
        return (self.employee_id, self.is_responsible)


@dataclass(frozen=True)
class TaskAssignmentPlan:
    """Immutable plan for creating a task with resolved executors.

    Produced by ``TaskAssigner.build_plan`` after validating all business
    rules. Passed to the task creation use case.

    Attributes:
        executors: List of resolved executor assignments.
        task_text: Validated task text.
        warnings: Advisory messages (e.g. duplicate executor detected).
    """

    executors: list[ExecutorAssignment] = field(default_factory=list)
    task_text: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def responsible(self) -> ExecutorAssignment | None:
        """Returns the primary responsible executor.

        Returns:
            The ``ExecutorAssignment`` with ``is_responsible=True``,
            or ``None`` when none is designated.
        """
        for ex in self.executors:
            if ex.is_responsible:
                return ex
        return None

    def to_api_executor_list(self) -> list[tuple[UUID, bool]]:
        """Converts all executors to the API tuple format.

        Returns:
            List of ``(employee_uuid, is_responsible)`` tuples for
            ``AbstractTaskRepository.create``.
        """
        return [ex.to_api_tuple() for ex in self.executors]


class TaskAssigner:
    """Pure domain service for building and validating task executor assignments.

    Encapsulates the business rules for assigning executors to tasks:
    - Exactly one responsible executor is required.
    - Duplicate executor UUIDs are not allowed.
    - Fired / inactive employees cannot be assigned.
    - At least one executor must be provided.

    No I/O — receives resolved ``Employee`` objects from the use case layer
    (which fetches them via the repository). Returns immutable value objects.
    """

    def build_plan(
        self,
        candidates: list[tuple[Employee, bool]],
        task_text: str,
    ) -> TaskAssignmentPlan:
        """Build and validate a task assignment plan from resolved employees.

        Validates all business rules and returns an immutable plan if
        the input is valid.

        Args:
            candidates: List of ``(Employee, is_responsible)`` tuples.
                Each tuple pairs a resolved ``Employee`` with a flag
                indicating whether they are the responsible executor.
            task_text: The task instruction text. Must be non-empty.

        Returns:
            ``TaskAssignmentPlan`` with validated executors and warnings.

        Raises:
            TaskCreationError: When validation fails (no executors, no
                responsible executor, duplicate UUIDs, or empty text).
        """
        if not task_text.strip():
            raise TaskCreationError("Текст поручения не может быть пустым")

        if not candidates:
            raise TaskCreationError("Необходимо указать хотя бы одного исполнителя")

        seen_ids: set[UUID] = set()
        assignments: list[ExecutorAssignment] = []
        warnings: list[str] = []
        responsible_count = 0

        for employee, is_responsible in candidates:
            if not employee.is_active:
                raise TaskCreationError(
                    f"Сотрудник '{employee.short_name}' неактивен и не может "
                    "быть назначен исполнителем"
                )

            if employee.id in seen_ids:
                warnings.append(
                    f"Дубликат исполнителя '{employee.short_name}' проигнорирован"
                )
                continue

            seen_ids.add(employee.id)

            if is_responsible:
                responsible_count += 1

            assignments.append(
                ExecutorAssignment(
                    employee_id=employee.id,
                    display_name=employee.short_name,
                    is_responsible=is_responsible,
                )
            )

        if not assignments:
            raise TaskCreationError(
                "После фильтрации дубликатов не осталось исполнителей"
            )

        if responsible_count == 0:
            if len(assignments) == 1:
                assignments = [
                    ExecutorAssignment(
                        employee_id=assignments[0].employee_id,
                        display_name=assignments[0].display_name,
                        is_responsible=True,
                    )
                ]
                warnings.append(
                    f"Единственный исполнитель '{assignments[0].display_name}' "
                    "автоматически назначен ответственным"
                )
            else:
                raise TaskCreationError(
                    "Необходимо указать ответственного исполнителя "
                    "(ровно один executor с is_responsible=True)"
                )

        if responsible_count > 1:
            raise TaskCreationError(
                f"Допускается только один ответственный исполнитель, "
                f"передано: {responsible_count}"
            )

        return TaskAssignmentPlan(
            executors=assignments,
            task_text=task_text.strip(),
            warnings=warnings,
        )

    @staticmethod
    def validate_executor_statuses(
        employees: list[Employee],
    ) -> list[Employee]:
        """Filter out inactive employees with warnings.

        Used when resolving a list of executor names to validate that all
        resolved employees are available for task assignment.

        Args:
            employees: List of resolved ``Employee`` entities.

        Returns:
            List of active employees only.

        Raises:
            EmployeeResolutionError: When the filtered list is empty.
        """
        active = [e for e in employees if e.is_active]
        if not active:
            inactive_names = [e.short_name for e in employees]
            raise EmployeeResolutionError(
                query=", ".join(inactive_names),
                candidates=None,
            )
        return active

    @staticmethod
    def infer_task_status_label(status: TaskStatus) -> str:
        """Returns a Russian-language label for a task status.

        Used for LLM context injection and user-facing messages.

        Args:
            status: The ``TaskStatus`` enum value.

        Returns:
            Human-readable Russian status label.

        Example:
            >>> TaskAssigner.infer_task_status_label(TaskStatus.OVERDUE)
            'Просрочено'
        """
        _labels: dict[TaskStatus, str] = {
            TaskStatus.NEW: "Новое",
            TaskStatus.IN_PROGRESS: "В работе",
            TaskStatus.COMPLETED: "Исполнено",
            TaskStatus.OVERDUE: "Просрочено",
            TaskStatus.ON_CHECK: "На проверке",
            TaskStatus.REVISION: "На доработке",
            TaskStatus.ENDLESS: "Бессрочное",
        }
        return _labels.get(status, status.value)
