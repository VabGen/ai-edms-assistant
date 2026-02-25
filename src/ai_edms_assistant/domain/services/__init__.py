# src/ai_edms_assistant/domain/services/__init__.py
"""Pure domain services for the AI EDMS Assistant.

Domain services contain business logic that does not naturally belong
to a single entity. They are stateless, have no I/O, and operate only
on domain entities and value objects.

Services:
    AppealValidator: Validates appeal required fields and business rules.
    DocumentComparer: Compares two document versions and returns a diff.
    TaskAssigner: Builds and validates task executor assignment plans.

Supporting types:
    AppealValidationResult: Result of an appeal validation check.
    ComparisonResult: Structured diff between two document versions.
    FieldDiff: Single field change in a comparison.
    ChangeType: Type of field change (ADDED, REMOVED, MODIFIED).
    TaskAssignmentPlan: Validated plan for task executor assignment.
    ExecutorAssignment: Single resolved executor for a task.
"""

from .appeal_validator import AppealValidationResult, AppealValidator
from .document_comparer import ChangeType, ComparisonResult, DocumentComparer, FieldDiff
from .task_assigner import ExecutorAssignment, TaskAssignmentPlan, TaskAssigner

__all__ = [
    "AppealValidator",
    "AppealValidationResult",
    "DocumentComparer",
    "ComparisonResult",
    "FieldDiff",
    "ChangeType",
    "TaskAssigner",
    "TaskAssignmentPlan",
    "ExecutorAssignment",
]
