# src/ai_edms_assistant/domain/services/document_comparer.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..entities.document import Document


class ChangeType(StrEnum):
    """Type of change detected between two document versions.

    Attributes:
        ADDED: Field was absent in the base version, present in the new one.
        REMOVED: Field was present in the base, absent in the new one.
        MODIFIED: Field value changed between versions.
        UNCHANGED: Field value is identical in both versions.
    """

    ADDED = "ADDED"
    REMOVED = "REMOVED"
    MODIFIED = "MODIFIED"
    UNCHANGED = "UNCHANGED"


@dataclass(frozen=True)
class FieldDiff:
    """Immutable representation of a single field change between versions.

    Attributes:
        field_name: Python attribute name of the changed field.
        change_type: Type of change (ADDED, REMOVED, MODIFIED).
        old_value: Value in the base (older) document. ``None`` for ADDED.
        new_value: Value in the new (current) document. ``None`` for REMOVED.
    """

    field_name: str
    change_type: ChangeType
    old_value: object = None
    new_value: object = None

    def as_text(self) -> str:
        """Returns a human-readable description of this change.

        Returns:
            Formatted change description in Russian for LLM context.

        Example:
            >>> diff = FieldDiff("short_summary", ChangeType.MODIFIED, "Старое", "Новое")
            >>> diff.as_text()
            'short_summary: "Старое" → "Новое"'
        """
        match self.change_type:
            case ChangeType.ADDED:
                return f"{self.field_name}: добавлено '{self.new_value}'"
            case ChangeType.REMOVED:
                return f"{self.field_name}: удалено '{self.old_value}'"
            case ChangeType.MODIFIED:
                return f"{self.field_name}: '{self.old_value}' → '{self.new_value}'"
            case _:
                return f"{self.field_name}: без изменений"


@dataclass(frozen=True)
class ComparisonResult:
    """Immutable result of comparing two document versions.

    Attributes:
        base_doc_id: UUID string of the older (base) document.
        new_doc_id: UUID string of the newer document.
        diffs: List of detected field changes.
        summary: Short human-readable summary for LLM context.
    """

    base_doc_id: str
    new_doc_id: str
    diffs: list[FieldDiff] = field(default_factory=list)
    summary: str = ""

    @property
    def has_changes(self) -> bool:
        """Returns True when at least one field changed.

        Returns:
            ``True`` when ``diffs`` contains at least one non-UNCHANGED entry.
        """
        return any(d.change_type != ChangeType.UNCHANGED for d in self.diffs)

    @property
    def changed_fields(self) -> list[FieldDiff]:
        """Returns only the diffs where the value actually changed.

        Returns:
            List of ``FieldDiff`` with ``change_type != UNCHANGED``.
        """
        return [d for d in self.diffs if d.change_type != ChangeType.UNCHANGED]


class DocumentComparer:
    """Pure domain service for comparing two document versions.

    Compares a set of meaningful fields between two ``Document`` entities
    and returns a structured diff. No I/O — takes only domain entities
    as input and returns value objects.

    Used by ``CompareDocumentsUseCase`` and the ``comparison_tool``.
    """

    _COMPARABLE_FIELDS: tuple[str, ...] = (
        "short_summary",
        "summary",
        "note",
        "status",
        "reg_number",
        "reg_date",
        "days_execution",
        "correspondent_name",
        "responsible_executor",
        "control_flag",
        "dsp_flag",
        "pages_count",
    )

    def compare(self, base: Document, new: Document) -> ComparisonResult:
        """Compare two document versions and return a structured diff.

        Iterates over ``_COMPARABLE_FIELDS`` and detects changes between
        the base (older) and new (current) document versions.

        Args:
            base: The older / base document version.
            new: The newer / current document version.

        Returns:
            ``ComparisonResult`` with a list of ``FieldDiff`` entries and
            a human-readable summary string.
        """
        diffs: list[FieldDiff] = []

        for field_name in self._COMPARABLE_FIELDS:
            old_val = self._get_field_value(base, field_name)
            new_val = self._get_field_value(new, field_name)

            if old_val == new_val:
                continue

            if old_val is None and new_val is not None:
                change_type = ChangeType.ADDED
            elif old_val is not None and new_val is None:
                change_type = ChangeType.REMOVED
            else:
                change_type = ChangeType.MODIFIED

            diffs.append(
                FieldDiff(
                    field_name=field_name,
                    change_type=change_type,
                    old_value=old_val,
                    new_value=new_val,
                )
            )

        summary = self._build_summary(diffs)
        return ComparisonResult(
            base_doc_id=str(base.id),
            new_doc_id=str(new.id),
            diffs=diffs,
            summary=summary,
        )

    def compare_to_text(self, base: Document, new: Document) -> str:
        """Compare two documents and return a plain-text diff for LLM injection.

        Convenience wrapper over ``compare`` that serializes the result
        to a human-readable string suitable for inclusion in LLM prompts.

        Args:
            base: The older document version.
            new: The newer document version.

        Returns:
            Multi-line string describing all detected changes.
            Returns "Различий не обнаружено" when documents are identical.
        """
        result = self.compare(base, new)
        if not result.has_changes:
            return "Различий не обнаружено"
        lines = [f"Обнаружено изменений: {len(result.changed_fields)}"]
        lines.extend(d.as_text() for d in result.changed_fields)
        return "\n".join(lines)

    @staticmethod
    def _get_field_value(doc: Document, field_name: str) -> object:
        """Safely extract a field value, normalizing complex types to strings.

        Args:
            doc: The document to extract from.
            field_name: Python attribute name.

        Returns:
            Scalar value, string representation of complex types, or ``None``.
        """
        val = getattr(doc, field_name, None)
        if val is None:
            return None
        if hasattr(val, "name"):
            return val.name
        if hasattr(val, "value"):
            return val.value
        if hasattr(val, "id"):
            return str(getattr(val, "id"))
        return val

    @staticmethod
    def _build_summary(diffs: list[FieldDiff]) -> str:
        """Build a compact Russian-language summary of the diff list.

        Args:
            diffs: List of detected field differences.

        Returns:
            Summary string, e.g. 'Изменено 3 поля: status, reg_number, note'.
        """
        if not diffs:
            return "Различий не обнаружено"
        names = ", ".join(d.field_name for d in diffs[:5])
        suffix = f" и ещё {len(diffs) - 5}" if len(diffs) > 5 else ""
        return f"Изменено {len(diffs)} поля: {names}{suffix}"
