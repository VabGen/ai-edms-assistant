# src/ai_edms_assistant/domain/services/document_comparer.py
"""DocumentComparer domain service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from ..entities.document import Document


# ---------------------------------------------------------------------------
# Enums & value objects
# ---------------------------------------------------------------------------


class ChangeType(StrEnum):
    """Type of change detected between two document versions.

    Attributes:
        ADDED: Field was absent in base, present in new.
        REMOVED: Field was present in base, absent in new.
        MODIFIED: Field value changed between versions.
        UNCHANGED: Field value is identical.
    """

    ADDED = "ADDED"
    REMOVED = "REMOVED"
    MODIFIED = "MODIFIED"
    UNCHANGED = "UNCHANGED"


# Русские имена полей для LLM-читаемого вывода
_FIELD_LABELS: dict[str, str] = {
    "short_summary": "Краткое содержание",
    "summary": "Текст документа",
    "note": "Примечание",
    "status": "Статус",
    "prev_status": "Предыдущий статус",
    "reg_number": "Рег. номер",
    "reg_date": "Дата регистрации",
    "out_reg_number": "Исходящий рег. номер",
    "out_reg_date": "Дата исходящей регистрации",
    "days_execution": "Срок исполнения (дней)",
    "correspondent_name": "Наименование корреспондента",
    "responsible_executor": "Ответственный исполнитель",
    "who_signed": "Кем подписан",
    "control_flag": "На контроле",
    "dsp_flag": "Гриф ДСП",
    "pages_count": "Кол-во страниц",
    "exemplar_count": "Кол-во экземпляров",
    "document_type_name": "Вид документа",
    "document_category": "Категория документа",
    "current_bpmn_task_name": "Текущий этап маршрута",
    "delivery_method_name": "Способ доставки",
    "contract_sum": "Сумма договора",
    "contract_number": "Номер договора",
    "contract_start_date": "Дата вступления в силу",
    "contract_duration_end": "Дата окончания договора",
    "contract_signing_date": "Дата подписания договора",
    "currency_id": "Валюта",
    "place_meeting": "Место совещания",
    "date_meeting": "Дата совещания",
    "enable_access_grief": "Гриф доступа",
}

ComparisonFocus = Literal["metadata", "content", "contract", "all"]


@dataclass(frozen=True)
class FieldDiff:
    """Immutable representation of a single field change between versions.

    Attributes:
        field_name: Python attribute name of the changed field.
        field_label: Russian human-readable field name for LLM.
        change_type: Type of change (ADDED, REMOVED, MODIFIED).
        old_value: Value in the base (older) document.
        new_value: Value in the new (current) document.
    """

    field_name: str
    change_type: ChangeType
    old_value: object = None
    new_value: object = None
    field_label: str = ""

    def as_text(self) -> str:
        """Returns a human-readable Russian description of this change.

        Returns:
            Formatted change description for LLM context injection.

        Example:
            >>> diff = FieldDiff("status", ChangeType.MODIFIED,
            ...                  "DRAFT", "REGISTERED", "Статус")
            >>> diff.as_text()
            'Статус: "DRAFT" → "REGISTERED"'
        """
        label = self.field_label or self.field_name
        match self.change_type:
            case ChangeType.ADDED:
                return f"{label}: добавлено '{self.new_value}'"
            case ChangeType.REMOVED:
                return f"{label}: удалено '{self.old_value}'"
            case ChangeType.MODIFIED:
                return f"{label}: '{self.old_value}' → '{self.new_value}'"
            case _:
                return f"{label}: без изменений"


@dataclass(frozen=True)
class ComparisonResult:
    """Immutable result of comparing two document versions.

    Attributes:
        base_doc_id: UUID string of the older (base) document.
        new_doc_id: UUID string of the newer document.
        diffs: List of ALL detected field changes (including UNCHANGED).
        summary: Short human-readable summary for LLM context.
        base_version_number: Version number of base doc (None if unknown).
        new_version_number: Version number of new doc (None if unknown).
    """

    base_doc_id: str
    new_doc_id: str
    diffs: list[FieldDiff] = field(default_factory=list)
    summary: str = ""
    base_version_number: int | None = None
    new_version_number: int | None = None

    @property
    def has_changes(self) -> bool:
        """Returns True when at least one field changed."""
        return any(d.change_type != ChangeType.UNCHANGED for d in self.diffs)

    @property
    def changed_fields(self) -> list[FieldDiff]:
        """Returns only diffs where the value actually changed."""
        return [d for d in self.diffs if d.change_type != ChangeType.UNCHANGED]

    def as_llm_context(self) -> str:
        """Build a rich multi-line LLM-readable context block.

        Returns a full comparison report suitable for injection into
        the LLM system prompt or user message. Includes version numbers
        when available.

        Returns:
            Formatted multi-line string in Russian.
        """
        lines: list[str] = []

        # ── Header ────────────────────────────────────────────────────
        if self.base_version_number is not None and self.new_version_number is not None:
            lines.append(
                f"Сравнение версии {self.base_version_number} "
                f"→ версии {self.new_version_number}"
            )
        else:
            lines.append(
                f"Сравнение документа {self.base_doc_id[:8]}... "
                f"→ {self.new_doc_id[:8]}..."
            )

        # ── Summary ───────────────────────────────────────────────────
        changed = self.changed_fields
        if not changed:
            lines.append("Различий не обнаружено.")
            return "\n".join(lines)

        lines.append(f"Обнаружено изменений: {len(changed)}")
        lines.append("")

        # ── Diffs ─────────────────────────────────────────────────────
        for diff in changed:
            lines.append(f"  • {diff.as_text()}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main domain service
# ---------------------------------------------------------------------------


class DocumentComparer:
    """Pure domain service for comparing two document versions.

    Stateless. No I/O. Accepts ``Document`` entities, returns value objects.

    Supports:
    - Full field comparison (30+ fields)
    - Focused comparison by ``focus`` parameter
    - Version-aware comparison via ``compare_versions()``
    - LLM-ready context generation

    Used by:
        - ``CompareDocumentsUseCase``
        - ``ComparisonTool``
    """

    # ── Full field set (30 полей) ──────────────────────────────────────────
    _COMPARABLE_FIELDS: tuple[str, ...] = (
        # ── Metadata
        "status",
        "prev_status",
        "reg_number",
        "reg_date",
        "out_reg_number",
        "out_reg_date",
        "document_type_name",
        "document_category",
        "current_bpmn_task_name",
        "days_execution",
        # ── Content
        "short_summary",
        "summary",
        "note",
        # ── Participants
        "correspondent_name",
        "responsible_executor",
        "who_signed",
        "delivery_method_name",
        # ── Control & flags
        "control_flag",
        "dsp_flag",
        "enable_access_grief",
        # ── Physical
        "pages_count",
        "exemplar_count",
        # ── Contract
        "contract_sum",
        "contract_number",
        "contract_start_date",
        "contract_duration_end",
        "contract_signing_date",
        "currency_id",
        # ── Meeting
        "place_meeting",
        "date_meeting",
    )

    # ── Fields grouped by focus ────────────────────────────────────────────
    _FIELDS_BY_FOCUS: dict[str, tuple[str, ...]] = {
        "metadata": (
            "status", "prev_status", "reg_number", "reg_date",
            "out_reg_number", "out_reg_date", "document_type_name",
            "document_category", "current_bpmn_task_name", "days_execution",
            "control_flag", "dsp_flag", "enable_access_grief",
        ),
        "content": (
            "short_summary", "summary", "note",
            "correspondent_name", "responsible_executor", "who_signed",
            "delivery_method_name", "pages_count", "exemplar_count",
        ),
        "contract": (
            "contract_sum", "contract_number", "contract_start_date",
            "contract_duration_end", "contract_signing_date", "currency_id",
            "status", "reg_number",
        ),
        "meeting": (
            "place_meeting", "date_meeting", "status", "short_summary",
        ),
    }

    def compare(
        self,
        base: Document,
        new: Document,
        focus: ComparisonFocus = "all",
    ) -> ComparisonResult:
        """Compare two document versions and return a structured diff.

        Args:
            base: Older / base document version.
            new: Newer / current document version.
            focus: Field group to compare:
                   "metadata" — регистрация, статус, тип документа;
                   "content"  — текст, исполнитель, доставка;
                   "contract" — договорные поля;
                   "all"      — все 30 полей (по умолчанию).

        Returns:
            ``ComparisonResult`` with diffs and LLM-ready summary.
        """
        fields = (
            self._COMPARABLE_FIELDS
            if focus == "all"
            else self._FIELDS_BY_FOCUS.get(focus, self._COMPARABLE_FIELDS)
        )

        diffs: list[FieldDiff] = []

        for field_name in fields:
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
                    field_label=_FIELD_LABELS.get(field_name, field_name),
                )
            )

        summary = self._build_summary(diffs)
        return ComparisonResult(
            base_doc_id=str(base.id),
            new_doc_id=str(new.id),
            diffs=diffs,
            summary=summary,
        )

    def compare_versions(
        self,
        base: Document,
        new: Document,
        base_version_number: int | None = None,
        new_version_number: int | None = None,
        focus: ComparisonFocus = "all",
    ) -> ComparisonResult:
        """Compare two versions of the same document.

        Wraps ``compare()`` and enriches the result with version numbers
        for LLM context display (e.g. "Версия 1 → Версия 2").

        Args:
            base: Older document version entity.
            new: Newer document version entity.
            base_version_number: Human-readable version number (1, 2, 3…).
            new_version_number: Human-readable version number of new doc.
            focus: Field group to compare (default "all").

        Returns:
            ``ComparisonResult`` enriched with version number metadata.
        """
        result = self.compare(base=base, new=new, focus=focus)
        # Возвращаем новый dataclass с version_number полями
        return ComparisonResult(
            base_doc_id=result.base_doc_id,
            new_doc_id=result.new_doc_id,
            diffs=result.diffs,
            summary=result.summary,
            base_version_number=base_version_number,
            new_version_number=new_version_number,
        )

    def compare_to_text(
        self,
        base: Document,
        new: Document,
        focus: ComparisonFocus = "all",
    ) -> str:
        """Compare two documents and return a plain-text diff for LLM injection.

        Args:
            base: The older document version.
            new: The newer document version.
            focus: Field group filter.

        Returns:
            Multi-line Russian string of all detected changes.
        """
        result = self.compare(base=base, new=new, focus=focus)
        return result.as_llm_context()

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _get_field_value(doc: Document, field_name: str) -> object:
        """Safely extract and normalize a field value for comparison.

        Handles all complex types that appear in Document entity:
        - Enum → .value (str)
        - UserInfo / any object with .name → name str
        - Decimal → str with 2 decimal places
        - datetime → ISO date string (date part only)
        - list → sorted comma-joined string
        - None → None

        Args:
            doc: The document to extract from.
            field_name: Python attribute name.

        Returns:
            Comparable scalar value or None.
        """
        val = getattr(doc, field_name, None)
        if val is None:
            return None

        # Enum (DocumentStatus, DocumentCategory, etc.)
        if hasattr(val, "value") and isinstance(val.value, str):
            return val.value

        # UserInfo and similar objects with .name
        if hasattr(val, "name") and isinstance(getattr(val, "name"), str):
            return val.name

        # Objects with .id only (UUID references)
        if hasattr(val, "id") and not hasattr(val, "value"):
            return str(val.id)

        # Decimal → formatted string
        if isinstance(val, Decimal):
            return f"{val:.2f}"

        # datetime → date only (versions differ in time, not date)
        if isinstance(val, datetime):
            return val.strftime("%Y-%m-%d")

        # list → sorted joined string for stable comparison
        if isinstance(val, list):
            if not val:
                return None
            return ", ".join(sorted(str(x) for x in val))

        return val

    @staticmethod
    def _build_summary(diffs: list[FieldDiff]) -> str:
        """Build a compact Russian-language summary of the diff list.

        Args:
            diffs: List of non-UNCHANGED field differences.

        Returns:
            Summary string, e.g. 'Изменено 3 поля: Статус, Рег. номер, Текст'.
        """
        if not diffs:
            return "Различий не обнаружено"

        labels = [_FIELD_LABELS.get(d.field_name, d.field_name) for d in diffs[:5]]
        names = ", ".join(labels)
        suffix = f" и ещё {len(diffs) - 5}" if len(diffs) > 5 else ""
        return f"Изменено {len(diffs)} поля/полей: {names}{suffix}"