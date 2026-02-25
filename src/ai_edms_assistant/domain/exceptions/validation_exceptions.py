# src/ai_edms_assistant/domain/exceptions/validation_exceptions.py
from __future__ import annotations

from .base import DomainError


class FilterValidationError(DomainError):
    """Raised when a domain filter object fails business-level validation.

    Mirrors Java ``DocumentFilterValidator`` mutual-exclusion rules.
    Raised from ``DocumentFilter.validate()`` before any API call is made.

    This is a domain error, not a Pydantic ``ValidationError`` — it checks
    business invariants (e.g. "author_current_user conflicts with author_id"),
    not type constraints.

    Attributes:
        field: Optional name of the conflicting filter field.

    Example:
        >>> filter = DocumentFilter(author_id=uuid, author_current_user=True)
        >>> filter.validate()
        FilterValidationError: author_current_user конфликтует с author_id
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message=message, code="FILTER_VALIDATION_ERROR")
        self.field = field


class AppealValidationError(DomainError):
    """Raised when an appeal (обращение) fails domain validation rules.

    Used by ``AppealValidator`` service when checking required fields before
    the autofill or extraction workflow begins.

    Attributes:
        missing_fields: List of field names that failed validation.

    Example:
        >>> raise AppealValidationError(
        ...     message="Обязательные поля обращения не заполнены",
        ...     missing_fields=["applicant_name", "citizen_type_id"],
        ... )
    """

    def __init__(
        self,
        message: str,
        missing_fields: list[str] | None = None,
    ) -> None:
        super().__init__(message=message, code="APPEAL_VALIDATION_ERROR")
        self.missing_fields = missing_fields or []


class ExtractionValidationError(DomainError):
    """Raised when extracted NLP data fails domain validation.

    Used by ``AppealExtractor`` when the model returns data that cannot
    be mapped to domain entities (e.g. invalid UUID format, unknown enum).

    Attributes:
        field: The field name that failed extraction validation.
        raw_value: The raw extracted value that could not be validated.

    Example:
        >>> raise ExtractionValidationError(
        ...     field="declarant_type",
        ...     raw_value="COMPANY",  # not in DeclarantType enum
        ... )
    """

    def __init__(
        self,
        field: str,
        raw_value: str | None = None,
    ) -> None:
        raw = f" (получено: '{raw_value}')" if raw_value else ""
        super().__init__(
            message=f"Извлечённое значение поля '{field}' не прошло валидацию{raw}",
            code="EXTRACTION_VALIDATION_ERROR",
        )
        self.field = field
        self.raw_value = raw_value


class EmployeeResolutionError(DomainError):
    """Raised when the agent cannot resolve an employee name to a unique record.

    Two variants:
    - **Not found**: no employee matches the query at all.
    - **Ambiguous**: multiple employees match and the agent cannot decide.

    Attributes:
        query: The name or identifier that was searched.
        candidates: List of candidate name strings when ambiguous.

    Example:
        >>> raise EmployeeResolutionError(
        ...     query="Иванов",
        ...     candidates=["Иванов И.И. (Бухгалтерия)", "Иванов П.П. (ИТ-отдел)"],
        ... )
    """

    def __init__(
        self,
        query: str,
        candidates: list[str] | None = None,
    ) -> None:
        if candidates:
            candidates_text = "; ".join(candidates)
            message = (
                f"Неоднозначное совпадение для '{query}'. "
                f"Уточните кого имели в виду: {candidates_text}"
            )
            code = "EMPLOYEE_RESOLUTION_AMBIGUOUS"
        else:
            message = f"Сотрудник '{query}' не найден в системе"
            code = "EMPLOYEE_NOT_FOUND"

        super().__init__(message=message, code=code)
        self.query = query
        self.candidates = candidates or []
