# src/ai_edms_assistant/domain/services/appeal_validator.py
from __future__ import annotations

from dataclasses import dataclass, field

from ..entities.appeal import DocumentAppeal
from ..entities.document import Document, DocumentCategory
from ..exceptions.validation_exceptions import AppealValidationError


@dataclass(frozen=True)
class AppealValidationResult:
    """Immutable result of an appeal validation check.

    Attributes:
        is_valid: Whether the appeal passed all validation rules.
        missing_required: List of required field names that are absent.
        warnings: List of non-blocking advisory messages.
    """

    is_valid: bool
    missing_required: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class AppealValidator:
    """Pure domain service for validating appeal (обращение) data.

    Encapsulates all business rules for appeal validation without any I/O.
    Called by use cases before invoking the autofill tool or submitting
    data to the EDMS API.

    Design:
        - Stateless: all methods are effectively pure functions on the inputs.
        - No I/O: never calls repositories or external APIs.
        - Raises ``AppealValidationError`` only via ``validate_or_raise``.
          Use ``validate`` for soft checks that return a result object.
    """

    _REQUIRED_FIELDS: tuple[str, ...] = (
        "applicant_name",
        "declarant_type",
        "citizen_type_id",
    )

    _RECOMMENDED_FIELDS: tuple[str, ...] = (
        "email",
        "phone",
        "full_address",
        "geo_location",
        "description",
    )

    def validate(self, appeal: DocumentAppeal) -> AppealValidationResult:
        """Validate an appeal against domain business rules.

        Checks required fields and generates advisory warnings for
        recommended-but-missing fields.

        Args:
            appeal: The ``DocumentAppeal`` entity to validate.

        Returns:
            ``AppealValidationResult`` with ``is_valid``, ``missing_required``,
            and ``warnings`` populated.
        """
        missing: list[str] = []
        warnings: list[str] = []

        for field_name in self._REQUIRED_FIELDS:
            if not getattr(appeal, field_name, None):
                missing.append(field_name)

        for field_name in self._RECOMMENDED_FIELDS:
            val = getattr(appeal, field_name, None)
            if not val:
                warnings.append(f"Рекомендуемое поле '{field_name}' не заполнено")

        if appeal.anonymous and appeal.applicant_name:
            warnings.append(
                "Обращение помечено как анонимное, но указано имя заявителя"
            )

        if appeal.geo_location is None and appeal.full_address is None:
            warnings.append(
                "Не указан адрес заявителя (ни структурированный, ни текстовый)"
            )

        return AppealValidationResult(
            is_valid=len(missing) == 0,
            missing_required=missing,
            warnings=warnings,
        )

    def validate_or_raise(self, appeal: DocumentAppeal) -> None:
        """Validate and raise an exception on the first failure.

        Convenience method for use cases that want strict validation.

        Args:
            appeal: The ``DocumentAppeal`` entity to validate.

        Raises:
            AppealValidationError: When required fields are missing.
        """
        result = self.validate(appeal)
        if not result.is_valid:
            raise AppealValidationError(
                message="Обязательные поля обращения не заполнены",
                missing_fields=result.missing_required,
            )

    def validate_document_is_appeal(self, document: Document) -> None:
        """Validate that a document is an appeal category.

        Args:
            document: The ``Document`` aggregate to check.

        Raises:
            AppealValidationError: When the document is not an appeal.
        """
        if document.document_category != DocumentCategory.APPEAL:
            raise AppealValidationError(
                message=(
                    f"Документ '{document.id}' не является обращением "
                    f"(категория: {document.document_category})"
                )
            )
        if document.appeal is None:
            raise AppealValidationError(
                message=f"Документ '{document.id}' не содержит данных обращения"
            )
