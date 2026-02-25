# src/ai_edms_assistant/domain/value_objects/extraction_result.py
from __future__ import annotations

from typing import Any

from ..entities.base import DomainModel


class FieldExtractionResult(DomainModel):
    """Immutable result of extracting a single field from a document.

    Carries the extracted value, a confidence score, and the raw source
    text that the extractor used. Immutable because extraction results
    are point-in-time snapshots that must not be modified after creation.

    Attributes:
        field_name: Name of the target domain field (e.g. ``"applicant_name"``).
        value: Extracted and normalized value. ``None`` when extraction failed.
        confidence: Confidence score in range [0.0, 1.0].
            ``1.0`` means the extractor is certain.
            ``0.0`` means no reliable value was found.
        source_text: The raw text snippet from which the value was extracted.
            Used for debugging and human review.
        extraction_method: Name of the extractor that produced this result
            (e.g. ``"regex"``, ``"llm"``, ``"spacy"``).

    Example:
        >>> result = FieldExtractionResult(
        ...     field_name="applicant_name",
        ...     value="Иванов Иван Иванович",
        ...     confidence=0.95,
        ...     source_text="Заявитель: Иванов Иван Иванович",
        ... )
        >>> result.is_confident
        True
    """

    field_name: str
    value: Any | None = None
    confidence: float = 0.0
    source_text: str | None = None
    extraction_method: str | None = None

    _CONFIDENCE_THRESHOLD: float = 0.7

    @property
    def is_confident(self) -> bool:
        """Returns True when confidence exceeds the acceptance threshold (0.7).

        The 0.7 threshold is chosen to balance precision and recall for
        the appeal autofill workflow — fields with lower confidence are
        flagged as warnings rather than applied automatically.

        Returns:
            ``True`` when ``confidence >= 0.7`` and ``value`` is not ``None``.
        """
        return self.value is not None and self.confidence >= self._CONFIDENCE_THRESHOLD

    @property
    def is_empty(self) -> bool:
        """Returns True when no value was extracted.

        Returns:
            ``True`` when ``value`` is ``None``.
        """
        return self.value is None


class ExtractionResult(DomainModel):
    """Aggregated result of extracting multiple fields from a document.

    Groups all ``FieldExtractionResult`` instances from a single extraction
    run. Used as the output of ``AppealExtractor`` and the input of
    ``AppealAutofillTool``.

    Immutable — extraction results represent a completed analysis snapshot.

    Attributes:
        source_document_id: UUID string of the document from which text
            was extracted. Used for traceability.
        fields: List of individual field extraction results.
        extraction_model: Name of the LLM / NLP model used.
        warnings: List of non-fatal issues encountered during extraction
            (e.g. low confidence, ambiguous value).

    Example:
        >>> result = ExtractionResult(
        ...     source_document_id="abc-123",
        ...     fields=[
        ...         FieldExtractionResult(field_name="applicant_name", value="Иванов", confidence=0.9),
        ...         FieldExtractionResult(field_name="phone", value=None, confidence=0.0),
        ...     ],
        ... )
        >>> result.confident_fields
        {'applicant_name': 'Иванов'}
    """

    source_document_id: str | None = None
    fields: list[FieldExtractionResult] = []
    extraction_model: str | None = None
    warnings: list[str] = []

    @property
    def confident_fields(self) -> dict[str, Any]:
        """Returns a dict of field_name → value for all confident results.

        Only includes fields where ``is_confident=True``. Used by the
        autofill tool to determine which fields can be applied automatically.

        Returns:
            Dict mapping field names to their extracted values.
            Empty dict when no fields pass the confidence threshold.

        Example:
            {'applicant_name': 'Иванов', 'declarant_type': 'INDIVIDUAL'}
        """
        return {f.field_name: f.value for f in self.fields if f.is_confident}

    @property
    def missing_fields(self) -> list[str]:
        """Returns names of fields where extraction found no value.

        Used to generate user-facing warnings about incomplete extraction.

        Returns:
            List of field name strings where ``value`` is ``None``.
        """
        return [f.field_name for f in self.fields if f.is_empty]

    @property
    def overall_confidence(self) -> float:
        """Returns the average confidence across all extracted fields.

        Returns:
            Mean confidence in [0.0, 1.0]. Returns 0.0 when no fields.
        """
        if not self.fields:
            return 0.0
        return sum(f.confidence for f in self.fields) / len(self.fields)

    @property
    def has_warnings(self) -> bool:
        """Returns True when there are non-fatal extraction warnings.

        Returns:
            ``True`` when the ``warnings`` list is non-empty.
        """
        return bool(self.warnings)
