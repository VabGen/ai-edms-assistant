# src/ai_edms_assistant/application/ports/nlp_port.py
from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.value_objects.extraction_result import ExtractionResult


class AbstractNLPExtractor(ABC):
    """Port (interface) for Natural Language Processing extractors.

    Defines the contract for extracting structured data from unstructured
    text. Used by the appeal autofill tool and document analysis processors.

    Implementations:
        - ``AppealExtractor`` (LLM-based, uses OpenAI for field extraction)
        - ``SpacyExtractor`` (rule-based, uses SpaCy for NER)
        - ``TransformersExtractor`` (local model-based)

    Architecture:
        - Application layer receives ``AbstractNLPExtractor`` via DI.
        - Infrastructure provides concrete extractors implementing this interface.
        - Switch extractors by changing DI config — enables A/B testing different
          extraction strategies without touching use case code.

    Example:
        >>> # In an autofill use case
        >>> class ExtractAppealDataUseCase:
        ...     def __init__(self, extractor: AbstractNLPExtractor) -> None:
        ...         self._extractor = extractor
        ...
        ...     async def execute(self, text: str) -> ExtractionResult:
        ...         return await self._extractor.extract_appeal_fields(text)
    """

    @abstractmethod
    async def extract_appeal_fields(
        self,
        text: str,
        document_id: str | None = None,
    ) -> ExtractionResult:
        """Extract structured appeal fields from unstructured text.

        Attempts to identify and extract:
        - Applicant name (``applicant_name``)
        - Declarant type (``declarant_type``: INDIVIDUAL / ENTITY)
        - Contact info (``email``, ``phone``)
        - Geographic location (``geo_location`` components)
        - Description / subject (``description``, ``question_category``)

        Args:
            text: Unstructured text content from the appeal document.
            document_id: Optional source document UUID string (for traceability).

        Returns:
            ``ExtractionResult`` containing all detected fields with confidence
            scores and warnings for low-confidence / missing fields.
        """

    @abstractmethod
    async def extract_entities(
        self,
        text: str,
        entity_types: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Extract named entities from text.

        General-purpose NER extraction — identifies people, organizations,
        locations, dates, etc.

        Args:
            text: Input text to analyze.
            entity_types: Optional list of entity types to extract
                (e.g. ``["PERSON", "ORG", "LOC"]``). Extracts all types
                when ``None``.

        Returns:
            Dict mapping entity type → list of extracted entity strings.
        """

    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """Returns the name of the extraction backend.

        Returns:
            Extractor identifier, e.g. ``"llm"``, ``"spacy"``, ``"transformers"``.
        """

    @property
    @abstractmethod
    def supports_confidence_scores(self) -> bool:
        """Returns True when the extractor provides confidence scores.

        Returns:
            ``True`` for LLM and ML-based extractors, ``False`` for rule-based.
        """
