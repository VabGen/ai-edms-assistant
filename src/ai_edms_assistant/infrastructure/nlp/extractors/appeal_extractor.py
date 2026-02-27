# src/ai_edms_assistant/infrastructure/nlp/extractors/appeal_extractor.py
"""Appeal-specific NLP extractor.

Extends BaseExtractor with appeal domain extraction:
- Declarant type detection (INDIVIDUAL / ENTITY)
- Address normalization
- Appeal subject classification

Architecture:
    Infrastructure Layer
    Inherits: BaseExtractor (regex-based)
    Implements: AbstractNLPExtractor port
"""

from __future__ import annotations

import logging
from datetime import datetime

from ....domain.value_objects.nlp_entities import Entity, EntityType
from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class AppealExtractor(BaseExtractor):
    """Appeal-domain NLP extractor.

    Adds appeal-specific entity recognition on top of BaseExtractor:
    - ``declarant_type``: физическое / юридическое лицо
    - ``address_parts``: structured address components
    - ``subject_keywords``: appeal topic classification hints

    Suitable for use in autofill scenarios (AppealAutofillTool).
    """

    # ── Declarant type keywords ───────────────────────────────────────────────
    _INDIVIDUAL_KEYWORDS: frozenset[str] = frozenset(
        {
            "физическое лицо",
            "физлицо",
            "гражданин",
            "гражданка",
            "заявитель",
            "житель",
            "жительница",
        }
    )
    _ENTITY_KEYWORDS: frozenset[str] = frozenset(
        {
            "юридическое лицо",
            "юрлицо",
            "организация",
            "предприятие",
            "ооо",
            "зао",
            "оао",
            "уп",
            "рупп",
            "рупп",
            "ип ",
        }
    )

    def extract_all(
        self,
        text: str,
        base_date: datetime | None = None,
    ) -> dict[str, list[Entity]]:
        """Extract all entities including appeal-specific ones.

        Args:
            text: Appeal text (Russian).
            base_date: Base datetime for relative date resolution.

        Returns:
            Dict with standard entity types + appeal-specific keys:
                - ``declarant_type``: list with one PERSON-typed entity
                  holding "INDIVIDUAL" or "ENTITY" as value.
        """
        result = super().extract_all(text, base_date)

        # ── Declarant type detection ──────────────────────────────────────────
        declarant = self._detect_declarant_type(text)
        if declarant:
            result["declarant_type"] = [declarant]

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_declarant_type(self, text: str) -> Entity | None:
        """Detect whether applicant is individual or legal entity.

        Args:
            text: Appeal text.

        Returns:
            Entity with value "INDIVIDUAL" or "ENTITY", or None.
        """
        text_lower = text.lower()

        for keyword in self._ENTITY_KEYWORDS:
            if keyword in text_lower:
                return Entity(
                    type=EntityType.PERSON,
                    value="ENTITY",
                    raw_text=keyword,
                    confidence=0.85,
                    normalized_value="ENTITY",
                )

        for keyword in self._INDIVIDUAL_KEYWORDS:
            if keyword in text_lower:
                return Entity(
                    type=EntityType.PERSON,
                    value="INDIVIDUAL",
                    raw_text=keyword,
                    confidence=0.8,
                    normalized_value="INDIVIDUAL",
                )

        return None
