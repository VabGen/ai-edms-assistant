# src/ai_edms_assistant/application/processors/appeal_processor.py
"""Appeal text processor for cleaning and normalization.

This is a basic implementation with simple text cleaning.
For production, integrate SpaCy for advanced NLP preprocessing.
"""

from __future__ import annotations

import logging
import re

from .base_processor import AbstractProcessor

logger = logging.getLogger(__name__)


class AppealProcessor(AbstractProcessor[str, str]):
    """Preprocess appeal text for NLP extraction (basic implementation).

    Performs:
        - Whitespace normalization
        - Remove extra newlines
        - Trim excessive spaces
        - Basic punctuation cleanup

    Planned features (TODO):
        - Sentence segmentation
        - Lemmatization (Russian)
        - Named entity recognition
        - Phone/email extraction
        - Address normalization
    """

    async def process(self, input_data: str) -> str:
        """Clean and normalize appeal text.

        Args:
            input_data: Raw appeal text from document or file.

        Returns:
            Cleaned and normalized text.
        """
        if not input_data:
            logger.warning("AppealProcessor received empty input")
            return ""

        text = input_data

        # 1. Normalize whitespace
        text = self._normalize_whitespace(text)

        # 2. Remove excessive newlines (more than 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 3. Trim leading/trailing whitespace
        text = text.strip()

        logger.info(f"AppealProcessor: cleaned {len(input_data)} → {len(text)} chars")

        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace (multiple spaces → single space).

        Args:
            text: Input text.

        Returns:
            Text with normalized whitespace.
        """
        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)

        # Replace tabs with spaces
        text = text.replace("\t", " ")

        # Remove trailing spaces on each line
        lines = [line.rstrip() for line in text.split("\n")]

        return "\n".join(lines)


# Production implementation example (commented out):
"""
import spacy
from typing import Dict, Any

class AppealProcessor(AbstractProcessor[str, Dict[str, Any]]):
    def __init__(self):
        self.nlp = spacy.load("ru_core_news_lg")

    async def process(self, input_data: str) -> Dict[str, Any]:
        # Clean text
        text = self._normalize_whitespace(input_data)

        # Process with SpaCy
        doc = self.nlp(text)

        # Extract entities
        entities = {
            "persons": [ent.text for ent in doc.ents if ent.label_ == "PER"],
            "locations": [ent.text for ent in doc.ents if ent.label_ in ("LOC", "GPE")],
            "orgs": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
        }

        # Extract contact info with regex
        phones = re.findall(r"\\+?[0-9]{1,4}[-\\s]?\\(?[0-9]{1,4}\\)?[-\\s]?[0-9]{1,9}", text)
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", text)

        return {
            "cleaned_text": text,
            "sentences": [sent.text for sent in doc.sents],
            "entities": entities,
            "phones": phones,
            "emails": emails,
        }
"""
