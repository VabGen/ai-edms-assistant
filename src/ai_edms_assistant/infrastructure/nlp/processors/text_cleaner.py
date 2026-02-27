# src/ai_edms_assistant/infrastructure/nlp/processors/text_cleaner.py
"""Text preprocessing utilities for NLP pipeline.

Handles cleaning of raw document text before extraction:
- Whitespace normalization
- Control character removal
- Russian-language abbreviation expansion
- Action verb normalization (for SemanticDispatcher)

Architecture:
    Infrastructure Layer → no external deps
    Used by: SemanticDispatcher, BaseExtractor
"""

from __future__ import annotations

import re


class TextCleaner:
    """Stateless text cleaning and normalization utility.

    All methods are pure functions — no state, no side effects.
    Safe to use as a module-level singleton.
    """

    # ── Russian EDMS abbreviation expansions ──────────────────────────────────
    ABBREVIATIONS: dict[str, str] = {
        "док": "документ",
        "ознак": "ознакомление",
        "пор": "поручение",
        "исп": "исполнитель",
        "отв": "ответственный",
        "сов": "совещание",
        "дог": "договор",
        "сэд": "система электронного документооборота",
    }

    # ── Action verb synonyms → canonical form ────────────────────────────────
    ACTION_SYNONYMS: dict[str, str] = {
        "покажи": "найди",
        "выведи": "найди",
        "дай": "найди",
        "скажи": "опиши",
        "расскажи": "опиши",
        "объясни": "опиши",
    }

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple whitespace characters to single space.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text with single spaces and stripped edges.
        """
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def remove_control_chars(text: str) -> str:
        """Remove non-printable control characters except newlines.

        Args:
            text: Raw input text.

        Returns:
            Text without control characters.
        """
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    @classmethod
    def expand_abbreviations(cls, text: str) -> str:
        """Expand EDMS-domain abbreviations to full words.

        Args:
            text: Input text with possible abbreviations.

        Returns:
            Text with abbreviations expanded.
        """
        words = text.split()
        return " ".join(cls.ABBREVIATIONS.get(word.lower(), word) for word in words)

    @classmethod
    def normalize_actions(cls, text: str) -> str:
        """Replace informal action verbs with canonical forms.

        Args:
            text: Input text.

        Returns:
            Text with normalized action verbs.
        """
        text_lower = text.lower()
        for synonym, canonical in cls.ACTION_SYNONYMS.items():
            text_lower = re.sub(
                r"\b" + re.escape(synonym) + r"\b", canonical, text_lower
            )
        return text_lower

    @classmethod
    def clean(cls, text: str) -> str:
        """Full cleaning pipeline: control chars → whitespace → abbreviations.

        Args:
            text: Raw text input.

        Returns:
            Fully cleaned and normalized text.
        """
        text = cls.remove_control_chars(text)
        text = cls.normalize_whitespace(text)
        text = cls.expand_abbreviations(text)
        return text
