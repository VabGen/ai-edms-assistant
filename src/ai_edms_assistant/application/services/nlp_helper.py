# src/ai_edms_assistant/application/services/nlp_helper.py
"""NLP Helper Service for the application layer.

Provides a lightweight facade over NLP infrastructure for use in
edms_agent.py. Specifically: suggest_summarize_format() is called
during tool call injection to auto-select summarization strategy.

Architecture:
    Application Layer → Service
    Delegates to: infrastructure/nlp/extractors/base_extractor.py
    Used by: application/agents/edms_agent.py (_orchestrate method)

Design:
    Intentionally stateless — all methods are @staticmethod so they can
    be called without instantiation. Falls back gracefully if NLP
    infrastructure is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NLPHelperService:
    """Stateless NLP helper for agent orchestration layer.

    Provides summarization format suggestion without requiring
    a fully configured NLP extractor instance.

    All methods are static — no constructor needed.

    Example:
        >>> suggestion = NLPHelperService.suggest_summarize_format(text)
        >>> summary_type = suggestion["recommended"]  # "extractive"
    """

    @staticmethod
    def suggest_summarize_format(text: str) -> dict[str, Any]:
        """Recommend summarization format based on text heuristics.

        Heuristics (in priority order):
            1. Empty text → "abstractive"
            2. Long text >5000 chars OR many digits → "thesis"
            3. Short text <5 newlines → "abstractive"
            4. Default → "extractive"

        Falls back to "extractive" on any error — never raises.

        Args:
            text: Document text to analyse. May be empty or None.

        Returns:
            Dict with:
                - ``recommended``: "extractive" | "abstractive" | "thesis"
                - ``reason``: Russian human-readable explanation.
                - ``stats``: {"chars": int, "lines": int}

        Example:
            >>> NLPHelperService.suggest_summarize_format("короткий текст")
            {"recommended": "abstractive", "reason": "...", "stats": {...}}
        """
        try:
            from ...infrastructure.nlp.extractors.base_extractor import BaseExtractor

            extractor = BaseExtractor()
            return extractor.suggest_summarize_format(text or "")

        except Exception as exc:
            logger.debug(
                "nlp_helper_fallback",
                extra={"error": str(exc)},
            )
            return NLPHelperService._fallback_suggest(text or "")

    @staticmethod
    def _fallback_suggest(text: str) -> dict[str, Any]:
        """Built-in heuristic when NLP infrastructure is unavailable.

        Args:
            text: Input text.

        Returns:
            Summarization recommendation dict.
        """
        import re

        if not text:
            return {
                "recommended": "abstractive",
                "reason": "Текст пуст",
                "stats": {"chars": 0, "lines": 0},
            }

        length = len(text)
        lines = text.count("\n")
        digit_count = len(re.findall(r"\d+", text))

        if length > 5000 or digit_count > 20:
            recommended = "thesis"
            reason = "Объёмный текст — тезисный план."
        elif lines < 5:
            recommended = "abstractive"
            reason = "Компактный текст — краткий пересказ."
        else:
            recommended = "extractive"
            reason = "Конкретный текст — ключевые факты."

        return {
            "recommended": recommended,
            "reason": reason,
            "stats": {"chars": length, "lines": lines},
        }
