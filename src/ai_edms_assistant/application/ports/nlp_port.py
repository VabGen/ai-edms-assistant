# src/ai_edms_assistant/application/ports/nlp_port.py
"""Abstract NLP port for application layer.

Defines the interface between application and infrastructure NLP adapters.
Following Dependency Inversion Principle — application depends on abstraction,
not on concrete SpaCy/regex implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractNLPExtractor(ABC):
    """Port interface for NLP text extraction operations.

    All concrete implementations (AppealExtractor, etc.) must satisfy
    this interface so the application layer stays infrastructure-agnostic.
    """

    @abstractmethod
    def extract_all(
        self,
        text: str,
        base_date: Any | None = None,
    ) -> dict[str, list[Any]]:
        """Extract all named entities from text.

        Args:
            text: Source text for extraction.
            base_date: Base datetime for relative date normalization.

        Returns:
            Dict mapping entity type names to lists of Entity objects.
            Example: {"dates": [...], "persons": [...]}
        """
        ...

    @abstractmethod
    def suggest_summarize_format(self, text: str) -> dict[str, Any]:
        """Analyse text structure and recommend summarization format.

        Args:
            text: Text to analyse.

        Returns:
            Dict with keys:
                - ``recommended``: str — "extractive" | "abstractive" | "thesis"
                - ``reason``: str — human-readable explanation
                - ``stats``: dict — {"chars": int, "lines": int}
        """
        ...
