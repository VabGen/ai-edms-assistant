# edms_ai_assistant/tools/summarization.py
"""
EDMS AI Assistant — Document Summarisation Tool.

Выполняет интеллектуальный анализ текста документов через LLM.
"""

from __future__ import annotations

import json
import logging
import re
from enum import StrEnum
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.llm import get_chat_model

logger = logging.getLogger(__name__)

# ─── Константы ───────────────────────────────────────────────────────────────
_MAX_TEXT_LENGTH: int = 12_000
_HEAD_FRACTION: float = 0.67

_SUMMARY_TYPE_ALIASES: dict[str, str] = {
    # extractive
    "факты": "extractive",
    "ключевые факты": "extractive",
    "extractive": "extractive",
    "1": "extractive",
    # abstractive
    "пересказ": "abstractive",
    "краткий пересказ": "abstractive",
    "abstractive": "abstractive",
    "2": "abstractive",
    # thesis
    "тезисы": "thesis",
    "тезисный план": "thesis",
    "thesis": "thesis",
    "3": "thesis",
}


# ─── Domain enumerations ──────────────────────────────────────────────────────


class SummarizeType(StrEnum):
    """Canonical summarisation format identifiers."""

    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    THESIS = "thesis"


# ─── Input schema ─────────────────────────────────────────────────────────────


class SummarizeInput(BaseModel):
    """Validated input schema for the doc_summarize_text tool.

    Attributes:
        text: Document text to summarise (1–50 000 chars). May arrive as a
            JSON envelope from doc_get_file_content — the tool unwraps it.
        summary_type: Summarisation format. When None the tool ALWAYS
            returns requires_choice so the user selects explicitly.
            The agent must NOT auto-fill this field via NLP.
    """

    text: str = Field(
        ...,
        description="Текст документа для суммаризации (или JSON-обёртка от doc_get_file_content)",
        min_length=1,
        max_length=50_000,
    )
    summary_type: SummarizeType | None = Field(
        None,
        description=(
            "Формат суммаризации: "
            "extractive (ключевые факты и данные), "
            "abstractive (краткий пересказ своими словами), "
            "thesis (структурированный тезисный план). "
            "Если None — пользователь ОБЯЗАН выбрать формат (Human-in-the-Loop)."
        ),
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Strip and validate text is not blank.

        Args:
            v: Raw text value.

        Returns:
            Stripped text string.

        Raises:
            ValueError: If text is blank after stripping.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("Текст документа не может быть пустым.")
        return stripped


# ─── Tool ─────────────────────────────────────────────────────────────────────


@tool("doc_summarize_text", args_schema=SummarizeInput)
async def doc_summarize_text(
    text: str,
    summary_type: SummarizeType | None = None,
) -> dict[str, Any]:
    """Perform intelligent summarisation of document text via LLM.

    Human-in-the-Loop contract:
        When ``summary_type`` is None this tool returns ``requires_choice``
        with three labelled options and a heuristic recommendation.
        The agent MUST show this selection to the user and wait for their
        explicit choice before calling the tool again.

    Supported formats when ``summary_type`` is provided:
    - ``extractive`` : Key facts, dates, amounts as a numbered list.
    - ``abstractive``: Concise 1–2 paragraph plain-language retelling.
    - ``thesis``     : Numbered section-by-section thesis plan.

    Internal pipeline:
    1. Unwrap JSON envelope if text arrived from doc_get_file_content.
    2. Normalise summary_type aliases (e.g. «тезисы» → thesis).
    3. If summary_type is None → return requires_choice (user must choose).
    4. Truncate text with head+tail strategy if > 12 000 chars.
    5. Invoke LLM chain and return the result.

    Args:
        text: Document text (plain or JSON-wrapped).
        summary_type: Desired format or None to trigger user selection.

    Returns:
        Dict with one of three statuses:
        - ``requires_choice``: summary_type was None. Contains:
            - options: list of {key, label, description} for UI rendering.
            - hint: heuristically recommended format key.
            - hint_reason: explanation of the recommendation.
        - ``success``: Analysis completed. Contains:
            - content: The summary string.
            - meta: {format_used, text_length, was_truncated}.
        - ``error``: Validation or LLM call failed. Contains:
            - message: Human-readable error description.
    """
    logger.info(
        "doc_summarize_text called",
        extra={
            "text_length": len(text),
            "summary_type": summary_type.value if summary_type else None,
        },
    )

    try:
        clean_text = _unwrap_json_envelope(text)

        if summary_type is None:
            return _build_requires_choice_response(clean_text)

        normalised_type = _normalise_summary_type(summary_type)
        return await _execute_summarization(clean_text, normalised_type)

    except ValueError as exc:
        logger.warning("Validation error in doc_summarize_text: %s", exc)
        return {"status": "error", "message": f"Ошибка валидации: {exc}"}
    except Exception as exc:
        logger.error("doc_summarize_text failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"Не удалось проанализировать документ: {exc}",
        }


# ─── Private helpers ──────────────────────────────────────────────────────────


def _normalise_summary_type(value: Any) -> SummarizeType:
    """Resolve user-provided aliases to a canonical SummarizeType.

    Handles string aliases, integer shortcuts, and existing SummarizeType
    instances. Falls back to EXTRACTIVE for unknown values with a warning.

    Args:
        value: Raw summary_type value from caller.

    Returns:
        Canonical SummarizeType enum member.
    """
    if isinstance(value, SummarizeType):
        return value
    raw = str(value).strip().lower() if value else ""
    canonical = _SUMMARY_TYPE_ALIASES.get(raw, raw)
    try:
        return SummarizeType(canonical)
    except ValueError:
        logger.warning("Unknown summary_type '%s' — falling back to extractive", value)
        return SummarizeType.EXTRACTIVE


def _unwrap_json_envelope(text: str) -> str:
    """Extract plain text from a JSON content envelope if present.

    doc_get_file_content wraps content as ``{"content": "...", ...}``.
    This function transparently unwraps it so the LLM receives clean text.

    Args:
        text: Raw text, potentially JSON-encoded.

    Returns:
        Extracted plain-text string, or original text if not JSON.
    """
    clean = text.strip()
    if not (clean.startswith("{") and clean.endswith("}")):
        return clean
    try:
        data: dict[str, Any] = json.loads(clean)
        for key in ("content", "text", "document_info", "text_preview"):
            extracted = data.get(key)
            if extracted and isinstance(extracted, str) and len(extracted) > 10:
                logger.debug(
                    "Unwrapped JSON envelope via key '%s' (%d chars)",
                    key,
                    len(extracted),
                )
                return extracted.strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return clean


def _heuristic_recommendation(text: str) -> dict[str, str]:
    """Heuristically recommend a summarisation format for the given text.

    Rules (applied in priority order):
    1. Many numeric groups (>20) or large text (>5000 chars) → thesis.
    2. Very short text (<5 lines) → abstractive.
    3. Default → extractive.

    This recommendation is shown to the user as a HINT only —
    the final choice always belongs to the user.

    Args:
        text: Source document text.

    Returns:
        Dict with keys: recommended (str key), reason (Russian explanation).
    """
    if not text:
        return {
            "recommended": "abstractive",
            "reason": "Текст пуст или очень короткий.",
        }

    chars = len(text)
    lines = text.count("\n")
    numeric_groups = len(re.findall(r"\d+", text))

    if chars > 5_000 or numeric_groups > 20:
        return {
            "recommended": "thesis",
            "reason": (
                f"Объёмный документ ({chars} симв.) или много числовых данных "
                f"({numeric_groups} чисел) — тезисный план удобнее для навигации."
            ),
        }
    if lines < 5:
        return {
            "recommended": "abstractive",
            "reason": f"Компактный текст ({lines} строк) — краткого пересказа достаточно.",
        }
    return {
        "recommended": "extractive",
        "reason": "Структурированный текст с конкретными данными — список фактов будет полезнее.",
    }


def _build_requires_choice_response(text: str) -> dict[str, Any]:
    """Build a requires_choice response with format options and a recommendation.

    Always called when summary_type is None. The response is structured
    for the frontend to render a selection widget, and for the agent
    to present labelled options in the chat.

    Args:
        text: Pre-cleaned document text (used for heuristic recommendation).

    Returns:
        Dict with status=requires_choice, options list, hint, and hint_reason.
    """
    hint = _heuristic_recommendation(text)

    return {
        "status": "requires_choice",
        "message": "Выберите формат анализа документа:",
        "options": [
            {
                "key": "extractive",
                "label": "Ключевые факты",
                "description": "Конкретные данные, даты, суммы, имена — нумерованным списком.",
            },
            {
                "key": "abstractive",
                "label": "Краткий пересказ",
                "description": "Суть документа своими словами в 1–2 абзацах.",
            },
            {
                "key": "thesis",
                "label": "Тезисный план",
                "description": "Структурированный план с разделами и подпунктами.",
            },
        ],
        "hint": hint["recommended"],
        "hint_reason": hint["reason"],
    }


def _truncate_for_llm(text: str, max_length: int = _MAX_TEXT_LENGTH) -> str:
    """Truncate text while preserving beginning and end context.

    Strategy: first 67 % + last 33 % of ``max_length`` characters,
    separated by a visible ellipsis marker.

    Args:
        text: Source text.
        max_length: Character budget.

    Returns:
        Original text if within budget, otherwise truncated version.
    """
    if len(text) <= max_length:
        return text
    head = int(max_length * _HEAD_FRACTION)
    tail = max_length - head
    truncated = (
        text[:head] + "\n\n[... часть содержимого пропущена ...]\n\n" + text[-tail:]
    )
    logger.debug(
        "Text truncated for LLM: %d → %d chars (head=%d, tail=%d)",
        len(text),
        len(truncated),
        head,
        tail,
    )
    return truncated


def _build_llm_prompt(summary_type: SummarizeType) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate for the requested summarisation format.

    Each format has a distinct system instruction that drives the LLM
    to produce the appropriate output structure.

    Args:
        summary_type: Canonical summarisation format.

    Returns:
        Ready-to-invoke ChatPromptTemplate.
    """
    instructions: dict[SummarizeType, str] = {
        SummarizeType.EXTRACTIVE: (
            "Выдели ключевые факты: конкретные даты, суммы, имена, сроки и обязательства. "
            "Оформи СТРОГО нумерованным списком. "
            "Каждый пункт — одна конкретная мысль, не более двух предложений."
        ),
        SummarizeType.ABSTRACTIVE: (
            "Напиши связный краткий пересказ сути документа своими словами (1–2 абзаца). "
            "Сохрани ключевую информацию, не используй технические детали. "
            "Пиши как для руководителя, который видит документ впервые."
        ),
        SummarizeType.THESIS: (
            "Сформируй структурированный тезисный план документа. "
            "Используй нумерацию разделов (1., 1.1., 1.2.) и подпункты. "
            "Каждый тезис — одно ёмкое предложение."
        ),
    }

    system_msg = (
        "Ты — ведущий аналитик системы электронного документооборота (СЭД). "
        f"Задача: {instructions[summary_type]} "
        "Отвечай строго на русском языке. "
        "НЕ начинай со слов «В данном документе», «Данный текст», «Документ» — "
        "сразу переходи к содержанию."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("user", "ТЕКСТ ДОКУМЕНТА:\n{text}\n\nРЕЗУЛЬТАТ АНАЛИЗА:"),
        ]
    )


async def _execute_summarization(
    text: str,
    summary_type: SummarizeType,
) -> dict[str, Any]:
    """Execute the LLM summarisation pipeline.

    Args:
        text: Pre-cleaned document text.
        summary_type: Validated summarisation format.

    Returns:
        Dict with status=success and the summary content, or status=error.
    """
    if len(text) < 50:
        return {
            "status": "success",
            "content": "Текст слишком короткий для глубокого анализа.",
            "meta": {
                "format_used": summary_type.value,
                "text_length": len(text),
                "was_truncated": False,
            },
        }

    was_truncated = len(text) > _MAX_TEXT_LENGTH
    processing_text = _truncate_for_llm(text)

    llm = get_chat_model()
    try:
        llm_for_summary = llm.bind_tools([])
    except Exception:
        llm_for_summary = llm

    prompt = _build_llm_prompt(summary_type)
    chain = prompt | llm_for_summary | StrOutputParser()

    logger.info(
        "Invoking LLM for summarization",
        extra={
            "format": summary_type.value,
            "text_length": len(processing_text),
            "was_truncated": was_truncated,
        },
    )

    summary: str = await chain.ainvoke({"text": processing_text})
    summary = summary.strip()

    logger.info(
        "Summarization completed",
        extra={"format": summary_type.value, "summary_length": len(summary)},
    )

    return {
        "status": "success",
        "content": summary,
        "meta": {
            "format_used": summary_type.value,
            "text_length": len(text),
            "was_truncated": was_truncated,
        },
    }
