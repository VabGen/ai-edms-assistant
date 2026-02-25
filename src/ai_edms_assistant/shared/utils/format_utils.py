# src/ai_edms_assistant/shared/utils/format_utils.py
"""
Text formatting utilities for document responses.

Migrated from edms_ai_assistant/utils/format_utils.py.
Provides format_document_response() for LLM output cleanup.
"""

from __future__ import annotations

import re

# Фразы-«мусор», которые LLM иногда добавляет в вывод
_JUNK_PATTERNS: list[str] = [
    r"Похоже, произошла ошибка при попытке извлечь содержание вложения\.",
    r"Я буду использовать другой подход для предоставления информации о документе\.",
    r"Для получения более подробного содержания файла необходимо его извлечь и проанализировать\.",
    r"Для получения более подробной информации о содержании вложения необходимо обратиться к соответствующему инструменту или сервису, который поддерживает извлечение содержимого документов\.",
    r"Для получения более подробного содержания файла необходимо использовать дополнительный инструмент 'summarize_attachment_tool_wrapped'\.",
]

# Строки с техническими полями, которые не нужны пользователю
_UNWANTED_PREFIXES: list[str] = [
    "ID документа:",
    "ID вложения:",
    "Размер:",
    "Дата загрузки:",
    "ID:",
    "UUID",
]


def format_document_response(text_content: str) -> str:
    """
    Clean and format an LLM-generated document response for display.

    Steps:
    1. Unescape escape sequences (\\n → newline, etc.)
    2. Remove known LLM junk phrases
    3. Filter lines containing internal UUID/ID fields
    4. Ensure markdown heading presence
    5. Collapse excess blank lines

    Args:
        text_content: Raw LLM response text.

    Returns:
        Cleaned, display-ready markdown string.
    """
    # Шаг 1: Unescape
    content = (
        text_content.replace(r"\n", "\n").replace(r"\t", "    ").replace(r"\"", '"')
    )

    # Шаг 2: Убираем junk-фразы
    for pattern in _JUNK_PATTERNS:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE | re.DOTALL).strip()

    # Шаг 3: Фильтруем строки с техническими полями
    filtered: list[str] = []
    for line in content.split("\n"):
        stripped = line.strip()
        if any(
            stripped.startswith(f"- **{kw}**") or stripped.startswith(f"- {kw}")
            for kw in _UNWANTED_PREFIXES
        ):
            continue
        if stripped:
            filtered.append(line.rstrip())

    content = "\n".join(filtered)

    # Шаг 4: Гарантируем заголовок
    if content.strip() and not content.strip().startswith("#"):
        content = "## Информация о Документе\n\n" + content

    # Шаг 5: Схлопываем лишние пустые строки
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    return content
