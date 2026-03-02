# src/ai_edms_assistant/application/agents/prompts/intent_guides.py
"""Intent-specific prompt guides for EdmsDocumentAgent.
"""

INTENT_GUIDES: dict[str, str] = {
    "create_introduction": """
<introduction_guide>
При "requires_disambiguation": перечисли найденных сотрудников, дождись выбора.
Повторный вызов: introduction_create_tool(selected_employee_ids=[uuid1, uuid2])
</introduction_guide>""",

    "create_task": """
<task_guide>
executor_last_names: обязателен (минимум 1).
planed_date_end: ISO 8601 UTC — "2026-03-01T23:59:59Z".
Если дата не указана → текущая дата + 7 дней.
При "requires_disambiguation" → перечисли найденных, дождись выбора.
</task_guide>""",

    "summarize": """
<summarize_guide>
Шаг 1: doc_get_file_content(attachment_id=<uuid>) — извлечь текст
Шаг 2: doc_summarize_text(text=<текст>) — summary_type оставь пустым
Инструмент сам предложит пользователю выбрать формат (факты / пересказ / тезисы).
</summarize_guide>""",

    "compare": """
<compare_guide>
РЕЖИМ 1 — Версии одного документа:
  Используй когда пользователь спрашивает: "сколько версий", "покажи версии",
  "что изменилось", "история изменений", "предыдущая версия" и т.п.
  Вызов: doc_compare(document_id=<id из document_context>)
  → инструмент вернёт список версий + сравнение крайних версий.

  Конкретные версии:
  doc_compare(document_id=<id>, version_1=1, version_2=2)

РЕЖИМ 2 — Два разных документа:
  doc_compare(document_id_1=<UUID1>, document_id_2=<UUID2>)

Параметр comparison_focus (опционально):
  "all"      → все поля (по умолчанию)
  "metadata" → рег. номер, статус, тип документа
  "content"  → текст, исполнитель, примечание
  "contract" → договорные поля (сумма, даты, валюта)

ВАЖНО:
  - Если <document_context> содержит <versioning> — документ имеет версии.
  - ВСЕГДА вызывай doc_compare для версионных запросов.
  - НЕ отвечай из памяти "нет данных о версиях" — данные только в API.
</compare_guide>""",
}