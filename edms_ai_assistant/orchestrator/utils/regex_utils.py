# orchestrator/utils/regex_utils.py
"""Общие regex-паттерны для переиспользования."""
from __future__ import annotations

import re

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
DOC_NUMBER_RE = re.compile(r"^DOC-\d{1,10}$", re.IGNORECASE)
JWT_RE = re.compile(r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$")
