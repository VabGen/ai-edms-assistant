# src/ai_edms_assistant/shared/utils/json_encoder.py
"""
Custom JSON encoder for EDMS domain types.

Migrated from edms_ai_assistant/utils/json_encoder.py.
Supports: UUID, datetime, Enum, Pydantic v1/v2 models.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from uuid import UUID


class CustomJSONEncoder(json.JSONEncoder):
    """
    JSON encoder extended for EDMS domain types.

    Handles:
        - ``UUID``     → str
        - ``datetime`` → ISO 8601 (appends 'Z' for naive datetimes)
        - ``Enum``     → value
        - Pydantic v2 models (``model_dump``) → dict
        - Pydantic v1 models (``dict``)       → dict
    """

    def default(self, obj: object) -> object:
        # Pydantic v2
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        # Pydantic v1
        if hasattr(obj, "dict"):
            return obj.dict()
        # UUID
        if isinstance(obj, UUID):
            return str(obj)
        # datetime — с timezone или naive (добавляем Z)
        if isinstance(obj, datetime):
            return obj.isoformat() if obj.tzinfo else obj.isoformat() + "Z"
        # Enum
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def dumps(obj: object, **kwargs) -> str:
    """
    Serialize obj to JSON string using CustomJSONEncoder.

    Args:
        obj:    Python object to serialize.
        **kwargs: Forwarded to json.dumps (e.g. indent=2).

    Returns:
        JSON string.
    """
    return json.dumps(obj, cls=CustomJSONEncoder, ensure_ascii=False, **kwargs)
