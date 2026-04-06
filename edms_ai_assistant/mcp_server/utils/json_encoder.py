# edms_ai_assistant/utils/json_encoder.py
import json
from datetime import datetime
from enum import Enum
from uuid import UUID


class CustomJSONEncoder(json.JSONEncoder):
    """
    Кастомный JSON encoder для сериализации специальных типов данных.

    Поддерживает:
    - UUID → str
    - datetime → ISO 8601 с timezone (для java.time.Instant)
    - Enum → value
    - Pydantic models → dict
    """

    def default(self, obj):
        # Pydantic V2 models
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")

        # Pydantic V1 models (legacy)
        if hasattr(obj, "dict"):
            return obj.dict()

        # UUID → string
        if isinstance(obj, UUID):
            return str(obj)

        # datetime → ISO 8601 с timezone
        if isinstance(obj, datetime):
            # Если datetime уже имеет timezone info
            if obj.tzinfo is not None:
                return obj.isoformat()
            else:
                # Naive datetime → добавляем 'Z' (UTC timezone)
                return obj.isoformat() + "Z"

        # Enum → value
        if isinstance(obj, Enum):
            return obj.value

        # Fallback к стандартному encoder
        return json.JSONEncoder.default(self, obj)
