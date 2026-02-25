# src/ai_edms_assistant/application/tools/base_tool.py
from __future__ import annotations

from abc import ABC
from typing import Any

from langchain_core.tools import BaseTool


class AbstractEdmsTool(BaseTool, ABC):
    """Base class for all EDMS-specific LangChain tools.

    Extends LangChain's ``BaseTool`` with common patterns:
    - Dependency injection via constructor
    - Structured error handling
    - Consistent return format

    Subclasses must define:
        - ``args_schema``: Pydantic model for input validation
        - ``_run`` or ``_arun``: Tool execution logic
    """

    def _handle_error(self, error: Exception) -> dict[str, Any]:
        """Standard error response format for all tools.

        Args:
            error: The caught exception.

        Returns:
            Dict with ``status="error"`` and ``message``.
        """
        return {"status": "error", "message": str(error)}

    def _success_response(
        self, data: Any, message: str | None = None
    ) -> dict[str, Any]:
        """Standard success response format.

        Args:
            data: Tool output data.
            message: Optional human-readable message.

        Returns:
            Dict with ``status="success"``, ``data``, and optional ``message``.
        """
        response = {"status": "success", "data": data}
        if message:
            response["message"] = message
        return response
