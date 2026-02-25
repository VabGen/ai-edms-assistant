# src/ai_edms_assistant/shared/logging/agent_tracer.py
"""
Agent Reasoning Trace logger.

Writes structured JSON traces for each agent step:
  - tool_call      : agent decided to call a tool
  - tool_result    : tool returned a result
  - llm_step       : LLM reasoning step (text chunk)
  - agent_final    : final agent response

These traces are separate from application logs and can be shipped
to a dedicated observability backend (LangSmith, custom dashboard).

Usage::

    from ai_edms_assistant.shared.logging.agent_tracer import AgentTracer

    tracer = AgentTracer(thread_id="user_123_doc_abc")
    tracer.tool_call("search_documents", {"reg_number": "ИН-001"})
    tracer.tool_result("search_documents", {"found": 3})
    tracer.final_answer("Найдено 3 документа по запросу...")
"""

from __future__ import annotations

import time
from typing import Any

import structlog

_tracer_logger = structlog.get_logger("agent.trace")


class AgentTracer:
    """
    Per-conversation structured trace emitter.

    Each method emits one JSON log line with a consistent schema:
        {
          "event":     "<event_name>",
          "thread_id": "<thread_id>",
          "step":      <monotonic step counter>,
          "elapsed_ms": <ms since tracer creation>,
          ...additional fields...
        }

    Attributes:
        thread_id:  Conversation / thread identifier.
    """

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self._start = time.perf_counter()
        self._step = 0

    def _emit(self, event: str, **kwargs: Any) -> None:
        """Emit a single trace record."""
        self._step += 1
        elapsed = round((time.perf_counter() - self._start) * 1000, 1)
        _tracer_logger.info(
            event,
            thread_id=self.thread_id,
            step=self._step,
            elapsed_ms=elapsed,
            **kwargs,
        )

    def tool_call(self, tool_name: str, inputs: dict[str, Any]) -> None:
        """
        Record an agent tool-call decision.

        Args:
            tool_name: Name of the tool being called.
            inputs:    Serialized tool input arguments.
        """
        self._emit("tool_call", tool=tool_name, inputs=inputs)

    def tool_result(
        self,
        tool_name: str,
        result: Any,
        error: str | None = None,
    ) -> None:
        """
        Record a tool execution result.

        Args:
            tool_name: Name of the tool that returned.
            result:    Tool output (any JSON-serializable value).
            error:     Error message if the tool raised an exception.
        """
        self._emit(
            "tool_result",
            tool=tool_name,
            result=result,
            error=error,
        )

    def llm_step(self, content: str, token_count: int | None = None) -> None:
        """
        Record an intermediate LLM reasoning step.

        Args:
            content:     LLM text output for this step.
            token_count: Optional token count for cost tracking.
        """
        self._emit("llm_step", content=content[:500], token_count=token_count)

    def final_answer(self, content: str, status: str = "success") -> None:
        """
        Record the agent's final answer to the user.

        Args:
            content: Final response text.
            status:  "success" | "error" | "requires_action".
        """
        self._emit("agent_final", content=content[:1000], status=status)

    def error(self, message: str, exc: Exception | None = None) -> None:
        """
        Record an agent-level error.

        Args:
            message: Human-readable error description.
            exc:     Optional exception for traceback context.
        """
        self._emit(
            "agent_error",
            message=message,
            exc_type=type(exc).__name__ if exc else None,
        )
