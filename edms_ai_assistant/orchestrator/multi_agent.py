# orchestrator/multi_agent.py
"""
Мульти-агентная координация EDMS AI Assistant.

ИЗМЕНЕНИЕ: использует LLMClient (llm.py) вместо прямого anthropic.AsyncAnthropic.
Модели берутся из .env через resolve_model() / model_for_role().

Пять специализированных агентов + координатор.
Инструменты вызываются через MCPClient по HTTP.

Агенты:
    PlannerAgent    — JSON-план для задач с ≥2 шагами
    ResearcherAgent — только read-only MCP операции
    ExecutorAgent   — только write MCP операции
    ReviewerAgent   — проверяет результаты при risk_level=high
    ExplainerAgent  — финальный ответ на русском

MultiAgentCoordinator:
    Простой запрос     → ReAct в одном цикле → Explainer
    Средний (read)     → Researcher → Explainer
    Средний (write)    → Researcher → Executor → Reviewer → Explainer
    Сложный (≥3 шагов) → Planner → [R|E]* → Reviewer → Explainer
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from edms_ai_assistant.llm_client import LLMClient, LLMResponse, get_llm_client

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERS = 8


# ── Датаклассы ────────────────────────────────────────────────────────────


@dataclass
class AgentResult:
    """Результат работы одного агента."""

    agent_name: str
    success: bool
    output: dict[str, Any]
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    model_used: str = ""
    latency_ms: int = 0


@dataclass
class PlanStep:
    """Один шаг плана выполнения."""

    step: int
    action: str
    tool: str
    args: dict[str, Any]
    depends_on: list[int] = field(default_factory=list)
    can_fail: bool = False
    fallback: str = ""
    executed: bool = False
    failed: bool = False
    result: dict[str, Any] | None = None


@dataclass
class ExecutionPlan:
    """Полный план выполнения запроса."""

    plan_id: str
    complexity: str
    estimated_steps: int
    steps: list[PlanStep]
    risk_level: str
    requires_review: bool


# ── Базовый агент ─────────────────────────────────────────────────────────


class BaseAgent(ABC):
    """
    Базовый класс для всех специализированных агентов.

    Использует LLMClient (поддерживает Ollama и Anthropic).
    Инструменты вызываются через MCPClient по HTTP.
    """

    name: str = "base"
    allowed_tools: list[str] = []
    role: str = "executor"  # planner | researcher | executor | reviewer | explainer

    def __init__(self, mcp_client: Any, llm_client: LLMClient) -> None:
        self._mcp = mcp_client
        self._llm = llm_client
        self._model = llm_client.model_for_role(self.role)

    @property
    def model(self) -> str:
        return self._model

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Системный промпт агента."""
        ...

    @abstractmethod
    async def run(self, context: dict[str, Any]) -> AgentResult:
        """Выполнить задачу агента."""
        ...

    def _get_tools_schema(self) -> list[dict[str, Any]]:
        """Фильтрует инструменты из MCPClient по allowed_tools."""
        all_tools: list[dict[str, Any]] = self._mcp.cached_tools if hasattr(self._mcp, "cached_tools") else []
        if not self.allowed_tools:
            return []
        return [t for t in all_tools if t.get("name") in set(self.allowed_tools)]

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Вызывает LLM через LLMClient."""
        return await self._llm.create(
            model=self._model,
            messages=messages,
            tools=tools or None,
            system=self.system_prompt,
            max_tokens=max_tokens,
        )

    async def _react_loop(
        self,
        initial_message: str,
        context: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        ReAct-цикл для агентов с инструментами.

        Returns:
            (текст_ответа, список_вызовов_инструментов)
        """
        tools = self._get_tools_schema()
        messages: list[dict[str, Any]] = [{"role": "user", "content": initial_message}]
        tool_calls_log: list[dict[str, Any]] = []
        last_text = ""

        for _ in range(_MAX_TOOL_ITERS):
            response = await self._call_llm(messages, tools=tools or None)

            assistant_blocks: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "text" and block.text:
                    last_text = block.text
                    assistant_blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input or {},
                    })

            messages.append({"role": "assistant", "content": assistant_blocks})

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason != "tool_use":
                break

            # Выполняем tool_use
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                result = await self._mcp.call_tool(block.name, block.input or {})
                tool_calls_log.append({"tool": block.name, "args": block.input, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        return last_text.strip(), tool_calls_log

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Извлекает первый JSON-объект из текста."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _text_from_response(self, response: LLMResponse) -> str:
        """Извлекает весь текст из ответа LLM."""
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


# ── PlannerAgent ──────────────────────────────────────────────────────────


class PlannerAgent(BaseAgent):
    """Строит JSON-план для многошаговых задач."""

    name = "planner"
    allowed_tools: list[str] = []
    role = "planner"

    @property
    def system_prompt(self) -> str:
        return """Ты — планировщик задач для корпоративной EDMS.

Построй JSON-план для выполнения запроса пользователя.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
READ:  get_document, search_documents, get_document_history, get_workflow_status, get_analytics
WRITE: create_document, update_document_status, assign_document

ПРАВИЛА:
- Сначала READ (получить данные), потом WRITE
- risk_level=high → requires_review=true (всегда)
- Деструктивные статусы (rejected, archived) → risk_level=high
- Максимум 10 шагов

ФОРМАТ ОТВЕТА — строго JSON без комментариев:
{
  "plan_id": "<uuid>",
  "complexity": "simple|medium|complex",
  "estimated_steps": <int>,
  "steps": [
    {
      "step": 1,
      "action": "<описание на русском>",
      "tool": "<mcp_tool_name>",
      "args": {},
      "depends_on": [],
      "can_fail": false,
      "fallback": "<действие если провалился>"
    }
  ],
  "risk_level": "low|medium|high",
  "requires_review": false
}"""

    async def run(self, context: dict[str, Any]) -> AgentResult:
        start_ts = time.monotonic()
        query = context.get("query", "")

        user_msg = (
            f"Запрос: {query}\n"
            f"Намерение: {context.get('intent', 'unknown')}\n"
            f"Сущности: {json.dumps(context.get('entities', {}), ensure_ascii=False)}\n\n"
            "Построй план выполнения."
        )

        try:
            response = await self._call_llm(
                [{"role": "user", "content": user_msg}],
                max_tokens=2048,
            )
            reasoning = self._text_from_response(response)
            plan_data = self._extract_json(reasoning)

            if not plan_data:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    output={},
                    error="Планировщик не вернул валидный JSON",
                    model_used=self._model,
                    latency_ms=int((time.monotonic() - start_ts) * 1000),
                )

            steps = [
                PlanStep(
                    step=s.get("step", i + 1),
                    action=s.get("action", ""),
                    tool=s.get("tool", ""),
                    args=s.get("args", {}),
                    depends_on=s.get("depends_on", []),
                    can_fail=s.get("can_fail", False),
                    fallback=s.get("fallback", ""),
                )
                for i, s in enumerate(plan_data.get("steps", []))
            ]

            plan = ExecutionPlan(
                plan_id=plan_data.get("plan_id", ""),
                complexity=plan_data.get("complexity", "medium"),
                estimated_steps=plan_data.get("estimated_steps", len(steps)),
                steps=steps[:10],
                risk_level=plan_data.get("risk_level", "low"),
                requires_review=plan_data.get("requires_review", False),
            )

            return AgentResult(
                agent_name=self.name,
                success=True,
                output={"plan": plan},
                reasoning=reasoning,
                model_used=self._model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )

        except Exception as exc:
            logger.error("PlannerAgent error: %s", exc, exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={},
                error=str(exc),
                model_used=self._model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )


# ── ResearcherAgent ───────────────────────────────────────────────────────


class ResearcherAgent(BaseAgent):
    """Только read-only MCP-операции."""

    name = "researcher"
    allowed_tools = [
        "get_document", "search_documents",
        "get_document_history", "get_workflow_status", "get_analytics",
    ]
    role = "researcher"

    @property
    def system_prompt(self) -> str:
        return """Ты — исследователь в корпоративной EDMS.
Собери необходимую информацию через read-only инструменты.
Отвечай на русском языке. Структурируй данные для следующего агента.
Если данных недостаточно — сообщи что именно не найдено."""

    async def run(self, context: dict[str, Any]) -> AgentResult:
        start_ts = time.monotonic()
        query = context.get("query", "")
        step = context.get("plan_step", {})

        msg = (
            f"Запрос: {query}\n"
            f"Задача: {step.get('action', 'собери информацию')}\n"
            f"Параметры: {json.dumps(step.get('args', {}), ensure_ascii=False)}\n\n"
            "Собери необходимые данные."
        )

        try:
            text, tool_calls = await self._react_loop(msg, context)
            return AgentResult(
                agent_name=self.name,
                success=True,
                output={"research_results": text, "tool_calls": tool_calls},
                reasoning=text,
                tool_calls=tool_calls,
                model_used=self._model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )
        except Exception as exc:
            logger.error("ResearcherAgent error: %s", exc, exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={},
                error=str(exc),
                model_used=self._model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )


# ── ExecutorAgent ─────────────────────────────────────────────────────────


class ExecutorAgent(BaseAgent):
    """Только write MCP-операции."""

    name = "executor"
    allowed_tools = [
        "create_document", "update_document_status", "assign_document",
    ]
    role = "executor"

    @property
    def system_prompt(self) -> str:
        return """Ты — исполнитель операций в корпоративной EDMS.
КРИТИЧЕСКИЕ ПРАВИЛА:
1. Все write-операции необратимы — проверь параметры перед вызовом
2. При изменении статуса на rejected/archived — обязателен comment
3. Используй данные из research_results для параметров
4. Если данных недостаточно — НЕ выполняй операцию, сообщи об этом
Сообщи что именно было сделано после выполнения."""

    async def run(self, context: dict[str, Any]) -> AgentResult:
        start_ts = time.monotonic()
        step = context.get("plan_step", {})
        research = context.get("research_results", "")

        msg = (
            f"Запрос: {context.get('query', '')}\n"
            f"Собранная информация: {research}\n"
            f"Действие: {step.get('action', '')}\n"
            f"Параметры: {json.dumps(step.get('args', {}), ensure_ascii=False)}\n\n"
            "Выполни действие."
        )

        try:
            text, tool_calls = await self._react_loop(msg, context)
            success = any(
                tc.get("result", {}).get("success", False) for tc in tool_calls
            ) if tool_calls else bool(text)

            return AgentResult(
                agent_name=self.name,
                success=success,
                output={"execution_results": text, "tool_calls": tool_calls},
                reasoning=text,
                tool_calls=tool_calls,
                model_used=self._model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )
        except Exception as exc:
            logger.error("ExecutorAgent error: %s", exc, exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={},
                error=str(exc),
                model_used=self._model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )


# ── ReviewerAgent ─────────────────────────────────────────────────────────


class ReviewerAgent(BaseAgent):
    """Проверяет результаты при risk_level=high."""

    name = "reviewer"
    allowed_tools: list[str] = []
    role = "reviewer"

    @property
    def system_prompt(self) -> str:
        return """Ты — ревьюер результатов в корпоративной EDMS.
Проверь выполненные действия по чеклисту:
□ Все шаги плана выполнены?
□ Результаты соответствуют исходному запросу?
□ Нет побочных эффектов?
□ Данные целостны (ID валидны, статусы корректны)?
□ Пользователь получит понятный ответ?

ФОРМАТ ОТВЕТА — строго JSON:
{
  "approved": true,
  "confidence": 0.95,
  "issues": [],
  "suggestions": [],
  "must_redo": [],
  "explanation": "<заключение на русском>"
}"""

    async def run(self, context: dict[str, Any]) -> AgentResult:
        start_ts = time.monotonic()

        plan = context.get("plan")
        plan_desc = ""
        if isinstance(plan, ExecutionPlan):
            plan_desc = "\n".join(
                f"Шаг {s.step}: {s.action} — {'✅' if s.executed and not s.failed else '❌'}"
                for s in plan.steps
            )

        msg = (
            f"Запрос: {context.get('query', '')}\n"
            f"План:\n{plan_desc}\n"
            f"Результаты исследования: {context.get('research_results', '')}\n"
            f"Результаты выполнения: {context.get('execution_results', '')}\n\n"
            "Проверь корректность и верни JSON-оценку."
        )

        try:
            response = await self._call_llm(
                [{"role": "user", "content": msg}], max_tokens=1024
            )
            reasoning = self._text_from_response(response)
            review_data = self._extract_json(reasoning) or {
                "approved": True,
                "confidence": 0.7,
                "explanation": reasoning,
            }

            return AgentResult(
                agent_name=self.name,
                success=True,
                output={"review": review_data},
                reasoning=reasoning,
                model_used=self._model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )

        except Exception as exc:
            logger.error("ReviewerAgent error: %s", exc, exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={"review": {"approved": False, "explanation": str(exc)}},
                error=str(exc),
                model_used=self._model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )


# ── ExplainerAgent ────────────────────────────────────────────────────────


class ExplainerAgent(BaseAgent):
    """Формирует финальный human-readable ответ на русском."""

    name = "explainer"
    allowed_tools: list[str] = []
    role = "explainer"

    @property
    def system_prompt(self) -> str:
        return """Ты — ИИ-ассистент корпоративной EDMS.
Сформулируй понятный финальный ответ пользователю на русском языке.

СТРУКТУРА ОТВЕТА:
1. Краткий итог (1-2 предложения): что сделано/найдено
2. Детали (если нужно): числа, статусы, имена
3. Следующие шаги (если применимо)

ПРАВИЛА:
- Только русский (технические термины допустимы)
- Не показывай UUID, HTTP-коды, stack traces
- Если что-то не выполнено — объясни причину и предложи альтернативу
- Будь конкретен: называй документы, статусы, числа"""

    async def run(self, context: dict[str, Any]) -> AgentResult:
        start_ts = time.monotonic()

        parts = [f"Запрос: {context.get('query', '')}\n"]
        if context.get("research_results"):
            parts.append(f"Найденная информация:\n{context['research_results']}\n")
        if context.get("execution_results"):
            parts.append(f"Выполненные действия:\n{context['execution_results']}\n")
        if context.get("review"):
            rev = context["review"]
            if not rev.get("approved", True):
                parts.append(f"Результат проверки: НЕ ОДОБРЕНО\n{rev.get('explanation', '')}\n")
        if context.get("error_message"):
            parts.append(f"Ошибка: {context['error_message']}\n")

        parts.append("Сформулируй финальный ответ пользователю.")

        try:
            response = await self._call_llm(
                [{"role": "user", "content": "\n".join(parts)}],
                max_tokens=1024,
            )
            final_text = self._text_from_response(response)
            return AgentResult(
                agent_name=self.name,
                success=True,
                output={"final_response": final_text},
                reasoning=final_text,
                model_used=self._model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )
        except Exception as exc:
            logger.error("ExplainerAgent error: %s", exc, exc_info=True)
            fallback = "Произошла ошибка при формировании ответа. Попробуйте переформулировать запрос."
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={"final_response": fallback},
                error=str(exc),
                model_used=self._model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )


# ── MultiAgentCoordinator ─────────────────────────────────────────────────


_READ_INTENTS = {"get_document", "get_history", "get_workflow_status", "get_analytics", "search_documents"}
_WRITE_INTENTS = {"create_document", "update_status", "assign_document"}
_COMPLEX_INTENTS = {"get_workflow_status", "get_analytics"}


class MultiAgentCoordinator:
    """
    Маршрутизирует запрос к нужному набору агентов.

    Принимает llm_client вместо anthropic_client для поддержки Ollama.
    """

    def __init__(self, mcp_client: Any, llm_client: LLMClient | None = None) -> None:
        client = llm_client or get_llm_client()
        args = (mcp_client, client)
        self._planner = PlannerAgent(*args)
        self._researcher = ResearcherAgent(*args)
        self._executor = ExecutorAgent(*args)
        self._reviewer = ReviewerAgent(*args)
        self._explainer = ExplainerAgent(*args)
        self._mcp = mcp_client

    async def route(
        self,
        nlu_result: Any,
        context: dict[str, Any],
    ) -> AgentResult:
        """Маршрутизирует запрос к агентам по сложности и намерению."""
        intent = nlu_result.intent
        confidence = nlu_result.confidence
        bypass_llm = nlu_result.bypass_llm

        is_write = intent in _WRITE_INTENTS
        is_simple = bypass_llm or (
            intent in _READ_INTENTS
            and confidence > 0.85
            and not is_write
        )
        is_complex = intent not in _READ_INTENTS | _WRITE_INTENTS

        agent_context = {
            **context,
            "intent": intent,
            "entities": {
                "document_ids": nlu_result.entities.document_ids,
                "statuses": nlu_result.entities.statuses,
                "document_types": nlu_result.entities.document_types,
            },
        }

        logger.info(
            "MultiAgentCoordinator.route: intent=%s confidence=%.2f "
            "simple=%s write=%s complex=%s",
            intent, confidence, is_simple, is_write, is_complex,
        )

        if is_simple:
            return await self._route_simple(nlu_result, agent_context)
        elif is_write:
            return await self._route_write(agent_context)
        elif is_complex:
            return await self._route_complex(agent_context)
        else:
            return await self._route_read(agent_context)

    async def _route_simple(self, nlu_result: Any, context: dict[str, Any]) -> AgentResult:
        if nlu_result.bypass_llm and nlu_result.required_tool:
            result = await self._mcp.call_tool(
                nlu_result.required_tool, nlu_result.tool_args or {}
            )
            context["research_results"] = json.dumps(
                result.get("data", result), ensure_ascii=False
            )
        return await self._explainer.run(context)

    async def _route_read(self, context: dict[str, Any]) -> AgentResult:
        r = await self._researcher.run(context)
        context["research_results"] = r.output.get("research_results", "")
        if not r.success:
            context["error_message"] = r.error
        return await self._explainer.run(context)

    async def _route_write(self, context: dict[str, Any]) -> AgentResult:
        r = await self._researcher.run(context)
        context["research_results"] = r.output.get("research_results", "")

        e = await self._executor.run(context)
        context["execution_results"] = e.output.get("execution_results", "")

        rev = await self._reviewer.run(context)
        context["review"] = rev.output.get("review", {})

        return await self._explainer.run(context)

    async def _route_complex(self, context: dict[str, Any]) -> AgentResult:
        plan_result = await self._planner.run(context)
        if not plan_result.success:
            context["error_message"] = f"Планировщик: {plan_result.error}"
            return await self._explainer.run(context)

        plan: ExecutionPlan = plan_result.output["plan"]
        context["plan"] = plan
        all_research: list[str] = []
        all_execution: list[str] = []

        _RESEARCHER_TOOLS = set(ResearcherAgent.allowed_tools)
        _EXECUTOR_TOOLS = set(ExecutorAgent.allowed_tools)

        for step in plan.steps:
            if step.depends_on and not all(
                any(s.step == dep and s.executed and not s.failed for s in plan.steps)
                for dep in step.depends_on
            ):
                logger.warning("Skipping step %d: dependencies not met", step.step)
                step.failed = True
                continue

            step_ctx = {
                **context,
                "plan_step": {"action": step.action, "tool": step.tool, "args": step.args},
            }

            if step.tool in _RESEARCHER_TOOLS:
                res = await self._researcher.run(step_ctx)
                if res.success:
                    all_research.append(res.output.get("research_results", ""))
                    step.executed = True
                    step.result = res.output
                else:
                    step.failed = True
                    if not step.can_fail:
                        context["error_message"] = f"Шаг {step.step} провалился: {res.error}"
                        break

            elif step.tool in _EXECUTOR_TOOLS:
                res = await self._executor.run(step_ctx)
                if res.success:
                    all_execution.append(res.output.get("execution_results", ""))
                    step.executed = True
                    step.result = res.output
                else:
                    step.failed = True
                    if not step.can_fail:
                        context["error_message"] = f"Шаг {step.step} (write) провалился"
                        break

        context["research_results"] = "\n---\n".join(filter(None, all_research))
        context["execution_results"] = "\n---\n".join(filter(None, all_execution))

        if plan.requires_review:
            rev = await self._reviewer.run(context)
            context["review"] = rev.output.get("review", {})

        return await self._explainer.run(context)