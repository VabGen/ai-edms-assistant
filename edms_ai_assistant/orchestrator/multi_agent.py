"""
orchestrator/multi_agent.py — Мульти-агентная координация EDMS AI Assistant.

Пять специализированных агентов + координатор:
  - PlannerAgent: строит JSON-план для сложных задач (claude-opus)
  - ResearcherAgent: только read-only MCP операции
  - ExecutorAgent: только write MCP операции
  - ReviewerAgent: проверяет результаты (risk_level=high)
  - ExplainerAgent: формирует финальный ответ на русском

MultiAgentCoordinator маршрутизирует запрос к нужному набору агентов.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import anthropic

from nlp_preprocessor import NLUResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_MAX_TOKENS: int = 4096
TOOL_TIMEOUT: float = 30.0

# Модели по ролям
MODEL_PLANNER = os.getenv("MODEL_PLANNER", "claude-opus-4-5")
MODEL_RESEARCHER = os.getenv("MODEL_RESEARCHER", "claude-haiku-4-5")
MODEL_EXECUTOR = os.getenv("MODEL_EXECUTOR", "claude-sonnet-4-6")
MODEL_REVIEWER = os.getenv("MODEL_REVIEWER", "claude-opus-4-5")
MODEL_EXPLAINER = os.getenv("MODEL_EXPLAINER", "claude-haiku-4-5")


# ---------------------------------------------------------------------------
# Датаклассы
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """
    Результат работы агента.

    Поля:
        agent_name: имя агента
        success: успешно ли выполнена задача
        output: результирующие данные
        reasoning: текст рассуждения агента
        tool_calls: список вызовов инструментов с результатами
        error: сообщение об ошибке (если success=False)
        model_used: использованная модель
        latency_ms: время выполнения в миллисекундах
    """
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
    result: dict[str, Any] | None = None
    executed: bool = False
    failed: bool = False


@dataclass
class ExecutionPlan:
    """
    Полный план выполнения сложного запроса.

    Поля:
        plan_id: уникальный ID плана
        complexity: оценка сложности (simple/medium/complex)
        estimated_steps: ожидаемое число шагов
        steps: список шагов плана
        risk_level: уровень риска (low/medium/high)
        requires_review: нужна ли проверка Reviewer
    """
    plan_id: str
    complexity: str
    estimated_steps: int
    steps: list[PlanStep]
    risk_level: str
    requires_review: bool


# ---------------------------------------------------------------------------
# MCP Tool Caller (абстракция над HTTP-вызовами MCP)
# ---------------------------------------------------------------------------

class MCPToolCaller:
    """
    Вызов MCP-инструментов через HTTP.
    Используется агентами для выполнения инструментальных действий.
    """

    def __init__(self, mcp_url: str) -> None:
        self._mcp_url = mcp_url.rstrip("/")

    async def call(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Вызвать MCP-инструмент.

        Параметры:
            tool_name: имя инструмента из реестра
            args: аргументы инструмента

        Возвращает:
            Результат выполнения инструмента
        """
        import httpx
        try:
            async with httpx.AsyncClient(timeout=TOOL_TIMEOUT) as client:
                response = await client.post(
                    f"{self._mcp_url}/call",
                    json={"tool": tool_name, "args": args},
                )
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            log.error("MCP call failed: tool=%s error=%s", tool_name, exc)
            return {"success": False, "error": {"code": "MCP_ERROR", "message": str(exc)}}


# ---------------------------------------------------------------------------
# Базовый агент
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Базовый класс для всех специализированных агентов.

    Каждый агент имеет:
    - Определённую роль и список разрешённых инструментов
    - Шаблон системного промпта
    - Метод run() для выполнения задачи
    """

    name: str = "base_agent"
    allowed_tools: list[str] = []
    model: str = "claude-haiku-4-5"

    def __init__(self, mcp_caller: MCPToolCaller) -> None:
        self._mcp = mcp_caller
        self._client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Системный промпт агента."""
        ...

    @abstractmethod
    async def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Выполнить задачу агента.

        Параметры:
            context: контекст задачи (запрос, история, инструменты и т.д.)

        Возвращает:
            AgentResult с результатом выполнения
        """
        ...

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> anthropic.types.Message:
        """
        Вызвать Anthropic API.

        Параметры:
            messages: история диалога
            tools: доступные инструменты (в формате Anthropic)
            max_tokens: максимальное количество токенов ответа

        Возвращает:
            Ответ от Anthropic API
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": self.system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        return await self._client.messages.create(**kwargs)

    def _extract_reasoning(self, response: anthropic.types.Message) -> str:
        """Извлечь текстовое рассуждение из ответа LLM."""
        text_blocks = [
            block.text
            for block in response.content
            if hasattr(block, "text")
        ]
        return "\n".join(text_blocks)

    def _mcp_tools_to_anthropic(self, tool_names: list[str]) -> list[dict[str, Any]]:
        """
        Конвертировать список имён MCP-инструментов в формат Anthropic tools.

        Использует упрощённые схемы (полная схема загружается из реестра).
        """
        # Упрощённые схемы для быстрого старта
        # В production загружались бы из tools_registry.json
        tool_schemas: dict[str, dict[str, Any]] = {
            "get_document": {
                "name": "get_document",
                "description": "Получить документ по ID",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "include_history": {"type": "boolean"},
                        "include_attachments": {"type": "boolean"},
                    },
                    "required": ["document_id"],
                },
            },
            "search_documents": {
                "name": "search_documents",
                "description": "Поиск документов по фильтрам",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "status": {"type": "array", "items": {"type": "string"}},
                        "document_type": {"type": "array", "items": {"type": "string"}},
                        "date_from": {"type": "string"},
                        "date_to": {"type": "string"},
                        "page": {"type": "integer"},
                        "page_size": {"type": "integer"},
                    },
                },
            },
            "create_document": {
                "name": "create_document",
                "description": "Создать новый документ (необратимо)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "document_type": {"type": "string"},
                        "content": {"type": "string"},
                        "assignees": {"type": "array", "items": {"type": "string"}},
                        "department": {"type": "string"},
                        "due_date": {"type": "string"},
                    },
                    "required": ["title", "document_type"],
                },
            },
            "update_document_status": {
                "name": "update_document_status",
                "description": "Изменить статус документа",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "new_status": {"type": "string"},
                        "comment": {"type": "string"},
                        "notify_assignees": {"type": "boolean"},
                    },
                    "required": ["document_id", "new_status"],
                },
            },
            "get_document_history": {
                "name": "get_document_history",
                "description": "История изменений документа",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "event_types": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer"},
                    },
                    "required": ["document_id"],
                },
            },
            "assign_document": {
                "name": "assign_document",
                "description": "Назначить ответственных за документ",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "assignees": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string"},
                                    "role": {"type": "string"},
                                },
                            },
                        },
                        "replace_existing": {"type": "boolean"},
                    },
                    "required": ["document_id", "assignees"],
                },
            },
            "get_analytics": {
                "name": "get_analytics",
                "description": "Аналитика и статистика по документам",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "metric_type": {"type": "string"},
                        "date_from": {"type": "string"},
                        "date_to": {"type": "string"},
                        "group_by": {"type": "string"},
                    },
                    "required": ["metric_type"],
                },
            },
            "get_workflow_status": {
                "name": "get_workflow_status",
                "description": "Статус рабочего процесса документа",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "include_completed": {"type": "boolean"},
                    },
                    "required": ["document_id"],
                },
            },
        }

        return [
            tool_schemas[name]
            for name in tool_names
            if name in tool_schemas
        ]


# ---------------------------------------------------------------------------
# Специализированные агенты
# ---------------------------------------------------------------------------

class PlannerAgent(BaseAgent):
    """
    Агент-планировщик для сложных многошаговых задач.

    Всегда использует claude-opus для максимального качества планирования.
    Строит JSON-план с шагами, зависимостями и оценкой рисков.
    """

    name = "planner"
    allowed_tools = []  # Плановщик не вызывает инструменты
    model = MODEL_PLANNER

    @property
    def system_prompt(self) -> str:
        return """Ты — планировщик задач для корпоративной системы документооборота (EDMS).

Твоя задача: разбить сложный запрос пользователя на чёткий пошаговый план выполнения.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
- get_document: получить документ по ID (read)
- search_documents: поиск документов (read)
- get_document_history: история документа (read)
- get_workflow_status: статус workflow (read)
- get_analytics: аналитика (read)
- create_document: создать документ (write, НЕОБРАТИМО)
- update_document_status: изменить статус (write, НЕОБРАТИМО)
- assign_document: назначить ответственных (write)

ПРАВИЛА:
1. Всегда начинай с read-операций (получить информацию), потом write
2. Деструктивные операции (create, update_status с rejected/archived) → risk_level=high → requires_review=true
3. Максимум 10 шагов; если больше — разбей на подзадачи
4. Каждый шаг должен иметь чёткое действие и инструмент

ФОРМАТ ОТВЕТА — строго JSON:
{
  "plan_id": "<uuid>",
  "complexity": "simple|medium|complex",
  "estimated_steps": <int>,
  "steps": [
    {
      "step": <int>,
      "action": "<описание действия на русском>",
      "tool": "<mcp_tool_name>",
      "args": {<параметры>},
      "depends_on": [<step_ids>],
      "can_fail": <bool>,
      "fallback": "<действие если шаг провалился>"
    }
  ],
  "risk_level": "low|medium|high",
  "requires_review": <bool>
}

Не добавляй пояснений вне JSON."""

    async def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Построить план выполнения запроса.

        Параметры:
            context: словарь с ключами 'query', 'nlu_result', 'entities'

        Возвращает:
            AgentResult с полем output['plan'] типа ExecutionPlan
        """
        start_ts = time.monotonic()
        query = context.get("query", "")
        few_shot = context.get("few_shot_examples", "")

        user_message = f"""Запрос пользователя: {query}

Распознанное намерение: {context.get('intent', 'unknown')}
Найденные сущности: {json.dumps(context.get('entities', {}), ensure_ascii=False)}

{few_shot}

Построй план выполнения."""

        messages = [{"role": "user", "content": user_message}]

        try:
            response = await self._call_llm(messages, max_tokens=2048)
            reasoning = self._extract_reasoning(response)

            # Парсим JSON-план
            plan_json = self._extract_json(reasoning)
            if not plan_json:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    output={},
                    error="Не удалось разобрать план (невалидный JSON)",
                    model_used=self.model,
                    latency_ms=int((time.monotonic() - start_ts) * 1000),
                )

            plan = self._json_to_plan(plan_json)
            return AgentResult(
                agent_name=self.name,
                success=True,
                output={"plan": plan},
                reasoning=reasoning,
                model_used=self.model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )

        except Exception as exc:
            log.error("PlannerAgent error: %s", exc, exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={},
                error=str(exc),
                model_used=self.model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Извлечь JSON из текста ответа."""
        import re
        # Ищем JSON в тексте
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _json_to_plan(self, data: dict[str, Any]) -> ExecutionPlan:
        """Конвертировать JSON в ExecutionPlan."""
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
            for i, s in enumerate(data.get("steps", []))
        ]
        return ExecutionPlan(
            plan_id=data.get("plan_id", str(uuid.uuid4())),
            complexity=data.get("complexity", "medium"),
            estimated_steps=data.get("estimated_steps", len(steps)),
            steps=steps[:10],  # не более 10
            risk_level=data.get("risk_level", "low"),
            requires_review=data.get("requires_review", False),
        )


class ResearcherAgent(BaseAgent):
    """
    Агент-исследователь: только read-only MCP-операции.

    Получает информацию из EDMS, не изменяя данных.
    Использует лёгкую модель для экономии токенов.
    """

    name = "researcher"
    allowed_tools = [
        "get_document",
        "search_documents",
        "get_document_history",
        "get_workflow_status",
        "get_analytics",
    ]
    model = MODEL_RESEARCHER

    @property
    def system_prompt(self) -> str:
        return """Ты — исследователь в системе документооборота EDMS. 
Твоя задача: собрать необходимую информацию из системы, используя только read-операции.

ПРАВИЛА:
1. Используй только доступные тебе инструменты (без write-операций)
2. Собери все данные необходимые для ответа на запрос
3. При нескольких документах — сначала ищи, потом читай конкретный
4. Отвечай на русском языке
5. Структурируй данные для передачи следующему агенту

Если данных недостаточно — чётко укажи что именно не найдено."""

    async def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Собрать информацию с помощью read-операций.

        Параметры:
            context: контекст задачи с запросом и найденными сущностями

        Возвращает:
            AgentResult с собранными данными в output['research_results']
        """
        start_ts = time.monotonic()
        query = context.get("query", "")
        entities = context.get("entities", {})

        user_message = f"""Запрос: {query}
Сущности: {json.dumps(entities, ensure_ascii=False)}
Шаг плана: {context.get('plan_step', {}).get('action', 'собери информацию')}

Собери необходимые данные из системы."""

        messages = [{"role": "user", "content": user_message}]
        tools = self._mcp_tools_to_anthropic(self.allowed_tools)
        tool_calls_log: list[dict[str, Any]] = []

        # ReAct цикл для исследователя
        max_iterations = 5
        for iteration in range(max_iterations):
            response = await self._call_llm(messages, tools=tools)

            has_tool_use = any(
                block.type == "tool_use"
                for block in response.content
                if hasattr(block, "type")
            )

            if not has_tool_use or response.stop_reason == "end_turn":
                reasoning = self._extract_reasoning(response)
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    output={"research_results": reasoning, "tool_calls": tool_calls_log},
                    reasoning=reasoning,
                    tool_calls=tool_calls_log,
                    model_used=self.model,
                    latency_ms=int((time.monotonic() - start_ts) * 1000),
                )

            # Выполняем tool calls
            assistant_blocks = [
                {"type": "text", "text": block.text}
                if hasattr(block, "text")
                else {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                for block in response.content
            ]
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_results = []
            for block in response.content:
                if not hasattr(block, "type") or block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input or {}

                log.info("ResearcherAgent calling: %s", tool_name)
                result = await self._mcp.call(tool_name, tool_input)

                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_input,
                    "result": result,
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            messages.append({"role": "user", "content": tool_results})

        reasoning = self._extract_reasoning(response)
        return AgentResult(
            agent_name=self.name,
            success=True,
            output={"research_results": reasoning, "tool_calls": tool_calls_log},
            reasoning=reasoning,
            tool_calls=tool_calls_log,
            model_used=self.model,
            latency_ms=int((time.monotonic() - start_ts) * 1000),
        )


class ExecutorAgent(BaseAgent):
    """
    Агент-исполнитель: только write MCP-операции.

    Выполняет изменения в системе. Использует среднюю модель
    для баланса качества и стоимости.
    """

    name = "executor"
    allowed_tools = [
        "create_document",
        "update_document_status",
        "assign_document",
    ]
    model = MODEL_EXECUTOR

    @property
    def system_prompt(self) -> str:
        return """Ты — исполнитель в системе документооборота EDMS.
Твоя задача: выполнить запрошенные изменения в системе.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Все write-операции необратимы — проверь параметры перед вызовом
2. При изменении статуса на 'rejected' или 'archived' — ОБЯЗАТЕЛЕН комментарий
3. Используй данные из research_results для формирования аргументов
4. Если данных недостаточно — НЕ выполняй операцию, сообщи об этом
5. Каждый вызов инструмента логируй с результатом

После выполнения операции чётко сообщи что было сделано."""

    async def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Выполнить запрошенные write-операции.

        Параметры:
            context: контекст с research_results и планом

        Возвращает:
            AgentResult с результатами выполненных операций
        """
        start_ts = time.monotonic()
        query = context.get("query", "")
        research = context.get("research_results", "")
        plan_step = context.get("plan_step", {})

        user_message = f"""Запрос: {query}
Собранная информация: {research}
Действие для выполнения: {plan_step.get('action', query)}
Параметры: {json.dumps(plan_step.get('args', {}), ensure_ascii=False)}

Выполни запрошенное действие."""

        messages = [{"role": "user", "content": user_message}]
        tools = self._mcp_tools_to_anthropic(self.allowed_tools)
        tool_calls_log: list[dict[str, Any]] = []

        max_iterations = 3
        last_response: anthropic.types.Message | None = None

        for iteration in range(max_iterations):
            response = await self._call_llm(messages, tools=tools)
            last_response = response

            has_tool_use = any(
                hasattr(block, "type") and block.type == "tool_use"
                for block in response.content
            )

            if not has_tool_use or response.stop_reason == "end_turn":
                break

            assistant_blocks = []
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_blocks.append({"type": "text", "text": block.text})
                elif hasattr(block, "type") and block.type == "tool_use":
                    assistant_blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_results = []
            for block in response.content:
                if not hasattr(block, "type") or block.type != "tool_use":
                    continue

                log.info("ExecutorAgent calling: %s", block.name)
                result = await self._mcp.call(block.name, block.input or {})
                tool_calls_log.append({
                    "tool": block.name,
                    "args": block.input,
                    "result": result,
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            messages.append({"role": "user", "content": tool_results})

        reasoning = self._extract_reasoning(last_response) if last_response else ""
        success = any(
            tc.get("result", {}).get("success", False) for tc in tool_calls_log
        ) if tool_calls_log else True

        return AgentResult(
            agent_name=self.name,
            success=success,
            output={"execution_results": reasoning, "tool_calls": tool_calls_log},
            reasoning=reasoning,
            tool_calls=tool_calls_log,
            model_used=self.model,
            latency_ms=int((time.monotonic() - start_ts) * 1000),
        )


class ReviewerAgent(BaseAgent):
    """
    Агент-ревьюер: проверяет результаты выполнения.

    Активируется при risk_level=high или деструктивных операциях.
    Использует тяжёлую модель для тщательного анализа.
    """

    name = "reviewer"
    allowed_tools = []  # Ревьюер не вызывает инструменты
    model = MODEL_REVIEWER

    @property
    def system_prompt(self) -> str:
        return """Ты — ревьюер результатов в системе документооборота EDMS.
Твоя задача: критически оценить выполненные действия и проверить их корректность.

ЧЕКЛИСТ ПРОВЕРКИ:
□ Все шаги плана выполнены?
□ Результаты соответствуют исходному запросу пользователя?
□ Нет побочных эффектов (лишних изменений)?
□ Данные целостны (ID валидны, статусы корректны)?
□ Пользователь получит понятный ответ?

ФОРМАТ ОТВЕТА — строго JSON:
{
  "approved": <bool>,
  "confidence": <float 0-1>,
  "issues": ["<описание проблемы>"],
  "suggestions": ["<рекомендация>"],
  "must_redo": [<step_ids>],
  "explanation": "<итоговое заключение на русском>"
}"""

    async def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Проверить результаты выполнения.

        Параметры:
            context: контекст с оригинальным запросом и результатами

        Возвращает:
            AgentResult с оценкой (approved/rejected) и замечаниями
        """
        start_ts = time.monotonic()
        original_query = context.get("query", "")
        plan = context.get("plan")
        results = context.get("execution_results", "")

        plan_desc = ""
        if plan and isinstance(plan, ExecutionPlan):
            plan_desc = "\n".join(
                f"Шаг {s.step}: {s.action} → {'✅' if s.executed and not s.failed else '❌'}"
                for s in plan.steps
            )

        user_message = f"""Оригинальный запрос: {original_query}

План выполнения:
{plan_desc}

Результаты:
{results}

Проверь корректность и верни JSON-оценку."""

        messages = [{"role": "user", "content": user_message}]

        try:
            response = await self._call_llm(messages, max_tokens=1024)
            reasoning = self._extract_reasoning(response)

            # Парсим JSON
            import re
            match = re.search(r"\{.*\}", reasoning, re.DOTALL)
            review_data: dict[str, Any] = {}
            if match:
                try:
                    review_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    review_data = {"approved": True, "confidence": 0.5, "explanation": reasoning}
            else:
                review_data = {"approved": True, "confidence": 0.7, "explanation": reasoning}

            return AgentResult(
                agent_name=self.name,
                success=True,
                output={"review": review_data},
                reasoning=reasoning,
                model_used=self.model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )

        except Exception as exc:
            log.error("ReviewerAgent error: %s", exc)
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={"review": {"approved": False, "explanation": str(exc)}},
                error=str(exc),
                model_used=self.model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )


class ExplainerAgent(BaseAgent):
    """
    Агент-объяснитель: формирует финальный ответ на русском.

    Всегда последний в цепочке. Переводит технические результаты
    в понятный пользователю текст.
    Использует лёгкую модель для экономии.
    """

    name = "explainer"
    allowed_tools = []
    model = MODEL_EXPLAINER

    @property
    def system_prompt(self) -> str:
        return """Ты — ИИ-ассистент корпоративной системы документооборота EDMS.
Твоя задача: сформировать понятный финальный ответ пользователю на русском языке.

СТРУКТУРА ОТВЕТА:
1. Краткий итог (1-2 предложения): что сделано/найдено
2. Детали (если необходимо): ключевые данные, числа, статусы
3. Следующие шаги (если применимо): что можно сделать дальше

ПРАВИЛА:
- Только русский язык (технические термины типа UUID допустимы)
- Не показывай технические детали ошибок — только понятное объяснение
- Структурируй длинные ответы маркированными списками
- Будь конкретен: называй документы, числа, статусы
- Если операция не выполнена — объясни причину и предложи альтернативу"""

    async def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Сформировать финальный ответ для пользователя.

        Параметры:
            context: полный контекст с результатами всех предыдущих агентов

        Возвращает:
            AgentResult с готовым ответом в output['final_response']
        """
        start_ts = time.monotonic()

        original_query = context.get("query", "")
        research_results = context.get("research_results", "")
        execution_results = context.get("execution_results", "")
        review = context.get("review", {})
        error_message = context.get("error_message", "")

        content_parts = [f"Исходный запрос пользователя: {original_query}\n"]

        if research_results:
            content_parts.append(f"Найденная информация:\n{research_results}\n")
        if execution_results:
            content_parts.append(f"Выполненные действия:\n{execution_results}\n")
        if review:
            explanation = review.get("explanation", "")
            approved = review.get("approved", True)
            if not approved:
                content_parts.append(f"Результат проверки: НЕ ОДОБРЕНО\n{explanation}\n")
        if error_message:
            content_parts.append(f"Произошла ошибка: {error_message}\n")

        content_parts.append("Сформируй финальный ответ пользователю.")
        user_message = "\n".join(content_parts)

        messages = [{"role": "user", "content": user_message}]

        try:
            response = await self._call_llm(messages, max_tokens=1024)
            final_text = self._extract_reasoning(response)

            return AgentResult(
                agent_name=self.name,
                success=True,
                output={"final_response": final_text},
                reasoning=final_text,
                model_used=self.model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )

        except Exception as exc:
            log.error("ExplainerAgent error: %s", exc)
            fallback = (
                "Произошла ошибка при формировании ответа. "
                "Пожалуйста, повторите запрос или обратитесь к администратору."
            )
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={"final_response": fallback},
                error=str(exc),
                model_used=self.model,
                latency_ms=int((time.monotonic() - start_ts) * 1000),
            )


# ---------------------------------------------------------------------------
# Координатор
# ---------------------------------------------------------------------------

class MultiAgentCoordinator:
    """
    Координатор мульти-агентной системы.

    Маршрутизирует запрос к нужному набору агентов в зависимости от сложности:
    - Простой (bypass_llm или 1 шаг): прямой вызов → Explainer
    - Средний (1-2 шага): Researcher/Executor → Explainer
    - Сложный (≥3 шагов или workflow): Planner → [Researcher|Executor] → Reviewer → Explainer
    - Деструктивные операции: всегда добавляет Reviewer
    """

    _WRITE_INTENTS = {"create_document", "update_status", "assign_document"}
    _COMPLEX_INTENTS = {"get_workflow_status", "get_analytics"}

    def __init__(self, mcp_url: str) -> None:
        self._mcp = MCPToolCaller(mcp_url)
        self._planner = PlannerAgent(self._mcp)
        self._researcher = ResearcherAgent(self._mcp)
        self._executor = ExecutorAgent(self._mcp)
        self._reviewer = ReviewerAgent(self._mcp)
        self._explainer = ExplainerAgent(self._mcp)

    async def route(
        self,
        nlu_result: NLUResult,
        context: dict[str, Any],
    ) -> AgentResult:
        """
        Маршрутизировать запрос к агентам.

        Параметры:
            nlu_result: результат NLU-анализа
            context: полный контекст (query, user_profile, session, few_shot...)

        Возвращает:
            Финальный AgentResult с ответом пользователю
        """
        is_write = nlu_result.intent in self._WRITE_INTENTS
        is_complex = (
            nlu_result.intent in self._COMPLEX_INTENTS
            or context.get("force_complex", False)
        )
        is_simple = (
            nlu_result.bypass_llm
            or (nlu_result.confidence > 0.85 and not is_write and not is_complex)
        )

        agent_context = {
            **context,
            "query": context.get("query", ""),
            "intent": nlu_result.intent,
            "entities": {
                "document_ids": nlu_result.entities.document_ids,
                "statuses": nlu_result.entities.statuses,
                "document_types": nlu_result.entities.document_types,
                "user_names": nlu_result.entities.user_names,
            },
        }

        log.info(
            "MultiAgentCoordinator.route: intent=%s confidence=%.2f simple=%s write=%s complex=%s",
            nlu_result.intent, nlu_result.confidence, is_simple, is_write, is_complex,
        )

        if is_simple:
            return await self._route_simple(nlu_result, agent_context)
        elif not is_write and not is_complex:
            return await self._route_medium_read(agent_context)
        elif is_write and not is_complex:
            return await self._route_medium_write(agent_context)
        else:
            return await self._route_complex(agent_context)

    async def _route_simple(
        self,
        nlu_result: NLUResult,
        context: dict[str, Any],
    ) -> AgentResult:
        """Простой маршрут: прямой вызов MCP → Explainer."""
        mcp_result: dict[str, Any] = {}

        if nlu_result.bypass_llm and nlu_result.required_tool:
            mcp_result = await self._mcp.call(
                nlu_result.required_tool,
                nlu_result.tool_args or {},
            )

        context["research_results"] = str(mcp_result.get("data", mcp_result))
        return await self._explainer.run(context)

    async def _route_medium_read(self, context: dict[str, Any]) -> AgentResult:
        """Средний маршрут (read): Researcher → Explainer."""
        research_result = await self._researcher.run(context)
        if not research_result.success:
            context["error_message"] = research_result.error
        else:
            context["research_results"] = research_result.output.get("research_results", "")
            context["tool_calls"] = research_result.tool_calls

        return await self._explainer.run(context)

    async def _route_medium_write(self, context: dict[str, Any]) -> AgentResult:
        """Средний маршрут (write): Researcher → Executor → Reviewer → Explainer."""
        # 1. Исследуем контекст
        research_result = await self._researcher.run(context)
        context["research_results"] = research_result.output.get("research_results", "")

        # 2. Выполняем операции
        exec_result = await self._executor.run(context)
        context["execution_results"] = exec_result.output.get("execution_results", "")

        # 3. Ревью (всегда для write)
        review_result = await self._reviewer.run(context)
        context["review"] = review_result.output.get("review", {})

        return await self._explainer.run(context)

    async def _route_complex(self, context: dict[str, Any]) -> AgentResult:
        """
        Сложный маршрут: Planner → [Researcher|Executor] → Reviewer → Explainer.
        """
        # 1. Планируем
        plan_result = await self._planner.run(context)
        if not plan_result.success:
            context["error_message"] = f"Планировщик не смог построить план: {plan_result.error}"
            return await self._explainer.run(context)

        plan: ExecutionPlan = plan_result.output["plan"]
        context["plan"] = plan

        all_research = []
        all_execution = []

        # 2. Выполняем шаги плана
        for step in plan.steps:
            # Проверяем зависимости
            deps_ok = all(
                any(s.step == dep and s.executed and not s.failed for s in plan.steps)
                for dep in step.depends_on
            )
            if step.depends_on and not deps_ok:
                log.warning("Skipping step %d: dependencies not met", step.step)
                step.failed = True
                continue

            step_context = {
                **context,
                "plan_step": {"action": step.action, "tool": step.tool, "args": step.args},
            }

            is_read_tool = step.tool in ResearcherAgent.allowed_tools
            is_write_tool = step.tool in ExecutorAgent.allowed_tools

            if is_read_tool:
                result = await self._researcher.run(step_context)
                if result.success:
                    all_research.append(result.output.get("research_results", ""))
                    step.executed = True
                else:
                    step.failed = True
                    if not step.can_fail:
                        context["error_message"] = f"Шаг {step.step} завершился с ошибкой"
                        break

            elif is_write_tool:
                result = await self._executor.run(step_context)
                if result.success:
                    all_execution.append(result.output.get("execution_results", ""))
                    step.executed = True
                else:
                    step.failed = True
                    if not step.can_fail:
                        context["error_message"] = f"Шаг {step.step} (запись) завершился с ошибкой"
                        break

        context["research_results"] = "\n---\n".join(all_research)
        context["execution_results"] = "\n---\n".join(all_execution)

        # 3. Ревью (если требуется)
        if plan.requires_review:
            review_result = await self._reviewer.run(context)
            context["review"] = review_result.output.get("review", {})

        # 4. Финальный ответ
        return await self._explainer.run(context)
