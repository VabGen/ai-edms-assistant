# src/ai_edms_assistant/interfaces/cli/commands.py
"""
CLI interactive REPL for EDMS AI Assistant with Rich UI and History support.

Usage:
    edms-cli --token <JWT> [--doc <UUID>]
    # Or set env var: export EDMS_JWT_TOKEN="..." && edms-cli
"""

from __future__ import annotations

import asyncio
import argparse
import os
import sys
from typing import List, Dict, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel

try:
    from ...application.agents import EdmsDocumentAgent
    from ...application.dto import AgentRequest
    from ...shared.logging.agent_tracer import AgentTracer
except ImportError as e:
    # Фоллбэк для запуска вне пакета, если возникнут проблемы с относительными импортами
    print(f"Import Error: {e}")
    print("Убедитесь, что запускаете команду из корня проекта как модуль:")
    print("python -m src.ai_edms_assistant.interfaces.cli.commands ...")
    sys.exit(1)

console = Console()


def _extract_content_from_result(result: dict[str, Any]) -> str:
    """
    Универсальная функция извлечения текста из ответа агента.
    Пытается найти текст в различных возможных полях и форматах.
    """
    # 1. Прямые поля
    if "content" in result and result["content"]:
        return str(result["content"])
    if "message" in result and result["message"]:
        return str(result["message"])
    if "output" in result and result["output"]:
        return str(result["output"])
    if "response" in result and result["response"]:
        return str(result["response"])

    messages = result.get("messages", [])
    if isinstance(messages, list) and messages:
        last_msg = messages[-1]

        if hasattr(last_msg, "content"):
            content = last_msg.content
            if content:
                return str(content)

        if isinstance(last_msg, dict):
            content = last_msg.get("content") or last_msg.get("text")
            if content:
                return str(content)

    if "final_answer" in result:
        val = result["final_answer"]
        if isinstance(val, dict) and "content" in val:
            return str(val["content"])
        return str(val)

    if "error" in result:
        return f"⚠️ Ошибка агента: {result['error']}"

    if "tool_calls" in result or "intermediate_steps" in result:
        return "_Агент выполнил действия, но не сформулировал итоговый ответ._"

    return ""


async def _chat_loop(token: str, context_id: str | None = None) -> None:
    """Interactive async chat loop with conversation history."""

    try:
        agent = EdmsDocumentAgent()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize Agent:[/bold red] {e}")
        return

    thread_id = f"cli_session_{os.getpid()}"
    tracer = AgentTracer(thread_id=thread_id)

    chat_history: List[Dict[str, str]] = []

    console.print(Panel.fit(
        "[bold blue]EDMS AI Assistant (CLI)[/bold blue]\n"
        f"Thread ID: [dim]{thread_id}[/dim]\n"
        f"Context Doc: [dim]{context_id or 'Global'}[/dim]\n"
        "Commands: [yellow]exit[/yellow], [yellow]clear[/yellow] (очистить историю)\n"
        "Type your message below:",
        title="System Ready",
        border_style="blue"
    ))
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")

            if not user_input:
                continue

            lower_input = user_input.lower().strip()

            # Команды управления
            if lower_input in ("exit", "quit", "q", "/exit"):
                console.print("[yellow]Exiting...[/yellow]")
                break

            if lower_input == "clear":
                chat_history.clear()
                console.print("[dim]History cleared.[/dim]\n")
                continue

            # Формирование запроса
            request = AgentRequest(
                message=user_input,
                user_token=token,
                context_ui_id=context_id,
                thread_id=thread_id,
                user_context={},
                chat_history=chat_history
            )

            # Индикатор загрузки
            with Live(Spinner("dots", text="Agent is thinking...", style="cyan"),
                      refresh_per_second=10, transient=True) as live:

                try:
                    result = await agent.chat(request)
                except Exception as e:
                    live.stop()
                    console.print(f"[bold red]Error during agent execution:[/bold red] {e}")
                    tracer.error("CLI Chat Error", exc=e)
                    continue

            # console.print(f"[dim]Raw Result: {result}[/dim]")

            status = result.get("status", "unknown")

            content = _extract_content_from_result(result)

            # Фоллбэк, если совсем пусто
            if not content:
                content = "_Пустой ответ от агента. Проверьте логи сервера._"
                if not status or status == "unknown":
                    status = "empty_response"

            # Обновляем историю только если есть осмысленный контент
            if content and not content.startswith("_") and "Ошибка агента" not in content:
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": content})
            elif content.startswith("_") or "Ошибка" in content:
                # Логируем проблему, но не добавляем мусор в историю контекста
                tracer.error("Bad agent response", message=content)

            # Формирование заголовка
            status_emoji = "✅" if status == "ok" else "⚠️"
            header = f"{status_emoji} Agent [{status}]"

            # Рендеринг Markdown
            md_content = Markdown(content)
            border_color = "green" if status == "ok" else "red"

            console.print(Panel(md_content, title=header, border_style=border_color))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user. Type 'exit' to quit.[/yellow]")
        except EOFError:
            break
        except Exception as e:
            console.print(f"[bold red]Critical Error:[/bold red] {e}")
            tracer.error("Critical CLI Loop Error", exc=e)
            break

    console.print("[dim]Session ended.[/dim]")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EDMS AI Assistant — Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  edms-cli --token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
  edms-cli --doc 550e8400-e29b-41d4-a716-446655440000
        """
    )

    parser.add_argument(
        "--token",
        default=os.getenv("EDMS_JWT_TOKEN"),
        help="JWT bearer token (or set EDMS_JWT_TOKEN env var)"
    )

    parser.add_argument(
        "--doc",
        default=None,
        metavar="UUID",
        help="UUID документа для фокусировки контекста"
    )

    args = parser.parse_args()

    if not args.token:
        console.print("[bold red]Error:[/bold red] JWT Token is required.")
        console.print("Provide via --token argument or EDMS_JWT_TOKEN environment variable.")
        sys.exit(1)

    try:
        asyncio.run(_chat_loop(args.token, args.doc))
    except KeyboardInterrupt:
        console.print("\nGoodbye!")
    except Exception as e:
        console.print(f"[bold red]Fatal Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# python -m src.ai_edms_assistant.interfaces.cli.commands --token "435110ed-fc1e-11f0-8e63-d2c3acf9bc01"