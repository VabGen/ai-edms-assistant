# src/ai_edms_assistant/interfaces/cli/commands.py
"""
CLI interactive REPL for EDMS AI Assistant.
"""

from __future__ import annotations

import asyncio
import argparse


async def _chat_loop(token: str, context_id: str | None = None) -> None:
    """Interactive async chat loop against EdmsDocumentAgent.

    Args:
        token:      JWT bearer token for EDMS API calls.
        context_id: Optional document UUID to set as UI context.
    """
    from ...application.agents import EdmsDocumentAgent
    from ...application.dto import AgentRequest

    agent = EdmsDocumentAgent()
    thread_id = f"cli_{id(agent)}"
    print("EDMS AI Assistant (CLI). Введите 'exit' для выхода.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            break

        if not user_input:
            continue

        request = AgentRequest(
            message=user_input,
            user_token=token,
            context_ui_id=context_id,
            thread_id=thread_id,
            user_context={},
        )

        result = await agent.chat(request)
        status = result.get("status", "ok")
        content = result.get("content") or result.get("message", "")
        print(f"Agent [{status}]: {content}\n")


def main() -> None:
    """CLI entry point registered in pyproject.toml scripts."""
    parser = argparse.ArgumentParser(description="EDMS AI Assistant — CLI режим")
    parser.add_argument("--token", required=True, help="JWT bearer токен")
    parser.add_argument(
        "--doc", default=None, metavar="UUID", help="UUID документа (контекст)"
    )
    args = parser.parse_args()
    asyncio.run(_chat_loop(args.token, args.doc))


if __name__ == "__main__":
    main()
