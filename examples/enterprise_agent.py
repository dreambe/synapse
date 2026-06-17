"""Enterprise-aware agent: KG grounding + mid-task KG tool + human escalation.

Offline end-to-end (fake knowledge graph + a callback 'human', scripted model).

    python examples/enterprise_agent.py
"""

from __future__ import annotations

import asyncio

from synapse import Agent, CallbackChannel, ScriptedModel, human_tool, tool

# --- a fake "knowledge graph" (stand-in for your MCP-backed KG) -------------

_KG = {
    "checkout": {"repo": "payments-svc", "scope": "backend", "redlines": "never log PII"},
}


def ground(requirement: str) -> str:
    for key, facts in _KG.items():
        if key in requirement:
            return f"Repo: {facts['repo']}. Scope: {facts['scope']}. Red line: {facts['redlines']}."
    return "No matching project found — ask a human which repo to use."


@tool
def business_knowledge(question: str) -> str:
    """Look up business knowledge from the enterprise knowledge graph."""
    return "Discounts are capped at 30% per company policy."


async def main() -> None:
    # The 'human' channel — in production this posts to a Feishu bot and waits.
    def human(question: str, context: str | None) -> str:
        return "Yes, proceed — cap the discount at 30% as policy says."

    agent = Agent(
        "dev",
        instructions=(
            "Implement the requirement within the business context and red lines. "
            "Use business_knowledge when unsure; call ask_human if you'd cross a red line."
        ),
        model=ScriptedModel(
            [
                [("business_knowledge", {"question": "max discount?"})],
                [("ask_human", {"question": "30% discount ok for checkout?"})],
                "Implemented in payments-svc backend: discount capped at 30%, no PII logged.",
            ]
        ),
        tools=[business_knowledge, human_tool(CallbackChannel(human))],
    )

    result = await agent.arun("add a checkout discount", grounding=ground)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
