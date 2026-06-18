"""v0.2 capabilities in one place: hooks, memory, guardrails, approval,
verification, and budgets — all offline with the scripted backend.

    python examples/advanced_agent.py
"""

from __future__ import annotations

from synapse import (
    Agent,
    CollectingHooks,
    InMemoryMemory,
    ScriptedModel,
    Verdict,
    block_keywords,
    redact,
    tool,
)


@tool(requires_approval=True)
def deploy(service: str) -> str:
    """Deploy a service to production (gated by approval)."""
    return f"deployed {service}"


def main() -> None:
    hooks = CollectingHooks()
    memory = InMemoryMemory()

    agent = Agent(
        "ops",
        instructions="You operate services carefully.",
        model=ScriptedModel(
            [
                [("remember", {"fact": "prod deploys need approval"})],
                [("deploy", {"service": "api"})],
                "Deploy complete. password is hunter2.",
            ]
        ),
        tools=[deploy],
        memory=memory,
    )

    # Verifier: require the final answer to mention "complete".
    def verify(output: str) -> Verdict:
        return Verdict(passed="complete" in output.lower(), feedback="say when complete")

    result = agent.run(
        "Deploy the api service.",
        hooks=hooks,
        approval=lambda name, inp: True,  # auto-approve in this demo
        verify=verify,
        input_guardrails=[block_keywords(["rm -rf"])],
        output_guardrails=[redact(r"hunter2")],
    )

    print("output     :", result.output)
    print("stop_reason:", result.stop_reason)
    print("remembered :", " | ".join(__import__("asyncio").run(memory.all())))
    print("events     :", [e[0] for e in hooks.events])


if __name__ == "__main__":
    main()
