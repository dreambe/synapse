"""Mid-run steering: correct or stop an agent while it is running.

Run the agent in one task and steer it from another — inject guidance that the
agent picks up at its next turn, or ask it to stop gracefully. Steering never
interrupts a turn mid-flight; it applies at the turn boundary.

    python examples/steering_agent.py
"""

from __future__ import annotations

import asyncio

from synapse import Agent, ScriptedModel, Steer, Steered, tool


async def main() -> None:
    steer = Steer()
    started = asyncio.Event()

    @tool
    async def research(topic: str) -> str:
        "Pretend to research a topic."
        started.set()
        await asyncio.sleep(0.05)
        return f"notes on {topic}"

    # Offline script: research, then answer. A real model would react to the
    # injected "[steering] ..." turn; here we just show it entering the transcript.
    agent = Agent(
        "researcher",
        instructions="Research the request, then summarize.",
        tools=[research],
        model=ScriptedModel([[("research", {"topic": "agents"})], "Summary: agents compose."]),
    )

    async def supervise() -> None:
        await started.wait()
        steer.send("also cover failure modes")  # nudge the next turn

    # Stream the run so we can see the Steered event as it lands.
    async def run() -> None:
        async for ev in agent.astream("research agents", steer=steer):
            if isinstance(ev, Steered):
                print(f"[supervisor steered]: {ev.text}")

    await asyncio.gather(run(), supervise())

    # A graceful stop (if we had wanted to cut it short):
    #   steer.stop("that's enough")  ->  result.stop_reason == "steered_stop"
    print("done.")


if __name__ == "__main__":
    asyncio.run(main())
