"""Operating an agent safely at scale: permissions, governance, durability.

Four mechanisms that turn a capable agent into one you can run as a service:

1. **Permission modes** — declare a run read-only (plan), ask-on-write, or auto.
2. **Governance** — per-tenant rate limit + quota + concurrency admission.
3. **Durable execution** — resume a crashed run without re-firing side effects.
4. **Compaction** — keep a long run inside the context window.

    python examples/governance_and_safety.py
"""

from __future__ import annotations

import asyncio

from synapse import (
    Agent,
    Governor,
    PermissionMode,
    RateLimited,
    ScriptedModel,
    permission_policy,
    tool,
)


@tool(side_effect="read")
def read_db(table: str) -> str:
    "Read-only query."
    return f"rows from {table}"


@tool  # default side_effect="write"
def drop_table(table: str) -> str:
    "Destructive!"
    return f"dropped {table}"


async def main() -> None:
    # 1) Permission mode: PLAN is read-only — the destructive tool is denied.
    agent = Agent(
        "dba",
        tools=[read_db, drop_table],
        model=ScriptedModel(
            [[("read_db", {"table": "users"}), ("drop_table", {"table": "users"})], "report ready"]
        ),
    )
    res = await agent.arun("investigate", permissions=permission_policy(PermissionMode.PLAN))
    from synapse import ToolResultBlock

    for m in res.messages:
        if isinstance(m.content, list):
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    print("  tool:", b.content)

    # 2) Governance: per-tenant admission control (burst of 1, then rate-limited).
    print("\n=== governance ===")
    gov = Governor(rate=1.0, burst=1.0, quota=100)
    async with gov.admit("tenant-A", block=False):
        print("  tenant-A: admitted")
    try:
        async with gov.admit("tenant-A", block=False):
            print("  tenant-A: admitted again")
    except RateLimited:
        print("  tenant-A: rate-limited (burst spent)")
    async with gov.admit("tenant-B", block=False):
        print("  tenant-B: admitted (isolated from A)")

    # 3) Durable execution and 4) compaction are wired via run kwargs:
    #    agent.arun(task, run_id="r1", journal=..., checkpointer=...)  # resumable
    #    agent.arun(task, compactor=Compactor(model, trigger_tokens=50_000))
    print("\nsee CHANGELOG / docs for durable resume + compaction wiring.")


if __name__ == "__main__":
    asyncio.run(main())
