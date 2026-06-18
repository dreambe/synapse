# Enterprise knowledge & human escalation

Make an agent *organizationally aware*: it grounds a one-line requirement in your
enterprise knowledge graph (which repo, frontend/backend, the red lines), queries
that knowledge mid-task when something is unclear, and escalates to a person when
it's stuck. The knowledge graph itself is assumed to exist — most likely behind
an **MCP** interface.

## 1. The knowledge graph is MCP tools

Expose KG capabilities as MCP tools (`locate_repository`, `get_redlines`,
`business_knowledge`, …) and adapt them into synapse tools:

```python
from synapse.mcp import tools_from_session
kg_tools = await tools_from_session(kg_mcp_session)   # your MCP client session
agent = Agent("dev", instructions="You implement requirements.", model=model, tools=kg_tools)
```

Write the tool descriptions prescriptively so the agent consults the KG
*during* execution when it's unsure — e.g. *"Call `get_redlines(area)` before
editing code in any area; call `business_knowledge(q)` whenever a requirement is
ambiguous."* This is the **mid-task** path: the model decides to ask.

## 2. Ground the plan before work starts

Don't rely only on the model deciding to look things up — for facts it must
always have (target repo, scope, red lines), use a **preflight grounding**
provider. It runs once before the loop and injects the result as business
context, so the agent has it while *decomposing* the task:

```python
async def ground(requirement: str) -> str:
    # query the KG MCP (or any RAG source) keyed on the requirement
    loc = await kg.locate_repository(requirement)
    redlines = await kg.get_redlines(loc["area"])
    return f"Repo: {loc['repo']}. Scope: {loc['scope']}. Red lines: {redlines}"

result = await agent.arun("add a checkout discount", grounding=ground)
```

The returned text appears in the system prompt under *"Business context (from
the knowledge graph)"*. Combine the two: grounding pins the must-have facts;
the KG tools cover ad-hoc questions that come up later.

## 3. Escalate to a human when stuck

Give the agent an `ask_human` tool backed by a `HumanChannel`. When it hits an
ambiguous requirement, missing access, or a red line, it asks a person and the
reply flows back into the run:

```python
from synapse import human_tool, CallbackChannel

async def feishu_ask(question: str, context: str | None) -> str:
    msg_id = await feishu.post_to_oncall(question)     # your Feishu/Lark bot
    return await feishu.await_reply(msg_id)            # block until a teammate replies

agent = Agent(
    "dev",
    instructions=(
        "Implement the requirement within the business context and red lines. "
        "If a requirement is ambiguous, you lack access, or you would cross a red "
        "line, call ask_human instead of guessing."
    ),
    model=model,
    tools=[*kg_tools, human_tool(CallbackChannel(feishu_ask))],
    grounding=ground,
)
```

`CallbackChannel` adapts any `(question, context) -> reply` function — the Feishu
glue (post to a bot, wait for a thread reply, return it) lives in your code, not
the framework. Implement `HumanChannel` directly for a custom transport.

## Putting it together

```
requirement ──▶ grounding(ground)  ─── KG MCP ──▶ "Repo X, backend, red lines…"
                     │ injected as business context
                     ▼
                run loop ──▶ KG tools (locate/redlines/business_knowledge)  ← mid-task questions
                     │
                     └─▶ ask_human ──▶ Feishu ──▶ teammate's reply ──▶ back into the run
```

See [`examples/enterprise_agent.py`](../examples/enterprise_agent.py) for an
offline end-to-end version (fake KG + a callback "human").

> The agent is only as grounded as the KG it queries and only as safe as the red
> lines it's told. Grounding and escalation are *mechanisms*; the quality of the
> business knowledge and the tool descriptions is what makes the agent smart.

## Multi-tenant isolation (sessions & memory)

When one agent serves many users/tenants, two things must not leak across them:

**Conversation history** — give each (tenant/user, conversation) its **own**
`Session`; never share one. The per-session lock and history are then naturally
isolated. Record the owner on `Session(scope=...)` for routing/audit. Over A2A,
key sessions by an unguessable, tenant-scoped `session_id`.

**Memory** — a single shared `Memory` would let one user's `recall` surface
another's facts. Use a `MemoryNamespace` to hand out an isolated store per
scope:

```python
from synapse import InMemoryNamespace, FileNamespace, VectorNamespace, HashingEmbedder

ns = FileNamespace("memory")                       # or InMemory / Vector namespace
# build the agent (or just its memory) per user — scopes are disjoint:
agent = Agent("assistant", memory=ns.scope(user_id), tools=[...])
```

`ns.scope(key)` returns the same store for the same key and a disjoint store for
different keys — so `ns.scope("alice")` can never recall `ns.scope("bob")`'s
facts. `FileNamespace` sanitizes the key into a flat per-scope file (no path
traversal). For semantic memory, `VectorNamespace(embedder)` partitions the
vector index per scope.

> Isolation is structural: distinct `Session` objects and distinct memory scopes.
> The framework gives you the partitions; deriving the scope key from your authn
> (tenant/user id) is your call — don't derive it from anything user-controllable.
