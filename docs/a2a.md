# Agent-to-agent (A2A)

synapse implements the **Agent2Agent protocol v0.3.0** (JSON-RPC 2.0), so it
interoperates with any A2A-compliant agent — not just other synapse agents.

## Serve an agent to the A2A ecosystem

```python
from synapse.a2a import create_a2a_app
app = create_a2a_app(agent)        # ASGI app
```

```bash
uvicorn yourmodule:app             # pip install "synapse[server]"
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/.well-known/agent-card.json` | Agent Card (discovery; legacy `/agent.json` also served) |
| `POST` | `/` | JSON-RPC 2.0 |

JSON-RPC methods: `message/send`, `message/stream` (SSE), `tasks/get`,
`tasks/cancel`, `tasks/resubscribe`, `tasks/pushNotificationConfig/set`+`get`.

Tasks carry the real lifecycle (`submitted → working → completed / canceled /
failed / …`), with `artifacts`, optional `history`, and `metadata` (token usage).

## Consume any A2A agent

```python
from synapse.a2a import A2AClient

client = A2AClient("https://some-a2a-agent.example.com")

client.card().name                       # discovery (any A2A agent)
client.ask("summarize the news")         # message/send → final text
task = client.send_message("...")        # full Task dict
client.get_task(task_id); client.cancel_task(task_id)

async for ev in client.astream_message("..."):   # message/stream (SSE)
    ...   # Task, status-update, artifact-update objects
```

## Cross-ecosystem delegation

Wrap a remote, standards-compliant agent as a local synapse tool — so a synapse
agent can delegate across vendors:

```python
coordinator = Agent("boss", tools=[A2AClient(url).as_tool()])
coordinator.run("delegate the research")
```

## Push notifications (signed + retrying)

Register a webhook; the agent POSTs the terminal task to it.

```python
# client side: register on a task
client.send_message("long job")  # then set a push config for that task id via
# tasks/pushNotificationConfig/set with {"url": ..., "token": "shared-secret"}
```

Delivery is **HMAC-signed** and **retried** with backoff. The receiver verifies:

```python
from synapse.a2a import push   # synapse.a2a.push

# in your webhook handler, with the raw request body + headers:
ok = push.verify_signature(
    body,
    secret="shared-secret",
    timestamp=headers["X-A2A-Timestamp"],
    signature=headers["X-A2A-Signature"],
)
```

Headers sent: `X-A2A-Signature` (`sha256=…` HMAC over `timestamp + "." + body`),
`X-A2A-Timestamp`, `X-A2A-Notification-Token`.

> Limitation: delivery is in-process (no durable outbox across restarts yet).

## Task persistence

By default tasks live in memory (lost on restart). Pass a `task_store` to
persist them — `FileTaskStore` survives restarts, or implement `TaskStore`:

```python
from synapse.a2a import A2ADispatcher, FileTaskStore
dispatcher = A2ADispatcher(agent, task_store=FileTaskStore("tasks"))
```

`create_a2a_app(agent)` uses an in-memory store; build the app around your own
dispatcher to swap in persistence.

## synapse-native convenience layer

For quick local/dev use there's also a lightweight `/run` + `/run/stream`
server. This is **not** the A2A wire format — it's a synapse convenience:

```python
from synapse.a2a import serve, RemoteAgent
serve(agent, port=8080)                       # stdlib, zero-dep
RemoteAgent("http://127.0.0.1:8080").run("hi")
```

When in doubt, use the A2A-compliant layer (`create_a2a_app` / `A2AClient`) for
interop; the native layer is just a shortcut.
