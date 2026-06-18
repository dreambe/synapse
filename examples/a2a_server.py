"""Serve an agent over HTTP, then call it from another process.

Terminal 1 — serve:

    python examples/a2a_server.py
    # or: synapse serve examples/a2a_server.py:agent --port 8080

Terminal 2 — call it:

    synapse card http://127.0.0.1:8080
    python -c "from synapse.a2a import RemoteAgent; \
print(RemoteAgent('http://127.0.0.1:8080').run('hello').output)"
"""

from __future__ import annotations

from synapse import Agent, EchoModel
from synapse.a2a import serve

agent = Agent(
    "echo-service",
    instructions="Echo whatever you receive.",
    description="A trivial agent that echoes its input. Useful for testing A2A.",
    model=EchoModel(),
)


if __name__ == "__main__":
    serve(agent, port=8080)
