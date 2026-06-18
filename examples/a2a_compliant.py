"""Serve a synapse agent over the A2A protocol (v0.3.0), and call it.

Serve (needs uvicorn — `pip install synapse[server]`):

    uvicorn examples.a2a_compliant:app

Then, from anywhere with an A2A-compliant client (including this one):

    python -c "from synapse.a2a import A2AClient; \
print(A2AClient('http://127.0.0.1:8000').ask('hello'))"

The agent card is at http://127.0.0.1:8000/.well-known/agent-card.json
"""

from __future__ import annotations

from synapse import Agent, EchoModel
from synapse.a2a import create_a2a_app

agent = Agent(
    "echo-a2a",
    description="A trivial A2A-compliant agent that echoes its input.",
    model=EchoModel(),
)

# ASGI application — point uvicorn at `examples.a2a_compliant:app`.
app = create_a2a_app(agent)
