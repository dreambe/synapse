"""Context engineering: keep long runs inside the window.

- :class:`Compactor` summarizes old turns once history grows past a threshold,
  replacing them with a single recap so the loop can keep going.
- :func:`select_tools` ranks a large tool set against a query so only the
  relevant schemas need to be put in front of the model (the building block for
  tool search).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .messages import Message, TextBlock

if TYPE_CHECKING:
    from .models.base import Model
    from .tool import Tool


class Compactor:
    """Summarizes earlier conversation when it grows too long.

    Triggers once the message count exceeds ``trigger_messages``; collapses
    everything except the last ``keep_recent`` messages into one recap turn.
    """

    def __init__(
        self,
        model: "Model",
        *,
        trigger_messages: int = 24,
        keep_recent: int = 8,
    ) -> None:
        self.model = model
        self.trigger_messages = trigger_messages
        self.keep_recent = keep_recent

    async def maybe_compact(self, messages: list[Message]) -> list[Message]:
        if len(messages) <= self.trigger_messages:
            return messages

        head = messages[: -self.keep_recent]
        tail = messages[-self.keep_recent :]

        transcript = "\n".join(f"{m.role}: {m.text}" for m in head if m.text)
        prompt = (
            "Summarize the following conversation so it can stand in for the "
            "original turns. Preserve decisions, facts, and open threads.\n\n"
            + transcript
        )
        response = await self.model.generate(
            system="You compress conversation history faithfully and concisely.",
            messages=[Message(role="user", content=prompt)],
            tools=[],
        )
        recap = Message(
            role="user",
            content=[TextBlock("[Summary of earlier conversation]\n" + response.message.text)],
        )
        return [recap, *tail]


def _score(query: str, tool: "Tool") -> int:
    terms = {w for w in query.lower().split() if w}
    haystack = f"{tool.name} {tool.description}".lower()
    return sum(1 for w in terms if w in haystack)


def select_tools(query: str, tools: list["Tool"], k: int = 5) -> list["Tool"]:
    """Return up to ``k`` tools most relevant to ``query`` (keyword overlap)."""
    scored = sorted(tools, key=lambda t: _score(query, t), reverse=True)
    relevant = [t for t in scored if _score(query, t) > 0]
    return (relevant or scored)[:k]
