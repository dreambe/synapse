"""Context engineering: keep long runs inside the window.

- :class:`Compactor` summarizes old turns once history grows past a threshold,
  replacing them with a single recap so the loop can keep going.
- :func:`select_tools` ranks a large tool set against a query so only the
  relevant schemas need to be put in front of the model (the building block for
  tool search).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .messages import Message, TextBlock

if TYPE_CHECKING:
    from .models.base import Model
    from .tool import Tool


def _estimate_tokens(messages: list[Message]) -> int:
    """A cheap, dependency-free token estimate (~4 chars/token)."""
    return sum(len(m.text) for m in messages if m.text) // 4


class Compactor:
    """Summarizes earlier conversation when it grows too long.

    Triggers when the message count exceeds ``trigger_messages`` *or* (if set)
    the estimated token count exceeds ``trigger_tokens`` — so a few very large
    turns compact as readily as many small ones. The original task (the first
    user turn) is **preserved verbatim**, the middle is summarized into one
    structured recap (decisions / facts / open threads / artifacts), and the last
    ``keep_recent`` turns are kept verbatim.

    Still lossy by nature: a summary is not the transcript. Keep ``keep_recent``
    generous for tasks where recent tool detail matters.
    """

    def __init__(
        self,
        model: "Model",
        *,
        trigger_messages: int = 24,
        trigger_tokens: Optional[int] = None,
        keep_recent: int = 8,
    ) -> None:
        self.model = model
        self.trigger_messages = trigger_messages
        self.trigger_tokens = trigger_tokens
        self.keep_recent = keep_recent

    def _should_compact(self, messages: list[Message]) -> bool:
        if len(messages) > self.trigger_messages:
            return True
        if self.trigger_tokens is not None and _estimate_tokens(messages) > self.trigger_tokens:
            return True
        return False

    async def maybe_compact(self, messages: list[Message]) -> list[Message]:
        # Need room for: the preserved first turn + a recap + the kept tail.
        if not self._should_compact(messages) or len(messages) <= self.keep_recent + 2:
            return messages

        first = messages[0]  # the original task — never summarized away
        middle = messages[1 : -self.keep_recent]
        tail = messages[-self.keep_recent :]
        if not middle:
            return messages

        transcript = "\n".join(f"{m.role}: {m.text}" for m in middle if m.text)
        prompt = (
            "Compress the conversation excerpt below into a faithful recap that "
            "can stand in for those turns. Use these sections, omitting any that "
            "are empty:\n"
            "- Decisions: choices made and why\n"
            "- Facts: concrete findings, values, file paths, identifiers\n"
            "- Open threads: what is still pending or unresolved\n"
            "- Artifacts: results produced or offloaded (with references)\n\n"
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
        return [first, recap, *tail]


def _score(query: str, tool: "Tool") -> int:
    terms = {w for w in query.lower().split() if w}
    haystack = f"{tool.name} {tool.description}".lower()
    return sum(1 for w in terms if w in haystack)


def select_tools(query: str, tools: list["Tool"], k: int = 5) -> list["Tool"]:
    """Return up to ``k`` tools most relevant to ``query``.

    Experimental / first cut: ranking is keyword overlap. Replace with an
    embedding or LLM ranker for large or nuanced tool sets.
    """
    scored = sorted(tools, key=lambda t: _score(query, t), reverse=True)
    relevant = [t for t in scored if _score(query, t) > 0]
    return (relevant or scored)[:k]
