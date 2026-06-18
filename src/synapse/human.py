"""Human-in-the-loop *assistance*: let an agent ask a person when it's stuck.

Distinct from tool *approval* (gating a tool call): this is escalation — when the
agent hits an ambiguous requirement, a missing permission, or a policy red line,
it calls the ``ask_human`` tool, a person answers (e.g. via a Feishu/Lark bot),
and the answer flows back into the run.

The transport is pluggable via :class:`HumanChannel`. ``CallbackChannel`` wraps
any function (sync or async), so the Feishu glue — post the question, wait for a
reply in the thread, return it — lives in your code, not the framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional, Union

from ._util import maybe_await
from .tool import Tool


class HumanChannel(ABC):
    """A way to reach a human and (optionally) get an answer back."""

    @abstractmethod
    async def ask(self, question: str, *, context: Optional[str] = None) -> str:
        """Pose a question to a human and return their reply."""

    async def notify(self, message: str) -> None:
        """Fire-and-forget notification (default: route through ``ask``)."""
        await self.ask(message)


class CallbackChannel(HumanChannel):
    """Adapt any callable ``(question, context) -> reply`` into a channel.

    Example (Feishu): post the question to a bot, block until a teammate replies
    in the thread, return that reply::

        channel = CallbackChannel(lambda q, ctx: feishu_ask_and_wait(q))
    """

    def __init__(self, fn: Callable[[str, Optional[str]], Union[str, Awaitable[str]]]) -> None:
        self._fn = fn

    async def ask(self, question: str, *, context: Optional[str] = None) -> str:
        return await maybe_await(self._fn(question, context))


def human_tool(
    channel: HumanChannel,
    *,
    name: str = "ask_human",
    description: Optional[str] = None,
) -> Tool:
    """Build an ``ask_human`` tool backed by ``channel``.

    Give the agent this tool and instruct it to call it when blocked — ambiguous
    requirements, missing access, or a red line it must not cross.
    """

    async def ask_human(question: str) -> str:
        return await channel.ask(question)

    return Tool(
        name=name,
        description=(
            description
            or "Ask a human for help when you are blocked: an ambiguous requirement, "
            "missing access or permissions, or a policy red line you must not cross. "
            "Returns the human's reply. Prefer this over guessing on irreversible work."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A specific question, with enough context for a human to answer.",
                }
            },
            "required": ["question"],
        },
        func=ask_human,
    )
