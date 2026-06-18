"""Steering: correct or interrupt a run while it is in flight.

A long agentic run shouldn't be a black box you can only kill and restart. A
:class:`Steer` channel lets a supervisor — a human, or another agent — inject
guidance into a *running* agent, or ask it to stop. Steering takes effect at the
next **turn boundary**, so the loop stays consistent: it never mutates a turn
mid-flight (a tool batch already in progress finishes first).

    steer = Steer()
    task = asyncio.create_task(agent.arun("big task", steer=steer))
    ...
    steer.send("focus on the API layer first")   # nudges the next model turn
    ...
    steer.stop()                                  # graceful stop at next boundary
    result = await task

Injected guidance appears as a ``[steering] …`` user turn in the transcript, and
streaming consumers also receive a :class:`~synapse.streaming.Steered` event.
"""

from __future__ import annotations

from collections import deque


class Steer:
    """A mid-run control channel: inject guidance or request a graceful stop.

    Sends are plain (sync) method calls, so a supervisor in any context — sync or
    async, the same task or another — can steer without awaiting. Queued guidance
    is drained by the run loop at each turn boundary.
    """

    def __init__(self) -> None:
        self._inbox: deque[str] = deque()
        self._stop = False
        self._stop_feedback = ""

    def send(self, text: str) -> None:
        """Queue guidance to inject before the agent's next model turn."""
        self._inbox.append(str(text))

    def stop(self, feedback: str = "") -> None:
        """Ask the run to stop gracefully at the next turn boundary."""
        self._stop = True
        self._stop_feedback = feedback

    @property
    def stop_requested(self) -> bool:
        return self._stop

    @property
    def stop_feedback(self) -> str:
        return self._stop_feedback

    @property
    def pending(self) -> int:
        return len(self._inbox)

    def drain(self) -> list[str]:
        """Remove and return all queued guidance (called by the run loop)."""
        out: list[str] = []
        while self._inbox:
            out.append(self._inbox.popleft())
        return out
