"""Transport-agnostic A2A dispatcher: maps A2A JSON-RPC methods onto a synapse
agent.

The dispatcher owns an in-memory task store and translates between the A2A
object model (:mod:`synapse.a2a.spec`) and the agent run loop. It is
transport-agnostic so it can be tested directly and bound to any HTTP server.

Supported methods (A2A v0.3.0): ``message/send``, ``message/stream`` (SSE),
``tasks/get``, ``tasks/cancel``, ``tasks/resubscribe``,
``tasks/pushNotificationConfig/set`` and ``/get``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator

from ..streaming import RunComplete, TextDelta
from . import jsonrpc
from .spec import (
    TERMINAL_STATES,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    _uuid,
)

if TYPE_CHECKING:
    from ..agent import Agent


class A2ADispatcher:
    """Serves one synapse :class:`~synapse.agent.Agent` over the A2A protocol."""

    def __init__(self, agent: "Agent") -> None:
        self.agent = agent
        self._tasks: dict[str, Task] = {}
        self._canceled: set[str] = set()
        self._push_configs: dict[str, dict] = {}

    # -- discovery -----------------------------------------------------------

    def agent_card(self, url: str) -> AgentCard:
        return AgentCard(
            name=self.agent.name,
            description=self.agent.description or self.agent.name,
            url=url,
            version=self.agent.version,
            capabilities=AgentCapabilities(streaming=True, push_notifications=True),
            skills=[
                AgentSkill(id=t.name, name=t.name, description=t.description, tags=[])
                for t in self.agent.tools
            ]
            or [AgentSkill(id=self.agent.name, name=self.agent.name, description=self.agent.description)],
        )

    # -- unary dispatch ------------------------------------------------------

    async def handle(self, payload: dict) -> dict:
        """Handle a unary JSON-RPC request, returning a response envelope."""
        try:
            request_id, method, params = jsonrpc.parse_request(payload)
        except jsonrpc.JSONRPCError as exc:
            return jsonrpc.error(payload.get("id"), exc)

        try:
            if method == "message/send":
                result = await self._message_send(params)
            elif method == "tasks/get":
                result = self._tasks_get(params).to_dict()
            elif method == "tasks/cancel":
                result = self._tasks_cancel(params).to_dict()
            elif method == "tasks/pushNotificationConfig/set":
                result = self._push_set(params)
            elif method == "tasks/pushNotificationConfig/get":
                result = self._push_get(params)
            elif method in ("message/stream", "tasks/resubscribe"):
                raise jsonrpc.JSONRPCError(
                    jsonrpc.INVALID_REQUEST, f"{method} requires a streaming transport"
                )
            else:
                raise jsonrpc.JSONRPCError(jsonrpc.METHOD_NOT_FOUND, f"unknown method {method!r}")
            return jsonrpc.success(request_id, result)
        except jsonrpc.JSONRPCError as exc:
            return jsonrpc.error(request_id, exc)
        except Exception as exc:  # noqa: BLE001
            return jsonrpc.error(request_id, jsonrpc.JSONRPCError(jsonrpc.INTERNAL_ERROR, str(exc)))

    # -- streaming dispatch --------------------------------------------------

    async def stream(self, payload: dict) -> AsyncIterator[dict]:
        """Handle ``message/stream`` / ``tasks/resubscribe``; yield JSON-RPC
        response envelopes (one per SSE ``data:`` frame)."""
        try:
            request_id, method, params = jsonrpc.parse_request(payload)
        except jsonrpc.JSONRPCError as exc:
            yield jsonrpc.error(payload.get("id"), exc)
            return

        if method == "tasks/resubscribe":
            task = self._tasks.get(params.get("id", ""))
            if task is None:
                yield jsonrpc.error(
                    request_id, jsonrpc.JSONRPCError(jsonrpc.TASK_NOT_FOUND, "task not found")
                )
            else:
                yield jsonrpc.success(request_id, task.to_dict())
            return

        if method != "message/stream":
            yield jsonrpc.error(
                request_id, jsonrpc.JSONRPCError(jsonrpc.METHOD_NOT_FOUND, f"unknown {method!r}")
            )
            return

        try:
            async for env in self._message_stream(request_id, params):
                yield env
        except jsonrpc.JSONRPCError as exc:
            yield jsonrpc.error(request_id, exc)
        except Exception as exc:  # noqa: BLE001
            yield jsonrpc.error(
                request_id, jsonrpc.JSONRPCError(jsonrpc.INTERNAL_ERROR, str(exc))
            )

    # -- method implementations ---------------------------------------------

    @staticmethod
    def _incoming(params: dict) -> tuple[Message, str]:
        if "message" not in params:
            raise jsonrpc.JSONRPCError(jsonrpc.INVALID_PARAMS, "missing 'message'")
        msg = Message.from_dict(params["message"])
        context_id = msg.context_id or _uuid()
        return msg, context_id

    async def _message_send(self, params: dict) -> dict:
        msg, context_id = self._incoming(params)
        task_id = msg.task_id or _uuid()
        result = await self.agent.arun(msg.text)
        answer = Message.agent_text(result.output, task_id=task_id, context_id=context_id)
        task = Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.COMPLETED, message=answer),
            artifacts=[Artifact(parts=[TextPart(result.output)], name="response")],
            history=[msg, answer],
        )
        self._tasks[task_id] = task
        await self._maybe_push(task)
        return task.to_dict()

    async def _message_stream(self, request_id: Any, params: dict) -> AsyncIterator[dict]:
        msg, context_id = self._incoming(params)
        task_id = msg.task_id or _uuid()

        submitted = Task(
            id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.SUBMITTED)
        )
        self._tasks[task_id] = submitted
        yield jsonrpc.success(request_id, submitted.to_dict())
        yield jsonrpc.success(
            request_id,
            TaskStatusUpdateEvent(
                task_id, context_id, TaskStatus(state=TaskState.WORKING), final=False
            ).to_dict(),
        )

        output = ""
        async for ev in self.agent.astream(msg.text):
            if task_id in self._canceled:
                break
            if isinstance(ev, TextDelta):
                yield jsonrpc.success(request_id, _artifact_chunk(task_id, context_id, ev.text))
            elif isinstance(ev, RunComplete):
                output = ev.result.output

        final_state = TaskState.CANCELED if task_id in self._canceled else TaskState.COMPLETED
        answer = Message.agent_text(output, task_id=task_id, context_id=context_id)
        final_status = TaskStatus(state=final_state, message=answer)
        self._tasks[task_id] = Task(
            id=task_id,
            context_id=context_id,
            status=final_status,
            artifacts=[Artifact(parts=[TextPart(output)], name="response")],
        )
        yield jsonrpc.success(
            request_id,
            TaskStatusUpdateEvent(task_id, context_id, final_status, final=True).to_dict(),
        )
        await self._maybe_push(self._tasks[task_id])

    def _tasks_get(self, params: dict) -> Task:
        task = self._tasks.get(params.get("id", ""))
        if task is None:
            raise jsonrpc.JSONRPCError(jsonrpc.TASK_NOT_FOUND, "task not found")
        return task

    def _tasks_cancel(self, params: dict) -> Task:
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if task is None:
            raise jsonrpc.JSONRPCError(jsonrpc.TASK_NOT_FOUND, "task not found")
        if task.status.state in TERMINAL_STATES:
            raise jsonrpc.JSONRPCError(jsonrpc.TASK_NOT_CANCELABLE, "task is not cancelable")
        self._canceled.add(task_id)
        task.status = TaskStatus(state=TaskState.CANCELED)
        return task

    def _push_set(self, params: dict) -> dict:
        task_id = params.get("taskId", "")
        config = params.get("pushNotificationConfig", {})
        self._push_configs[task_id] = config
        return {"taskId": task_id, "pushNotificationConfig": config}

    def _push_get(self, params: dict) -> dict:
        task_id = params.get("taskId", "")
        config = self._push_configs.get(task_id)
        if config is None:
            raise jsonrpc.JSONRPCError(jsonrpc.TASK_NOT_FOUND, "no push config for task")
        return {"taskId": task_id, "pushNotificationConfig": config}

    async def _maybe_push(self, task: Task) -> None:
        """Fire a signed, retrying webhook for a terminal task (surpasses the
        fire-and-forget baseline)."""
        config = self._push_configs.get(task.id)
        if not config or task.status.state not in TERMINAL_STATES:
            return
        from . import push

        await asyncio.to_thread(push.deliver, config, task.to_dict())


def _artifact_chunk(task_id: str, context_id: str, text: str) -> dict:
    return TaskArtifactUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        artifact=Artifact(parts=[TextPart(text)], name="response"),
        append=True,
        last_chunk=False,
    ).to_dict()
