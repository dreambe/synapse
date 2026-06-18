"""Run monitor: a live activity plane for many concurrent agents.

A long agentic system is hard to operate precisely because it is *opaque*: a
coordinator spawns sub-agents, each of which loads a skill, calls an MCP tool,
runs a script — and from the outside you see only the final answer. The
:class:`Monitor` makes that legible. It is a :class:`~synapse.observability.Hooks`
implementation you attach to a run (``hooks=monitor``); it turns the run's
lifecycle into a stream of structured :class:`ActivityEvent` s and keeps a live
:class:`RunView` per run.

Two design choices make it work for *many* agents at once:

- **Per-run identity without a signature change.** A run id is minted at
  ``on_run_start`` and tracked in a :class:`contextvars.ContextVar`. Because
  asyncio tasks copy their context, concurrent runs — and sub-agent runs that
  inherit the monitor — get distinct ids and a correct parent→child link, so
  the monitor can reconstruct the **agent tree** (who spawned whom).
- **Semantic classification.** Tool calls are classified by synapse's own
  naming conventions — ``ask_*`` → a sub-agent delegation, ``load_skill`` /
  ``read_skill_file`` → a skill, ``run_python`` → a script, ``fetch_result`` /
  ``write_plan`` → framework affordances — so the activity feed reads as
  "spawned researcher → loaded skill pandas → ran script", not a flat log.

The framework supplies the **data plane** (events, views, a subscribe stream)
plus a minimal reference dashboard (:func:`monitor_app`, a dependency-free ASGI
page). A production console is an app concern built on the same data.

To capture the *inside* of a sub-agent (not just the spawning tool call),
forward the monitor when exposing the agent as a tool::

    agent.as_tool(hooks=monitor)
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .observability import Hooks

# The run currently executing in this asyncio context (None at the top level).
_current_run: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "synapse_current_run", default=None
)


def classify_tool(name: str) -> str:
    """Map a tool name to an activity kind using synapse's conventions."""
    if name.startswith("ask_"):
        return "subagent"
    if name in ("load_skill", "read_skill_file"):
        return "skill"
    if name in ("run_python", "code_execution", "execute_python"):
        return "script"
    if name in ("write_plan", "update_step"):
        return "plan"
    if name in ("fetch_result",):
        return "fetch"
    if name in ("remember", "recall"):
        return "memory"
    if name in ("ask_human", "escalate"):
        return "human"
    if name.startswith("mcp_") or "." in name:
        return "mcp"
    return "tool"


@dataclass
class ActivityEvent:
    """One thing that happened in a run, with enough context to place it."""

    run_id: str
    parent_run_id: Optional[str]
    agent: str
    kind: str  # run_start | run_end | model | tool_start | tool_end
    name: str = ""
    detail: str = ""
    is_error: bool = False
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunView:
    """Live state of a single run, including its place in the agent tree."""

    run_id: str
    agent: str
    parent_run_id: Optional[str] = None
    status: str = "running"  # running | done | error
    stop_reason: str = ""
    started: float = field(default_factory=time.time)
    ended: Optional[float] = None
    iterations: int = 0
    tokens: int = 0
    activity: list[ActivityEvent] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        return None if self.ended is None else self.ended - self.started

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "parent_run_id": self.parent_run_id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "started": self.started,
            "ended": self.ended,
            "duration": self.duration,
            "iterations": self.iterations,
            "tokens": self.tokens,
            "activity": [a.to_dict() for a in self.activity],
        }


class Monitor(Hooks):
    """Collects live activity across many runs; the observability data plane.

    Attach as ``hooks=monitor`` to each top-level run (and forward to sub-agents
    via ``agent.as_tool(hooks=monitor)`` to see inside them). Inspect with
    :meth:`snapshot` / :meth:`tree`, or live-tail with :meth:`subscribe`.
    """

    def __init__(self, *, history: int = 200, per_run_activity: int = 500) -> None:
        self._runs: dict[str, RunView] = {}
        self._order: list[str] = []
        self._history = history
        self._per_run_activity = per_run_activity
        self._subscribers: set[asyncio.Queue[ActivityEvent]] = set()
        self._tokens: dict[str, contextvars.Token] = {}

    # -- queries ------------------------------------------------------------

    def snapshot(self) -> list[dict]:
        """All known runs (newest first) as plain dicts."""
        return [self._runs[r].to_dict() for r in reversed(self._order) if r in self._runs]

    def active(self) -> list[dict]:
        return [v for v in self.snapshot() if v["status"] == "running"]

    def tree(self) -> list[dict]:
        """Root runs with their sub-agent runs nested under ``children``."""
        children: dict[Optional[str], list[str]] = {}
        for rid in self._order:
            view = self._runs.get(rid)
            if view is None:
                continue
            children.setdefault(view.parent_run_id, []).append(rid)

        def build(rid: str) -> dict:
            node = self._runs[rid].to_dict()
            node["children"] = [build(c) for c in children.get(rid, [])]
            return node

        roots = [rid for rid in self._order if self._runs.get(rid) and
                 self._runs[rid].parent_run_id not in self._runs]
        return [build(r) for r in roots]

    # -- live streaming -----------------------------------------------------

    def subscribe(self) -> "asyncio.Queue[ActivityEvent]":
        """Register a queue that receives every subsequent :class:`ActivityEvent`."""
        q: asyncio.Queue[ActivityEvent] = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[ActivityEvent]") -> None:
        self._subscribers.discard(q)

    def _publish(self, event: ActivityEvent) -> None:
        view = self._runs.get(event.run_id)
        if view is not None:
            view.activity.append(event)
            if len(view.activity) > self._per_run_activity:
                del view.activity[0]
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # a slow consumer must not stall the run
                pass

    def _prune(self) -> None:
        while len(self._order) > self._history:
            oldest = self._order.pop(0)
            self._runs.pop(oldest, None)

    # -- hook callbacks -----------------------------------------------------

    async def on_run_start(self, agent: str, user_input: str) -> None:
        parent = _current_run.get()
        run_id = "run_" + uuid.uuid4().hex[:12]
        self._tokens[run_id] = _current_run.set(run_id)
        self._runs[run_id] = RunView(run_id=run_id, agent=agent, parent_run_id=parent)
        self._order.append(run_id)
        self._prune()
        self._publish(ActivityEvent(run_id, parent, agent, "run_start", detail=user_input[:200]))

    async def on_model_response(self, response: Any) -> None:
        run_id = _current_run.get()
        if run_id is None or run_id not in self._runs:
            return
        view = self._runs[run_id]
        view.iterations += 1
        self._publish(
            ActivityEvent(run_id, view.parent_run_id, view.agent, "model",
                          detail=getattr(response, "stop_reason", "") or "")
        )

    async def on_tool_start(self, name: str, tool_input: dict) -> None:
        run_id = _current_run.get()
        if run_id is None or run_id not in self._runs:
            return
        view = self._runs[run_id]
        self._publish(
            ActivityEvent(run_id, view.parent_run_id, view.agent, "tool_start",
                          name=name, detail=classify_tool(name))
        )

    async def on_tool_end(self, name: str, result: str, is_error: bool) -> None:
        run_id = _current_run.get()
        if run_id is None or run_id not in self._runs:
            return
        view = self._runs[run_id]
        self._publish(
            ActivityEvent(run_id, view.parent_run_id, view.agent, "tool_end",
                          name=name, detail=classify_tool(name), is_error=is_error)
        )

    async def on_run_end(self, result: Any) -> None:
        run_id = _current_run.get()
        token = self._tokens.pop(run_id, None) if run_id else None
        if token is not None:
            _current_run.reset(token)
        if run_id is None or run_id not in self._runs:
            return
        view = self._runs[run_id]
        view.status = "error" if getattr(result, "stop_reason", "") == "error" else "done"
        view.stop_reason = getattr(result, "stop_reason", "") or ""
        view.iterations = getattr(result, "iterations", view.iterations)
        usage = getattr(result, "usage", None)
        view.tokens = getattr(usage, "total_tokens", 0) if usage else 0
        view.ended = time.time()
        self._publish(
            ActivityEvent(view.run_id, view.parent_run_id, view.agent, "run_end",
                          detail=view.stop_reason)
        )


# -- reference dashboard (minimal, dependency-free ASGI) --------------------

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]


def monitor_app(monitor: Monitor):
    """A minimal ASGI dashboard over ``monitor`` (run under ``synapse[server]``).

    Routes:
        GET /              → a single self-contained HTML page (live tree + feed)
        GET /api/runs      → the run tree as JSON
        GET /api/stream    → Server-Sent Events of every activity event
    """

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":  # pragma: no cover - websockets etc.
            return

        method, path = scope["method"], scope["path"]
        if method == "GET" and path == "/":
            await _send(send, 200, "text/html; charset=utf-8", _PAGE.encode())
            return
        if method == "GET" and path == "/api/runs":
            body = json.dumps({"runs": monitor.tree()}).encode()
            await _send(send, 200, "application/json", body)
            return
        if method == "GET" and path == "/api/stream":
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream"),
                            (b"cache-control", b"no-cache")],
            })
            q = monitor.subscribe()
            try:
                while True:
                    event = await q.get()
                    frame = f"data: {json.dumps(event.to_dict())}\n\n".encode()
                    await send({"type": "http.response.body", "body": frame, "more_body": True})
            except asyncio.CancelledError:  # pragma: no cover - client disconnect
                raise
            finally:
                monitor.unsubscribe(q)
            return

        await _send(send, 404, "application/json", b'{"error":"not found"}')

    return app


async def _send(send: Send, status: int, content_type: str, body: bytes) -> None:
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", content_type.encode()),
                    (b"content-length", str(len(body)).encode())],
    })
    await send({"type": "http.response.body", "body": body})


# A single dependency-free page: poll the tree, live-tail the SSE feed.
_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>synapse · run monitor</title>
<style>
 body{font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;margin:0;background:#0d1117;color:#c9d1d9}
 header{padding:10px 16px;border-bottom:1px solid #21262d;font-weight:600}
 .wrap{display:flex;height:calc(100vh - 43px)}
 .col{overflow:auto;padding:12px 16px}
 #runs{flex:1;border-right:1px solid #21262d}
 #feed{width:42%}
 .run{margin:2px 0;padding:4px 8px;border-left:2px solid #30363d;border-radius:3px}
 .run.running{border-left-color:#d29922}.run.done{border-left-color:#2ea043}.run.error{border-left-color:#f85149}
 .agent{color:#79c0ff;font-weight:600}.meta{color:#8b949e}
 .kids{margin-left:18px}
 .ev{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .k-subagent{color:#d2a8ff}.k-skill{color:#7ee787}.k-mcp{color:#ffa657}.k-script{color:#ff7b72}
 .k-plan{color:#79c0ff}.k-run_start,.k-run_end{color:#8b949e}.err{color:#f85149}
 h2{font-size:12px;text-transform:uppercase;color:#8b949e;letter-spacing:.05em}
</style></head>
<body>
<header>synapse · run monitor <span class="meta" id="stat"></span></header>
<div class="wrap">
 <div class="col" id="runs"><h2>agents</h2><div id="tree"></div></div>
 <div class="col" id="feed"><h2>activity</h2><div id="events"></div></div>
</div>
<script>
const KIND={subagent:'spawn',skill:'skill',mcp:'mcp',script:'script',plan:'plan',
            fetch:'fetch',memory:'memory',human:'human',tool:'tool'};
function fmtRun(r){
  const dur=r.duration?(' '+r.duration.toFixed(1)+'s'):'';
  const tok=r.tokens?(' · '+r.tokens+'tok'):'';
  let h=`<div class="run ${r.status}"><span class="agent">${r.agent}</span> `+
        `<span class="meta">${r.status}${r.stop_reason?(' · '+r.stop_reason):''}`+
        ` · ${r.iterations} iter${tok}${dur}</span>`;
  if(r.children&&r.children.length)
    h+=`<div class="kids">`+r.children.map(fmtRun).join('')+`</div>`;
  return h+`</div>`;
}
async function poll(){
  try{const d=await (await fetch('/api/runs')).json();
    document.getElementById('tree').innerHTML=d.runs.map(fmtRun).join('')||'<span class="meta">no runs yet</span>';
    const n=JSON.stringify(d.runs).split('"run_id"').length-1;
    document.getElementById('stat').textContent='· '+n+' runs';
  }catch(e){}
}
function addEvent(e){
  const box=document.getElementById('events');
  const t=new Date(e.ts*1000).toLocaleTimeString();
  let label=e.kind;
  if(e.kind==='tool_start')label='→ '+(KIND[e.detail]||'tool')+' '+e.name;
  else if(e.kind==='tool_end')label='✓ '+e.name;
  else if(e.kind==='run_start')label='▶ run';
  else if(e.kind==='run_end')label='■ '+(e.detail||'end');
  else if(e.kind==='model')label='· model';
  const cls='k-'+(e.kind==='tool_start'||e.kind==='tool_end'?e.detail:e.kind);
  const div=document.createElement('div');
  div.className='ev '+(e.is_error?'err':cls);
  div.textContent=`${t} [${e.agent}] ${label}`;
  box.prepend(div);
  while(box.childNodes.length>300)box.removeChild(box.lastChild);
}
const es=new EventSource('/api/stream');
es.onmessage=ev=>{addEvent(JSON.parse(ev.data));poll();};
poll();setInterval(poll,2000);
</script>
</body></html>
"""
