"""Permission modes: a structured policy for what an agent may *do*.

Per-tool `requires_approval` + an `approval=` callback already gives
human-in-the-loop gating. A :class:`PermissionPolicy` raises that to a declared
**mode** that governs the whole run — the structural-safety equivalent of a
frontier agent's plan / accept-edits / bypass modes — so "this run is read-only"
is one decision enforced by construction, not a careful habit per tool.

Modes (a tool's :attr:`~synapse.tool.Tool.side_effect` — ``"read"`` vs the safe
default ``"write"`` — drives classification):

- ``PLAN``   — read-only: allow read tools, **deny** every mutating tool. The
  agent can investigate and plan but cannot change anything.
- ``ASK``    — allow reads; **escalate** writes to the approval callback (deny if
  none is configured — fail safe).
- ``AUTO``   — allow everything (still honoring explicit ``deny`` and a tool's
  own ``requires_approval``). Sensible default once you trust the agent.
- ``BYPASS`` — allow everything, explicitly and loudly. Use only when you mean it.

``allow`` / ``deny`` name-sets always win over the mode, so you can carve
exceptions either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tool import Tool


class PermissionMode(str, Enum):
    PLAN = "plan"
    ASK = "ask"
    AUTO = "auto"
    BYPASS = "bypass"


class Permission(str, Enum):
    """The decision for a single tool call."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"  # escalate to the approval callback


@dataclass
class PermissionPolicy:
    """A declared, run-wide policy mapping each tool call to allow/deny/ask."""

    mode: PermissionMode = PermissionMode.AUTO
    allow: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)

    def decide(self, tool: "Tool") -> Permission:
        if tool.name in self.deny:
            return Permission.DENY
        if tool.name in self.allow:
            return Permission.ALLOW
        is_read = tool.side_effect == "read"
        if self.mode is PermissionMode.BYPASS or self.mode is PermissionMode.AUTO:
            return Permission.ALLOW
        if self.mode is PermissionMode.PLAN:
            return Permission.ALLOW if is_read else Permission.DENY
        # ASK mode
        return Permission.ALLOW if is_read else Permission.ASK


def permission_policy(
    mode: "PermissionMode | str" = PermissionMode.AUTO,
    *,
    allow: "set[str] | None" = None,
    deny: "set[str] | None" = None,
) -> PermissionPolicy:
    """Build a :class:`PermissionPolicy`. ``mode`` accepts the enum or its string."""
    return PermissionPolicy(
        mode=PermissionMode(mode),
        allow=set(allow or ()),
        deny=set(deny or ()),
    )
