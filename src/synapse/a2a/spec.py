"""Strictly A2A-compliant data types (Agent2Agent protocol, v0.3.0).

These mirror the official A2A JSON-RPC objects exactly — field names, ``kind``
discriminators, and ``TaskState`` literals — so synapse agents interoperate
with any A2A-compliant peer (and vice versa). See https://a2a-protocol.org.

Reference: A2A specification v0.3.0 (JSON-RPC 2.0 transport).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

PROTOCOL_VERSION = "0.3.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex


class TaskState:
    """The exact A2A ``TaskState`` string literals."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
    UNKNOWN = "unknown"


TERMINAL_STATES = {
    TaskState.COMPLETED,
    TaskState.CANCELED,
    TaskState.FAILED,
    TaskState.REJECTED,
}


# -- Parts ------------------------------------------------------------------


@dataclass
class TextPart:
    text: str
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"kind": "text", "text": self.text}
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class DataPart:
    data: dict
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"kind": "data", "data": self.data}
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class FilePart:
    """A file part: either bytes (base64) or a URI, per A2A FileWithBytes/Uri."""

    name: Optional[str] = None
    mime_type: Optional[str] = None
    bytes: Optional[str] = None
    uri: Optional[str] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        file_obj: dict[str, Any] = {}
        if self.name is not None:
            file_obj["name"] = self.name
        if self.mime_type is not None:
            file_obj["mimeType"] = self.mime_type
        if self.bytes is not None:
            file_obj["bytes"] = self.bytes
        if self.uri is not None:
            file_obj["uri"] = self.uri
        d: dict[str, Any] = {"kind": "file", "file": file_obj}
        if self.metadata:
            d["metadata"] = self.metadata
        return d


Part = Any  # TextPart | DataPart | FilePart


def part_from_dict(data: dict) -> Part:
    kind = data.get("kind")
    if kind == "text":
        return TextPart(text=data.get("text", ""), metadata=data.get("metadata"))
    if kind == "data":
        return DataPart(data=data.get("data", {}), metadata=data.get("metadata"))
    if kind == "file":
        f = data.get("file", {})
        return FilePart(
            name=f.get("name"),
            mime_type=f.get("mimeType"),
            bytes=f.get("bytes"),
            uri=f.get("uri"),
            metadata=data.get("metadata"),
        )
    raise ValueError(f"unknown part kind: {kind!r}")


def part_to_text(part: Part) -> str:
    if isinstance(part, TextPart):
        return part.text
    if isinstance(part, DataPart):
        import json

        return json.dumps(part.data)
    return ""


# -- Message ----------------------------------------------------------------


@dataclass
class Message:
    role: str  # "user" | "agent"
    parts: list[Part]
    message_id: str = field(default_factory=_uuid)
    task_id: Optional[str] = None
    context_id: Optional[str] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "kind": "message",
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
            "messageId": self.message_id,
        }
        if self.task_id is not None:
            d["taskId"] = self.task_id
        if self.context_id is not None:
            d["contextId"] = self.context_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @property
    def text(self) -> str:
        return "".join(part_to_text(p) for p in self.parts)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            parts=[part_from_dict(p) for p in data.get("parts", [])],
            message_id=data.get("messageId", _uuid()),
            task_id=data.get("taskId"),
            context_id=data.get("contextId"),
            metadata=data.get("metadata"),
        )

    @classmethod
    def user_text(cls, text: str) -> "Message":
        return cls(role="user", parts=[TextPart(text)])

    @classmethod
    def agent_text(cls, text: str, *, task_id: str | None = None, context_id: str | None = None):
        return cls(role="agent", parts=[TextPart(text)], task_id=task_id, context_id=context_id)


# -- Task & status ----------------------------------------------------------


@dataclass
class TaskStatus:
    state: str
    message: Optional[Message] = None
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"state": self.state, "timestamp": self.timestamp}
        if self.message is not None:
            d["message"] = self.message.to_dict()
        return d


@dataclass
class Artifact:
    parts: list[Part]
    artifact_id: str = field(default_factory=_uuid)
    name: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "artifactId": self.artifact_id,
            "parts": [p.to_dict() for p in self.parts],
        }
        if self.name is not None:
            d["name"] = self.name
        if self.description is not None:
            d["description"] = self.description
        return d


@dataclass
class Task:
    id: str
    context_id: str
    status: TaskStatus
    artifacts: list[Artifact] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "kind": "task",
            "id": self.id,
            "contextId": self.context_id,
            "status": self.status.to_dict(),
        }
        if self.artifacts:
            d["artifacts"] = [a.to_dict() for a in self.artifacts]
        if self.history:
            d["history"] = [m.to_dict() for m in self.history]
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class TaskStatusUpdateEvent:
    task_id: str
    context_id: str
    status: TaskStatus
    final: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": "status-update",
            "taskId": self.task_id,
            "contextId": self.context_id,
            "status": self.status.to_dict(),
            "final": self.final,
        }


@dataclass
class TaskArtifactUpdateEvent:
    task_id: str
    context_id: str
    artifact: Artifact
    append: bool = False
    last_chunk: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": "artifact-update",
            "taskId": self.task_id,
            "contextId": self.context_id,
            "artifact": self.artifact.to_dict(),
            "append": self.append,
            "lastChunk": self.last_chunk,
        }


# -- Agent Card -------------------------------------------------------------


@dataclass
class AgentSkill:
    id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    examples: Optional[list[str]] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
        }
        if self.examples:
            d["examples"] = self.examples
        return d


@dataclass
class AgentCapabilities:
    streaming: bool = True
    push_notifications: bool = False
    state_transition_history: bool = False

    def to_dict(self) -> dict:
        return {
            "streaming": self.streaming,
            "pushNotifications": self.push_notifications,
            "stateTransitionHistory": self.state_transition_history,
        }


@dataclass
class AgentCard:
    name: str
    description: str
    url: str
    version: str = "0.1.0"
    protocol_version: str = PROTOCOL_VERSION
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    default_input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    skills: list[AgentSkill] = field(default_factory=list)
    provider: Optional[dict] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "protocolVersion": self.protocol_version,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": self.capabilities.to_dict(),
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "skills": [s.to_dict() for s in self.skills],
        }
        if self.provider is not None:
            d["provider"] = self.provider
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCard":
        caps = data.get("capabilities", {})
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            url=data["url"],
            version=data.get("version", "0.1.0"),
            protocol_version=data.get("protocolVersion", PROTOCOL_VERSION),
            capabilities=AgentCapabilities(
                streaming=caps.get("streaming", False),
                push_notifications=caps.get("pushNotifications", False),
                state_transition_history=caps.get("stateTransitionHistory", False),
            ),
            default_input_modes=data.get("defaultInputModes", ["text/plain"]),
            default_output_modes=data.get("defaultOutputModes", ["text/plain"]),
            skills=[
                AgentSkill(
                    id=s["id"],
                    name=s.get("name", s["id"]),
                    description=s.get("description", ""),
                    tags=s.get("tags", []),
                    examples=s.get("examples"),
                )
                for s in data.get("skills", [])
            ],
            provider=data.get("provider"),
        )


WELL_KNOWN_PATH = "/.well-known/agent-card.json"
WELL_KNOWN_PATH_LEGACY = "/.well-known/agent.json"
