"""JSON-RPC 2.0 envelopes and A2A error codes."""

from __future__ import annotations

from typing import Any

# Standard JSON-RPC 2.0 errors
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# A2A-specific errors
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002
PUSH_NOTIFICATION_NOT_SUPPORTED = -32003
UNSUPPORTED_OPERATION = -32004
CONTENT_TYPE_NOT_SUPPORTED = -32005
INVALID_AGENT_RESPONSE = -32006


class JSONRPCError(Exception):
    """An error to be serialized into a JSON-RPC error response."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


def success(request_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error(request_id: Any, exc: JSONRPCError) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": exc.to_dict()}


def parse_request(payload: dict) -> tuple[Any, str, dict]:
    """Validate a JSON-RPC request envelope; return (id, method, params)."""
    if payload.get("jsonrpc") != "2.0" or "method" not in payload:
        raise JSONRPCError(INVALID_REQUEST, "invalid JSON-RPC 2.0 request")
    return payload.get("id"), payload["method"], payload.get("params") or {}
