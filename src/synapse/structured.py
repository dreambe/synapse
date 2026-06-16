"""Structured outputs: make an agent produce typed, validated results.

Two provider-neutral pieces:

- **Output schema** — give a run a JSON Schema (or a Pydantic model). The
  schema is described to the model, and the final answer is parsed + validated
  into ``RunResult.parsed``. Works with any backend (instruction + validation),
  and composes with ``verify=`` to iterate until valid.
- **Tool-input validation** — optionally validate a tool call's arguments
  against the tool's own schema *before* executing, so a malformed call returns
  a correctable error to the model instead of raising.

The validator covers the common JSON-Schema subset (type, required, properties,
items, enum) with no dependency; Pydantic models are supported when present.
"""

from __future__ import annotations

import json
from typing import Any, Optional

_JSON_TYPES = {
    "string": str,
    "object": dict,
    "array": list,
}


def _type_ok(data: Any, t: str) -> bool:
    if t == "integer":
        return isinstance(data, int) and not isinstance(data, bool)
    if t == "number":
        return isinstance(data, (int, float)) and not isinstance(data, bool)
    if t == "boolean":
        return isinstance(data, bool)
    if t == "null":
        return data is None
    py = _JSON_TYPES.get(t)
    return isinstance(data, py) if py else True


def validate_json(data: Any, schema: dict, path: str = "$") -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []
    t = schema.get("type")
    if t and not _type_ok(data, t):
        errors.append(f"{path}: expected {t}")
        return errors
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: {data!r} not in enum")
    if (t == "object" or "properties" in schema) and isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                errors.append(f"{path}.{req}: required")
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                errors.extend(validate_json(data[key], sub, f"{path}.{key}"))
    if t == "array" and isinstance(data, list) and "items" in schema:
        for i, item in enumerate(data):
            errors.extend(validate_json(item, schema["items"], f"{path}[{i}]"))
    return errors


def schema_instruction(schema: dict) -> str:
    return (
        "Respond with ONLY a single JSON value conforming to this JSON Schema — "
        "no prose, no markdown fences:\n" + json.dumps(schema)
    )


def _extract_json(text: str) -> str:
    s = text.strip()
    if "```" in s:
        # take the content of the first fenced block
        block = s.split("```", 2)
        if len(block) >= 2:
            inner = block[1]
            if inner.startswith("json"):
                inner = inner[4:]
            s = inner.strip()
    # fall back to the first {...} or [...] span
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = s.find(open_c)
        end = s.rfind(close_c)
        if start != -1 and end > start:
            return s[start : end + 1]
    return s


def parse_output(text: str, schema: dict) -> tuple[Optional[Any], list[str]]:
    """Parse ``text`` as JSON and validate against ``schema``."""
    try:
        data = json.loads(_extract_json(text))
    except (ValueError, TypeError) as exc:
        return None, [f"invalid JSON: {exc}"]
    errors = validate_json(data, schema)
    return (data if not errors else None), errors


def model_schema(response_model: type) -> Optional[dict]:
    """JSON Schema from a Pydantic model (or None if not one)."""
    fn = getattr(response_model, "model_json_schema", None)
    return fn() if callable(fn) else None


def instantiate(response_model: type, data: Any) -> Any:
    """Build a Pydantic instance from validated data; fall back to the data."""
    validator = getattr(response_model, "model_validate", None)
    if callable(validator):
        try:
            return validator(data)
        except Exception:  # noqa: BLE001 - validation handled upstream
            return data
    return data
