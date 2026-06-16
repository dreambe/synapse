"""Pluggable LLM backends."""

from __future__ import annotations

from .anthropic import DEFAULT_MODEL, AnthropicModel
from .base import Model, ModelResponse
from .scripted import EchoModel, ScriptedModel


def default_model() -> Model:
    """The backend used when an agent is created without one: Claude."""
    return AnthropicModel()


__all__ = [
    "Model",
    "ModelResponse",
    "AnthropicModel",
    "EchoModel",
    "ScriptedModel",
    "DEFAULT_MODEL",
    "default_model",
]
