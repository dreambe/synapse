"""Exception types raised across the synapse framework."""

from __future__ import annotations


class SynapseError(Exception):
    """Base class for all synapse errors."""


class ConfigurationError(SynapseError):
    """Raised when an agent, model, or tool is misconfigured."""


class ModelError(SynapseError):
    """Raised when an LLM backend fails to produce a response."""


class ToolError(SynapseError):
    """Raised when a tool cannot be built or executed."""


class RegistryError(SynapseError):
    """Raised when an agent lookup in a registry fails."""


class A2AError(SynapseError):
    """Raised when an agent-to-agent call fails."""


class MaxIterationsExceeded(SynapseError):
    """Raised when an agent loop runs longer than its iteration budget."""
