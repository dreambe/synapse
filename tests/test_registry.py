from synapse import Agent, AgentRegistry, EchoModel
from synapse.errors import RegistryError

import pytest


def test_register_and_get():
    reg = AgentRegistry()
    a = Agent("alpha", model=EchoModel())
    reg.register(a)
    assert reg.get("alpha") is a
    assert "alpha" in reg
    assert reg.names() == ["alpha"]
    assert len(reg) == 1


def test_duplicate_name_rejected():
    reg = AgentRegistry()
    reg.register(Agent("dup", model=EchoModel()))
    with pytest.raises(RegistryError):
        reg.register(Agent("dup", model=EchoModel()))


def test_missing_name_raises():
    reg = AgentRegistry()
    with pytest.raises(RegistryError):
        reg.get("ghost")
