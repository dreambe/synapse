from synapse import tool
from synapse.errors import ToolError
from synapse.tool import Tool

import pytest


def test_tool_schema_from_signature():
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.description == "Add two integers."
    schema = add.to_schema()
    assert schema["input_schema"]["properties"] == {
        "a": {"type": "integer"},
        "b": {"type": "integer"},
    }
    assert schema["input_schema"]["required"] == ["a", "b"]
    assert add(a=2, b=3) == 5


def test_optional_params_not_required():
    @tool
    def greet(name: str, loud: bool = False) -> str:
        """Greet someone."""
        return name.upper() if loud else name

    assert greet.parameters["required"] == ["name"]
    assert "loud" in greet.parameters["properties"]
    assert greet.parameters["properties"]["loud"]["type"] == "boolean"


def test_custom_name_and_description():
    @tool(name="sum_two", description="Custom desc")
    def add(a: int, b: int) -> int:
        return a + b

    assert add.name == "sum_two"
    assert add.description == "Custom desc"


def test_var_args_rejected():
    with pytest.raises(ToolError):

        @tool
        def bad(*args):
            return args
