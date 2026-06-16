from synapse import Agent, ScriptedModel, Session, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_agent_runs_tool_then_answers():
    # First turn asks for the tool; second turn gives the final answer.
    model = ScriptedModel([[("add", {"a": 19, "b": 23})], "The answer is 42."])
    agent = Agent("calc", instructions="You do math.", model=model, tools=[add])

    result = agent.run("What is 19 + 23?")

    assert result.output == "The answer is 42."
    assert result.iterations == 2
    assert result.stop_reason == "end_turn"
    # the tool result was fed back to the model on the second call
    assert any(
        b.content == "42"
        for msg in result.messages
        for b in msg.content
        if getattr(b, "type", None) == "tool_result"
    )


def test_unknown_tool_reports_error_to_model():
    model = ScriptedModel([[("nonexistent", {})], "done"])
    agent = Agent("x", model=model, tools=[add])
    result = agent.run("go")
    error_results = [
        b
        for msg in result.messages
        for b in msg.content
        if getattr(b, "type", None) == "tool_result" and b.is_error
    ]
    assert error_results and "nonexistent" in error_results[0].content


def test_max_iterations_guard():
    # Always asks for a tool — should stop at the iteration budget.
    model = ScriptedModel([[("add", {"a": 1, "b": 1})]] * 10)
    agent = Agent("loop", model=model, tools=[add])
    result = agent.run("go", max_iterations=3)
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3


def test_session_preserves_history():
    model = ScriptedModel(["first answer", "second answer"])
    agent = Agent("chat", model=model)
    session = Session()

    agent.run("hello", session=session)
    agent.run("again", session=session)

    # both user turns and both assistant turns are retained
    user_turns = [m for m in session.messages if m.role == "user"]
    assert len(user_turns) == 2


def test_decorator_attaches_tool():
    agent = Agent("calc")

    @agent.tool
    def multiply(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    assert "multiply" in agent.tool_map
