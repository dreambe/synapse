"""Tests for skills: discovery, progressive disclosure, bundled files."""

from __future__ import annotations

from synapse import Agent, ScriptedModel, Skill, load_skills
from synapse.messages import ImageBlock, ToolResultBlock


def _make_skill(dir_path, name, description, body, files=None):
    skill_dir = dir_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}")
    for fname, content in (files or {}).items():
        (skill_dir / fname).write_text(content)
    return skill_dir


def test_from_directory_parses_front_matter(tmp_path):
    _make_skill(tmp_path, "fill-pdf", "Fill PDF forms", "Step 1. Do X.", {"t.txt": "tmpl"})
    skill = Skill.from_directory(tmp_path / "fill-pdf")
    assert skill.name == "fill-pdf"
    assert skill.description == "Fill PDF forms"
    assert "Step 1" in skill.instructions
    assert "t.txt" in skill.resources


def test_load_skills_discovers_all(tmp_path):
    _make_skill(tmp_path, "a", "skill a", "body a")
    _make_skill(tmp_path, "b", "skill b", "body b")
    (tmp_path / "not-a-skill").mkdir()  # no SKILL.md → ignored
    skills = load_skills(tmp_path)
    assert sorted(s.name for s in skills) == ["a", "b"]


def test_agent_puts_descriptions_in_context_and_adds_tools(tmp_path):
    _make_skill(tmp_path, "math", "advanced math help", "Use the quadratic formula.")
    agent = Agent("a", instructions="You help.", model=ScriptedModel(["ok"]), skills=load_skills(tmp_path))
    # description is in context by default (progressive disclosure)
    assert "math: advanced math help" in agent.instructions
    # full instructions are NOT dumped into context
    assert "quadratic formula" not in agent.instructions
    assert "load_skill" in agent.tool_map


async def test_load_skill_reveals_full_instructions(tmp_path):
    _make_skill(tmp_path, "math", "math help", "Use the quadratic formula.")
    agent = Agent(
        "a",
        model=ScriptedModel([[("load_skill", {"name": "math"})], "done"]),
        skills=load_skills(tmp_path),
    )
    result = await agent.arun("solve x^2-1=0")
    loaded = [
        b.content
        for m in result.messages
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]
    assert any("quadratic formula" in c for c in loaded if isinstance(c, str))


async def test_read_skill_file_returns_image(tmp_path):
    # a 1x1 png-ish file; mimetypes keys off the .png extension
    _make_skill(tmp_path, "vision", "vision skill", "see chart.png", {"chart.png": "fakepng"})
    agent = Agent(
        "a",
        model=ScriptedModel(
            [[("read_skill_file", {"skill": "vision", "filename": "chart.png"})], "done"]
        ),
        skills=load_skills(tmp_path),
    )
    result = await agent.arun("read the chart")
    tool_results = [
        b for m in result.messages for b in m.content if isinstance(b, ToolResultBlock)
    ]
    assert isinstance(tool_results[0].content, list)
    assert isinstance(tool_results[0].content[0], ImageBlock)


async def test_read_skill_file_blocks_traversal(tmp_path):
    _make_skill(tmp_path, "s", "s", "body")
    (tmp_path / "secret.txt").write_text("top secret")
    agent = Agent(
        "a",
        model=ScriptedModel(
            [[("read_skill_file", {"skill": "s", "filename": "../secret.txt"})], "done"]
        ),
        skills=load_skills(tmp_path),
    )
    result = await agent.arun("read")
    results = [
        b.content
        for m in result.messages
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]
    assert any("no file" in c for c in results if isinstance(c, str))
