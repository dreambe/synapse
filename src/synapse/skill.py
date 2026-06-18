"""Skills: packaged, on-demand expertise (progressive disclosure).

A *skill* is a folder with a ``SKILL.md`` — YAML-ish front matter (``name``,
``description``) plus a body of instructions — and optional bundled resource
files. Following the progressive-disclosure model: a skill's short
*description* sits in the agent's context by default (cheap), and the agent
loads the full instructions only when it judges the skill relevant, by calling
the ``load_skill`` tool. Bundled files are fetched on demand via
``read_skill_file`` (which returns text, images, or documents — multimodal).

    skills/
      fill-pdf/
        SKILL.md          # --- name: fill-pdf / description: ... --- + instructions
        template.pdf      # bundled resource (optional)

    agent = Agent("assistant", skills=load_skills("skills"))
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .messages import DocumentBlock, ImageBlock
from .tool import Tool


@dataclass
class Skill:
    """A unit of on-demand expertise."""

    name: str
    description: str
    instructions: str = ""
    path: Optional[Path] = None
    resources: list[str] = field(default_factory=list)

    @classmethod
    def from_directory(cls, path: str | Path) -> "Skill":
        p = Path(path)
        md = p / "SKILL.md"
        if not md.exists():
            raise FileNotFoundError(f"no SKILL.md in {p}")
        meta, body = _parse_front_matter(md.read_text())
        resources = sorted(
            f.name for f in p.iterdir() if f.is_file() and f.name != "SKILL.md"
        )
        return cls(
            name=meta.get("name") or p.name,
            description=meta.get("description", ""),
            instructions=body.strip(),
            path=p,
            resources=resources,
        )


def load_skills(directory: str | Path) -> list[Skill]:
    """Discover every skill (subdirectory with a ``SKILL.md``) under ``directory``."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return [
        Skill.from_directory(sub)
        for sub in sorted(d.iterdir())
        if sub.is_dir() and (sub / "SKILL.md").exists()
    ]


def skill_catalog(skills: list[Skill]) -> str:
    """The always-in-context summary: skill names + descriptions."""
    lines = "\n".join(f"- {s.name}: {s.description}" for s in skills)
    return (
        "You have access to the following skills. Call `load_skill(name)` to load a "
        "skill's full instructions when it is relevant to the task:\n" + lines
    )


def _within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def skill_tools(skills: list[Skill]) -> list[Tool]:
    """Build the ``load_skill`` / ``read_skill_file`` tools over ``skills``."""
    by_name = {s.name: s for s in skills}

    def load_skill(name: str) -> str:
        skill = by_name.get(name)
        if skill is None:
            return f"no skill named {name!r}; available: {', '.join(by_name)}"
        text = skill.instructions or "(this skill has no instructions)"
        if skill.resources:
            text += "\n\nBundled files (use read_skill_file): " + ", ".join(skill.resources)
        return text

    def read_skill_file(skill: str, filename: str):
        s = by_name.get(skill)
        if s is None or s.path is None:
            return f"no skill named {skill!r}"
        target = s.path / filename
        if not _within(s.path, target) or not target.is_file():
            return f"no file {filename!r} in skill {skill!r}"
        mime = mimetypes.guess_type(target.name)[0] or ""
        if mime.startswith("image/"):
            return ImageBlock.from_file(target)
        if mime.startswith("text/") or mime in ("application/json", "application/xml"):
            return target.read_text()
        return DocumentBlock.from_file(target)

    return [
        Tool(
            name="load_skill",
            description="Load a skill's full instructions by name when it is relevant.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Skill name."}},
                "required": ["name"],
            },
            func=load_skill,
        ),
        Tool(
            name="read_skill_file",
            description="Read a bundled file from a skill (text, image, or document).",
            parameters={
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": "Skill name."},
                    "filename": {"type": "string", "description": "Bundled file name."},
                },
                "required": ["skill", "filename"],
            },
            func=read_skill_file,
        ),
    ]


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Parse optional ``---`` front matter (simple ``key: value`` lines)."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        meta: dict[str, str] = {}
        i = 1
        while i < len(lines) and lines[i].strip() != "---":
            if ":" in lines[i]:
                key, _, value = lines[i].partition(":")
                meta[key.strip()] = value.strip()
            i += 1
        body = "\n".join(lines[i + 1 :]) if i < len(lines) else ""
        return meta, body
    return {}, text
