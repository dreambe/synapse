"""Skills with progressive disclosure — offline demo.

Only the skill *description* is in context until the agent loads it.

    python examples/skill_agent.py
"""

from __future__ import annotations

from pathlib import Path

from synapse import Agent, ScriptedModel, load_skills

SKILLS_DIR = Path(__file__).parent / "skills"


def main() -> None:
    skills = load_skills(SKILLS_DIR)

    # Offline: pretend the model decides the 'changelog' skill is relevant,
    # loads it, then writes the entry.
    agent = Agent(
        "release-bot",
        instructions="You help maintain a project.",
        model=ScriptedModel(
            [
                [("load_skill", {"name": "changelog"})],
                "### Added\n- Skills with progressive disclosure",
            ]
        ),
        skills=skills,
    )

    print("Skill descriptions in context:")
    print("  " + agent.instructions.splitlines()[-1])
    print("\nResult:")
    print(agent.run("Write a changelog entry for the new skills feature.").output)


if __name__ == "__main__":
    main()
