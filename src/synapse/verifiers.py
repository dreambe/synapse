"""Outcome verifiers: pass only if a real check succeeds.

`verify=` accepts any callable returning a Verdict, so verification can *run the
tests/build* against the artifacts the agent produced — not just inspect the
final text. Combine with ``max_verify_rounds`` for "produce → check → iterate".

    agent.run("fix the failing test", verify=command_verifier(["pytest", "-q"]),
              max_verify_rounds=3)
"""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence

from .runtime import Verdict


def command_verifier(
    argv: Sequence[str],
    *,
    cwd: Optional[str] = None,
    timeout: float = 120.0,
):
    """A verifier that runs ``argv`` and passes iff it exits 0.

    The command's stdout/stderr become the verifier feedback, so a failing
    build/test loops back into the agent on the next round.
    """

    async def verify(_output: str) -> Verdict:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            return Verdict(False, f"check timed out after {timeout}s")
        except FileNotFoundError as exc:
            return Verdict(False, f"check command not found: {exc}")
        text = out.decode("utf-8", "replace")[-4000:]
        if proc.returncode == 0:
            return Verdict(True, "check passed")
        return Verdict(False, f"check failed (exit {proc.returncode}):\n{text}")

    return verify
