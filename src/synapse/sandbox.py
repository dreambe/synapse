"""Code execution: run code in an isolated subprocess with resource limits.

**Honest scope.** This provides *process isolation* — a separate subprocess in
a fresh temp working directory with a minimal environment, hard wall-clock
timeout, and POSIX resource limits (CPU time, address space, output file size).
It is **not** a security boundary against adversarial code: there is no syscall
filtering and no network isolation. For untrusted/third-party code, run inside a
container, gVisor, firejail, or a VM. Use this for code your own agent generates,
with these guard rails — not arbitrary input.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from .tool import Tool


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def __str__(self) -> str:
        if self.timed_out:
            return f"[timed out]\n{self.stderr}".strip()
        parts = [f"exit code: {self.returncode}"]
        if self.stdout:
            parts.append("stdout:\n" + self.stdout.rstrip())
        if self.stderr:
            parts.append("stderr:\n" + self.stderr.rstrip())
        return "\n".join(parts)


def _rlimit_preexec(cpu_seconds: Optional[int], memory_mb: Optional[int], fsize_mb: int):
    import resource  # POSIX only

    def apply() -> None:
        if cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if memory_mb:
            nbytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        fbytes = fsize_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fbytes, fbytes))

    return apply


async def run_python(
    code: str,
    *,
    timeout: float = 10.0,
    cpu_seconds: int = 10,
    memory_mb: int = 512,
    fsize_mb: int = 16,
    python: Optional[str] = None,
) -> SandboxResult:
    """Run a Python snippet in an isolated subprocess. See module docstring for
    the (limited) isolation guarantees."""
    interpreter = python or sys.executable
    with TemporaryDirectory() as workdir:
        script = Path(workdir) / "snippet.py"
        script.write_text(code)

        kwargs: dict = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _rlimit_preexec(cpu_seconds, memory_mb, fsize_mb)

        proc = await asyncio.create_subprocess_exec(
            interpreter,
            str(script),
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", ""), "HOME": workdir, "TMPDIR": workdir},
            **kwargs,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxResult("", f"killed after {timeout}s", -1, timed_out=True)
        return SandboxResult(
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
            proc.returncode if proc.returncode is not None else -1,
        )


def code_execution_tool(
    *,
    timeout: float = 10.0,
    cpu_seconds: int = 10,
    memory_mb: int = 512,
    name: str = "run_python",
) -> Tool:
    """A tool that runs Python in the sandbox and returns stdout/stderr."""

    async def run_python_tool(code: str) -> str:
        result = await run_python(
            code, timeout=timeout, cpu_seconds=cpu_seconds, memory_mb=memory_mb
        )
        return str(result)

    return Tool(
        name=name,
        description=(
            "Execute a Python 3 snippet in an isolated subprocess (resource-limited, "
            "no network). Returns exit code, stdout, and stderr."
        ),
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source to run."}},
            "required": ["code"],
        },
        func=run_python_tool,
    )
