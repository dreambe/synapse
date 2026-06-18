"""Command-line entry point: ``synapse run|serve|card``.

Agents are referenced as ``module:attribute`` or ``path/to/file.py:attribute``::

    synapse run examples.basic_agent:agent "What is 19 * 23?"
    synapse serve examples/basic_agent.py:agent --port 8080
    synapse card http://localhost:8080
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from typing import Any


def _load_object(spec: str) -> Any:
    module_part, sep, attr = spec.partition(":")
    if not sep:
        raise SystemExit(f"target must be 'module:attr' or 'file.py:attr', got {spec!r}")

    if module_part.endswith(".py"):
        mod_spec = importlib.util.spec_from_file_location("_synapse_target", module_part)
        if mod_spec is None or mod_spec.loader is None:
            raise SystemExit(f"cannot load module from {module_part!r}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_part)
        except ImportError as exc:
            raise SystemExit(f"cannot import {module_part!r}: {exc}") from exc

    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise SystemExit(f"{module_part!r} has no attribute {attr!r}") from exc


def _cmd_run(args: argparse.Namespace) -> int:
    agent = _load_object(args.target)
    result = agent.run(args.input, max_iterations=args.max_iterations)
    if args.json:
        print(json.dumps(
            {
                "output": result.output,
                "agent": result.agent,
                "iterations": result.iterations,
                "stop_reason": result.stop_reason,
            },
            indent=2,
        ))
    else:
        print(result.output)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    agent = _load_object(args.target)

    # Prefer uvicorn (native async, no thread ceiling) when available.
    try:
        import uvicorn
    except ImportError:
        from .a2a import serve

        print(
            "synapse: uvicorn not installed — using the stdlib threaded server. "
            "Install `synapse[server]` for native-async high-concurrency serving."
        )
        serve(agent, host=args.host, port=args.port)
        return 0

    from .a2a.asgi import create_app

    print(f"synapse: serving agent {agent.name!r} on http://{args.host}:{args.port} (uvicorn)")
    uvicorn.run(create_app(agent), host=args.host, port=args.port, log_level="warning")
    return 0


def _cmd_card(args: argparse.Namespace) -> int:
    from .a2a import fetch_card

    print(json.dumps(fetch_card(args.url).to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synapse", description="The synapse agent framework.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an agent on a single input")
    p_run.add_argument("target", help="agent reference, e.g. mymod:agent")
    p_run.add_argument("input", help="the task or question for the agent")
    p_run.add_argument("--max-iterations", type=int, default=12)
    p_run.add_argument("--json", action="store_true", help="emit the full result as JSON")
    p_run.set_defaults(handler=_cmd_run)

    p_serve = sub.add_parser("serve", help="serve an agent over HTTP (A2A)")
    p_serve.add_argument("target", help="agent reference, e.g. mymod:agent")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.set_defaults(handler=_cmd_serve)

    p_card = sub.add_parser("card", help="fetch a remote agent's card")
    p_card.add_argument("url", help="base URL of a served agent")
    p_card.set_defaults(handler=_cmd_card)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
