from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from xcode.harness.execution_env import ExecutionEnv, SubprocessExecutionEnv
from xcode.harness.skills import ToolInput, ToolSpec
from .output_accumulator import OutputAccumulator
from .shell_adapter import ShellSpec, build_shell_argv, detect_shell

logger = logging.getLogger("xcode.coding_agent.tools.bash")

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120


SpawnHook = Callable[[str, Path], tuple[str, Path]]
"""Hook to adjust command and cwd before execution. Receives (command, cwd), returns (command, cwd)."""


def build_bash_tool(
    project_root: Path,
    cancel_event: threading.Event | None = None,
    shell_spec: ShellSpec | None = None,
    env: ExecutionEnv | None = None,
    command_prefix: str | None = None,
    spawn_hook: SpawnHook | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ToolSpec:
    root = project_root.resolve()
    spec = shell_spec or detect_shell()
    env = env or SubprocessExecutionEnv()

    def bash(data: ToolInput) -> str:
        command = str(data.get("command") or data.get("input") or "").strip()
        if not command:
            raise ValueError("command is required")
        if command_prefix:
            command = f"{command_prefix}\n{command}"
        if spawn_hook:
            command, cwd = spawn_hook(command, root)
        else:
            cwd = root
        timeout = _parse_timeout(data.get("timeout", DEFAULT_TIMEOUT_SECONDS))

        argv = build_shell_argv(spec, command)
        result = env.run(argv, cwd=cwd, timeout=timeout, cancel_event=cancel_event)

        acc = OutputAccumulator()
        for raw in [result.stdout.encode(), result.stderr.encode()]:
            if raw:
                acc.append(raw)

        output = acc.snapshot()
        acc.close()

        if result.timed_out:
            output += f"\nCommand timed out after {timeout}s"
        elif result.cancelled:
            output += "\nCommand cancelled"
        elif result.returncode not in (0, None):
            output = f"exit code: {result.returncode}\n{output}"

        if on_progress:
            on_progress(output)
        return output

    return ToolSpec(
        name="bash",
        description="Run a shell command in the project root.",
        input_hint='JSON: {"command": "git status --short", "timeout": 30}',
        handler=bash,
        risk="low",
        prompt_snippet="Run a shell command in the project root",
        prompt_guidelines=(
            "Bash output may be truncated; use the reported full output path when present.",
        ),
        schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 30.",
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_SECONDS,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        counts_as_progress=True,
    )


def _parse_timeout(value: Any) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be an integer") from exc
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    return timeout
