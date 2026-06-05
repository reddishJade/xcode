from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from xcode.harness.skills import ToolInput, ToolSpec
from .output_accumulator import OutputAccumulator
from .shell_adapter import ShellSpec, build_shell_argv, detect_shell

logger = logging.getLogger("xcode.coding_agent.tools.bash")

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120
POLL_INTERVAL = 0.1
TERMINATE_GRACE_SECONDS = 3


def _kill_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=TERMINATE_GRACE_SECONDS)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        if sys.platform != "win32":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            _taskkill(proc)
        proc.wait()


def _taskkill(proc: subprocess.Popen) -> None:
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
        capture_output=True,
        timeout=5,
    )


def _close_pipes(proc: subprocess.Popen) -> None:
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                logger.debug("failed to close process pipe", exc_info=True)


class BashOperations(Protocol):
    def exec_command(
        self,
        argv: list[str],
        cwd: Path,
        timeout: int,
        cancel_event: threading.Event | None,
    ) -> str: ...


SpawnHook = Callable[[str, Path], tuple[str, Path]]
"""Hook to adjust command and cwd before execution. Receives (command, cwd), returns (command, cwd)."""


def build_bash_tool(
    project_root: Path,
    cancel_event: threading.Event | None = None,
    shell_spec: ShellSpec | None = None,
    bash_ops: BashOperations | None = None,
    command_prefix: str | None = None,
    spawn_hook: SpawnHook | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ToolSpec:
    root = project_root.resolve()
    spec = shell_spec or detect_shell()

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
        popen_kwargs: dict[str, Any] = {}
        if sys.platform != "win32":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            argv,
            shell=False,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
        acc = OutputAccumulator()
        cancelled = False
        timed_out = False

        def _drain(pipe):
            try:
                for raw in pipe:
                    acc.append(raw)
            except Exception:
                logger.debug("error draining process output", exc_info=True)

        out_thread = threading.Thread(target=_drain, args=(proc.stdout,), daemon=True)
        err_thread = threading.Thread(target=_drain, args=(proc.stderr,), daemon=True)
        out_thread.start()
        err_thread.start()
        last_progress = 0.0

        try:
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                now = time.monotonic()
                if on_progress and now - last_progress >= 0.5:
                    on_progress(acc.snapshot())
                    last_progress = now
                if now >= deadline:
                    _kill_process(proc)
                    timed_out = True
                    break
                if cancel_event is not None and cancel_event.is_set():
                    _kill_process(proc)
                    cancelled = True
                    break
                time.sleep(POLL_INTERVAL)
            out_thread.join(timeout=2)
            err_thread.join(timeout=2)
        except Exception:
            _kill_process(proc)
            raise
        finally:
            _close_pipes(proc)

        output = acc.snapshot()
        acc.close()

        prefixed = False
        if timed_out:
            output += f"\nCommand timed out after {timeout}s"
            prefixed = True
        if cancelled:
            output += "\nCommand cancelled"
            prefixed = True
        if not prefixed and proc.returncode not in (0, None):
            output = f"exit code: {proc.returncode}\n{output}"
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
