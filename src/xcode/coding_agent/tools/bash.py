"""bash 工具：命令执行、workdir、timeout_ms、流式输出。"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xcode.harness.execution_env import (
    ExecutionEnv,
    ExecutionResult,
    SubprocessExecutionEnv,
)
from xcode.harness.skills import ToolInput, ToolSpec
from .cygpath import to_windows as cygpath_to_windows
from .output_accumulator import OutputAccumulator
from .shell_adapter import ShellSpec, build_shell_argv, detect_shell
from ._constants import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
)

logger = logging.getLogger("xcode.coding_agent.tools.bash")


SpawnHook = Callable[[str, Path], tuple[str, Path]]
"""修改 command 和 cwd 的钩子。输入 (command, cwd)，返回 (command, cwd)。"""

SpawnEnvHook = Callable[[str, Path, dict[str, str]], dict[str, str]]
"""修改环境变量的钩子。输入 (command, cwd, env)，返回 env。"""


@dataclass(frozen=True)
class BashRequest:
    command: str
    timeout: int  # 毫秒
    workdir: str | None = None


@dataclass(frozen=True)
class BashExecutionPlan:
    command: str
    cwd: Path
    timeout: int  # 毫秒


# 最大 timeout_ms 值（300 秒）
_MAX_TIMEOUT_MS = 300_000


def build_bash_tool(
    project_root: Path,
    cancel_event: threading.Event | None = None,
    shell_spec: ShellSpec | None = None,
    env: ExecutionEnv | None = None,
    command_prefix: str | None = None,
    spawn_hook: SpawnHook | None = None,
    env_hook: SpawnEnvHook | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ToolSpec:
    root = project_root.resolve()
    spec = shell_spec or detect_shell()
    env = env or SubprocessExecutionEnv()
    shell_syntax = spec.syntax

    def bash(data: ToolInput) -> str:
        request = _parse_bash_request(data)
        plan = _build_bash_execution_plan(
            request,
            root,
            command_prefix=command_prefix,
            spawn_hook=spawn_hook,
        )
        # 计算最终环境变量
        cmd_env: dict[str, str] | None = None
        if env_hook:
            cmd_env = env_hook(plan.command, plan.cwd, dict(**os.environ))

        acc = OutputAccumulator(on_progress=on_progress)
        try:
            result = env.run(
                build_shell_argv(spec, plan.command),
                cwd=plan.cwd,
                timeout=plan.timeout,
                cancel_event=cancel_event,
                env=cmd_env,
                on_progress=lambda chunk: acc.append(
                    chunk.encode("utf-8", errors="replace")
                ),
            )
        finally:
            pass

        # 如果运行环境未通过 on_progress 推送输出（如 mock），
        # 从返回结果中回填 accumulator
        if acc.total_bytes == 0 and (result.stdout or result.stderr):
            for raw in [result.stdout.encode(), result.stderr.encode()]:
                if raw:
                    acc.append(raw)

        output = _render_bash_output(result, acc, plan.timeout)
        return output

    return ToolSpec(
        name="bash",
        description="Run a shell command.",
        input_hint='JSON: {"command": "git status --short", "timeout_ms": 30000}',
        handler=bash,
        prompt_snippet=_build_prompt_snippet(spec, shell_syntax),
        prompt_guidelines=_build_prompt_guidelines(spec, shell_syntax, root),
        schema=_build_schema(shell_syntax),
        counts_as_progress=True,
    )


# ── Schema ──


def _build_schema(shell_syntax: str) -> dict[str, Any]:
    """按 shell 类型生成参数 Schema。"""
    props: dict[str, Any] = {
        "command": {
            "type": "string",
            "description": "Shell command to run.",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds. Default 30. (deprecated, use timeout_ms)",
            "minimum": 1,
            "maximum": MAX_TIMEOUT_SECONDS,
        },
        "timeout_ms": {
            "type": "integer",
            "description": "Timeout in milliseconds. Default 30000. Max 300000.",
            "minimum": 1,
            "maximum": _MAX_TIMEOUT_MS,
        },
        "workdir": {
            "type": "string",
            "description": (
                "Working directory relative to project root."
                " Defaults to project root."
            ),
        },
    }
    return {
        "type": "object",
        "properties": props,
        "required": ["command"],
        "additionalProperties": False,
    }


# ── Prompt ──


def _build_prompt_snippet(spec: ShellSpec, syntax: str) -> str:
    """按 shell 类型生成 prompt 摘要。"""
    name = spec.name
    if syntax == "powershell":
        return f"Run a PowerShell ({name}) command in the project root."
    if syntax == "cmd":
        return "Run a cmd.exe command in the project root."
    return f"Run a shell ({name}) command in the project root."


def _build_prompt_guidelines(
    spec: ShellSpec,
    syntax: str,
    root: Path,
) -> tuple[str, ...]:
    """按 shell 类型生成详细使用指引。"""
    guidelines: list[str] = [
        "Bash output may be truncated; use the reported full output path when present.",
    ]

    if syntax == "powershell":
        ps_ver = spec.ps_kind or "powershell"
        if ps_ver == "powershell":
            # PowerShell 5.1 不支持 &&
            guidelines.extend([
                "PowerShell 5.1 detected: use `;` to chain commands (`&&` is NOT supported).",
                "Example: `cd src; dir` or `Set-Location src; Get-ChildItem`",
            ])
        else:
            # pwsh 7+
            guidelines.extend([
                "PowerShell 7+ detected: use `&&` or `;` to chain commands.",
                "Example: `cd src && dir` or `cd src; dir`",
            ])
        guidelines.extend([
            "Use `|` for pipelines (Out-File, Select-Object, etc.).",
            "Quote paths with spaces using single quotes (').",
            "Use `cd` to change directory before running commands.",
            f"Common file commands: Get-Content (read), Set-Content (write), "
            f"Remove-Item (del), Copy-Item, Move-Item, New-Item, Out-File",
        ])
    elif syntax == "cmd":
        guidelines.extend([
            "Use `&&` or `&` to chain commands in cmd.exe.",
            "Example: `cd src && dir`",
            "Use `cd /d` to change drives (e.g. `cd /d D:/projects`).",
            f"Common file commands: type (read), copy (read+write), "
            f"del/erase (delete), move/ren (write), mkdir (write), dir (list)",
        ])
    else:
        guidelines.extend([
            "Chain commands with `&&` (stop on error) or `;` (always continue).",
            "Example: `cd src && cat file.txt`",
            "Use `cd` to change directory before running commands.",
            "Use single quotes (') for paths with spaces in bash.",
            f"Common file commands: cat (read), cp (read+write), mv (write), "
            f"rm (delete), grep/rg (search), curl/wget (download), tar (archive)",
        ])

    # workdir 指引
    guidelines.append(
        "Use the `workdir` parameter to run commands in a subdirectory."
        f" Project root: {root.as_posix()}"
    )

    # 运行时变量
    import sys as _sys
    import tempfile as _tempfile

    guidelines.append(
        f"If not specified, commands time out after "
        f"{DEFAULT_TIMEOUT_SECONDS * 1000}ms."
    )
    guidelines.append(
        f"OS: {_sys.platform}, Shell: {spec.name}, "
        f"Temp dir: {_tempfile.gettempdir()}"
    )

    return tuple(guidelines)


# ── 请求解析 ──


def _parse_bash_request(data: ToolInput) -> BashRequest:
    command = str(data.get("command") or data.get("input") or "").strip()
    if not command:
        raise ValueError("command is required")
    return BashRequest(
        command=command,
        timeout=_parse_timeout(data),
        workdir=_parse_workdir(data),
    )


def _parse_workdir(data: ToolInput) -> str | None:
    raw = data.get("workdir")
    if raw is None:
        return None
    workdir = str(raw).strip()
    return workdir if workdir else None


def _parse_timeout(data: ToolInput) -> int:
    """解析 timeout（秒级向下兼容）和 timeout_ms（毫秒级，优先）。

    优先级：timeout_ms > timeout > DEFAULT_TIMEOUT_SECONDS
    """
    timeout_ms = data.get("timeout_ms")
    if timeout_ms is not None:
        try:
            value = int(timeout_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_ms must be an integer") from exc
        if value <= 0:
            raise ValueError("timeout_ms must be positive")
        if value > _MAX_TIMEOUT_MS:
            raise ValueError(f"timeout_ms must be <= {_MAX_TIMEOUT_MS}")
        return value

    timeout = data.get("timeout", DEFAULT_TIMEOUT_SECONDS)
    try:
        value = int(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be an integer") from exc
    if value <= 0:
        raise ValueError("timeout must be positive")
    # 秒 → 毫秒转换
    return value * 1000


# ── 执行计划 ──


def _build_bash_execution_plan(
    request: BashRequest,
    root: Path,
    *,
    command_prefix: str | None,
    spawn_hook: SpawnHook | None,
) -> BashExecutionPlan:
    command = request.command
    if command_prefix:
        command = f"{command_prefix}\n{command}"
    cwd = _resolve_workdir(root, request.workdir)
    if spawn_hook:
        command, cwd = spawn_hook(command, cwd)
    return BashExecutionPlan(command=command, cwd=cwd, timeout=request.timeout)


def _resolve_workdir(root: Path, workdir: str | None) -> Path:
    """安全解析 workdir。"""
    if workdir is None:
        return root
    # 安全解析（禁止绝对路径、.. 逃逸）
    from xcode.harness.skills import resolve_project_path

    try:
        return resolve_project_path(root, workdir)
    except ValueError:
        logger.warning("invalid workdir %r, falling back to project root", workdir)
        return root


# ── 输出渲染 ──


def _render_bash_output(
    result: ExecutionResult,
    acc: OutputAccumulator,
    timeout_ms: int,
) -> str:
    try:
        output = acc.snapshot()
    finally:
        acc.close()

    if result.timed_out:
        output += f"\nCommand timed out after {timeout_ms}ms"
    elif result.cancelled:
        output += "\nCommand cancelled"
    elif result.returncode not in (0, None):
        output = f"exit code: {result.returncode}\n{output}"
    return output
