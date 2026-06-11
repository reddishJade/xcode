from __future__ import annotations

import logging
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
from xcode.harness.observability.permissions import PermissionDecision
from .output_accumulator import OutputAccumulator
from .shell_adapter import ShellSpec, build_shell_argv, detect_shell

logger = logging.getLogger("xcode.coding_agent.tools.bash")

# 命令执行超时配置
DEFAULT_TIMEOUT_SECONDS = 30  # 默认超时：适配常规命令（ls/git/npm 等）
MAX_TIMEOUT_SECONDS = 120  # 最大超时：防止长时间挂起阻塞 Agent 循环


SpawnHook = Callable[[str, Path], tuple[str, Path]]
"""Hook to adjust command and cwd before execution. Receives (command, cwd), returns (command, cwd)."""


@dataclass(frozen=True)
class BashRequest:
    command: str
    timeout: int


@dataclass(frozen=True)
class BashExecutionPlan:
    command: str
    cwd: Path
    timeout: int


# 危险命令模式（拒绝执行）
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs.",
    "> /dev/sda",
    "dd if=",
    "chmod -R 777",
    "chown -R root",
]

# 高风险写操作命令前缀（需要用户确认）
HIGH_RISK_WRITE_COMMANDS = [
    "rm ",
    "rm\t",
    "mv ",
    "mv\t",
    "git reset --hard",
    "git clean -f",
    "git push --force",
    "git push -f",
]


def _bash_risk_evaluator(tool_input: dict[str, Any]) -> PermissionDecision:
    """评估 bash 命令的风险级别。

    返回：
    - "deny"  — 危险命令，直接拒绝
    - "ask"   — 高风险写操作，需要用户确认
    - "allow" — 普通命令，放行
    """
    command = str(tool_input.get("command", "")).strip().lower()

    # 1. 检查危险命令模式
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return "deny"

    # 2. 检查高风险写操作
    for prefix in HIGH_RISK_WRITE_COMMANDS:
        if command.startswith(prefix):
            return "ask"

    # 3. 普通命令放行
    return "allow"


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
        request = _parse_bash_request(data)
        plan = _build_bash_execution_plan(
            request,
            root,
            command_prefix=command_prefix,
            spawn_hook=spawn_hook,
        )
        result = env.run(
            build_shell_argv(spec, plan.command),
            cwd=plan.cwd,
            timeout=plan.timeout,
            cancel_event=cancel_event,
        )
        output = _render_bash_output(result, plan.timeout)
        if on_progress:
            on_progress(output)
        return output

    return ToolSpec(
        name="bash",
        description="Run a shell command in the project root.",
        input_hint='JSON: {"command": "git status --short", "timeout": 30}',
        handler=bash,
        risk="high",
        risk_evaluator=_bash_risk_evaluator,
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


def _parse_bash_request(data: ToolInput) -> BashRequest:
    command = str(data.get("command") or data.get("input") or "").strip()
    if not command:
        raise ValueError("command is required")
    return BashRequest(
        command=command,
        timeout=_parse_timeout(data.get("timeout", DEFAULT_TIMEOUT_SECONDS)),
    )


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
    cwd = root
    if spawn_hook:
        command, cwd = spawn_hook(command, root)
    return BashExecutionPlan(command=command, cwd=cwd, timeout=request.timeout)


def _render_bash_output(result: ExecutionResult, timeout: int) -> str:
    acc = OutputAccumulator()
    try:
        for raw in [result.stdout.encode(), result.stderr.encode()]:
            if raw:
                acc.append(raw)
        output = acc.snapshot()
    finally:
        acc.close()

    if result.timed_out:
        output += f"\nCommand timed out after {timeout}s"
    elif result.cancelled:
        output += "\nCommand cancelled"
    elif result.returncode not in (0, None):
        output = f"exit code: {result.returncode}\n{output}"
    return output


def _parse_timeout(value: Any) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be an integer") from exc
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    return timeout
