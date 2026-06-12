from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

from xcode.agent.types import ShellCallOutputContent
from xcode.harness.execution_env import ExecutionEnv, ExecutionResult
from xcode.harness.execution_env import SubprocessExecutionEnv
from xcode.harness.observability.permissions import PermissionDecision
from xcode.harness.skills import (
    AGENT_CONTENT_BLOCKS_METADATA_KEY,
    ToolInput,
    ToolOutput,
    ToolSpec,
)

from ._constants import (
    DANGEROUS_PATTERNS,
    DEFAULT_TIMEOUT_SECONDS,
    HIGH_RISK_WRITE_COMMANDS,
    MAX_TIMEOUT_SECONDS,
)
from .shell_adapter import ShellSpec, build_shell_argv, detect_shell

DEFAULT_MAX_OUTPUT_LENGTH = 4096


def build_native_shell_tool(
    project_root: Path,
    cancel_event: threading.Event | None = None,
    shell_spec: ShellSpec | None = None,
    env: ExecutionEnv | None = None,
    skills: tuple[dict[str, str], ...] = (),
) -> ToolSpec:
    """构造 OpenAI Responses builtin shell 的本地执行桥。"""
    root = project_root.resolve()
    spec = shell_spec or detect_shell()
    execution_env = env or SubprocessExecutionEnv()

    def shell(data: ToolInput) -> ToolOutput:
        commands = _parse_commands(data)
        timeout = _parse_timeout_seconds(data.get("timeout_ms"))
        max_output_length = _parse_max_output_length(data.get("max_output_length"))
        output_items = [
            _run_command(
                command,
                spec,
                root,
                timeout,
                max_output_length,
                execution_env,
                cancel_event,
            )
            for command in commands
        ]
        text = _format_output_text(commands, output_items)
        content_block = ShellCallOutputContent(
            output=output_items,
            max_output_length=max_output_length,
        )
        return ToolOutput(
            text,
            metadata={AGENT_CONTENT_BLOCKS_METADATA_KEY: [content_block]},
        )

    return ToolSpec(
        name="shell",
        description="Run OpenAI Responses shell commands in the project root.",
        input_hint='JSON: {"commands": ["python --version"], "timeout_ms": 120000}',
        handler=shell,
        risk="high",
        risk_evaluator=_native_shell_risk_evaluator,
        group="core",
        schema={
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Shell commands to run in order.",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in milliseconds.",
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_SECONDS * 1000,
                },
                "max_output_length": {
                    "type": "integer",
                    "description": "Maximum stdout/stderr characters per stream.",
                    "minimum": 1,
                },
            },
            "required": ["commands"],
            "additionalProperties": False,
        },
        execution_mode="sequential",
        counts_as_progress=True,
        builtin=_builtin_shell_definition(skills),
    )


def _run_command(
    command: str,
    spec: ShellSpec,
    root: Path,
    timeout: int,
    max_output_length: int | None,
    execution_env: ExecutionEnv,
    cancel_event: threading.Event | None,
) -> dict[str, Any]:
    """执行单条 shell 命令并转为官方 output 片段。"""
    result = execution_env.run(
        build_shell_argv(spec, command),
        cwd=root,
        timeout=timeout,
        cancel_event=cancel_event,
    )
    return {
        "stdout": _limit_output(result.stdout, max_output_length),
        "stderr": _limit_output(result.stderr, max_output_length),
        "outcome": _outcome(result),
    }


def _parse_commands(data: ToolInput) -> list[str]:
    """解析 Responses shell action.commands。"""
    raw_commands = data.get("commands")
    if raw_commands is None:
        raw_commands = data.get("command")

    if isinstance(raw_commands, list):
        commands = [str(command).strip() for command in raw_commands]
    elif raw_commands is None:
        commands = []
    else:
        commands = [str(raw_commands).strip()]

    commands = [command for command in commands if command]
    if not commands:
        raise ValueError("commands is required")
    return commands


def _parse_timeout_seconds(value: object) -> int:
    """将官方 timeout_ms 转为本地 ExecutionEnv 秒级超时。"""
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    timeout_ms = _parse_int(value, "timeout_ms")
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")
    return min(MAX_TIMEOUT_SECONDS, max(1, math.ceil(timeout_ms / 1000)))


def _parse_max_output_length(value: object) -> int | None:
    """解析 max_output_length。"""
    if value is None:
        return None
    max_output_length = _parse_int(value, "max_output_length")
    if max_output_length <= 0:
        raise ValueError("max_output_length must be positive")
    return max_output_length


def _parse_int(value: object, field_name: str) -> int:
    """解析 JSON 标量整数。"""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
    raise ValueError(f"{field_name} must be an integer")


def _limit_output(text: str, max_output_length: int | None) -> str:
    """按官方请求长度限制裁剪输出。"""
    if max_output_length is None or len(text) <= max_output_length:
        return text
    return text[:max_output_length]


def _outcome(result: ExecutionResult) -> dict[str, Any]:
    """构造 OpenAI shell output outcome。"""
    if result.timed_out or result.cancelled:
        return {"type": "timeout"}
    return {"type": "exit", "exit_code": result.returncode}


def _format_output_text(
    commands: list[str],
    output_items: list[dict[str, Any]],
) -> str:
    """生成人类可读的 shell 执行摘要。"""
    lines: list[str] = []
    for command, item in zip(commands, output_items, strict=True):
        lines.append(f"$ {command}")
        stdout = item.get("stdout")
        stderr = item.get("stderr")
        outcome = item.get("outcome")
        if stdout:
            lines.append(str(stdout))
        if stderr:
            lines.append(str(stderr))
        lines.append(f"outcome: {outcome}")
    return "\n".join(lines)


def _builtin_shell_definition(
    skills: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    """构造 Responses builtin shell 定义。"""
    environment: dict[str, Any] = {"type": "local"}
    if skills:
        environment["skills"] = [dict(skill) for skill in skills]
    return {"type": "shell", "environment": environment}


def _native_shell_risk_evaluator(tool_input: dict[str, Any]) -> PermissionDecision:
    """复用 bash 风险规则评估 Responses shell 命令。"""
    raw_commands = tool_input.get("commands")
    if isinstance(raw_commands, list):
        command = "\n".join(str(item) for item in raw_commands).strip().lower()
    else:
        command = str(raw_commands or tool_input.get("command") or "").strip().lower()

    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return "deny"
    for prefix in HIGH_RISK_WRITE_COMMANDS:
        if command.startswith(prefix):
            return "ask"
    return "allow"
