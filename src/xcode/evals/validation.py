from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

from .schema import EvalTask, GraderResult


@dataclass(frozen=True)
class ValidationCommandResult:
    """单条验证命令的执行结果。"""

    command: str
    passed: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


def run_validation(
    task: EvalTask,
    project_root: Path | None,
) -> tuple[tuple[GraderResult, ...], tuple[ValidationCommandResult, ...]]:
    """执行 task 声明的验证命令并返回 grader。"""
    specs = validation_commands(task)
    if not specs:
        return (), ()
    if project_root is None:
        return (
            (
                GraderResult(
                    name="validation:project_root",
                    passed=False,
                    details="project_root is unavailable",
                ),
            ),
            (),
        )
    timeout_seconds = validation_timeout_seconds(task)
    results = tuple(
        _run_command(spec, cwd=project_root, timeout_seconds=timeout_seconds)
        for spec in specs
    )
    graders = tuple(
        GraderResult(
            name=f"validation_command:{index}",
            passed=result.passed,
            details="" if result.passed else _validation_failure_details(result),
        )
        for index, result in enumerate(results, start=1)
    )
    return graders, results


def validation_commands(task: EvalTask) -> tuple[str | tuple[str, ...], ...]:
    """读取 task.metadata.validation.commands。"""
    validation = task.metadata.validation
    return validation.commands if validation is not None else ()
def validation_timeout_seconds(task: EvalTask) -> float:
    """读取验证命令超时时间，默认 60 秒。"""
    validation = task.metadata.validation
    if validation is None:
        return 60.0
    return max(float(validation.timeout_seconds), 1.0)


def validation_results_to_dict(
    results: tuple[ValidationCommandResult, ...],
) -> list[dict[str, Any]]:
    """转换验证结果为 report 可序列化结构。"""
    return [
        {
            "command": result.command,
            "passed": result.passed,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        }
        for result in results
    ]


def _run_command(
    command: str | tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> ValidationCommandResult:
    display_command = command if isinstance(command, str) else " ".join(command)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            shell=isinstance(command, str),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ValidationCommandResult(
            command=display_command,
            passed=False,
            returncode=None,
            stdout=_coerce_output(exc.stdout),
            stderr=_coerce_output(exc.stderr),
            timed_out=True,
        )
    return ValidationCommandResult(
        command=display_command,
        passed=completed.returncode == 0,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _validation_failure_details(result: ValidationCommandResult) -> str:
    if result.timed_out:
        return f"timed out: {result.command}"
    excerpt = (result.stderr or result.stdout).strip().splitlines()
    detail = excerpt[-1] if excerpt else "no output"
    return f"exit {result.returncode}: {detail}"


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
