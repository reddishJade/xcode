"""以进程退出状态判定任务成功。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import time

from benchmarks.models import CommandSpec

_MAX_CAPTURE_CHARS = 8_000


@dataclass(frozen=True)
class CommandOutcome:
    """一次验证命令的可序列化结果。"""

    argv: tuple[str, ...]
    passed: bool
    returncode: int | None
    duration_seconds: float
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_command(spec: CommandSpec, workspace: Path) -> CommandOutcome:
    """在任务工作区执行无 shell 的验证命令。"""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(spec.argv),
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandOutcome(
            argv=spec.argv,
            passed=False,
            returncode=None,
            duration_seconds=time.perf_counter() - started,
            stdout=_tail(_as_text(exc.stdout)),
            stderr=_tail(_as_text(exc.stderr)),
            timed_out=True,
        )
    return CommandOutcome(
        argv=spec.argv,
        passed=completed.returncode == 0,
        returncode=completed.returncode,
        duration_seconds=time.perf_counter() - started,
        stdout=_tail(completed.stdout),
        stderr=_tail(completed.stderr),
    )


def _tail(value: str) -> str:
    return value if len(value) <= _MAX_CAPTURE_CHARS else value[-_MAX_CAPTURE_CHARS:]


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
