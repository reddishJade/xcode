"""使用 bubblewrap 建立 Agent 不可访问控制面的真实执行边界。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any

from xcode.harness.config import XcodeRuntimeConfig

from .schema import ResourceUsage, Task, Trial


class IsolationError(RuntimeError):
    """OS 隔离或隔离 worker 未能完成。"""


@dataclass(frozen=True)
class IsolatedExecution:
    """从隔离进程返回的 Agent 阶段摘要。"""

    started_at: datetime
    finished_at: datetime
    usage: ResourceUsage
    termination_reason: str
    answer: str
    error_detail: str | None
    watchdog_reason: str | None
    result: dict[str, Any]


class BubblewrapExecutor:
    """只挂载 workspace、运行时和专用输出目录。"""

    def __init__(
        self,
        *,
        xcode_source: Path,
        virtualenv: Path,
        python_runtime: Path,
        uv_executable: Path,
    ) -> None:
        self._xcode_source = xcode_source.resolve()
        self._virtualenv = virtualenv.resolve()
        self._python_runtime = python_runtime.resolve()
        # 保留启动器路径，避免宿主运行时替换 uv 版本后缓存失效的真实路径。
        self._uv_executable = uv_executable.absolute()

    def run(
        self,
        *,
        task: Task,
        trial: Trial,
        runtime_config: XcodeRuntimeConfig,
        workspace: Path,
        output: Path,
    ) -> IsolatedExecution:
        """运行 worker；runtime_config 仅经 stdin 传递，不写入 Agent 工作区。"""
        workspace = workspace.resolve()
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        request = json.dumps(
            {
                "task": task.model_dump(mode="json"),
                "trial": trial.model_dump(mode="json"),
                "runtime_config": runtime_config.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        command = self._command(workspace=workspace, output=output)
        try:
            completed = subprocess.run(
                command,
                input=request,
                check=False,
                capture_output=True,
                text=True,
                timeout=trial.budget.wall_time_seconds + 30,
            )
        except subprocess.TimeoutExpired as error:
            raise IsolationError(
                "isolated Xcode exceeded wall-clock timeout "
                f"({trial.budget.wall_time_seconds + 30}s grace included)"
            ) from error
        if completed.returncode != 0:
            raise IsolationError(
                "isolated Xcode failed "
                f"(exit={completed.returncode}): {completed.stderr[-4000:]}"
            )
        summary_path = output / "execution.json"
        trace_path = output / "trace.jsonl"
        if not summary_path.is_file() or not trace_path.is_file():
            raise IsolationError("isolated worker omitted execution or trace artifact")
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            return IsolatedExecution(
                started_at=datetime.fromisoformat(payload["started_at"]),
                finished_at=datetime.fromisoformat(payload["finished_at"]),
                usage=ResourceUsage.model_validate(payload["usage"]),
                termination_reason=str(payload["termination_reason"]),
                answer=str(payload["answer"]),
                error_detail=payload.get("error_detail"),
                watchdog_reason=payload.get("watchdog_reason"),
                result=dict(payload["result"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise IsolationError(
                f"invalid isolated execution summary: {error}"
            ) from error

    def _command(self, *, workspace: Path, output: Path) -> tuple[str, ...]:
        python = self._virtualenv / "bin/python"
        return (
            "bwrap",
            "--clearenv",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--ro-bind",
            "/etc",
            "/etc",
            "--dir",
            "/run",
            "--dir",
            "/run/systemd",
            "--ro-bind",
            "/run/systemd/resolve",
            "/run/systemd/resolve",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/runtime",
            "--ro-bind",
            str(self._xcode_source),
            "/runtime/xcode",
            "--dir",
            "/home",
            "--dir",
            "/home/dwei",
            "--dir",
            "/home/dwei/workspace",
            "--dir",
            "/home/dwei/workspace/xcode",
            "--ro-bind",
            str(self._virtualenv),
            str(self._virtualenv),
            "--dir",
            "/home/dwei/.local",
            "--dir",
            "/home/dwei/.local/share",
            "--dir",
            "/home/dwei/.local/share/uv",
            "--dir",
            "/home/dwei/.local/share/uv/python",
            "--ro-bind",
            str(self._python_runtime),
            str(self._python_runtime),
            "--dir",
            "/tools",
            "--ro-bind",
            str(self._uv_executable),
            "/tools/uv",
            "--bind",
            str(workspace),
            "/workspace",
            "--bind",
            str(output),
            "/output",
            "--dir",
            "/home/eval",
            "--setenv",
            "HOME",
            "/home/eval",
            "--setenv",
            "PYTHONPATH",
            "/runtime",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTEST_ADDOPTS",
            "-p no:cacheprovider",
            "--setenv",
            "RUFF_CACHE_DIR",
            "/tmp/ruff-cache",
            "--setenv",
            "UV_CACHE_DIR",
            "/tmp/uv-cache",
            "--setenv",
            "VIRTUAL_ENV",
            str(self._virtualenv),
            "--setenv",
            "PATH",
            f"/tools:{self._virtualenv}/bin:/usr/bin:/bin",
            "--unshare-pid",
            "--die-with-parent",
            "--chdir",
            "/workspace",
            str(python),
            "-m",
            "xcode.evals.isolated_worker",
        )
