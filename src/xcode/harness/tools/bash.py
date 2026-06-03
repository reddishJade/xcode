from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
import tempfile
from pathlib import Path
from typing import Any

from ..skills import ToolInput, ToolSpec
from .shell_adapter import ShellSpec, build_shell_argv, detect_shell

logger = logging.getLogger("xcode.harness.tools.bash")

MAX_OUTPUT_BYTES = 50_000
MAX_OUTPUT_LINES = 2_000
DEFAULT_TIMEOUT_SECONDS = 30
POLL_INTERVAL = 0.1
TERMINATE_GRACE_SECONDS = 3


class OutputAccumulator:
    """管理 bash 输出，支持尾部截断和全量持久化到临时文件。"""

    def __init__(
        self, max_bytes: int = MAX_OUTPUT_BYTES, max_lines: int = MAX_OUTPUT_LINES
    ):
        self._max_bytes = max_bytes
        self._max_lines = max_lines
        self._chunks: list[bytes] = []
        self._total_lines = 0
        self._total_bytes = 0
        self._truncated = False
        self._full_path: str | None = None
        self._file: Any = None

    def append(self, chunk: bytes) -> None:
        """摄入原始字节块。"""
        self._chunks.append(chunk)
        self._total_bytes += len(chunk)
        if not self._truncated:
            self._total_lines += chunk.count(b"\n")
        if (
            self._total_lines > self._max_lines
            or self._total_bytes > self._max_bytes * 2
        ):
            self._persist_full()

    def _persist_full(self) -> None:
        """将完整输出写入临时文件并释放内存缓冲。"""
        if self._file is None:
            self._file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".log", prefix="xcode-bash-"
            )
            self._full_path = self._file.name
            for chunk in self._chunks:
                self._file.write(chunk)
            self._file.flush()
        else:
            for chunk in self._chunks:
                self._file.write(chunk)
            self._file.flush()
        self._truncated = True
        self._chunks = []

    def snapshot(self) -> str:
        """返回截断后的尾部文本，带截断标记。"""
        if self._chunks:
            text = b"".join(self._chunks).decode("utf-8", errors="replace")
        else:
            text = ""

        if not text and not self._truncated:
            return "(no output)"

        lines = text.splitlines()
        total_lines = len(lines)

        # 尾部截断：保留最后 max_lines 行，最后 max_bytes 字节
        truncated_by: str | None = None
        output_lines = total_lines
        if output_lines > self._max_lines:
            lines = lines[-self._max_lines :]
            truncated_by = "lines"
            output_lines = len(lines)

        output = "\n".join(lines)
        output_bytes = len(output.encode("utf-8"))
        if output_bytes > self._max_bytes:
            # 尾部字节截断
            encoded = output.encode("utf-8")
            output = encoded[-self._max_bytes :].decode("utf-8", errors="replace")
            # 找到第一个换行开始位置
            first_newline = output.find("\n")
            if first_newline > 0:
                output = output[first_newline + 1 :]
            truncated_by = "bytes"

        if self._full_path and (truncated_by or total_lines > output_lines):
            footer = (
                f"\n[Showing {output_lines} of {total_lines} lines"
                f" ({self._max_bytes // 1024}KB limit)."
                f" Full output: {self._full_path}]"
            )
            return output + footer

        return output

    def close(self) -> None:
        """清理临时文件。"""
        if self._file is not None:
            try:
                self._file.close()
                if self._full_path:
                    os.unlink(self._full_path)
            except Exception:
                logger.debug("failed to clean up temp file %s", self._full_path, exc_info=True)


def _kill_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if sys.platform != "win32":
            _kill_process_group(proc, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        if sys.platform != "win32":
            _kill_process_group(proc, signal.SIGKILL)
        else:
            _taskkill(proc)
        proc.wait()


def _kill_process_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        killpg = getattr(os, "killpg")
        getpgid = getattr(os, "getpgid")
        killpg(getpgid(proc.pid), sig)
    except ProcessLookupError:
        pass
    except OSError:
        proc.kill()


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


def build_bash_tool(
    project_root: Path,
    cancel_event: threading.Event | None = None,
    shell_spec: ShellSpec | None = None,
) -> ToolSpec:
    root = project_root.resolve()
    spec = shell_spec or detect_shell()

    def bash(data: ToolInput) -> str:
        command = str(data.get("command") or data.get("input") or "").strip()
        if not command:
            raise ValueError("command is required")
        timeout = _parse_timeout(data.get("timeout", DEFAULT_TIMEOUT_SECONDS))

        argv = build_shell_argv(spec, command)
        popen_kwargs: dict[str, Any] = {}
        if sys.platform != "win32":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            argv,
            shell=False,
            cwd=root,
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

        try:
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if time.monotonic() >= deadline:
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
        schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 30.",
                    "minimum": 1,
                    "maximum": 120,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )


def _parse_timeout(value: Any) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be an integer") from exc
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    return timeout
