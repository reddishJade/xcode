from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


logger = logging.getLogger(__name__)

# 进程管理常量
POLL_INTERVAL = 0.1  # 轮询间隔：平衡响应性与 CPU 占用
TERMINATE_GRACE_SECONDS = 3  # 优雅终止宽限期：允许进程清理资源


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False
    cancelled: bool = False


@runtime_checkable
class ExecutionEnv(Protocol):
    def run(
        self,
        argv: list[str],
        cwd: Path,
        timeout: int = 30,
        cancel_event: threading.Event | None = None,
    ) -> ExecutionResult: ...


class SubprocessExecutionEnv:
    def run(
        self,
        argv: list[str],
        cwd: Path,
        timeout: int = 30,
        cancel_event: threading.Event | None = None,
    ) -> ExecutionResult:
        proc = _start_process(argv, cwd)
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        lock = threading.Lock()
        cancelled = False
        timed_out = False

        def _drain(src: Iterable[bytes] | None, dest: list[bytes]) -> None:
            if src is None:
                return
            try:
                for raw in src:
                    with lock:
                        dest.append(raw)
            except Exception:
                logger.debug("error draining process output", exc_info=True)

        out_thread = threading.Thread(
            target=_drain, args=(proc.stdout, stdout_chunks), daemon=True
        )
        err_thread = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_chunks), daemon=True
        )
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

        with lock:
            stdout_text = b"".join(stdout_chunks).decode(errors="replace")
            stderr_text = b"".join(stderr_chunks).decode(errors="replace")

        return ExecutionResult(
            stdout=stdout_text,
            stderr=stderr_text,
            returncode=proc.returncode,
            timed_out=timed_out,
            cancelled=cancelled,
        )


def _start_process(argv: list[str], cwd: Path) -> subprocess.Popen[bytes]:
    if sys.platform == "win32":
        return subprocess.Popen(
            argv,
            shell=False,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    return subprocess.Popen(
        argv,
        shell=False,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _kill_process(proc: subprocess.Popen) -> None:
    """终止进程，优雅终止失败后强制杀死。

    平台差异处理：
    - Unix: 发送 SIGTERM 到进程组，优雅终止子进程树
    - Windows: 使用 terminate() 发送 CTRL_BREAK_EVENT

    两阶段终止策略：
    1. 先发送 SIGTERM，等待 TERMINATE_GRACE_SECONDS（3秒）
    2. 若进程仍未退出，升级为 SIGKILL（Unix）或 taskkill /F（Windows）

    设计原因：
    优雅终止允许进程清理资源（关闭文件、刷新缓冲），
    强制杀死确保不会无限等待挂起的进程。
    """
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
