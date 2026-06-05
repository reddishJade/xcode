from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


logger = logging.getLogger(__name__)

POLL_INTERVAL = 0.1
TERMINATE_GRACE_SECONDS = 3


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
        popen_kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            argv,
            shell=False,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        lock = threading.Lock()
        cancelled = False
        timed_out = False

        def _drain(src: Any, dest: list[bytes]) -> None:
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


class SandboxExecutionEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path, int]] = []
        self._results: list[ExecutionResult] = []
        self._result_index = 0

    def enqueue(self, result: ExecutionResult) -> None:
        self._results.append(result)

    def run(
        self,
        argv: list[str],
        cwd: Path,
        timeout: int = 30,
        cancel_event: threading.Event | None = None,
    ) -> ExecutionResult:
        self.calls.append((argv, cwd, timeout))
        if self._result_index < len(self._results):
            result = self._results[self._result_index]
            self._result_index += 1
            return result
        return ExecutionResult(
            stdout="", stderr="", returncode=0, timed_out=False, cancelled=False
        )


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
