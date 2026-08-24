from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from ._process import POLL_INTERVAL, close_pipes, kill_process, start_process
from .result import ExecutionResult
from .sandbox import CommandSandbox


class SubprocessShell:
    def __init__(self, sandbox: CommandSandbox | None = None) -> None:
        self._sandbox = sandbox

    def run(
        self,
        argv: list[str],
        cwd: Path,
        timeout: int = 30_000,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[str], None] | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        logger = __import__("logging").getLogger(__name__)
        finalize_sandbox: Callable[[], str | None] | None = None
        if self._sandbox is not None:
            command = self._sandbox.wrap(argv, cwd)
            argv = list(command.argv)
            cwd = command.cwd
            finalize_sandbox = command.finalize
        try:
            proc = start_process(argv, cwd, env=env)
        except OSError:
            if finalize_sandbox is not None:
                finalize_sandbox()
            raise
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        lock = threading.Lock()
        cancelled = False
        timed_out = False
        sandbox_violation: str | None = None

        def _drain(
            src: Iterable[bytes] | None,
            dest: list[bytes],
            is_stderr: bool = False,
        ) -> None:
            if src is None:
                return
            try:
                for raw in src:
                    with lock:
                        dest.append(raw)
                    if on_progress:
                        text = raw.decode("utf-8", errors="replace")
                        on_progress(text)
            except Exception:
                logger.debug("error draining process output", exc_info=True)

        out_thread = threading.Thread(
            target=_drain, args=(proc.stdout, stdout_chunks, False), daemon=True
        )
        err_thread = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_chunks, True), daemon=True
        )
        out_thread.start()
        err_thread.start()

        try:
            deadline = time.monotonic() + (timeout / 1000.0)
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    kill_process(proc)
                    timed_out = True
                    break
                if cancel_event is not None and cancel_event.is_set():
                    kill_process(proc)
                    cancelled = True
                    break
                time.sleep(POLL_INTERVAL)
            out_thread.join(timeout=2)
            err_thread.join(timeout=2)
        except Exception:
            kill_process(proc)
            raise
        finally:
            close_pipes(proc)
            if finalize_sandbox is not None:
                sandbox_violation = finalize_sandbox()

        with lock:
            stdout_text = b"".join(stdout_chunks).decode(errors="replace")
            stderr_text = b"".join(stderr_chunks).decode(errors="replace")
        returncode = proc.returncode
        if sandbox_violation is not None:
            if on_progress is not None:
                on_progress(f"\n{sandbox_violation}\n")
            if stderr_text and not stderr_text.endswith("\n"):
                stderr_text += "\n"
            stderr_text += sandbox_violation
            if returncode == 0:
                returncode = 126

        return ExecutionResult(
            stdout=stdout_text,
            stderr=stderr_text,
            returncode=returncode,
            timed_out=timed_out,
            cancelled=cancelled,
        )
