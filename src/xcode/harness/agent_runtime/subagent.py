from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
import itertools
from pathlib import Path
from typing import Literal

from ..config import PROFILE_SUBAGENT
from ..skills import ToolInput, ToolSpec
from .async_worker import IsolatedAsyncWorker

"""子 Agent 任务工具。

父 Agent 只接收子 Agent 的摘要结果；子 Agent 使用独立事件循环运行，避免把
父 Agent 的 async 上下文和 Windows 线程事件循环细节混在一起。
"""

SubagentStatus = Literal["running", "done", "cancelled", "failed"]
RunChild = Callable[[str, str, Path | None], Awaitable[str]]


@dataclass
class SubagentJob:
    id: str
    prompt: str
    created_at: datetime
    timeout_seconds: float | None
    future: concurrent.futures.Future[str]
    isolation: str = "context"
    cwd_override: Path | None = None
    worktree_task_id: str | None = None

    def status(self) -> SubagentStatus:
        if self.future.cancelled():
            return "cancelled"
        if not self.future.done():
            return "running"
        try:
            self.future.result()
        except concurrent.futures.CancelledError:
            return "cancelled"
        except Exception:
            return "failed"
        return "done"


class ManagedSubagentRunner:
    def __init__(
        self,
        run_child: RunChild,
        timeout_seconds: float | None = 120,
        available_profiles: tuple[str, ...] = (PROFILE_SUBAGENT,),
        default_profile: str = PROFILE_SUBAGENT,
        worktree_runner=None,
        worker: IsolatedAsyncWorker | None = None,
    ) -> None:
        self.run_child = run_child
        self.timeout_seconds = timeout_seconds
        self.available_profiles = available_profiles
        self.default_profile = default_profile
        self.worktree_runner = worktree_runner
        self._worker = worker or IsolatedAsyncWorker(name="xcode-subagent-worker")
        self._jobs: dict[str, SubagentJob] = {}
        self._ids = itertools.count(1)
        self._closing = False

    def submit(
        self,
        prompt: str,
        model_profile: str | None = None,
        isolation: str | None = None,
    ) -> str:
        if self._closing:
            raise RuntimeError("subagent runner is shutting down")

        profile = (model_profile or self.default_profile).strip()
        if profile not in self.available_profiles:
            raise ValueError(_unknown_profile(profile, self.available_profiles))
        isolation_mode, cwd_override, worktree_task_id = self._resolve_isolation(
            prompt, isolation
        )

        async def entry() -> str:
            coro = self.run_child(prompt, profile, cwd_override)
            if self.timeout_seconds is None:
                return await coro
            return await asyncio.wait_for(coro, timeout=self.timeout_seconds)

        job_id = f"subagent-{next(self._ids)}"
        future = self._worker.submit(entry())
        self._jobs[job_id] = SubagentJob(
            id=job_id,
            prompt=prompt,
            created_at=datetime.now(),
            timeout_seconds=self.timeout_seconds,
            future=future,
            isolation=isolation_mode,
            cwd_override=cwd_override,
            worktree_task_id=worktree_task_id,
        )
        return job_id

    def status(self, job_id: str) -> str:
        job = self._jobs.get(job_id)
        if job is None:
            return "unknown"
        return job.status()

    def result(self, job_id: str, timeout: float | None = None) -> str:
        job = self._require_job(job_id)
        try:
            return job.future.result(timeout=timeout)
        finally:
            if job.future.done():
                self._jobs.pop(job_id, None)

    def cancel(self, job_id: str) -> str:
        job = self._jobs.get(job_id)
        if job is None:
            return f"unknown job: {job_id}"
        ok = job.future.cancel()
        if job.future.done():
            self._jobs.pop(job_id, None)
        return "cancel requested" if ok else "already completed"

    def sweep_finished(self) -> None:
        finished = [job_id for job_id, job in self._jobs.items() if job.future.done()]
        for job_id in finished:
            self._jobs.pop(job_id, None)

    def shutdown(self, drain_timeout: float = 2.0) -> None:
        self._closing = True
        futures = [job.future for job in self._jobs.values() if not job.future.done()]
        for future in futures:
            future.cancel()
        if futures:
            concurrent.futures.wait(futures, timeout=drain_timeout)
        self._worker.close()
        self._jobs.clear()

    def _resolve_isolation(
        self, prompt: str, isolation: str | None
    ) -> tuple[str, Path | None, str | None]:
        isolation_mode = (isolation or "context").strip() or "context"
        if isolation_mode == "worktree":
            if self.worktree_runner is None:
                raise ValueError("worktree isolation requires the worktree tool group")
            task = self.worktree_runner.create(_task_name(prompt))
            return isolation_mode, Path(task.path).resolve(), task.id
        if isolation_mode not in ("context", "none"):
            raise ValueError(f"unknown subagent isolation: {isolation_mode}")
        return isolation_mode, None, None

    def _require_job(self, job_id: str) -> SubagentJob:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"unknown subagent job: {job_id}") from None


def build_managed_subagent_tools(runner: ManagedSubagentRunner) -> tuple[ToolSpec, ...]:
    def submit_subagent(data: ToolInput) -> str:
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")
        model_profile = str(data.get("model_profile", runner.default_profile)).strip()
        isolation = str(data.get("isolation", "context")).strip()
        try:
            job_id = runner.submit(prompt, model_profile, isolation)
        except ValueError as exc:
            return str(exc)
        return f"subagent job {job_id} submitted"

    def check_subagent(data: ToolInput) -> str:
        job_id = str(data.get("job_id", "")).strip()
        if not job_id:
            raise ValueError("job_id is required")
        status = runner.status(job_id)
        if status in ("unknown", "running", "cancelled"):
            return f"status={status}"
        if status == "failed":
            try:
                runner.result(job_id)
            except Exception as exc:
                return f"status=failed\n{type(exc).__name__}: {exc}"
        try:
            return f"status=done\n{runner.result(job_id)}"
        except KeyError as exc:
            return str(exc)

    def cancel_subagent(data: ToolInput) -> str:
        job_id = str(data.get("job_id", "")).strip()
        if not job_id:
            raise ValueError("job_id is required")
        return runner.cancel(job_id)

    return (
        ToolSpec(
            "submit_subagent",
            "Submit a subagent task. Use isolation=worktree to run from an isolated worktree cwd.",
            'JSON: {"prompt":"...", "model_profile":"subagent", "isolation":"context|worktree"}',
            submit_subagent,
            group="subagent",
        ),
        ToolSpec(
            "check_subagent",
            "Check a subagent job.",
            'JSON: {"job_id":"..."}',
            check_subagent,
            group="subagent",
        ),
        ToolSpec(
            "cancel_subagent",
            "Cancel a subagent job.",
            'JSON: {"job_id":"..."}',
            cancel_subagent,
            group="subagent",
        ),
    )


def _unknown_profile(model_profile: str, profiles) -> str:
    available = ", ".join(sorted(profiles)) or "(none)"
    return f"unknown model_profile: {model_profile}; available: {available}"


def _task_name(prompt: str) -> str:
    return prompt.strip().splitlines()[0][:40] or "subagent"
