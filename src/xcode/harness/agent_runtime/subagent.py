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
from ...agent.messages import BranchSummaryMessage

"""子 Agent 任务工具。

父 Agent 只接收子 Agent 的摘要结果；子 Agent 使用独立事件循环运行，避免把
父 Agent 的 async 上下文和 Windows 线程事件循环细节混在一起。
"""

SubagentStatus = Literal["running", "done", "cancelled", "failed"]
RunChild = Callable[[str, str, Path | None], Awaitable[str]]
SubagentLifecycleCallback = Callable[["SubagentLifecycleEvent"], None]


@dataclass(frozen=True)
class SubagentStartEvent:
    """子 Agent 任务开始事件。"""

    job_id: str
    prompt: str
    model_profile: str
    isolation: str
    cwd_override: Path | None = None
    worktree_task_id: str | None = None
    type: str = "subagent_start"


@dataclass(frozen=True)
class SubagentEndEvent:
    """子 Agent 任务结束事件。"""

    job_id: str
    status: SubagentStatus
    error: str | None = None
    type: str = "subagent_end"


type SubagentLifecycleEvent = SubagentStartEvent | SubagentEndEvent


@dataclass
class SubagentJob:
    """子 Agent 任务状态。"""

    id: str
    prompt: str
    model_profile: str
    created_at: datetime
    timeout_seconds: float | None
    future: concurrent.futures.Future[str]
    isolation: str = "context"
    cwd_override: Path | None = None
    worktree_task_id: str | None = None
    end_event_emitted: bool = False

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
        lifecycle_callback: SubagentLifecycleCallback | None = None,
    ) -> None:
        self.run_child = run_child
        self.timeout_seconds = timeout_seconds
        self.available_profiles = available_profiles
        self.default_profile = default_profile
        self.worktree_runner = worktree_runner
        self.lifecycle_callback = lifecycle_callback
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
            model_profile=profile,
            created_at=datetime.now(),
            timeout_seconds=self.timeout_seconds,
            future=future,
            isolation=isolation_mode,
            cwd_override=cwd_override,
            worktree_task_id=worktree_task_id,
        )
        self._emit_start(self._jobs[job_id])
        return job_id

    def status(self, job_id: str) -> str:
        job = self._jobs.get(job_id)
        if job is None:
            return "unknown"
        return job.status()

    def result(self, job_id: str, timeout: float | None = None) -> str:
        job = self._require_job(job_id)
        try:
            result = job.future.result(timeout=timeout)
        except concurrent.futures.CancelledError:
            self._emit_end(job, "cancelled")
            raise
        except Exception as exc:
            self._emit_end(job, "failed", f"{type(exc).__name__}: {exc}")
            raise
        else:
            self._emit_end(job, "done")
            return result
        finally:
            if job.future.done():
                self._jobs.pop(job_id, None)

    def cancel(self, job_id: str) -> str:
        job = self._jobs.get(job_id)
        if job is None:
            return f"unknown job: {job_id}"
        ok = job.future.cancel()
        if ok:
            self._emit_end(job, "cancelled")
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

    def _emit_start(self, job: SubagentJob) -> None:
        if self.lifecycle_callback is None:
            return
        self.lifecycle_callback(
            SubagentStartEvent(
                job_id=job.id,
                prompt=job.prompt,
                model_profile=job.model_profile,
                isolation=job.isolation,
                cwd_override=job.cwd_override,
                worktree_task_id=job.worktree_task_id,
            )
        )

    def _emit_end(
        self,
        job: SubagentJob,
        status: SubagentStatus,
        error: str | None = None,
    ) -> None:
        if self.lifecycle_callback is None or job.end_event_emitted:
            return
        job.end_event_emitted = True
        self.lifecycle_callback(
            SubagentEndEvent(job_id=job.id, status=status, error=error)
        )


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
            raw_result = runner.result(job_id)
            job = runner._jobs.get(job_id)
            prompt = job.prompt if job else ""
            return f"status=done\n{build_branch_summary(job_id, prompt, raw_result).summary}"
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
            schema=_submit_subagent_schema(),
        ),
        ToolSpec(
            "check_subagent",
            "Check a subagent job.",
            'JSON: {"job_id":"..."}',
            check_subagent,
            group="subagent",
            schema=_job_id_schema(),
        ),
        ToolSpec(
            "cancel_subagent",
            "Cancel a subagent job.",
            'JSON: {"job_id":"..."}',
            cancel_subagent,
            group="subagent",
            schema=_job_id_schema(),
        ),
    )


def _submit_subagent_schema() -> dict[str, object]:
    """返回 submit_subagent 的参数 schema。"""
    return {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "model_profile": {"type": "string"},
            "isolation": {
                "type": "string",
                "enum": ["context", "worktree"],
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }


def _job_id_schema() -> dict[str, object]:
    """返回只接受 job_id 的子 Agent 工具参数 schema。"""
    return {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
        "additionalProperties": False,
    }


def _unknown_profile(model_profile: str, profiles) -> str:
    available = ", ".join(sorted(profiles)) or "(none)"
    return f"unknown model_profile: {model_profile}; available: {available}"


def build_branch_summary(
    job_id: str,
    prompt: str,
    result: str,
) -> BranchSummaryMessage:
    lines = result.strip().splitlines()
    summary_lines = [f"Task: {prompt[:80]}"]
    output_lines = [line for line in lines if line.strip()][:5]
    if output_lines:
        summary_lines.append("Results:")
        summary_lines.extend(f"  {line}" for line in output_lines)
    if len(lines) > 5:
        summary_lines.append(f"  (... {len(lines) - 5} more lines)")
    return BranchSummaryMessage(
        summary="\n".join(summary_lines),
        from_id=job_id,
    )


def _task_name(prompt: str) -> str:
    return prompt.strip().splitlines()[0][:40] or "subagent"
