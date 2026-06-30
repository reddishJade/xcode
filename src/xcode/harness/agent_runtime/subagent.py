from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
import itertools
from pathlib import Path
import threading
from typing import Literal

from ..config import PROFILE_SUBAGENT
from ..skills import ToolInput, ToolOutput, ToolSpec
from .async_worker import IsolatedAsyncWorker

"""子 Agent 委派工具。

模型只看到一个 ``delegate_task`` 工具。工具执行期间同步等待子 Agent 完成，
并把子 Agent 的进度通过 tool update 流给终端；最终只把完成摘要返回给父模型。
"""

SubagentIsolation = Literal["context", "worktree"]
SubagentStatus = Literal["running", "done", "failed"]
SubagentUpdate = Callable[[str], None]
RunChild = Callable[[str, str, Path | None, SubagentUpdate | None], Awaitable[str]]


class SubagentBusyError(RuntimeError):
    """active subagent 额度已满。"""


@dataclass(frozen=True)
class DelegatedTaskResult:
    """一次委派运行的最终结果。"""

    run_id: str
    prompt: str
    model_profile: str
    isolation: SubagentIsolation
    status: SubagentStatus
    answer: str
    started_at: datetime
    finished_at: datetime
    cwd_override: Path | None = None
    worktree_task_id: str | None = None
    error: str | None = None


class DelegatedTaskRunner:
    """执行一次子 Agent 委派，并负责并发额度与 worktree 隔离。"""

    def __init__(
        self,
        run_child: RunChild,
        timeout_seconds: float | None = 120,
        available_profiles: tuple[str, ...] = (PROFILE_SUBAGENT,),
        default_profile: str = PROFILE_SUBAGENT,
        worktree_runner=None,
        worker: IsolatedAsyncWorker | None = None,
        max_active_jobs: int = 4,
    ) -> None:
        self.run_child = run_child
        self.timeout_seconds = timeout_seconds
        self.available_profiles = available_profiles
        self.default_profile = default_profile
        self.worktree_runner = worktree_runner
        self.max_active_jobs = max(1, max_active_jobs)
        self._worker = worker or IsolatedAsyncWorker(name="xcode-subagent-worker")
        self._active_run_ids: set[str] = set()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._closing = False

    @property
    def active_job_count(self) -> int:
        """返回当前占用 subagent 执行额度的运行数。"""
        with self._lock:
            return len(self._active_run_ids)

    def delegate(
        self,
        prompt: str,
        *,
        model_profile: str | None = None,
        isolation: str | None = None,
        on_update: SubagentUpdate | None = None,
    ) -> DelegatedTaskResult:
        if self._closing:
            raise RuntimeError("subagent runner is shutting down")
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt is required")
        profile = (model_profile or self.default_profile).strip()
        if profile not in self.available_profiles:
            raise ValueError(_unknown_profile(profile, self.available_profiles))

        run_id = self._reserve_run_id()
        started_at = datetime.now()
        try:
            isolation_mode, cwd_override, worktree_task_id = self._resolve_isolation(
                clean_prompt, isolation
            )
            self._emit(
                on_update,
                _format_update(
                    run_id,
                    "started",
                    f"profile={profile} isolation={isolation_mode}",
                ),
            )
            answer = self._run_child_blocking(
                clean_prompt,
                profile,
                cwd_override,
                lambda text: self._emit(
                    on_update, _format_update(run_id, "event", text)
                ),
            )
        except Exception as exc:
            finished_at = datetime.now()
            self._emit(
                on_update,
                _format_update(run_id, "failed", f"{type(exc).__name__}: {exc}"),
            )
            return DelegatedTaskResult(
                run_id=run_id,
                prompt=clean_prompt,
                model_profile=profile,
                isolation=_coerce_isolation(isolation),
                status="failed",
                answer="",
                started_at=started_at,
                finished_at=finished_at,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._release_run_id(run_id)

        finished_at = datetime.now()
        self._emit(on_update, _format_update(run_id, "done", "completed"))
        return DelegatedTaskResult(
            run_id=run_id,
            prompt=clean_prompt,
            model_profile=profile,
            isolation=isolation_mode,
            status="done",
            answer=answer,
            started_at=started_at,
            finished_at=finished_at,
            cwd_override=cwd_override,
            worktree_task_id=worktree_task_id,
        )

    def shutdown(self) -> None:
        self._closing = True
        self._worker.close()
        with self._lock:
            self._active_run_ids.clear()

    def _reserve_run_id(self) -> str:
        with self._lock:
            if len(self._active_run_ids) >= self.max_active_jobs:
                raise SubagentBusyError(
                    "subagent busy: "
                    f"{len(self._active_run_ids)}/{self.max_active_jobs} active"
                )
            run_id = f"subagent-{next(self._ids)}"
            self._active_run_ids.add(run_id)
            return run_id

    def _release_run_id(self, run_id: str) -> None:
        with self._lock:
            self._active_run_ids.discard(run_id)

    def _run_child_blocking(
        self,
        prompt: str,
        profile: str,
        cwd_override: Path | None,
        on_update: SubagentUpdate | None,
    ) -> str:
        async def entry() -> str:
            coro = self.run_child(prompt, profile, cwd_override, on_update)
            if self.timeout_seconds is None:
                return await coro
            return await asyncio.wait_for(coro, timeout=self.timeout_seconds)

        future = self._worker.submit(entry())
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            raise RuntimeError("subagent run was cancelled") from None

    def _resolve_isolation(
        self, prompt: str, isolation: str | None
    ) -> tuple[SubagentIsolation, Path | None, str | None]:
        isolation_mode = _coerce_isolation(isolation)
        if isolation_mode == "worktree":
            if self.worktree_runner is None:
                raise ValueError("worktree isolation requires the worktree tool group")
            task = self.worktree_runner.create(_task_name(prompt))
            return isolation_mode, Path(task.path).resolve(), task.id
        return isolation_mode, None, None

    def _emit(self, on_update: SubagentUpdate | None, text: str) -> None:
        if on_update is not None:
            on_update(text)


def build_delegate_task_tools(runner: DelegatedTaskRunner) -> tuple[ToolSpec, ...]:
    def delegate_task(data: ToolInput, on_update: Callable[[str], None] | None) -> str:
        result = runner.delegate(
            str(data.get("prompt", "")).strip(),
            model_profile=str(
                data.get("model_profile", runner.default_profile)
            ).strip(),
            isolation=str(data.get("isolation", "context")).strip(),
            on_update=on_update,
        )
        return _render_delegate_result(result)

    return (
        ToolSpec(
            "delegate_task",
            (
                "Delegate a complete task to a subagent and wait for the final "
                "result. Progress streams to the user; only the final result is "
                "returned to the parent model. Do not poll for status."
            ),
            'JSON: {"description":"short label","prompt":"...", "model_profile":"subagent", "isolation":"context|worktree"}',
            lambda data: delegate_task(data, None),
            group="subagent",
            counts_as_progress=True,
            schema=_delegate_task_schema(),
            streaming_handler=delegate_task,
            prompt_guidelines=(
                "Use delegate_task for substantial independent investigation or implementation tasks.",
                "Do not poll delegated tasks; delegate_task waits for completion and streams progress to the user.",
                "Give the subagent a complete prompt with scope, expected output, and relevant files.",
            ),
        ),
    )


def _delegate_task_schema() -> dict[str, object]:
    """返回 delegate_task 的参数 schema。"""
    return {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short 3-7 word label for the delegated task.",
            },
            "prompt": {
                "type": "string",
                "description": "Complete task prompt for the subagent.",
            },
            "model_profile": {"type": "string"},
            "isolation": {
                "type": "string",
                "enum": ["context", "worktree"],
            },
        },
        "required": ["description", "prompt"],
        "additionalProperties": False,
    }


def _render_delegate_result(result: DelegatedTaskResult) -> str:
    if result.status == "failed":
        return ToolOutput(
            f'<task id="{result.run_id}" state="error">\n'
            f"<summary>Delegated task failed</summary>\n"
            f"<task_error>{result.error or 'unknown error'}</task_error>\n"
            "</task>",
            is_error=True,
        )
    answer = result.answer.strip() or "(no output)"
    lines = [
        f'<task id="{result.run_id}" state="completed">',
        f"<summary>Delegated task completed: {result.prompt[:80]}</summary>",
    ]
    if result.worktree_task_id:
        lines.append(f"<worktree>{result.worktree_task_id}</worktree>")
    lines.extend(["<task_result>", answer, "</task_result>", "</task>"])
    return "\n".join(lines)


def _format_update(run_id: str, status: str, message: str) -> str:
    clean = " ".join(message.strip().split())
    return f"[{run_id}] {status}: {clean}" if clean else f"[{run_id}] {status}"


def _coerce_isolation(isolation: str | None) -> SubagentIsolation:
    value = (isolation or "context").strip() or "context"
    if value not in ("context", "worktree"):
        raise ValueError(f"unknown subagent isolation: {value}")
    return value


def _unknown_profile(model_profile: str, profiles: tuple[str, ...]) -> str:
    available = ", ".join(sorted(profiles)) or "(none)"
    return f"unknown model_profile: {model_profile}; available: {available}"


def _task_name(prompt: str) -> str:
    return prompt.strip().splitlines()[0][:40] or "subagent"
