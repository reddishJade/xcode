from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

from xcode.harness.agent_runtime.async_worker import IsolatedAsyncWorker
from xcode.harness.agent_runtime import (
    ManagedSubagentRunner,
    SubagentEndEvent,
    SubagentStartEvent,
    build_managed_subagent_tools,
)
from xcode.harness.agent_runtime.subagent import SubagentBusyError
import pytest
class XcodeSubagentToolTests:
    def test_active_job_limit_returns_busy_and_releases_on_finish(self) -> None:
        """完成的 job 自动释放独立 subagent 额度。"""
        release = threading.Event()

        async def run_child(
            prompt: str, _model_profile: str, _cwd_override: Path | None
        ) -> str:
            if prompt == "hold":
                await asyncio.to_thread(release.wait)
            return prompt

        runner = ManagedSubagentRunner(
            run_child,
            timeout_seconds=1,
            max_active_jobs=1,
        )
        try:
            first = runner.submit("hold")
            assert runner.active_job_count == 1
            with pytest.raises(SubagentBusyError, match="1/1 active"):
                runner.submit("blocked")

            release.set()
            self._wait_status(runner, first, "done")
            self._wait_active_count(runner, 0)

            second = runner.submit("next")
            self._wait_status(runner, second, "done")
            assert runner.result(second) == "next"
        finally:
            runner.shutdown()

    def test_busy_tool_result_is_explicit(self) -> None:
        """submit_subagent 在额度耗尽时返回明确 busy 状态。"""
        release = threading.Event()

        async def run_child(
            _prompt: str, _model_profile: str, _cwd_override: Path | None
        ) -> str:
            await asyncio.to_thread(release.wait)
            return "done"

        runner = ManagedSubagentRunner(
            run_child,
            timeout_seconds=1,
            max_active_jobs=1,
        )
        try:
            tools = {tool.name: tool for tool in build_managed_subagent_tools(runner)}
            tools["submit_subagent"].handler({"prompt": "first"})

            output = tools["submit_subagent"].handler({"prompt": "second"})

            assert "subagent busy" in output
        finally:
            release.set()
            runner.shutdown()

    def test_timeout_releases_active_job_limit(self) -> None:
        """超时失败后可立即提交新的 subagent job。"""
        calls = 0

        async def run_child(
            _prompt: str, _model_profile: str, _cwd_override: Path | None
        ) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(999)
            return "done"

        runner = ManagedSubagentRunner(
            run_child,
            timeout_seconds=0.01,
            max_active_jobs=1,
        )
        try:
            first = runner.submit("slow")
            self._wait_status(runner, first, "failed")
            self._wait_active_count(runner, 0)

            second = runner.submit("fast")
            self._wait_status(runner, second, "done")
            assert runner.result(second) == "done"
        finally:
            runner.shutdown()

    def test_cancel_releases_active_job_limit(self) -> None:
        """取消完成后释放额度，允许后续提交。"""

        async def run_child(
            prompt: str, _model_profile: str, _cwd_override: Path | None
        ) -> str:
            if prompt == "slow":
                await asyncio.sleep(999)
            return "done"

        runner = ManagedSubagentRunner(
            run_child,
            timeout_seconds=None,
            max_active_jobs=1,
        )
        try:
            first = runner.submit("slow")
            assert runner.cancel(first) == "cancel requested"
            self._wait_active_count(runner, 0)

            second = runner.submit("fast")
            self._wait_status(runner, second, "done")
            assert runner.result(second) == "done"
        finally:
            runner.shutdown()

    def test_managed_subagent_runner_tracks_real_async_lifecycle(self) -> None:
        async def run_child(
            prompt: str, model_profile: str, cwd_override: Path | None
        ) -> str:
            await asyncio.sleep(0.05)
            return f"{model_profile}:{prompt}:{cwd_override}"

        runner = ManagedSubagentRunner(run_child, timeout_seconds=1)
        try:
            job_id = runner.submit("work")

            assert runner.status(job_id) == "running"
            self._wait_status(runner, job_id, "done")
            assert runner.result(job_id) == "subagent:work:None"
            assert runner.status(job_id) == "unknown"
        finally:
            runner.shutdown()

    def test_managed_subagent_tools_submit_and_check(self) -> None:
        async def run_child(
            prompt: str, _model_profile: str, _cwd_override: Path | None
        ) -> str:
            return f"done {prompt}"

        runner = ManagedSubagentRunner(run_child, timeout_seconds=1)
        try:
            tools = {tool.name: tool for tool in build_managed_subagent_tools(runner)}

            submitted = tools["submit_subagent"].handler({"prompt": "work"})
            job_id = submitted.split()[2]
            self._wait_status(runner, job_id, "done")
            checked = tools["check_subagent"].handler({"job_id": job_id})

            assert "status=done" in checked
            assert "done work" in checked
            assert runner.status(job_id) == "unknown"
        finally:
            runner.shutdown()

    def test_managed_subagent_runner_emits_lifecycle_events(self) -> None:
        events: list[SubagentStartEvent | SubagentEndEvent] = []

        async def run_child(
            prompt: str, model_profile: str, cwd_override: Path | None
        ) -> str:
            return f"{model_profile}:{prompt}:{cwd_override}"

        runner = ManagedSubagentRunner(
            run_child,
            timeout_seconds=1,
            lifecycle_callback=events.append,
        )
        try:
            job_id = runner.submit("work")
            self._wait_status(runner, job_id, "done")

            assert runner.result(job_id) == "subagent:work:None"

            assert [event.type for event in events] == ["subagent_start", "subagent_end"]
            start = events[0]
            end = events[1]
            assert isinstance(start, SubagentStartEvent)
            assert isinstance(end, SubagentEndEvent)
            assert isinstance(start, SubagentStartEvent)
            assert isinstance(end, SubagentEndEvent)
            assert start.job_id == job_id
            assert start.model_profile == "subagent"
            assert start.isolation == "context"
            assert end.job_id == job_id
            assert end.status == "done"
        finally:
            runner.shutdown()

    def test_managed_subagent_runner_passes_model_profile(self) -> None:
        seen: list[tuple[str, str]] = []

        async def run_child(
            prompt: str, model_profile: str, _cwd_override: Path | None
        ) -> str:
            seen.append((prompt, model_profile))
            return f"{model_profile}:{prompt}"

        runner = ManagedSubagentRunner(
            run_child,
            timeout_seconds=1,
            available_profiles=("main", "subagent"),
        )
        try:
            job_id = runner.submit("work", "main")
            self._wait_status(runner, job_id, "done")

            assert runner.result(job_id) == "main:work"
            assert seen == [("work", "main")]
        finally:
            runner.shutdown()

    def test_managed_subagent_tools_reject_unknown_profile(self) -> None:
        async def run_child(
            prompt: str, model_profile: str, _cwd_override: Path | None
        ) -> str:
            return f"{model_profile}:{prompt}"

        runner = ManagedSubagentRunner(
            run_child,
            timeout_seconds=1,
            available_profiles=("subagent",),
        )
        try:
            tools = {tool.name: tool for tool in build_managed_subagent_tools(runner)}

            output = tools["submit_subagent"].handler(
                {"prompt": "work", "model_profile": "missing"}
            )

            assert "unknown model_profile: missing" in output
        finally:
            runner.shutdown()

    def test_worktree_isolation_passes_cwd_override_to_child(self) -> None:
        seen: list[tuple[str, str, Path | None]] = []

        class FakeWorktreeRunner:
            def create(self, name: str):
                return type(
                    "Task",
                    (),
                    {
                        "id": "wt123",
                        "path": Path("sandbox").resolve(),
                        "branch": f"xcode/{name}",
                    },
                )()

        async def run_child(
            prompt: str, model_profile: str, cwd_override: Path | None
        ) -> str:
            seen.append((prompt, model_profile, cwd_override))
            return "done"

        runner = ManagedSubagentRunner(
            run_child,
            timeout_seconds=1,
            available_profiles=("subagent",),
            worktree_runner=FakeWorktreeRunner(),
        )
        try:
            job_id = runner.submit("work on files", isolation="worktree")
            self._wait_status(runner, job_id, "done")

            assert runner.result(job_id) == "done"
            assert seen == [("work on files", "subagent", Path("sandbox").resolve())]
        finally:
            runner.shutdown()

    def test_timeout_cancels_child_coroutine(self) -> None:
        cancelled = []

        async def run_child(
            _prompt: str, _model_profile: str, _cwd_override: Path | None
        ) -> str:
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return "never"

        runner = ManagedSubagentRunner(run_child, timeout_seconds=0.01)
        try:
            job_id = runner.submit("slow")
            self._wait_status(runner, job_id, "failed")

            with pytest.raises(TimeoutError):
                runner.result(job_id)
            assert cancelled == [True]
        finally:
            runner.shutdown()

    def test_shutdown_cancels_running_job_without_hanging(self) -> None:
        async def run_child(
            _prompt: str, _model_profile: str, _cwd_override: Path | None
        ) -> str:
            await asyncio.sleep(999)
            return "never"

        runner = ManagedSubagentRunner(run_child, timeout_seconds=None)
        job_id = runner.submit("slow")
        assert runner.status(job_id) == "running"

        started = time.perf_counter()
        runner.shutdown(drain_timeout=0.1)

        assert time.perf_counter() - started < 2.0
        assert runner.status(job_id) == "unknown"
        assert runner.active_job_count == 0

    def test_windows_worker_uses_selector_event_loop(self) -> None:
        if not hasattr(asyncio, "SelectorEventLoop"):
            pytest.skip("SelectorEventLoop is unavailable")

        worker = IsolatedAsyncWorker()
        with (
            patch.object(sys, "platform", "win32"),
            patch(
                "xcode.harness.agent_runtime.async_worker.asyncio.SelectorEventLoop",
                side_effect=asyncio.new_event_loop,
            ) as selector_loop,
        ):
            try:
                future = worker.submit(_immediate("ok"))
                assert future.result(timeout=1) == "ok"
            finally:
                worker.close()

        assert selector_loop.called

    def _wait_status(
        self, runner: ManagedSubagentRunner, job_id: str, expected: str
    ) -> None:
        for _ in range(100):
            if runner.status(job_id) == expected:
                return
            time.sleep(0.01)
        pytest.fail(f"subagent did not reach {expected}: {runner.status(job_id)}")

    def _wait_active_count(self, runner: ManagedSubagentRunner, expected: int) -> None:
        """等待 active job 额度释放。"""
        for _ in range(100):
            if runner.active_job_count == expected:
                return
            time.sleep(0.01)
        pytest.fail(
            f"active subagent count did not reach {expected}: {runner.active_job_count}"
        )

async def _immediate(value: str) -> str:
    return value

if __name__ == "__main__":
    pytest.main()
