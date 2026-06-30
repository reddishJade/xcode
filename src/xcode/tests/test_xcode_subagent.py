from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

import pytest

from xcode.harness.agent_runtime import (
    SubagentRunner,
    build_subagent_tools,
)
from xcode.harness.agent_runtime.async_worker import IsolatedAsyncWorker
from xcode.harness.agent_runtime.subagent import SubagentBusyError


class XcodeSubagentToolTests:
    def test_subagent_waits_for_result_and_streams_updates(self) -> None:
        updates: list[str] = []

        async def run_child(
            prompt: str,
            model_profile: str,
            cwd_override: Path | None,
            on_update,
        ) -> str:
            assert cwd_override is None
            if on_update is not None:
                on_update(f"tool: read_file {prompt}")
            return f"{model_profile}:{prompt}"

        runner = SubagentRunner(run_child, timeout_seconds=1)
        try:
            result = runner.delegate(
                "inspect ai layer",
                model_profile="subagent",
                on_update=updates.append,
            )

            assert result.status == "done"
            assert result.answer == "subagent:inspect ai layer"
            assert any("started" in update for update in updates)
            assert any(
                "tool: read_file inspect ai layer" in update for update in updates
            )
            assert any("done" in update for update in updates)
        finally:
            runner.shutdown()

    def test_subagent_tool_is_single_public_subagent_tool(self) -> None:
        async def run_child(
            prompt: str,
            _model_profile: str,
            _cwd_override: Path | None,
            _on_update,
        ) -> str:
            return f"done {prompt}"

        runner = SubagentRunner(run_child, timeout_seconds=1)
        try:
            tools = build_subagent_tools(runner)
            assert [tool.name for tool in tools] == ["subagent"]

            output = tools[0].streaming_handler(
                {"description": "Inspect", "prompt": "work"},
                None,
            )

            assert 'state="completed"' in output
            assert "done work" in output
            assert "status=running" not in output
        finally:
            runner.shutdown()

    def test_active_job_limit_returns_busy_and_releases_on_finish(self) -> None:
        release = threading.Event()

        async def run_child(
            prompt: str,
            _model_profile: str,
            _cwd_override: Path | None,
            _on_update,
        ) -> str:
            if prompt == "hold":
                await asyncio.to_thread(release.wait)
            return prompt

        runner = SubagentRunner(
            run_child,
            timeout_seconds=1,
            max_active_jobs=1,
        )
        errors: list[BaseException] = []

        def hold_run() -> None:
            try:
                runner.delegate("hold")
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=hold_run)
        try:
            thread.start()
            self._wait_active_count(runner, 1)
            with pytest.raises(SubagentBusyError, match="1/1 active"):
                runner.delegate("blocked")

            release.set()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert errors == []
            self._wait_active_count(runner, 0)

            assert runner.delegate("next").answer == "next"
        finally:
            release.set()
            runner.shutdown()

    def test_subagent_returns_error_result_and_releases_limit(self) -> None:
        calls = 0

        async def run_child(
            _prompt: str,
            _model_profile: str,
            _cwd_override: Path | None,
            _on_update,
        ) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(999)
            return "done"

        runner = SubagentRunner(
            run_child,
            timeout_seconds=0.01,
            max_active_jobs=1,
        )
        try:
            first = runner.delegate("slow")
            assert first.status == "failed"
            assert "TimeoutError" in (first.error or "")
            self._wait_active_count(runner, 0)

            second = runner.delegate("fast")
            assert second.status == "done"
            assert second.answer == "done"
        finally:
            runner.shutdown()

    def test_subagent_tool_marks_failed_runs_as_error(self) -> None:
        async def run_child(
            _prompt: str,
            _model_profile: str,
            _cwd_override: Path | None,
            _on_update,
        ) -> str:
            raise RuntimeError("boom")

        runner = SubagentRunner(run_child, timeout_seconds=1)
        try:
            tool = build_subagent_tools(runner)[0]

            output = tool.streaming_handler(
                {"description": "Explode", "prompt": "fail"},
                None,
            )

            assert 'state="error"' in output
            assert getattr(output, "is_error") is True
        finally:
            runner.shutdown()

    def test_subagent_passes_model_profile(self) -> None:
        seen: list[tuple[str, str]] = []

        async def run_child(
            prompt: str,
            model_profile: str,
            _cwd_override: Path | None,
            _on_update,
        ) -> str:
            seen.append((prompt, model_profile))
            return f"{model_profile}:{prompt}"

        runner = SubagentRunner(
            run_child,
            timeout_seconds=1,
            available_profiles=("main", "subagent"),
        )
        try:
            result = runner.delegate("work", model_profile="main")

            assert result.answer == "main:work"
            assert seen == [("work", "main")]
        finally:
            runner.shutdown()

    def test_subagent_rejects_unknown_profile(self) -> None:
        async def run_child(
            prompt: str,
            model_profile: str,
            _cwd_override: Path | None,
            _on_update,
        ) -> str:
            return f"{model_profile}:{prompt}"

        runner = SubagentRunner(
            run_child,
            timeout_seconds=1,
            available_profiles=("subagent",),
        )
        try:
            with pytest.raises(ValueError, match="unknown model_profile: missing"):
                runner.delegate("work", model_profile="missing")
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
            prompt: str,
            model_profile: str,
            cwd_override: Path | None,
            _on_update,
        ) -> str:
            seen.append((prompt, model_profile, cwd_override))
            return "done"

        runner = SubagentRunner(
            run_child,
            timeout_seconds=1,
            available_profiles=("subagent",),
            worktree_runner=FakeWorktreeRunner(),
        )
        try:
            result = runner.delegate("work on files", isolation="worktree")

            assert result.answer == "done"
            assert result.worktree_task_id == "wt123"
            assert seen == [("work on files", "subagent", Path("sandbox").resolve())]
        finally:
            runner.shutdown()

    def test_timeout_cancels_child_coroutine(self) -> None:
        cancelled = []

        async def run_child(
            _prompt: str,
            _model_profile: str,
            _cwd_override: Path | None,
            _on_update,
        ) -> str:
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return "never"

        runner = SubagentRunner(run_child, timeout_seconds=0.01)
        try:
            result = runner.delegate("slow")

            assert result.status == "failed"
            assert cancelled == [True]
        finally:
            runner.shutdown()

    def test_shutdown_cancels_running_job_without_hanging(self) -> None:
        async def run_child(
            _prompt: str,
            _model_profile: str,
            _cwd_override: Path | None,
            _on_update,
        ) -> str:
            await asyncio.sleep(999)
            return "never"

        runner = SubagentRunner(run_child, timeout_seconds=None)
        thread = threading.Thread(target=lambda: runner.delegate("slow"), daemon=True)
        thread.start()
        self._wait_active_count(runner, 1)

        started = time.perf_counter()
        runner.shutdown()

        assert time.perf_counter() - started < 2.0
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

    def _wait_active_count(self, runner: SubagentRunner, expected: int) -> None:
        """等待 active job 额度变化。"""
        for _ in range(100):
            if runner.active_job_count == expected:
                return
            time.sleep(0.01)
        pytest.fail(
            f"active subagent count did not reach {expected}: {runner.active_job_count}"
        )


async def _immediate(value: str) -> str:
    return value
