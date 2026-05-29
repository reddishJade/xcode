from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

from xcode.harness.agent_runtime.async_worker import IsolatedAsyncWorker
from xcode.harness.agent_runtime import (
    ManagedSubagentRunner,
    build_managed_subagent_tools,
)


class XcodeSubagentToolTests(unittest.TestCase):
    def test_managed_subagent_runner_tracks_real_async_lifecycle(self) -> None:
        async def run_child(
            prompt: str, model_profile: str, cwd_override: Path | None
        ) -> str:
            await asyncio.sleep(0.05)
            return f"{model_profile}:{prompt}:{cwd_override}"

        runner = ManagedSubagentRunner(run_child, timeout_seconds=1)
        try:
            job_id = runner.submit("work")

            self.assertEqual(runner.status(job_id), "running")
            self._wait_status(runner, job_id, "done")
            self.assertEqual(runner.result(job_id), "subagent:work:None")
            self.assertEqual(runner.status(job_id), "unknown")
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

            submitted = tools["submit_subagent"].handler('{"prompt":"work"}')
            job_id = submitted.split()[2]
            self._wait_status(runner, job_id, "done")
            checked = tools["check_subagent"].handler(f'{{"job_id":"{job_id}"}}')

            self.assertIn("status=done", checked)
            self.assertIn("done work", checked)
            self.assertEqual(runner.status(job_id), "unknown")
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

            self.assertEqual(runner.result(job_id), "main:work")
            self.assertEqual(seen, [("work", "main")])
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
                '{"prompt":"work","model_profile":"missing"}'
            )

            self.assertIn("unknown model_profile: missing", output)
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

            self.assertEqual(runner.result(job_id), "done")
            self.assertEqual(
                seen, [("work on files", "subagent", Path("sandbox").resolve())]
            )
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

            with self.assertRaises(TimeoutError):
                runner.result(job_id)
            self.assertEqual(cancelled, [True])
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
        self.assertEqual(runner.status(job_id), "running")

        started = time.perf_counter()
        runner.shutdown(drain_timeout=0.1)

        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(runner.status(job_id), "unknown")

    def test_windows_worker_uses_selector_event_loop(self) -> None:
        if not hasattr(asyncio, "SelectorEventLoop"):
            self.skipTest("SelectorEventLoop is unavailable")

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
                self.assertEqual(future.result(timeout=1), "ok")
            finally:
                worker.close()

        self.assertTrue(selector_loop.called)

    def _wait_status(
        self, runner: ManagedSubagentRunner, job_id: str, expected: str
    ) -> None:
        for _ in range(100):
            if runner.status(job_id) == expected:
                return
            time.sleep(0.01)
        self.fail(f"subagent did not reach {expected}: {runner.status(job_id)}")


async def _immediate(value: str) -> str:
    return value


if __name__ == "__main__":
    unittest.main()
