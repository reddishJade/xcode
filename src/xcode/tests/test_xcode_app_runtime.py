from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from xcode.harness.app import XcodeApp, _build_project_scoped_registry, build_app
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.config import (
    AgentRuntimeConfig,
    ObservabilityRuntimeConfig,
    PathsRuntimeConfig,
    ToolsRuntimeConfig,
    XcodeRuntimeConfig,
)
from xcode.harness.tools.shell_adapter import detect_shell
from xcode.harness.skills import ToolSpec


class XcodeAppRuntimeTests(unittest.TestCase):
    def test_app_async_ask_uses_native_async_agent(self) -> None:
        async def main():
            provider = MockProvider([])
            app = XcodeApp(agent=StructuredAgent(provider=provider, registry=()))
            return await app.aask("hello")

        import asyncio

        self.assertEqual(asyncio.run(main()), "child done")

    def test_default_tool_groups_hide_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=XcodeRuntimeConfig(),
            )

        names = {tool.name for tool in app.registry}

        self.assertIn("read_file", names)
        self.assertNotIn("lsp_diagnostics", names)
        self.assertNotIn("task", names)
        self.assertNotIn("create_worktree_task", names)
        self.assertNotIn("static_analysis", names)

    def test_default_tool_groups_do_not_construct_optional_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            with patch(
                "xcode.harness.app.build_worktree_tools", side_effect=AssertionError
            ):
                app = build_app(
                    project_root=Path(tmp),
                    runtime_config=XcodeRuntimeConfig(),
                )

        names = {tool.name for tool in app.registry}
        self.assertEqual(
            names,
            {
                "read_file",
                "write_file",
                "edit_file",
                "glob_files",
                "grep_search",
                "ls",
                "bash",
            },
        )

    def test_bash_tool_uses_agent_cancellation_event(self) -> None:
        captured = {}

        def fake_bash_tool(*_args, **kwargs):
            captured["cancel_event"] = kwargs.get("cancel_event")
            return ToolSpec("bash", "Run shell.", "command", lambda _value: "ok")

        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            with patch(
                "xcode.harness.tools.build_bash_tool", side_effect=fake_bash_tool
            ):
                app = build_app(
                    project_root=Path(tmp),
                    runtime_config=XcodeRuntimeConfig(),
                )

        self.assertIs(captured["cancel_event"], app.agent.cancellation_token.event)

    def test_enabling_single_optional_group_adds_only_that_group(self) -> None:
        runtime_config = XcodeRuntimeConfig(
            tools=ToolsRuntimeConfig(enabled_groups=("core", "worktree")),
        )
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=runtime_config,
            )

        names = {tool.name for tool in app.registry}
        self.assertIn("create_worktree_task", names)

    def test_build_app_consumes_runtime_config_defaults(self) -> None:
        runtime_config = XcodeRuntimeConfig(
            agent=AgentRuntimeConfig(max_steps=7, tool_workers=2),
            paths=PathsRuntimeConfig(sessions_dir=Path("sessions")),
            observability=ObservabilityRuntimeConfig(audit_path=Path("audit.jsonl")),
        )
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            root = Path(tmp)
            app = build_app(
                project_root=root,
                runtime_config=runtime_config,
            )

        self.assertEqual(app.agent.config.max_steps, 7)
        self.assertEqual(app.agent.config.tool_workers, 2)
        self.assertIsNotNone(app.agent.audit_logger)

    def test_build_app_discovers_project_root_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            root = Path(tmp)
            (root / "xcode.config.json").write_text(
                '{"agent":{"max_steps":6,"tool_workers":1}}',
                encoding="utf-8",
            )

            app = build_app(project_root=root)

        self.assertEqual(app.agent.config.max_steps, 6)
        self.assertEqual(app.agent.config.tool_workers, 1)

    def test_tool_execution_records_recent_tool_call_context(self) -> None:
        class ReadingProvider(ModelProvider):
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, messages, tools):
                from xcode.harness.agent_runtime.events import ToolCall, ToolCallReady

                self.calls += 1
                if self.calls == 1:
                    yield ToolCallReady(
                        [ToolCall("read-1", "read_file", {"path": "a.txt"})]
                    )
                else:
                    yield TextDelta("done")
                    yield FinalMessage("", "end_turn")

            def complete(self, prompt: str) -> str:
                return "done"

            def judge(self, prompt: str) -> str:
                return "ok"

        provider = ReadingProvider()
        bundle = SimpleNamespace(
            llm=provider,
            llms={"main": provider, "subagent": provider},
            embedding=object(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("hello", encoding="utf-8")
            with patch("xcode.harness.app._build_providers", return_value=bundle):
                app = build_app(project_root=root, runtime_config=XcodeRuntimeConfig())
            app.ask("read it")

            rendered = app.contextual_state.render()

        self.assertIn("recent_tool_results:", rendered)
        self.assertIn("- read_file:", rendered)

    def test_subagent_inherits_only_enabled_core_tools(self) -> None:
        seen_child_tools: list[list[str]] = []
        runtime_config = XcodeRuntimeConfig(
            tools=ToolsRuntimeConfig(enabled_groups=("core", "subagent")),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            _patched_provider_bundle(seen_child_tools),
        ):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=runtime_config,
            )

        tools = {tool.name: tool for tool in app.registry}
        self.assertIn("submit_subagent", tools)
        tools["submit_subagent"].handler('{"prompt":"inspect"}')

        import time

        for _ in range(100):
            if seen_child_tools:
                break
            time.sleep(0.01)

        self.assertTrue(seen_child_tools, "child tools should not be empty")
        self.assertIn("read_file", seen_child_tools[0])
        self.assertNotIn("submit_subagent", seen_child_tools[0])
        self.assertNotIn("create_worktree_task", seen_child_tools[0])

    def test_subagent_tools_are_not_created_when_group_disabled(self) -> None:
        runtime_config = XcodeRuntimeConfig(
            tools=ToolsRuntimeConfig(enabled_groups=("core",)),
        )
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=runtime_config,
            )

        names = {tool.name for tool in app.registry}

        self.assertNotIn("task", names)
        self.assertNotIn("submit_subagent", names)

    def test_scoped_child_registry_uses_override_for_file_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "wt"
            worktree.mkdir()
            (root / "marker.txt").write_text("root", encoding="utf-8")
            (worktree / "marker.txt").write_text("worktree", encoding="utf-8")

            tools = {
                tool.name: tool
                for tool in _build_project_scoped_registry(
                    project_root=worktree,
                    enabled={"core", "subagent", "worktree"},
                    contextual_state=None,
                    shell_spec=detect_shell(),
                )
            }

            self.assertEqual(
                tools["read_file"].handler('{"path":"marker.txt"}'), "worktree"
            )

    def test_scoped_child_registry_uses_override_for_bash(self) -> None:
        seen_cwds: list[Path] = []

        class FakePipe:
            def __iter__(self):
                return iter(())

            def close(self) -> None:
                pass

        class FakePopen:
            def __init__(self, *args, **kwargs):
                seen_cwds.append(Path(kwargs["cwd"]))
                self.stdout = FakePipe()
                self.stderr = FakePipe()
                self.returncode = 0

            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp) / "wt"
            worktree.mkdir()
            tools = {
                tool.name: tool
                for tool in _build_project_scoped_registry(
                    project_root=worktree,
                    enabled={"core", "subagent", "worktree"},
                    contextual_state=None,
                    shell_spec=detect_shell(),
                )
            }
            with patch("xcode.harness.tools.bash.subprocess.Popen", FakePopen):
                tools["bash"].handler('{"command":"pwd"}')

        self.assertEqual(seen_cwds, [worktree.resolve()])

    def test_worktree_subagent_runs_child_tools_in_worktree(self) -> None:
        seen_reads: list[str] = []

        class ReadingProvider(ModelProvider):
            async def stream(self, messages, tools):
                read_file = {tool.name: tool for tool in tools}["read_file"]
                seen_reads.append(read_file.handler('{"path":"marker.txt"}'))
                yield TextDelta("child done")
                yield FinalMessage("", "end_turn")

            def complete(self, prompt: str) -> str:
                return "done"

            def judge(self, prompt: str) -> str:
                return "ok"

        class FakeWorktreeRunner:
            def __init__(self, repo_root: Path) -> None:
                self.repo_root = repo_root

            def create(self, name: str):
                worktree = self.repo_root / "wt"
                worktree.mkdir()
                (worktree / "marker.txt").write_text("worktree", encoding="utf-8")
                return SimpleNamespace(
                    id="wt123", path=worktree, branch=f"xcode/{name}"
                )

        runtime_config = XcodeRuntimeConfig(
            tools=ToolsRuntimeConfig(enabled_groups=("core", "subagent", "worktree")),
        )
        provider = ReadingProvider()
        bundle = SimpleNamespace(
            llm=provider,
            llms={"main": provider, "subagent": provider},
            embedding=object(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "marker.txt").write_text("root", encoding="utf-8")
            with (
                patch("xcode.harness.app._build_providers", return_value=bundle),
                patch(
                    "xcode.harness.app.WorktreeTaskRunner",
                    FakeWorktreeRunner,
                ),
            ):
                app = build_app(project_root=root, runtime_config=runtime_config)

            tools = {tool.name: tool for tool in app.registry}
            submitted = tools["submit_subagent"].handler(
                '{"prompt":"inspect","isolation":"worktree"}'
            )
            job_id = submitted.split()[2]
            for _ in range(100):
                checked = tools["check_subagent"].handler(f'{{"job_id":"{job_id}"}}')
                if "status=done" in checked:
                    break
                import time

                time.sleep(0.01)
            else:
                self.fail("subagent did not finish")

        self.assertEqual(seen_reads, ["worktree"])


from xcode.harness.agent_runtime.provider import ModelProvider  # noqa: E402
from xcode.harness.agent_runtime.events import TextDelta, FinalMessage  # noqa: E402


class MockProvider(ModelProvider):
    def __init__(self, seen_child_tools: list[list[str]]):
        self.seen_child_tools = seen_child_tools

    async def stream(self, messages, tools):
        self.seen_child_tools.append([tool.name for tool in tools])
        yield TextDelta("child done")
        yield FinalMessage("", "end_turn")

    def complete(self, prompt: str) -> str:
        return "done"

    def judge(self, prompt: str) -> str:
        return "ok"


def _patched_provider_bundle(seen_child_tools: list[list[str]]):
    provider = MockProvider(seen_child_tools)
    bundle = SimpleNamespace(
        llm=provider,
        llms={
            "main": provider,
            "subagent": provider,
            "judge": provider,
            "refiner": provider,
        },
        embedding=object(),
    )
    return patch("xcode.harness.app._build_providers", return_value=bundle)


if __name__ == "__main__":
    unittest.main()
