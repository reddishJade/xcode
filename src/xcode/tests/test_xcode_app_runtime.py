from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from xcode.harness.assembly import build_project_scoped_registry
from xcode.harness.app import XcodeApp, build_app
from xcode.ai.events import (
    FinalMessage,
    Message,
    ProviderEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from xcode.ai.providers.protocol import StreamProvider
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.config import (
    AgentConfig,
    DaemonRuntimeConfig,
    ObservabilityRuntimeConfig,
    PathsRuntimeConfig,
    SecurityRuntimeConfig,
    SkillsRuntimeConfig,
    ToolsRuntimeConfig,
    XcodeRuntimeConfig,
)
from xcode.harness.agent_runtime.compaction import LayeredCompactor
from xcode.coding_agent.tools.shell_adapter import detect_shell
from xcode.harness.mcp import McpRuntimeRegistry
from xcode.harness.skills import ToolSpec


def _write_skill(directory: Path, name: str, description: str, body: str) -> None:
    """在指定技能目录写入最小 SKILL.md。"""
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}",
        encoding="utf-8",
    )


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
        with (
            tempfile.TemporaryDirectory() as tmp,
            _patched_provider_bundle([]),
            patch.object(Path, "home", return_value=Path(tmp) / "home"),
        ):
            with patch(
                "xcode.coding_agent.tools.worktree.WorktreeTaskRunner",
                side_effect=AssertionError,
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
                "find_files",
                "grep_search",
                "ls",
                "bash",
                "search_tools",
                "update_todo",
            },
        )

    def test_matching_task_loads_discovered_skill_before_execution(self) -> None:
        class SkillSelectingProvider(StreamProvider):
            """模拟模型按 catalog 指令激活匹配技能。"""

            def __init__(self) -> None:
                self.calls: list[tuple[list[Message], list[ToolDefinition]]] = []

            async def stream(
                self,
                messages: list[Message],
                tools: list[ToolDefinition],
                options: StreamOptions | None = None,
                **kwargs: Any,
            ) -> AsyncIterator[ProviderEvent]:
                self.calls.append((messages, tools))
                if len(self.calls) == 1:
                    yield ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="skill-1",
                                name="load_skill",
                                input={"name": "code-review"},
                            )
                        ]
                    )
                    yield FinalMessage(content="", stop_reason="tool_use")
                    return
                yield TextDelta(chunk="review complete")
                yield FinalMessage(content="", stop_reason="end_turn")

        provider = SkillSelectingProvider()
        bundle = SimpleNamespace(
            llm=provider,
            llms={"main": provider, "subagent": provider},
            embedding=object(),
        )
        runtime_config = XcodeRuntimeConfig(
            skills=SkillsRuntimeConfig(trust_project_skills=True),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / ".xcode" / "skills" / "review"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: code-review\n"
                "description: Review code changes.\n"
                "---\n\n"
                "Follow the review workflow.",
                encoding="utf-8",
            )
            with patch("xcode.harness.assembly.build_providers", return_value=bundle):
                app = build_app(project_root=root, runtime_config=runtime_config)
            result = app.agent.run("Review this code change.")

        self.assertEqual(result.answer, "review complete")
        self.assertEqual(len(provider.calls), 2)
        first_messages, first_tools = provider.calls[0]
        self.assertIn("load_skill", {tool.name for tool in first_tools})
        first_context = "\n".join(str(message) for message in first_messages)
        self.assertIn("call load_skill", first_context)
        self.assertIn("Review code changes.", first_context)
        second_context = "\n".join(str(message) for message in provider.calls[1][0])
        self.assertIn("Follow the review workflow.", second_context)

    def test_build_app_uses_project_relative_configured_skills_dir(self) -> None:
        """项目相对 paths.skills_dir 进入技能发现。"""
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            root = Path(tmp)
            _write_skill(
                root / "configured-skills" / "review",
                "configured-review",
                "Review configured code.",
                "CONFIGURED_BODY",
            )
            app = build_app(
                project_root=root,
                runtime_config=XcodeRuntimeConfig(
                    paths=PathsRuntimeConfig(skills_dir=Path("configured-skills")),
                ),
            )

        load_skill = next(tool for tool in app.registry if tool.name == "load_skill")
        output = load_skill.handler({"name": "configured-review"})
        self.assertIn("CONFIGURED_BODY", output)

    def test_build_app_uses_absolute_skills_dir_argument(self) -> None:
        """绝对 skills_dir API 参数进入技能发现。"""
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as skills_tmp,
            _patched_provider_bundle([]),
        ):
            root = Path(tmp)
            skills_dir = Path(skills_tmp)
            _write_skill(
                skills_dir / "review",
                "absolute-review",
                "Review absolute code.",
                "ABSOLUTE_BODY",
            )
            app = build_app(project_root=root, skills_dir=skills_dir)

        load_skill = next(tool for tool in app.registry if tool.name == "load_skill")
        output = load_skill.handler({"name": "absolute-review"})
        self.assertIn("ABSOLUTE_BODY", output)

    def test_explicit_skills_dir_wins_duplicate_name(self) -> None:
        """显式目录中的同名技能覆盖固定项目目录。"""
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            root = Path(tmp)
            explicit_dir = root / "configured-skills"
            _write_skill(
                explicit_dir / "review",
                "duplicate-review",
                "Configured review.",
                "CONFIGURED_BODY",
            )
            _write_skill(
                root / ".xcode" / "skills" / "review",
                "duplicate-review",
                "Project review.",
                "PROJECT_BODY",
            )
            app = build_app(
                project_root=root,
                skills_dir=explicit_dir,
                runtime_config=XcodeRuntimeConfig(
                    skills=SkillsRuntimeConfig(trust_project_skills=True),
                ),
            )

        load_skill = next(tool for tool in app.registry if tool.name == "load_skill")
        output = load_skill.handler({"name": "duplicate-review"})
        self.assertIn("CONFIGURED_BODY", output)
        self.assertNotIn("PROJECT_BODY", output)

    def test_disabled_skills_group_ignores_configured_directory(self) -> None:
        """禁用 skills group 时不扫描或注册显式目录。"""
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            root = Path(tmp)
            with self.assertNoLogs(
                "xcode.harness.skills_registry",
                level="WARNING",
            ):
                app = build_app(
                    project_root=root,
                    skills_dir=root / "missing-skills",
                    runtime_config=XcodeRuntimeConfig(
                        tools=ToolsRuntimeConfig(enabled_groups=("core",)),
                    ),
                )

        self.assertNotIn("load_skill", {tool.name for tool in app.registry})
        self.assertIn("update_todo", {tool.name for tool in app.registry})

    def test_default_runtime_discovers_configured_mcp_tools(self) -> None:
        mcp_tool = ToolSpec(
            name="mcp__demo__read",
            description="Read from demo MCP server.",
            input_hint="{}",
            handler=lambda _data: "ok",
            group="mcp",
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            _patched_provider_bundle([]),
            patch("xcode.harness.mcp.build_mcp_tools", return_value=(mcp_tool,)),
        ):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=XcodeRuntimeConfig(),
            )

        self.assertIn("mcp__demo__read", {tool.name for tool in app.registry})

    def test_runtime_registry_applies_dynamic_mcp_tool_snapshot(self) -> None:
        """MCP 发布的新 schema 会进入 agent、应用和工具搜索视图。"""
        old_tool = ToolSpec(
            name="mcp__demo__old",
            description="Old MCP tool.",
            input_hint="{}",
            handler=lambda _data: "old",
            group="mcp",
        )
        new_tool = ToolSpec(
            name="mcp__demo__new",
            description="New MCP tool.",
            input_hint="{}",
            handler=lambda _data: "new",
            group="mcp",
        )
        runtime_registries: list[McpRuntimeRegistry] = []

        def build_dynamic_tools(
            _project_root: Path,
            runtime_registry: McpRuntimeRegistry,
        ) -> tuple[ToolSpec, ...]:
            """记录装配使用的 MCP 运行时注册器。"""
            runtime_registries.append(runtime_registry)
            return (old_tool,)

        with (
            tempfile.TemporaryDirectory() as tmp,
            _patched_provider_bundle([]),
            patch(
                "xcode.harness.mcp.build_mcp_tools",
                side_effect=build_dynamic_tools,
            ),
        ):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=XcodeRuntimeConfig(),
            )

        runtime_registries[0].publish((new_tool,))

        self.assertIn("mcp__demo__new", {tool.name for tool in app.registry})
        self.assertNotIn("mcp__demo__old", {tool.name for tool in app.agent.registry})
        search_tool = next(tool for tool in app.registry if tool.name == "search_tools")
        self.assertIn("mcp__demo__new", search_tool.handler({"query": "new"}))
        app.close()

    def test_default_runtime_does_not_enable_experimental_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=XcodeRuntimeConfig(
                    daemon=DaemonRuntimeConfig(enabled=True),
                ),
            )

        self.assertIsNone(_layered_compactor(app).on_compact)
        self.assertIsNone(app.daemon)
        self.assertIsNone(app.mailbox)
        self.assertIsNone(app.progress)

    def test_removed_experimental_group_does_not_enable_memory(self) -> None:
        runtime_config = XcodeRuntimeConfig(
            tools=ToolsRuntimeConfig(enabled_groups=("core", "experimental")),
        )
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=runtime_config,
            )

        names = {tool.name for tool in app.registry}
        self.assertNotIn("create_worktree_task", names)
        self.assertNotIn("create_task", names)
        self.assertNotIn("send_mailbox_message", names)
        self.assertNotIn("save_task_progress", names)
        self.assertIsNone(_layered_compactor(app).on_compact)
        self.assertIsNone(app.daemon)
        self.assertIsNone(app.mailbox)
        self.assertIsNone(app.progress)

    def test_individual_experimental_feature_groups_enable_individual_features(
        self,
    ) -> None:
        runtime_config = XcodeRuntimeConfig(
            tools=ToolsRuntimeConfig(enabled_groups=("core", "memory")),
        )
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=runtime_config,
            )

        names = {tool.name for tool in app.registry}
        self.assertNotIn("create_worktree_task", names)
        self.assertNotIn("create_task", names)
        self.assertIsNotNone(_layered_compactor(app).on_compact)
        self.assertIsNone(app.daemon)
        self.assertIsNone(app.mailbox)
        self.assertIsNone(app.progress)

    def test_mailbox_group_adds_mailbox_tools_only(self) -> None:
        runtime_config = XcodeRuntimeConfig(
            tools=ToolsRuntimeConfig(enabled_groups=("core", "mailbox")),
        )
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=runtime_config,
            )

        names = {tool.name for tool in app.registry}
        self.assertIn("send_mailbox_message", names)
        self.assertIn("read_mailbox_messages", names)
        self.assertIn("acknowledge_mailbox_message", names)
        self.assertNotIn("save_task_progress", names)
        self.assertIsNone(_layered_compactor(app).on_compact)
        self.assertIsNotNone(app.mailbox)
        self.assertIsNone(app.progress)

    def test_progress_group_adds_progress_tools_only(self) -> None:
        runtime_config = XcodeRuntimeConfig(
            tools=ToolsRuntimeConfig(enabled_groups=("core", "progress")),
        )
        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            app = build_app(
                project_root=Path(tmp),
                runtime_config=runtime_config,
            )

        names = {tool.name for tool in app.registry}
        self.assertIn("save_task_progress", names)
        self.assertIn("resume_task_progress", names)
        self.assertNotIn("send_mailbox_message", names)
        self.assertIsNone(_layered_compactor(app).on_compact)
        self.assertIsNone(app.mailbox)
        self.assertIsNotNone(app.progress)

    def test_bash_tool_uses_agent_cancellation_event(self) -> None:
        captured = {}

        def fake_bash_tool(*_args, **kwargs):
            captured["cancel_event"] = kwargs.get("cancel_event")
            return ToolSpec(
                "bash",
                "Run shell.",
                "command",
                lambda _data: "ok",
                schema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            )

        with tempfile.TemporaryDirectory() as tmp, _patched_provider_bundle([]):
            with patch(
                "xcode.coding_agent.registry.build_bash_tool",
                side_effect=fake_bash_tool,
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
            agent=AgentConfig(max_steps=7, tool_workers=2),
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

    def test_security_approval_policy_never_allows_high_risk_tools(self) -> None:
        runtime_config = XcodeRuntimeConfig(
            security=SecurityRuntimeConfig(approval_policy="never"),
        )

        class WritingProvider(MockProvider):
            def __init__(self, seen_child_tools, transport=""):
                super().__init__(seen_child_tools, transport)
                self.stream_calls = []

            async def stream(self, messages, tools, options=None, **kwargs):
                self.stream_calls.append(messages)
                if len(self.stream_calls) == 1:
                    yield ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="write-1",
                                name="write_file",
                                input={"path": "ok.txt", "content": "ok"},
                            )
                        ]
                    )
                    yield FinalMessage(content="", stop_reason="tool_use")
                    return
                yield TextDelta(chunk="done")
                yield FinalMessage(content="", stop_reason="end_turn")

        provider = WritingProvider([])
        bundle = SimpleNamespace(
            llm=provider,
            llms={"main": provider, "subagent": provider, "fallback": provider},
            embedding=object(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("xcode.harness.assembly.build_providers", return_value=bundle):
                app = build_app(project_root=root, runtime_config=runtime_config)
            result = app.agent.run("write")

            self.assertEqual(result.answer, "done")
            self.assertEqual((root / "ok.txt").read_text(encoding="utf-8"), "ok")

    def test_security_approval_policy_always_blocks_low_risk_without_callback(
        self,
    ) -> None:
        runtime_config = XcodeRuntimeConfig(
            security=SecurityRuntimeConfig(approval_policy="always"),
        )

        class ReadingProvider(MockProvider):
            def __init__(self, seen_child_tools, transport=""):
                super().__init__(seen_child_tools, transport)
                self.stream_calls = []

            async def stream(self, messages, tools, options=None, **kwargs):
                self.stream_calls.append(messages)
                if len(self.stream_calls) == 2:
                    yield TextDelta(chunk="done")
                    yield FinalMessage(content="", stop_reason="end_turn")
                    return
                yield ToolCallEvent(
                    calls=[
                        ToolCall(id="read-1", name="read_file", input={"path": "a.txt"})
                    ]
                )
                yield FinalMessage(content="", stop_reason="tool_use")

        provider = ReadingProvider([])
        bundle = SimpleNamespace(
            llm=provider,
            llms={"main": provider, "subagent": provider, "fallback": provider},
            embedding=object(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("content", encoding="utf-8")
            with patch("xcode.harness.assembly.build_providers", return_value=bundle):
                app = build_app(project_root=root, runtime_config=runtime_config)
            result = app.agent.run("read")

        tool_result: dict[str, object] | None = None
        for message in result.messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                tool_result = content[0]
                break

        assert tool_result is not None
        self.assertEqual(tool_result["status"], "error")
        self.assertIn("requires approval", str(tool_result["content"]))

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
        class ReadingProvider(StreamProvider):
            def __init__(self) -> None:
                self.calls = 0

            async def stream(
                self,
                messages: list[Message],
                tools: list[ToolDefinition],
                options: StreamOptions | None = None,
                **kwargs: Any,
            ) -> AsyncIterator[ProviderEvent]:
                from xcode.ai.events import ToolCall, ToolCallEvent

                self.calls += 1
                if self.calls == 1:
                    yield ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="read-1", name="read_file", input={"path": "a.txt"}
                            )
                        ]
                    )
                else:
                    yield TextDelta(chunk="done")
                    yield FinalMessage(content="", stop_reason="end_turn")

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
            with patch("xcode.harness.assembly.build_providers", return_value=bundle):
                app = build_app(project_root=root, runtime_config=XcodeRuntimeConfig())
            app.ask("read it")

            contextual_state = app.contextual_state
            assert contextual_state is not None
            rendered = contextual_state.render()

        self.assertIn("recent_tool_results:", rendered)
        self.assertIn("- read_file:", rendered)
        self.assertIn("active_file: a.txt", rendered)

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
        tools["submit_subagent"].handler({"prompt": "inspect"})

        import time

        for _ in range(100):
            if seen_child_tools:
                break
            time.sleep(0.01)

        self.assertTrue(seen_child_tools, "child tools should not be empty")
        self.assertIn("read_file", seen_child_tools[0])
        self.assertNotIn("update_todo", seen_child_tools[0])
        self.assertNotIn("submit_subagent", seen_child_tools[0])
        self.assertNotIn("create_worktree_task", seen_child_tools[0])

    def test_subagent_can_explicitly_allow_session_todo_tool(self) -> None:
        """subagent 仅在明确 allowlist 时继承 update_todo。"""
        seen_child_tools: list[list[str]] = []
        runtime_config = XcodeRuntimeConfig(
            tools=ToolsRuntimeConfig(
                enabled_groups=("core", "subagent"),
                subagent_tool_allowlist=("update_todo",),
            ),
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
        tools["submit_subagent"].handler({"prompt": "inspect"})

        import time

        for _ in range(100):
            if seen_child_tools:
                break
            time.sleep(0.01)

        self.assertTrue(seen_child_tools)
        self.assertIn("update_todo", seen_child_tools[0])

    def test_subagent_receives_runtime_prompt_context(self) -> None:
        seen_messages: list[list[Message]] = []

        class CapturingProvider(StreamProvider):
            async def stream(
                self,
                messages: list[Message],
                tools: list[ToolDefinition],
                options: StreamOptions | None = None,
                **kwargs: Any,
            ) -> AsyncIterator[ProviderEvent]:
                seen_messages.append(messages)
                yield TextDelta(chunk="child done")
                yield FinalMessage(content="", stop_reason="end_turn")

            def complete(self, prompt: str) -> str:
                return "done"

            def judge(self, prompt: str) -> str:
                return "ok"

        provider = CapturingProvider()
        bundle = SimpleNamespace(
            llm=provider,
            llms={"main": provider, "subagent": provider},
            embedding=object(),
        )
        runtime_config = XcodeRuntimeConfig(
            tools=ToolsRuntimeConfig(enabled_groups=("core", "subagent")),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text(
                "Subagent must follow project rules.",
                encoding="utf-8",
            )
            with patch("xcode.harness.assembly.build_providers", return_value=bundle):
                app = build_app(project_root=root, runtime_config=runtime_config)

            tools = {tool.name: tool for tool in app.registry}
            submitted = tools["submit_subagent"].handler({"prompt": "inspect"})
            job_id = submitted.split()[2]
            for _ in range(100):
                checked = tools["check_subagent"].handler({"job_id": job_id})
                if "status=done" in checked:
                    break
                import time

                time.sleep(0.01)
            else:
                self.fail("subagent did not finish")

        self.assertTrue(seen_messages)
        first_msg = seen_messages[0][0]
        self.assertEqual(first_msg["role"], "system")
        combined_system = " ".join(
            str(m.get("content", "") or "")
            for m in seen_messages[0]
            if m.get("role") == "system"
        )
        self.assertIn("Subagent must follow project rules.", combined_system)
        self.assertIn("<git-preflight>", combined_system)
        self.assertIn("<cwd-info>", combined_system)

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
                for tool in build_project_scoped_registry(
                    project_root=worktree,
                    enabled={"core", "subagent", "worktree"},
                    contextual_state=None,
                    shell_spec=detect_shell(),
                )
            }

            self.assertEqual(
                tools["read_file"].handler({"path": "marker.txt"}), "worktree"
            )

    def test_scoped_child_registry_uses_override_for_bash(self) -> None:
        seen_cwds: list[Path] = []

        class TrackingEnv:
            def run(self, argv, cwd, timeout=30, cancel_event=None):
                seen_cwds.append(Path(cwd))
                from xcode.harness.execution_env import ExecutionResult

                return ExecutionResult(stdout="")

        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp) / "wt"
            worktree.mkdir()
            tools = {
                tool.name: tool
                for tool in build_project_scoped_registry(
                    project_root=worktree,
                    enabled={"core", "subagent", "worktree"},
                    contextual_state=None,
                    shell_spec=detect_shell(),
                    env=TrackingEnv(),
                )
            }
            tools["bash"].handler({"command": "pwd"})

        self.assertEqual(seen_cwds, [worktree.resolve()])

    def test_worktree_subagent_runs_child_tools_in_worktree(self) -> None:
        seen_reads: list[str] = []

        class ReadingProvider(StreamProvider):
            async def stream(
                self,
                messages: list[Message],
                tools: list[ToolDefinition],
                options: StreamOptions | None = None,
                **kwargs: Any,
            ) -> AsyncIterator[ProviderEvent]:
                if messages and messages[-1].get("role") == "tool":
                    seen_reads.append(str(messages[-1].get("content", "")))
                    yield TextDelta(chunk="child done")
                    yield FinalMessage(content="", stop_reason="end_turn")
                    return
                yield TextDelta(chunk="child done")
                yield ToolCallEvent(
                    calls=[
                        ToolCall(
                            id="read-1", name="read_file", input={"path": "marker.txt"}
                        )
                    ]
                )
                yield FinalMessage(content="", stop_reason="tool_use")

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
                patch("xcode.harness.assembly.build_providers", return_value=bundle),
                patch(
                    "xcode.coding_agent.tools.worktree.WorktreeTaskRunner",
                    lambda project_root: FakeWorktreeRunner(project_root),
                ),
            ):
                app = build_app(project_root=root, runtime_config=runtime_config)

            tools = {tool.name: tool for tool in app.registry}
            submitted = tools["submit_subagent"].handler(
                {"prompt": "inspect", "isolation": "worktree"}
            )
            job_id = submitted.split()[2]
            for _ in range(100):
                checked = tools["check_subagent"].handler({"job_id": job_id})
                if "status=done" in checked:
                    break
                import time

                time.sleep(0.01)
            else:
                self.fail("subagent did not finish")

        self.assertEqual(seen_reads, ["worktree"])


class MockProvider(StreamProvider):
    def __init__(
        self,
        seen_child_tools: list[list[str]],
        transport: str = "",
    ) -> None:
        self.seen_child_tools = seen_child_tools
        self.transport = transport

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        self.seen_child_tools.append([tool.name for tool in tools])
        yield TextDelta(chunk="child done")
        yield FinalMessage(content="", stop_reason="end_turn")

    def complete(self, prompt: str) -> str:
        return "done"

    def judge(self, prompt: str) -> str:
        return "ok"


def _patched_provider_bundle(
    seen_child_tools: list[list[str]],
    transport: str = "",
):
    provider = MockProvider(seen_child_tools, transport=transport)
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
    return patch("xcode.harness.assembly.build_providers", return_value=bundle)


def _layered_compactor(app: XcodeApp) -> LayeredCompactor:
    compactor = app.agent.compactor
    assert isinstance(compactor, LayeredCompactor)
    return compactor


if __name__ == "__main__":
    unittest.main()
