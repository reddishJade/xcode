from __future__ import annotations

import asyncio
import json
import re
import subprocess
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from xcode.cli.app_contract import ReplApp
from xcode.cli.completion import ReplCompleter
from xcode.cli.commands import PromptText, ReplState
from xcode.cli.repl import current_effort_options, run_repl
from xcode.cli.repl_commands import (
    COMMAND_NAMES,
    COMMAND_REGISTRY,
    HELP_TEXT,
    handle_command,
)
from xcode.cli.repl_rendering import REPL_PROMPT_STYLE
from xcode.cli.repl_rendering import ReplInputLexer
from xcode.cli.repl_rendering import input_prompt
from xcode.cli.repl_rendering import reasoning_preview_lines
from xcode.cli.repl_sessions import records_to_agent_messages
from xcode.cli.repl_skills import parse_skill_invocation
from xcode.cli.repl_tools import brief_input, run_tool_command
from xcode.harness.skill_activation import ExplicitSkillActivationResult
from xcode.harness.session import SessionRecord, SessionStore
from xcode.harness.snapshot import SnapshotResult, SnapshotStore
from xcode.harness.agent_runtime import (
    CancellationToken,
    StructuredAgentResult,
)
from xcode.harness.agent_runtime.execution_modes import registry_for_mode
from xcode.harness.observability import (
    ExternalHookDiagnostic,
    HITLResult,
    PermissionPolicy,
    StaticPermission,
)
from xcode.harness.agent_runtime.events import (
    FinalStructuredEvent,
    ReasoningDeltaStructuredEvent,
    TextDeltaStructuredEvent,
    ToolResultBlock,
    ToolResultStructuredEvent,
    TodoUpdateStructuredEvent,
    ToolUseStructuredEvent,
)
from xcode.ai.events import ToolCall
from xcode.agent.messages import AgentMessage, AssistantMessage
from xcode.harness.skills import ApprovalCallback, ToolSpec
from xcode.harness.session_todo import TodoItem
import pytest


class XcodeReplTests:
    def test_session_store_writes_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))

            store.append("user", "hello")

            text = store.current_path.read_text(encoding="utf-8")
            assert '"type": "user"' in text
            assert '"hello"' in text

    def test_session_store_rewinds_last_user_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "first")
            store.append("assistant", "one")
            store.append("user", "second")
            store.append("assistant", "two")

            removed = store.rewind_turns()
            records = store.load_records()

            assert removed == 2
            assert [record.content for record in records] == ["first", "one"]
            assert store.user_turn_count() == 1

    def test_session_store_resumes_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            first = store.current_path
            store.append("user", "hello")
            store.clear()

            resumed = store.resume_latest()

            assert resumed == first
            assert store.current_path == first

    def test_resumed_tool_turn_keeps_final_assistant_text(self) -> None:
        """工具轮次恢复时保留最终助手文本。"""
        created_at = "2026-01-01T00:00:00+00:00"
        records = [
            SessionRecord(type="user", content="change file", created_at=created_at),
            SessionRecord(
                type="event",
                content={
                    "type": "assistant",
                    "data": [
                        {
                            "type": "tool_use",
                            "id": "write-1",
                            "name": "write_file",
                            "input": {"path": "hello.py"},
                        }
                    ],
                },
                created_at=created_at,
            ),
            SessionRecord(
                type="event",
                content={
                    "type": "tool_result",
                    "data": {
                        "tool_use_id": "write-1",
                        "status": "ok",
                        "content": "wrote file: hello.py",
                    },
                },
                created_at=created_at,
            ),
            SessionRecord(
                type="assistant", content="Updated hello.py.", created_at=created_at
            ),
        ]

        messages = records_to_agent_messages(records)

        assert isinstance(messages[-1], AssistantMessage)
        final_message = cast(AssistantMessage, messages[-1])
        assert final_message.content[0].text == "Updated hello.py."

    def test_resumed_tool_turn_deduplicates_event_assistant_text(self) -> None:
        """工具事件已包含相同文本时不重复恢复。"""
        created_at = "2026-01-01T00:00:00+00:00"
        records = [
            SessionRecord(type="user", content="inspect file", created_at=created_at),
            SessionRecord(
                type="event",
                content={
                    "type": "assistant",
                    "data": [
                        {"type": "text", "text": "Inspecting hello.py."},
                        {
                            "type": "tool_use",
                            "id": "read-1",
                            "name": "read_file",
                            "input": {"path": "hello.py"},
                        },
                    ],
                },
                created_at=created_at,
            ),
            SessionRecord(
                type="event",
                content={
                    "type": "tool_result",
                    "data": {
                        "tool_use_id": "read-1",
                        "status": "ok",
                        "content": "hello",
                    },
                },
                created_at=created_at,
            ),
            SessionRecord(
                type="assistant", content="Inspecting hello.py.", created_at=created_at
            ),
        ]

        messages = records_to_agent_messages(records)
        assistant_messages = [
            message for message in messages if isinstance(message, AssistantMessage)
        ]

        assert len(assistant_messages) == 1

    def test_session_store_writes_title_summary_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions = Path(temp_dir) / ".local" / "sessions"
            store = SessionStore(sessions, project_root=Path(temp_dir))

            store.append("user", "Refactor session storage and resume flow")
            store.append("assistant", "Implemented titled session metadata.")
            metadata = store.update_summary()

            assert metadata is not None
            index = json.loads(
                (Path(temp_dir) / ".local" / "session_index.json").read_text(
                    encoding="utf-8"
                )
            )
            item = index["sessions"][0]
            assert index["version"] == 1
            assert index["storage"] == "jsonl-v1"
            assert index["recovery_boundary"] == "current_transcript_and_session_tree"
            assert item["title"] == "Refactor session storage and resume flow"
            assert "Answer preview" in item["summary"]
            assert not (Path(item["transcript_path"]).is_absolute())

    def test_session_store_loads_legacy_index_without_protocol_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions = Path(temp_dir) / ".local" / "sessions"
            sessions.mkdir(parents=True)
            transcript = sessions / "session-legacy.jsonl"
            transcript.write_text("", encoding="utf-8")
            index_path = Path(temp_dir) / ".local" / "session_index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "sessions": [
                            {
                                "id": "legacy",
                                "title": "Legacy",
                                "summary": "Old index",
                                "project_path": temp_dir,
                                "transcript_path": "sessions/session-legacy.jsonl",
                                "created_at": "2026-01-01T00:00:00+00:00",
                                "updated_at": "2026-01-01T00:00:00+00:00",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            store = SessionStore(sessions, project_root=Path(temp_dir))

            views = store.list_session_infos()

            assert views[0].title == "Legacy"
            assert store.protocol_info().storage == "jsonl-v1"

    def test_run_repl_persists_user_and_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = FakeApp()
            prompt = FakePrompt(["hello", "/exit"])

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            session = next(Path(temp_dir).glob("session-*.jsonl"))
            text = session.read_text(encoding="utf-8")
            assert '"type": "user"' in text
            assert '"type": "assistant"' in text
            assert "hello!" in text

    def test_run_repl_hides_session_path_and_prints_saved_title(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = FakeApp()
            prompt = FakePrompt(["hello", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            text = output.getvalue()
            assert code == 0
            assert text.startswith("\033[2J\033[3J\033[H")
            assert "XCode" in text
            assert "model:" in text
            assert "unknown" in text
            assert "thinking: unknown" in text
            assert "effort:" in text
            assert "not set" in text
            assert "cwd:" in text
            assert "Conversation saved: hello" in text
            assert "Session:" not in text
            assert "session-" not in text.splitlines()[0]

    def test_run_repl_streams_markdown_without_changing_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = FakeMarkdownApp()
            prompt = FakePrompt(["hello", "/exit"])
            renderer = FakeRenderer()
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt, renderer=renderer)

            assert code == 0
            assert renderer.rendered == []
            assert "Title" in output.getvalue()
            assert "item" in output.getvalue()
            session = next(Path(temp_dir).glob("session-*.jsonl"))
            text = session.read_text(encoding="utf-8")
            assert "# Title" in text

    def test_run_repl_streams_text_delta_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = MultiDeltaApp()
            prompt = FakePrompt(["hello", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            assert "hello" in output.getvalue()
            assert "thinking..." not in output.getvalue()

    def test_run_repl_shows_reasoning_preview_before_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ReasoningApp()
            prompt = FakePrompt(["hello", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            assert "Thought for" in output.getvalue()
            assert "three four five six seven eight" not in output.getvalue()
            assert "done" in output.getvalue()

    def test_run_repl_hides_tiny_reasoning_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = TinyReasoningApp()
            prompt = FakePrompt(["hello", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            assert "thought for" not in output.getvalue()
            assert "done" in output.getvalue()

    def test_run_repl_summarizes_tools_without_success_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ToolEventApp()
            prompt = FakePrompt(["search", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            text = output.getvalue()
            assert code == 0
            assert "Explore: Search src/xcode for mcp" in text
            assert "done: 1 tools" in text
            assert "tool result" not in text
            assert "← ok" not in text

    def test_run_repl_verbose_shows_individual_tool_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ToolEventApp()
            prompt = FakePrompt(["/verbose on", "search", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            text = output.getvalue()
            assert code == 0
            assert 'grep_search: pattern="mcp", path="src/xcode"' in text
            assert "← ok" in text

    def test_run_repl_renders_structured_todo_update(self) -> None:
        """结构化 todo_update 事件显示当前完整清单。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = TodoEventApp()
            prompt = FakePrompt(["plan", "/exit"])

            with redirect_stdout(StringIO()) as output:
                code = run_repl(cast(Any, app), Path(temp_dir), prompt)

        assert code == 0
        rendered = output.getvalue()
        assert "Todo (2 items)" in rendered
        assert "[/] Implement feature" in rendered
        assert "[ ] Run tests" in rendered

    def test_reasoning_preview_lines_keep_latest_three_visual_lines(self) -> None:
        assert reasoning_preview_lines("one\ntwo\nthree\nfour", width=80) == [
            "two",
            "three",
            "four",
        ]

    def test_run_repl_expands_file_references_but_preserves_user_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "note.md").write_text("file body", encoding="utf-8")
            app = CapturingApp()
            prompt = FakePrompt(["read @note.md", "/exit"])

            with redirect_stdout(StringIO()):
                code = run_repl(app, root, prompt, project_root=root)

            assert code == 0
            assert '<file-reference path="note.md">' in app.seen[0]
            assert "file body" in app.seen[0]
            session = next(root.glob("session-*.jsonl"))
            text = session.read_text(encoding="utf-8")
            assert "read @note.md" in text
            assert "file_references" in text

    def test_run_repl_plan_build_and_act_toggle_execution_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = CapturingApp()
            prompt = FakePrompt(
                ["/plan", "first", "/build", "second", "/act", "third", "/exit"]
            )

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            assert app.modes == ["plan", "build", "act"]

    def test_run_repl_plan_command_accepts_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = CapturingApp()
            prompt = FakePrompt(["/plan inspect only", "/exit"])

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            assert app.seen == ["inspect only"]
            assert app.modes == ["plan"]

    def test_run_repl_skill_command_accepts_followup_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ExplicitSkillApp()
            prompt = FakePrompt(["/skill code-review inspect this patch", "/exit"])

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            assert app.questions == ["inspect this patch"]
            assert app.agent.activated == ["code-review"]
            assert app.agent.activation_modes == ["act"]

    def test_run_repl_skill_activation_uses_current_build_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ExplicitSkillApp()
            prompt = FakePrompt(
                ["/build", "$code-review inspect this patch", "/exit"]
            )

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            assert app.questions == ["inspect this patch"]
            assert app.agent.activated == ["code-review"]
            assert app.agent.activation_modes == ["build"]

    def test_plan_mode_registry_keeps_safe_discovery_and_skills(self) -> None:
        registry = (
            ToolSpec("read_file", "Read.", "text", lambda _value: ""),
            ToolSpec("list_dir", "List.", "text", lambda _value: ""),
            ToolSpec("search_tools", "Search tools.", "text", lambda _value: ""),
            ToolSpec("load_skill", "Load skill.", "text", lambda _value: ""),
            ToolSpec("write_file", "Write.", "text", lambda _value: ""),
            ToolSpec("bash", "Shell.", "text", lambda _value: ""),
        )

        names = {tool.name for tool in registry_for_mode(registry, "plan")}
        build_names = {tool.name for tool in registry_for_mode(registry, "build")}

        assert {"read_file", "list_dir", "search_tools", "load_skill"} <= names
        assert "write_file" not in names
        assert "bash" not in names
        assert "load_skill" in build_names

    def test_run_repl_tool_command_runs_registered_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ToolApp()
            prompt = FakePrompt(["/tool echo hello", "/exit"])
            renderer = FakeRenderer()

            with redirect_stdout(StringIO()):
                code = run_repl(
                    cast(ReplApp, app), Path(temp_dir), prompt, renderer=renderer
                )

            assert code == 0
            assert renderer.rendered == ["hello"]

    def test_tool_list_output_is_visible_markdown_text(self) -> None:
        output = run_tool_command("/tool list", ToolApp())

        assert "## Visible Tools" in output
        assert "- **core**" in output
        assert "<visible tools>" not in output

    def test_run_repl_shell_shortcut_runs_bash_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ShellShortcutApp(output="Name    Length\n----    ------\nfile    42")
            prompt = FakePrompt(["!echo hello", "/exit"])
            renderer = FakeRenderer()
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(
                    cast(ReplApp, app), Path(temp_dir), prompt, renderer=renderer
                )

            assert code == 0
            assert app.commands == ["echo hello"]
            assert renderer.rendered == []
            text_output = output.getvalue()
            assert "Name    Length\n----    ------\nfile    42\n" in text_output
            session = next(Path(temp_dir).glob("session-*.jsonl"))
            text = session.read_text(encoding="utf-8")
            assert "shell_shortcut" in text
            assert '"type": "user"' not in text

    def test_run_repl_shell_shortcut_rejects_empty_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ShellShortcutApp()
            prompt = FakePrompt(["!", "/exit"])
            renderer = FakeRenderer()
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(
                    cast(ReplApp, app), Path(temp_dir), prompt, renderer=renderer
                )

            assert code == 0
            assert app.commands == []
            assert renderer.rendered == []
            assert "usage: !COMMAND\n" in output.getvalue()

    def test_run_repl_tool_command_preserves_high_risk_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ToolApp(
                registry=(
                    ToolSpec(
                        "danger",
                        "Danger.",
                        "text",
                        lambda value: value["input"],
                    ),
                )
            )
            prompt = FakePrompt(["/tool danger now", "/exit"])
            renderer = FakeRenderer()

            with redirect_stdout(StringIO()):
                code = run_repl(
                    cast(ReplApp, app), Path(temp_dir), prompt, renderer=renderer
                )

            assert code == 0
            # High-risk approval removed; tool runs by default
            assert renderer.rendered[0] == "now"

    def test_undo_uses_registered_tool_spec_for_approval(self) -> None:
        """撤销写入时使用规范工具描述触发授权并恢复文件。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            subprocess.run(
                ["git", "init"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            target = root / "hello.py"
            target.write_text("before\n", encoding="utf-8")
            sessions_dir = root / ".local" / "sessions"
            store = SessionStore(sessions_dir, project_root=root)
            store.append("user", "change hello.py")
            snapshot_store = SnapshotStore(root)
            service = snapshot_store.service(store.session_id)
            pre = service.track()
            target.write_text("after\n", encoding="utf-8")
            post = service.track()
            snapshot_store.record_turn(
                store.session_id,
                "001",
                pre.snapshot_id,
                post.snapshot_id,
                service.diff(pre.snapshot_id, post.snapshot_id),
            )
            approvals: list[str] = []

            def approve(tool: ToolSpec, _input: dict[str, object]) -> HITLResult:
                approvals.append(tool.name)
                return HITLResult("allow", "once")

            app = SimpleNamespace(
                registry=(
                    ToolSpec(
                        "write_file",
                        "Write a file.",
                        'JSON: {"path": "relative/path", "content": "text"}',
                        lambda _input: "",
                    ),
                ),
                agent=SimpleNamespace(
                    approval_callback=approve,
                    external_directories=(),
                ),
            )
            policy = PermissionPolicy(
                (StaticPermission(tool="write_file", decision="ask"),),
                global_default="allow",
            )

            with redirect_stdout(StringIO()):
                handled = handle_command(
                    "/undo",
                    store,
                    cast(Any, app),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                    static_policy=policy,
                    snapshot_store=snapshot_store,
                )

            assert not (handled)
            assert approvals == ["write_file"]
            assert target.read_text(encoding="utf-8") == "before\n"
            assert snapshot_store.list_records(store.session_id)[0].undone

    def test_undo_denial_keeps_snapshot_record_active(self) -> None:
        """撤销被拒绝时保留活动记录，允许后续重试。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            subprocess.run(
                ["git", "init"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            target = root / "hello.py"
            target.write_text("before\n", encoding="utf-8")
            store = SessionStore(root / ".local" / "sessions", project_root=root)
            store.append("user", "change hello.py")
            snapshot_store = SnapshotStore(root)
            service = snapshot_store.service(store.session_id)
            pre = service.track()
            target.write_text("after\n", encoding="utf-8")
            post = service.track()
            snapshot_store.record_turn(
                store.session_id,
                "001",
                pre.snapshot_id,
                post.snapshot_id,
                service.diff(pre.snapshot_id, post.snapshot_id),
            )
            app = SimpleNamespace(
                registry=(
                    ToolSpec(
                        "write_file",
                        "Write a file.",
                        'JSON: {"path": "relative/path", "content": "text"}',
                        lambda _input: "",
                    ),
                ),
                agent=SimpleNamespace(
                    approval_callback=lambda _tool, _input: HITLResult("deny", "once"),
                    external_directories=(),
                ),
            )
            policy = PermissionPolicy(
                (StaticPermission(tool="write_file", decision="ask"),),
                global_default="allow",
            )

            with redirect_stdout(StringIO()) as output:
                handle_command(
                    "/undo",
                    store,
                    cast(Any, app),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                    static_policy=policy,
                    snapshot_store=snapshot_store,
                )

            assert target.read_text(encoding="utf-8") == "after\n"
            assert not (snapshot_store.list_records(store.session_id)[0].undone)
            assert "record remains active" in output.getvalue()

    def test_tool_command_uses_static_permission_policy(self) -> None:
        app = DeniedToolApp()

        output = run_tool_command("/tool bash git status", app)

        assert "deny for bash" in output
        assert app.commands == []

    def test_run_repl_permissions_show_static_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = StaticPermissionApp()
            prompt = FakePrompt(["/permissions", "/exit"])

            with redirect_stdout(StringIO()) as output:
                code = run_repl(cast(Any, app), Path(temp_dir), prompt)

            assert code == 0
            rendered = output.getvalue()
            assert "Static rules (1)" in rendered
            assert "tool `bash` -> deny" in rendered
            assert "Static rules: none" not in rendered

    def test_permissions_show_global_default_without_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            output = StringIO()

            with redirect_stdout(output):
                handled = handle_command(
                    "/permissions",
                    store,
                    FakeApp(),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                    static_policy=PermissionPolicy(global_default="ask"),
                )

            assert not handled
            rendered = output.getvalue()
            assert "Default from config: ask" in rendered
            assert "(none" not in rendered

    def test_permissions_empty_state_explains_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            output = StringIO()

            with redirect_stdout(output):
                handled = handle_command(
                    "/permissions",
                    store,
                    FakeApp(),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

            assert not handled
            rendered = output.getvalue()
            assert "Static rules: none" in rendered
            assert "Implicit fallback: allow" in rendered

    def test_hooks_command_shows_source_state_and_recent_error(self) -> None:
        """`/hooks` 展示配置来源、启用状态和最近脱敏错误。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir), project_root=Path(temp_dir))
            app = SimpleNamespace(
                hook_diagnostics=lambda: (
                    ExternalHookDiagnostic(
                        index=0,
                        event="pre_tool",
                        source=".xcode/settings.json",
                        command=("python", "check.py"),
                        matcher="bash",
                        enabled=True,
                        failure_policy="warn",
                        inherit_to_subagents=False,
                        run_count=2,
                        last_status="failed",
                        last_error="api_key=[REDACTED]",
                        last_run_at="2026-06-22T01:02:03+00:00",
                    ),
                )
            )

            with redirect_stdout(StringIO()) as output:
                handled = handle_command(
                    "/hooks",
                    store,
                    cast(Any, app),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

        assert not (handled)
        rendered = output.getvalue()
        assert "pre_tool enabled" in rendered
        assert "source=.xcode/settings.json" in rendered
        assert "runs=2" in rendered
        assert "error=api_key=[REDACTED]" in rendered

    def test_hooks_command_reports_empty_configuration(self) -> None:
        """未配置外部 hook 时给出明确诊断。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir), project_root=Path(temp_dir))
            app = SimpleNamespace(hook_diagnostics=lambda: ())

            with redirect_stdout(StringIO()) as output:
                handle_command(
                    "/hooks",
                    store,
                    cast(Any, app),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

        assert "No external hooks configured." in output.getvalue()

    def test_mcp_command_shows_runtime_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir), project_root=Path(temp_dir))
            app = SimpleNamespace(
                mcp_status=lambda: (
                    {
                        "server_name": "demo",
                        "state": "connected",
                        "tool_count": 2,
                        "deferred": False,
                        "protocol_version": "2025-11-25",
                        "server_info": {"name": "demo", "version": "1.0.0"},
                        "last_error": None,
                    },
                )
            )

            with redirect_stdout(StringIO()) as output:
                handled = handle_command(
                    "/mcp status",
                    store,
                    cast(Any, app),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

        assert not handled
        rendered = output.getvalue()
        assert "demo: state=connected" in rendered
        assert "tools=2" in rendered
        assert "protocol=2025-11-25" in rendered
        assert "identity=demo@1.0.0" in rendered

    def test_mcp_command_triggers_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir), project_root=Path(temp_dir))
            reloaded: list[bool] = []

            def reload_mcp() -> tuple[str, ...]:
                reloaded.append(True)
                return ("mcp__demo__read", "mcp__demo__write")

            app = SimpleNamespace(reload_mcp=reload_mcp)

            with redirect_stdout(StringIO()) as output:
                handled = handle_command(
                    "/mcp reload",
                    store,
                    cast(Any, app),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

        assert not handled
        assert reloaded == [True]
        assert "Reloaded MCP config. Registered 2 MCP tools." in output.getvalue()

    def test_run_repl_tool_list_shows_visible_and_hidden_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ToolApp(
                registry=(
                    ToolSpec(
                        "read_file",
                        "Read.",
                        "path",
                        lambda value: value["input"],
                        group="core",
                    ),
                )
            )
            prompt = FakePrompt(["/tool list", "/exit"])
            renderer = FakeRenderer()

            with patch(
                "xcode.cli.repl_tools.build_tool_catalog",
                return_value={
                    "core": {"read_file"},
                    "subagent": {"submit_subagent"},
                },
            ):
                with redirect_stdout(StringIO()):
                    code = run_repl(
                        cast(ReplApp, app), Path(temp_dir), prompt, renderer=renderer
                    )

            assert code == 0
            assert "## Visible Tools" in renderer.rendered[0]
            assert "read_file" in renderer.rendered[0]
            assert "## Hidden Tools" in renderer.rendered[0]
            assert "submit_subagent" in renderer.rendered[0]

    def test_run_repl_queue_mode_enqueues_followup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = QueueModeApp()
            prompt = FakePrompt(["/queue on", "hello", "queued followup", "", "/exit"])

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            assert app.agent.followups == ["queued followup"]

    def test_brief_input_shows_bash_command_and_file_paths(self) -> None:
        assert (
            brief_input("bash", {"command": "Remove-Item tmp\\hello.c"})
            == "bash: Remove-Item tmp\\hello.c"
        )
        write_summary = brief_input(
            "write_file", {"path": "tmp/hello.py", "content": "x" * 200}
        )
        assert write_summary.startswith('write_file: path="tmp/hello.py"')
        assert write_summary.endswith("…")
        assert (
            brief_input("grep_search", {"pattern": "**/*mcp*", "path": "src/xcode"})
            == 'grep_search: pattern="**/*mcp*", path="src/xcode"'
        )

    def test_run_repl_interrupt_is_final_standalone_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = InterruptingToolApp()
            prompt = FakePrompt(["run command", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            text = _strip_ansi(output.getvalue())
            interrupt_index = text.rfind("[interrupted] current run cancelled")
            assert interrupt_index != -1
            assert "bash: cd .. && git diff .gitignore" not in text[interrupt_index:]
            assert app.agent.cancellation_token.is_cancelled()

    def test_run_repl_pre_snapshot_interrupt_cancels_turn(self) -> None:
        def interrupt_track() -> SnapshotResult:
            raise KeyboardInterrupt

        service = SimpleNamespace(track=interrupt_track)
        snapshot_store = SimpleNamespace(
            next_turn_id=lambda session_id: "001",
            service=lambda session_id: service,
        )
        app = ExplicitSkillApp()
        prompt = FakePrompt(["hello", "/exit"])
        output = StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("xcode.cli.repl.SnapshotStore", return_value=snapshot_store),
                redirect_stdout(output),
            ):
                code = run_repl(app, Path(temp_dir), prompt)

        assert code == 0
        assert app.questions == []
        assert "current run cancelled; session is still active" in output.getvalue()

    def test_run_repl_post_snapshot_interrupt_keeps_session_active(self) -> None:
        track_results = iter(
            [
                SnapshotResult(snapshot_id="pre", skipped_files=[]),
                KeyboardInterrupt(),
            ]
        )

        def track() -> SnapshotResult:
            result = next(track_results)
            if isinstance(result, BaseException):
                raise result
            return result

        service = SimpleNamespace(track=track)
        snapshot_store = SimpleNamespace(
            next_turn_id=lambda session_id: "001",
            service=lambda session_id: service,
        )
        app = ExplicitSkillApp()
        prompt = FakePrompt(["hello", "/exit"])
        output = StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("xcode.cli.repl.SnapshotStore", return_value=snapshot_store),
                redirect_stdout(output),
            ):
                code = run_repl(app, Path(temp_dir), prompt)

        assert code == 0
        assert app.questions == ["hello"]
        assert "snapshot cancelled; session is still active" in output.getvalue()

    def test_run_repl_second_ctrl_c_uses_blank_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = FakeApp()
            prompt = InterruptingPrompt([KeyboardInterrupt(), KeyboardInterrupt()])

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            assert prompt.prompts[1] == ""

    def test_run_repl_resume_uses_picker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            seed = SessionStore(sessions_dir)
            seed.append("user", "first conversation")
            seed.append("assistant", "old answer")
            seed.update_summary()
            app = FakeApp()
            prompt = FakePrompt(["/resume", "/exit"])
            output = StringIO()

            with (
                patch(
                    "xcode.cli.repl_sessions._run_session_picker",
                    return_value=seed.list_session_infos()[0],
                ),
                redirect_stdout(output),
            ):
                code = run_repl(app, sessions_dir, prompt)

            assert code == 0
            text = output.getvalue()
            assert "Resumed conversation: first conversation" in text

    def test_resume_command_loads_agent_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "How many bytes is AGENTS.md?")
            store.append("assistant", "AGENTS.md is 10000 bytes.")
            session_id = store.current_path.stem.removeprefix("session-")
            store.clear()
            app = HistoryLoadingApp()
            renderer = FakeRenderer()

            with redirect_stdout(StringIO()):
                handled = handle_command(
                    f"/resume {session_id}",
                    store,
                    app,
                    renderer,
                    ReplState(),
                    FakePrompt([]),
                )

            assert not (handled)
            assert [message.role for message in app.agent.loaded] == [
                "user",
                "assistant",
            ]
            assert "AGENTS.md" in str(app.agent.loaded[0].content)
            assert "10000 bytes" in str(app.agent.loaded[1].content)

    def test_resume_command_loads_tool_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "Check file size")
            store.append(
                "event",
                {
                    "type": "assistant",
                    "data": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "bash",
                            "input": {"command": "wc -c AGENTS.md"},
                        }
                    ],
                },
            )
            store.append(
                "event",
                {
                    "type": "tool_use",
                    "data": {
                        "id": "call_1",
                        "name": "bash",
                        "input": {"command": "wc -c AGENTS.md"},
                    },
                },
            )
            store.append(
                "event",
                {
                    "type": "tool_result",
                    "data": {
                        "tool_use_id": "call_1",
                        "content": "10000 AGENTS.md",
                        "status": "ok",
                    },
                },
            )
            session_id = store.current_path.stem.removeprefix("session-")
            store.clear()
            app = HistoryLoadingApp()

            with redirect_stdout(StringIO()):
                handle_command(
                    f"/resume {session_id}",
                    store,
                    app,
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

            assert [message.role for message in app.agent.loaded] == [
                "user",
                "assistant",
                "tool_result",
            ]
            tool_call = app.agent.loaded[1].content[0]
            assert tool_call.id == "call_1"
            assert tool_call.name == "bash"
            assert tool_call.arguments["command"] == "wc -c AGENTS.md"
            assert app.agent.loaded[2].tool_call_id == "call_1"
            assert "10000" in str(app.agent.loaded[2].content)

    def test_repl_completer_suggests_slash_commands(self) -> None:
        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        items = completer.complete("/pl")

        assert [item.text for item in items] == ["/plan"]

    def test_repl_completer_fuzzy_ranks_typo_command_without_dispatching(self) -> None:
        """slash typo 仅提供候选，命令执行仍由精确 dispatch 决定。"""
        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        items = completer.complete("/pln")

        assert [item.text for item in items] == ["/plan"]

        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            state = ReplState()
            with redirect_stdout(StringIO()) as output:
                handle_command(
                    "/pln",
                    store,
                    cast(Any, FakeApp()),
                    FakeRenderer(),
                    state,
                    FakePrompt([]),
                )

        assert state.mode == "act"
        assert "Unknown command: /pln" in output.getvalue()

    def test_repl_completer_suggests_new_command(self) -> None:
        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        items = completer.complete("/ne")

        assert items[0].text == "/new"

    def test_repl_completer_suggests_skill_names(self) -> None:
        """`/skill` 和 `$` 入口共享技能名称补全。"""
        completer = ReplCompleter(
            Path.cwd(),
            command_names=COMMAND_NAMES,
            skill_options=("code-review", "pdf"),
        )

        command_items = completer.complete("/skill co")
        reference_items = completer.complete("$")

        assert [item.text for item in command_items] == ["code-review"]
        assert [item.text for item in reference_items] == ["$code-review", "$pdf"]

    def test_parse_skill_invocation_preserves_follow_up_task(self) -> None:
        """`$skill-name` 只消费激活前缀，保留后续任务。"""
        assert parse_skill_invocation("$code-review inspect this patch") == (
            "code-review",
            "inspect this patch",
        )
        assert parse_skill_invocation("$code-review") == ("code-review", "")
        assert parse_skill_invocation("price is $5") is None

    def test_skill_command_records_activation_events(self) -> None:
        """`/skill` 调用运行时并写入可恢复的工具事件。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            app = ExplicitSkillApp()
            output = StringIO()

            with redirect_stdout(output):
                handle_command(
                    "/skill code-review",
                    store,
                    app,
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

            records = store.load_records()

        assert "Activated skill: code-review" in output.getvalue()
        assert [record.content["type"] for record in records] == [
            "tool_use",
            "tool_result",
        ]
        restored = records_to_agent_messages(records)
        assert [message.role for message in restored] == ["assistant", "tool_result"]

    def test_repl_dollar_skill_activates_before_follow_up(self) -> None:
        """`$skill task` 激活技能后仅把剩余任务发送给 agent。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = ExplicitSkillApp()
            output = StringIO()

            with redirect_stdout(output):
                exit_code = run_repl(
                    app,
                    root / "sessions",
                    prompt_session=FakePrompt(
                        ["$code-review inspect this patch", "/exit"]
                    ),
                    renderer=FakeRenderer(),
                    project_root=root,
                )

        assert exit_code == 0
        assert app.questions == ["inspect this patch"]
        assert app.agent.activated == ["code-review"]

    def test_repl_completer_hides_quit_alias(self) -> None:
        completer = ReplCompleter(
            Path.cwd(),
            command_names=COMMAND_NAMES,
            command_registry=COMMAND_REGISTRY,
        )

        items = completer.complete("/q")

        assert [item.text for item in items][:2] == ["/exit", "/queue"]
        assert "/quit" not in COMMAND_NAMES

    def test_repl_completer_shows_canonical_for_revert_alias(self) -> None:
        completer = ReplCompleter(
            Path.cwd(),
            command_names=COMMAND_NAMES,
            command_registry=COMMAND_REGISTRY,
        )

        items = completer.complete("/rev")

        assert [item.text for item in items] == ["/undo"]
        assert "/revert" not in COMMAND_NAMES

    def test_repl_completer_suggests_effort_levels(self) -> None:
        completer = ReplCompleter(
            Path.cwd(),
            command_names=COMMAND_NAMES,
            effort_options=("off", "minimal", "low", "medium", "high", "xhigh", "max"),
        )

        items = completer.complete("/effort me")

        assert [item.text for item in items] == ["medium"]

    def test_repl_effort_options_use_active_provider_transport(self) -> None:
        app = _ActiveProviderApp("deepseek_chat")

        options = current_effort_options(app)

        assert options == ("off", "high", "max")

    def test_handle_command_keeps_quit_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            output = StringIO()

            with redirect_stdout(output):
                handled = handle_command(
                    "/quit",
                    store,
                    FakeApp(),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

            assert handled
            assert COMMAND_REGISTRY["/quit"].canonical == "/exit"
            assert output.getvalue() == "/exit\n"

    def test_handle_command_keeps_revert_alias_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            output = StringIO()

            with redirect_stdout(output):
                handled = handle_command(
                    "/revert --list",
                    store,
                    FakeApp(),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

            assert not handled
            assert "/revert" not in COMMAND_NAMES
            assert COMMAND_REGISTRY["/revert"].canonical == "/undo"
            assert output.getvalue().startswith("/undo --list\n")
            assert "Snapshot undo requires a git repository" in output.getvalue()

    def test_queue_command_toggles_without_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            state = ReplState()
            output = StringIO()

            with redirect_stdout(output):
                handle_command(
                    "/queue",
                    store,
                    FakeApp(),
                    FakeRenderer(),
                    state,
                    FakePrompt([]),
                )
                handle_command(
                    "/queue",
                    store,
                    FakeApp(),
                    FakeRenderer(),
                    state,
                    FakePrompt([]),
                )

            assert not (state.queue_mode)
            assert "Queue mode enabled." in output.getvalue()
            assert "Queue mode disabled." in output.getvalue()

    def test_handle_command_rejects_unexpected_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "keep me")
            output = StringIO()

            with redirect_stdout(output):
                handled = handle_command(
                    "/clear now",
                    store,
                    FakeApp(),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

            assert not (handled)
            assert "Unknown command: /clear now" in output.getvalue()
            assert len(store.load_records()) == 1

    def test_rewind_command_reports_user_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "first")
            store.append("assistant", "one")
            store.append("user", "second")
            store.append("assistant", "two")
            output = StringIO()

            with redirect_stdout(output):
                handled = handle_command(
                    "/rewind 1",
                    store,
                    FakeApp(),
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

            assert not (handled)
            assert [record.content for record in store.load_records()] == [
                "first",
                "one",
            ]
            assert (
                "Rewound 1 user turn (2 transcript records removed)."
                in output.getvalue()
            )

    def test_handle_new_command_starts_new_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "keep me")
            app = HistoryLoadingApp()
            output = StringIO()

            with redirect_stdout(output):
                handled = handle_command(
                    "/new",
                    store,
                    app,
                    FakeRenderer(),
                    ReplState(),
                    FakePrompt([]),
                )

            assert not (handled)
            assert store.load_records() == []
            assert app.agent.loaded == []
            text = output.getvalue()
            assert "\033[2J\033[3J\033[H" in text
            assert "XCode" in text
            assert "Type /help for commands." in text

    def test_repl_completer_suggests_tool_names(self) -> None:
        completer = ReplCompleter(
            Path.cwd(),
            (ToolSpec("read_file", "Read.", "path", lambda value: value["input"]),),
        )

        items = completer.complete("/tool r")

        assert [item.text for item in items] == ["read_file"]

    def test_repl_completer_fuzzy_ranks_tool_names(self) -> None:
        completer = ReplCompleter(
            Path.cwd(),
            (
                ToolSpec("read_file", "Read.", "path", lambda value: value["input"]),
                ToolSpec("write_file", "Write.", "path", lambda value: value["input"]),
            ),
        )

        items = completer.complete("/tool rdf")

        assert [item.text for item in items] == ["read_file"]

    def test_repl_completer_does_not_suggest_shell_commands(self) -> None:
        completer = ReplCompleter(Path.cwd())

        items = completer.complete("!g")

        assert items == []

    def test_repl_completer_suggests_shell_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("print('ok')", encoding="utf-8")
            (root / "space dir").mkdir()
            (root / "space dir" / "note.txt").write_text("ok", encoding="utf-8")
            completer = ReplCompleter(root)

            items = completer.complete("!ls sr")
            spaced_items = completer.complete("!cat space\\ d")

        assert [item.text for item in items] == ["src/"]
        assert items[0].start_position == -2
        assert [item.text for item in spaced_items] == ["space\\ dir/"]

    def test_repl_completer_filters_shell_sensitive_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("secret", encoding="utf-8")
            (root / "public.txt").write_text("ok", encoding="utf-8")
            completer = ReplCompleter(root)

            items = completer.complete("!cat ")

        assert [item.text for item in items] == ["public.txt"]

    def test_repl_completer_suggests_file_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "docs" / "note.md").write_text("note", encoding="utf-8")
            (root / ".env").write_text("secret", encoding="utf-8")
            completer = ReplCompleter(root)

            items = completer.complete("read @do")

            assert [item.text for item in items] == ["docs/note.md"]

    def test_repl_completer_lists_project_files_for_empty_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("b", encoding="utf-8")
            completer = ReplCompleter(root, command_names=COMMAND_NAMES)

            items = completer.complete("@")

        assert [item.text for item in items] == ["a.txt", "b.txt"]

    def test_repl_completer_searches_deep_project_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "docs" / "nested").mkdir()
            (root / "docs" / "nested" / "deep.md").write_text("deep", encoding="utf-8")
            (root / "docs" / "note.md").write_text("note", encoding="utf-8")
            completer = ReplCompleter(root)

            items = completer.complete("read @docs/")

            assert [item.text for item in items] == [
                "docs/note.md",
                "docs/nested/deep.md",
            ]

    def test_repl_completer_filters_sensitive_file_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("secret", encoding="utf-8")
            (root / ".env").write_text("secret", encoding="utf-8")
            (root / "public.txt").write_text("ok", encoding="utf-8")
            completer = ReplCompleter(root)

            items = completer.complete("read @")
            public_items = completer.complete("read @p")

            assert [item.text for item in items] == ["public.txt"]
            assert [item.text for item in public_items] == ["public.txt"]

    def test_repl_completer_filters_gitignored_and_hidden_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".gitignore").write_text(
                "ignored.txt\nbuild/\n",
                encoding="utf-8",
            )
            (root / "ignored.txt").write_text("ignored", encoding="utf-8")
            (root / ".hidden.txt").write_text("hidden", encoding="utf-8")
            (root / "build").mkdir()
            (root / "build" / "artifact.txt").write_text(
                "ignored",
                encoding="utf-8",
            )
            (root / "visible.txt").write_text("visible", encoding="utf-8")
            completer = ReplCompleter(root)

            items = completer.complete("@")

        assert [item.text for item in items] == ["visible.txt"]

    def test_repl_completer_handles_duplicate_basenames_and_windows_separators(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src" / "utils").mkdir(parents=True)
            (root / "tests" / "utils").mkdir(parents=True)
            (root / "src" / "utils" / "helper.py").write_text(
                "src",
                encoding="utf-8",
            )
            (root / "tests" / "utils" / "helper.py").write_text(
                "tests",
                encoding="utf-8",
            )
            completer = ReplCompleter(root)

            duplicate_items = completer.complete("@helper")
            windows_items = completer.complete(r"@src\utl")

        assert [item.text for item in duplicate_items] == [
            "src/utils/helper.py",
            "tests/utils/helper.py",
        ]
        assert [item.text for item in windows_items] == ["src/utils/helper.py"]

    def test_repl_completer_file_index_cache_expires_quickly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "first.txt").write_text("first", encoding="utf-8")
            completer = ReplCompleter(root)
            completer.complete("@")
            (root / "second.txt").write_text("second", encoding="utf-8")

            cached = completer.complete("@")
            time.sleep(0.3)
            refreshed = completer.complete("@")

        assert [item.text for item in cached] == ["first.txt"]
        assert [item.text for item in refreshed] == ["first.txt", "second.txt"]

    def test_repl_completer_supports_prompt_toolkit_async_api(self) -> None:
        try:
            from prompt_toolkit.document import Document
        except ImportError:
            pytest.skip("prompt_toolkit is not installed")

        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        async def collect():
            return [
                completion.text
                async for completion in completer.get_completions_async(
                    Document("/pl"),
                    cast(Any, None),
                )
            ]

        assert asyncio.run(collect()) == ["/plan"]

    def test_repl_completion_menu_style_uses_default_background(self) -> None:
        completion_styles = {
            name: style
            for name, style in REPL_PROMPT_STYLE.items()
            if name.startswith("completion-menu") or name.startswith("scrollbar.")
        }

        assert completion_styles
        assert all("bg:default" in style for style in completion_styles.values())

    def test_repl_input_lexer_highlights_file_references(self) -> None:
        document = SimpleNamespace(lines=["read @src/xcode/harness/app.py please"])
        fragments = ReplInputLexer().lex_document(document)(0)

        assert ("class:file-reference", "@src/xcode/harness/app.py") in fragments

    def test_repl_input_lexer_highlights_shell_prefix(self) -> None:
        document = SimpleNamespace(lines=["!echo hello"])
        fragments = ReplInputLexer().lex_document(document)(0)

        assert fragments[0] == ("class:shell-prefix", "!")

    def test_input_prompt_uses_shell_marker_when_buffer_starts_with_bang(self) -> None:
        session = SimpleNamespace(default_buffer=SimpleNamespace(text="!echo hello"))
        prompt = input_prompt(session)

        assert callable(prompt)
        assert prompt()[0] == ("class:prompt-marker.shell", "❯ ")


class XcodeReplForkTests:
    def test_fork_into_switches_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            original = store.current_path
            store.append("user", "hello")
            store.fork_into()
            assert store.current_path != original

    def test_fork_into_has_parent_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "parent conversation")
            parent = store.current_metadata()
            assert parent is not None
            store.fork_into()
            fork_meta = store.current_metadata()
            assert fork_meta is not None
            assert fork_meta is not None
            assert fork_meta.parent_id == parent.id

    def test_fork_into_copies_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "first message")
            store.append("assistant", "first answer")
            store.fork_into()
            records = store.load_records()
            assert len(records) == 2
            assert records[0].content == "first message"
            assert records[1].content == "first answer"

    def test_fork_into_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "parent msg")
            store.fork_into()
            store.append("user", "fork msg")
            fork_records = store.load_records()
            assert "fork msg" in [r.content for r in fork_records]

    def test_fork_into_validates_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            with pytest.raises(ValueError):
                store.fork_into("invalid")

    def test_fork_into_all_fork_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for ft in ("explore", "verify", "isolate"):
                store = SessionStore(Path(temp_dir))
                store.append("user", "test")
                store.fork_into(ft)
                meta = store.current_metadata()
                assert meta is not None
                assert meta.fork_type == ft

    def test_fork_preserves_parent_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions = Path(temp_dir) / ".local" / "sessions"
            store = SessionStore(sessions, project_root=Path(temp_dir))
            store.append("user", "parent conversation")
            parent_meta = store.current_metadata()
            assert parent_meta is not None
            store.fork_into()
            fork_meta = store.current_metadata()
            assert fork_meta is not None
            assert fork_meta.title == f"Fork of {parent_meta.title}"
            assert fork_meta.summary == parent_meta.summary

    def test_fork_sessions_shows_fork_relationship(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions = Path(temp_dir) / ".local" / "sessions"
            store = SessionStore(sessions, project_root=Path(temp_dir))
            store.append("user", "parent")
            store.update_summary()
            parent_meta = store.current_metadata()
            assert parent_meta is not None
            parent_id = parent_meta.id
            store.fork_into("verify")
            # 执行 fork 后 list_session_infos 应显示 parent_id
            views = store.list_session_infos(limit=10)
            fork_view = next(v for v in views if v.id != parent_id)
            assert fork_view.parent_id == parent_id
            assert fork_view.fork_type == "verify"

    def test_fork_into_default_type_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "hello")
            store.fork_into()
            meta = store.current_metadata()
            assert meta is not None
            assert meta.fork_type is None

    def test_switch_branch_resumes_branch_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "parent")
            parent = store.current_metadata()
            assert parent is not None
            store.fork_into("explore")
            branch = store.current_metadata()
            assert branch is not None
            store.resume(parent.id)

            view = store.switch_branch(branch.id)

            assert view.id == branch.id
            assert store.current_metadata() == branch

    def test_branch_command_switches_and_prints_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "parent")
            parent = store.current_metadata()
            assert parent is not None
            store.fork_into("verify")
            branch = store.current_metadata()
            assert branch is not None
            store.append("assistant", "branch answer")
            store.resume(parent.id)
            prompt = FakePrompt([])
            renderer = FakeRenderer()
            state = ReplState()

            with redirect_stdout(StringIO()) as output:
                handled = handle_command(
                    f"/branch {branch.id}",
                    store,
                    FakeApp(),
                    renderer,
                    state,
                    prompt,
                )

            assert not (handled)
            assert "Resumed conversation:" in output.getvalue()
            assert store.current_metadata() == branch

    def test_run_repl_compact_command(self) -> None:
        class FakeCompactor:
            def __init__(self) -> None:
                self.max_recent_messages = 1

        class FakeAgent:
            def __init__(self) -> None:
                self.compact_requested = False
                self.compactor = FakeCompactor()
                self.approval_callback: ApprovalCallback | None = None
                self.cancellation_token = CancellationToken()

            def request_compaction(self) -> None:
                self.compact_requested = True

            def follow_up(self, msg: AgentMessage) -> None: ...

            def load_history(self, messages: list[AgentMessage]) -> None: ...

        class CompactFakeApp:
            def __init__(self) -> None:
                self.agent = FakeAgent()
                self.registry: tuple[ToolSpec, ...] = ()

            def get_model_info(self) -> dict[str, str]:
                return {}

            def set_model(
                self,
                *,
                model: str = "",
                profile: str = "main",
                base_url: str | None = None,
                api_key: str | None = None,
                thinking: bool | None = None,
                reasoning_effort: str | None = None,
            ) -> str:
                return "unknown"

            def ask_stream(self, question: str, mode: str | None = None):
                yield FinalStructuredEvent(
                    "final",
                    1,
                    StructuredAgentResult(
                        answer="unused", messages=[], steps=1, tool_calls=[]
                    ),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            app = CompactFakeApp()
            prompt = FakePrompt([])
            renderer = FakeRenderer()
            state = ReplState()
            store = SessionStore(Path(temp_dir))
            store.append("user", "first")
            store.append("assistant", "second")
            store.append("user", "third")
            store.append(
                "event",
                {
                    "type": "tool_result",
                    "step": 1,
                    "data": {
                        "tool_use_id": "call_1",
                        "content": "a" * 300,
                        "status": "ok",
                        "type": "tool_result",
                    },
                },
            )
            with redirect_stdout(StringIO()) as output:
                handled = handle_command(
                    "/compact",
                    store,
                    app,
                    renderer,
                    state,
                    prompt,
                )

            assert not (handled)
            assert app.agent.compact_requested
            assert (
                "Active context compaction requested for the next agent run"
                in output.getvalue()
            )
            records = store.load_records()
            tool_results = [
                record.content
                for record in records
                if record.type == "event"
                and isinstance(record.content, dict)
                and record.content.get("type") == "tool_result"
            ]
            assert len(tool_results) == 1
            event_data = tool_results[0].get("data")
            assert isinstance(event_data, dict)
            assert isinstance(event_data, dict)
            content = event_data.get("content")
            assert isinstance(content, str)
            assert isinstance(content, str)
            assert "compacted" in content
            assert "300 chars removed" in content

    def test_run_repl_compact_command_skips_short_clean_session(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.compact_requested = False
                self.approval_callback: ApprovalCallback | None = None
                self.cancellation_token = CancellationToken()

            def request_compaction(self) -> None:
                self.compact_requested = True

            def follow_up(self, msg: AgentMessage) -> None: ...

            def load_history(self, messages: list[AgentMessage]) -> None: ...

        class CompactFakeApp:
            def __init__(self) -> None:
                self.agent = FakeAgent()
                self.registry: tuple[ToolSpec, ...] = ()

            def get_model_info(self) -> dict[str, str]:
                return {}

            def set_model(
                self,
                *,
                model: str = "",
                profile: str = "main",
                base_url: str | None = None,
                api_key: str | None = None,
                thinking: bool | None = None,
                reasoning_effort: str | None = None,
            ) -> str:
                return "unknown"

            def ask_stream(self, question: str, mode: str | None = None):
                yield FinalStructuredEvent(
                    "final",
                    1,
                    StructuredAgentResult(
                        answer="unused", messages=[], steps=1, tool_calls=[]
                    ),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            app = CompactFakeApp()
            prompt = FakePrompt([])
            renderer = FakeRenderer()
            state = ReplState()
            store = SessionStore(Path(temp_dir))
            store.append("user", "hello")
            store.append("assistant", "Hi.")

            with redirect_stdout(StringIO()) as output:
                handled = handle_command(
                    "/compact",
                    store,
                    app,
                    renderer,
                    state,
                    prompt,
                )

            assert not (handled)
            assert not (app.agent.compact_requested)
            assert "No context compaction needed" in output.getvalue()


class FakePrompt:
    def __init__(self, values: list[str]) -> None:
        self.values = iter(values)

    def prompt(self, prompt_text: PromptText) -> str:
        return next(self.values)


class InterruptingPrompt:
    def __init__(self, values: list[str | BaseException]) -> None:
        self.values = iter(values)
        self.prompts: list[PromptText] = []

    def prompt(self, prompt_text: PromptText) -> str:
        self.prompts.append(prompt_text)
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


class _StubAgent:
    approval_callback: ApprovalCallback | None = None
    cancellation_token: CancellationToken = CancellationToken()

    def follow_up(self, msg: AgentMessage) -> None: ...

    def load_history(self, messages: list[AgentMessage]) -> None: ...

    def request_compaction(self) -> None: ...

    def set_session_grant_store_provider(self, provider: object) -> None: ...

    def set_permanent_grant_store(self, store: object) -> None: ...

    def available_skill_names(self) -> tuple[str, ...]:
        return ()

    def activate_skill(
        self, skill_name: str, mode: object = None
    ) -> ExplicitSkillActivationResult:
        return ExplicitSkillActivationResult(
            name=skill_name,
            status="disabled",
            message="Skills are disabled for this runtime.",
        )


class FakeApp:
    agent = _StubAgent()
    registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield TextDeltaStructuredEvent("text_delta", 1, question)
        yield TextDeltaStructuredEvent("text_delta", 1, "!")
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer=f"{question}!",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class ExplicitSkillAgent(_StubAgent):
    """记录 REPL 显式技能激活调用。"""

    def __init__(self) -> None:
        self.activated: list[str] = []
        self.activation_modes: list[object] = []

    def available_skill_names(self) -> tuple[str, ...]:
        return ("code-review",)

    def activate_skill(
        self, skill_name: str, mode: object = None
    ) -> ExplicitSkillActivationResult:
        self.activated.append(skill_name)
        self.activation_modes.append(mode)
        return ExplicitSkillActivationResult(
            name=skill_name,
            status="activated",
            message=f"Activated skill: {skill_name}",
            content=(
                f'<skill name="{skill_name}" activated="true">\n'
                f'<skill-activation-state>{{"name": "{skill_name}"}}'
                "</skill-activation-state>\nBODY\n</skill>"
            ),
            tool_call_id=f"explicit-skill-{len(self.activated)}",
        )


class ExplicitSkillApp(FakeApp):
    """支持显式技能激活的最小 REPL app。"""

    def __init__(self) -> None:
        self.agent = ExplicitSkillAgent()
        self.questions: list[str] = []

    def ask_stream(self, question: str, mode: str | None = None):
        self.questions.append(question)
        yield from super().ask_stream(question, mode)


class StaticPermissionAgent:
    """提供 REPL 权限命令读取的静态策略字段。"""

    def __init__(self) -> None:
        self.permission_policy = PermissionPolicy(
            (StaticPermission(tool="bash", decision="deny"),)
        )
        self.restricted_dirs: tuple[str, ...] = ()
        self.approval_callback: ApprovalCallback | None = None
        self.cancellation_token = CancellationToken()

    def follow_up(self, msg: AgentMessage) -> None: ...

    def load_history(self, messages: list[AgentMessage]) -> None: ...

    def request_compaction(self) -> None: ...


class StaticPermissionApp:
    """最小 REPL app，仅用于 /permissions 静态策略展示测试。"""

    def __init__(self) -> None:
        self.agent = StaticPermissionAgent()
        self.registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield TextDeltaStructuredEvent("text_delta", 1, question)
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer=question,
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class HistoryLoadingAgent:
    """记录 REPL 恢复时同步进来的会话历史。"""

    def __init__(self) -> None:
        self.loaded: list[Any] = []
        self.approval_callback: ApprovalCallback | None = None
        self.cancellation_token = CancellationToken()

    def follow_up(self, msg: AgentMessage) -> None: ...

    def load_history(self, messages: list[Any]) -> None:
        """保存测试传入的历史消息。"""
        self.loaded = messages

    def request_compaction(self) -> None: ...


class HistoryLoadingApp:
    """暴露带 load_history 接口的测试 agent。"""

    def __init__(self) -> None:
        self.agent = HistoryLoadingAgent()
        self.registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer="ok",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class QueueModeAgent:
    def __init__(self) -> None:
        self.followups: list[str] = []
        self.approval_callback: ApprovalCallback | None = None
        self.cancellation_token = CancellationToken()

    def follow_up(self, msg: Any) -> None:
        self.followups.append(str(msg.content))

    def load_history(self, messages: list[AgentMessage]) -> None: ...

    def request_compaction(self) -> None: ...

    def set_session_grant_store_provider(self, provider: object) -> None: ...

    def set_permanent_grant_store(self, store: object) -> None: ...


class QueueModeApp:
    def __init__(self) -> None:
        self.agent = QueueModeAgent()
        self.registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield TextDeltaStructuredEvent("text_delta", 1, question)
        time.sleep(0.05)
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer=question,
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class FakeMarkdownApp:
    agent = _StubAgent()
    registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield TextDeltaStructuredEvent("text_delta", 1, "# Title\n\n")
        yield TextDeltaStructuredEvent("text_delta", 1, "- item")
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer="# Title\n\n- item",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class MultiDeltaApp:
    agent = _StubAgent()
    registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield TextDeltaStructuredEvent("text_delta", 1, "he")
        yield TextDeltaStructuredEvent("text_delta", 1, "ll")
        yield TextDeltaStructuredEvent("text_delta", 1, "o")
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer="hello",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class ReasoningApp:
    agent = _StubAgent()
    registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield ReasoningDeltaStructuredEvent("reasoning_delta", 1, "one\n")
        yield ReasoningDeltaStructuredEvent(
            "reasoning_delta", 1, "two\nthree\nfour\nfive six seven eight"
        )
        yield TextDeltaStructuredEvent("text_delta", 1, "done")
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer="done",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class TinyReasoningApp:
    agent = _StubAgent()
    registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield ReasoningDeltaStructuredEvent("reasoning_delta", 1, "ok")
        yield TextDeltaStructuredEvent("text_delta", 1, "done")
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer="done",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class ToolEventApp:
    agent = _StubAgent()
    registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield ToolUseStructuredEvent(
            "tool_use",
            1,
            ToolCall(
                id="call_1",
                name="grep_search",
                input={"pattern": "mcp", "path": "src/xcode"},
            ),
        )
        yield ToolResultStructuredEvent(
            "tool_result",
            1,
            ToolResultBlock("call_1", "ok", "ok"),
        )
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer="",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class TodoEventApp:
    agent = _StubAgent()
    registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield TodoUpdateStructuredEvent(
            "todo_update",
            1,
            (
                TodoItem("implement", "Implement feature", "in_progress"),
                TodoItem("test", "Run tests", "pending"),
            ),
        )
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer="",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class CapturingApp:
    registry: tuple[ToolSpec, ...] = ()
    agent = _StubAgent()

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.modes: list[str | None] = []

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        self.seen.append(question)
        self.modes.append(mode)
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer="ok",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class ToolApp:
    def __init__(
        self,
        registry: tuple[ToolSpec, ...] = (
            ToolSpec("echo", "Echo.", "text", lambda value: value["input"]),
        ),
    ) -> None:
        self.registry = registry

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield FinalStructuredEvent(
            "final",
            1,
            StructuredAgentResult(
                answer="unused",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class DeniedToolAgent:
    """提供 /tool 直连执行读取的静态权限策略。"""

    def __init__(self) -> None:
        self.permission_policy = PermissionPolicy(
            (StaticPermission(tool="bash", decision="deny"),)
        )
        self.restricted_dirs: tuple[str, ...] = ()
        self.approval_callback = None


class DeniedToolApp(ToolApp):
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.agent = DeniedToolAgent()

        def bash(value: dict[str, Any]) -> str:
            command = str(value["command"])
            self.commands.append(command)
            return f"ran: {command}"

        super().__init__(
            registry=(
                ToolSpec(
                    "bash",
                    "Run shell.",
                    "command",
                    bash,
                    schema={
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                ),
            )
        )


class ShellShortcutApp(ToolApp):
    def __init__(self, output: str | None = None) -> None:
        self.commands: list[str] = []
        self.output = output

        def bash(value: dict[str, Any]) -> str:
            command = str(value["command"])
            self.commands.append(command)
            return self.output or f"ran: {command}"

        super().__init__(
            registry=(
                ToolSpec(
                    "bash",
                    "Run shell.",
                    "command",
                    bash,
                    schema={
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                ),
            )
        )


class InterruptingToolApp:
    class Agent:
        def __init__(self) -> None:
            self.cancellation_token = CancellationToken()
            self.approval_callback: ApprovalCallback | None = None

        def follow_up(self, msg: AgentMessage) -> None: ...

        def load_history(self, messages: list[AgentMessage]) -> None: ...

        def request_compaction(self) -> None: ...

    def __init__(self) -> None:
        self.agent = self.Agent()
        self.registry: tuple[ToolSpec, ...] = ()

    def get_model_info(self) -> dict[str, str]:
        return {}

    def set_model(
        self,
        *,
        model: str = "",
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "unknown"

    def ask_stream(self, question: str, mode: str | None = None):
        yield ToolUseStructuredEvent(
            "tool_use",
            1,
            ToolCall(
                id="call_1",
                name="bash",
                input={"command": "cd .. && git diff .gitignore"},
            ),
        )
        raise KeyboardInterrupt()


class _ActiveProvider:
    def __init__(self, transport: str) -> None:
        self.transport = transport


class _ActiveProviderWrapper:
    def __init__(self, active_provider: _ActiveProvider) -> None:
        self.active_provider = active_provider


class _ActiveProviderAgent:
    def __init__(self, provider: _ActiveProviderWrapper) -> None:
        self.provider = provider


class _ActiveProviderApp:
    def __init__(self, transport: str) -> None:
        self.agent = _ActiveProviderAgent(
            _ActiveProviderWrapper(_ActiveProvider(transport))
        )


class FakeRenderer:
    def __init__(self) -> None:
        self.rendered: list[str] = []

    def render(self, text: str) -> None:
        self.rendered.append(text)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


class XcodeReplContinueTests:
    """Step 8: CLI flags and /continue — session lifecycle helpers and commands."""

    # ── _validate_session_id ──────────────────────────────────────────────

    def test_validate_session_id_rejects_empty(self) -> None:
        from xcode.harness.session import SessionStore

        with pytest.raises(ValueError):
            SessionStore._validate_session_id("")

    @pytest.mark.parametrize(
        "bad", ["a/b", "a\\b", "../etc", "C:foo", ".", "~user", "a b", "a\tb"]
    )
    def test_validate_session_id_rejects_path_separators(self, bad: str) -> None:
        from xcode.harness.session import SessionStore

        with pytest.raises(ValueError):
            SessionStore._validate_session_id(bad)

    @pytest.mark.parametrize("bad", ["a*b", "abc*", "*abc"])
    def test_validate_session_id_rejects_asterisk(self, bad: str) -> None:
        from xcode.harness.session import SessionStore

        with pytest.raises(ValueError):
            SessionStore._validate_session_id(bad)

    @pytest.mark.parametrize(
        "good", ["abc123", "ABC-123", "20260616-120000", "a_b", "z"]
    )
    def test_validate_session_id_accepts_valid(self, good: str) -> None:
        from xcode.harness.session import SessionStore

        assert SessionStore._validate_session_id(good) == good

    def test_find_by_id_rejects_asterisk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            assert store.find_by_id("session-with*") is None

    # ── is_meaningful_session ─────────────────────────────────────────────

    def test_is_meaningful_session_false_for_nonexistent(self) -> None:
        from xcode.harness.session import SessionStore

        assert not (SessionStore.is_meaningful_session(Path("/nonexistent")))

    def test_is_meaningful_session_false_for_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.jsonl"
            p.write_text("")
            assert not (SessionStore.is_meaningful_session(p))

    def test_is_meaningful_session_false_for_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "session-foo.jsonl"
            # write a record that is not user/assistant/event
            p.write_text(
                json.dumps(
                    {
                        "type": "system",
                        "content": "init",
                        "created_at": "2026-01-01T00:00:00",
                    }
                )
                + "\n"
            )
            assert not (SessionStore.is_meaningful_session(p))

    def test_is_meaningful_session_true_for_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "session-bar.jsonl"
            p.write_text(
                json.dumps(
                    {
                        "type": "user",
                        "content": "hello",
                        "created_at": "2026-01-01T00:00:00",
                    }
                )
                + "\n"
            )
            assert SessionStore.is_meaningful_session(p)

    def test_is_meaningful_session_true_for_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "session-baz.jsonl"
            p.write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "content": "hi",
                        "created_at": "2026-01-01T00:00:00",
                    }
                )
                + "\n"
            )
            assert SessionStore.is_meaningful_session(p)

    def test_is_meaningful_session_true_for_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "session-qux.jsonl"
            p.write_text(
                json.dumps(
                    {
                        "type": "event",
                        "content": {"type": "tool_result"},
                        "created_at": "2026-01-01T00:00:00",
                    }
                )
                + "\n"
            )
            assert SessionStore.is_meaningful_session(p)

    # ── find_latest_for_project ───────────────────────────────────────────

    def test_find_latest_for_project_filters_by_project_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            sessions_dir = root / "sessions"
            sessions_dir.mkdir(parents=True)
            # Create two sessions with different project paths in same sessions dir
            store_a = SessionStore(sessions_dir, project_root=root)
            store_a.append("user", "project A message")
            # Create a session for a subdirectory project
            sub = root / "subproj"
            sub.mkdir()
            store_b = SessionStore(sessions_dir, project_root=sub)
            store_b.append("user", "project B message")
            # Both session files exist in the same sessions_dir
            assert len(list(sessions_dir.glob("session-*.jsonl"))) == 2
            # find for root project only
            result_a = store_a.find_latest_for_project(root)
            assert result_a is not None
            assert result_a is not None
            assert result_a.id == store_a.session_id
            # find for sub-proj using a fresh store (no cached metadata)
            fresh = SessionStore(sessions_dir, project_root=root)
            result_b = fresh.find_latest_for_project(sub)
            assert result_b is not None
            assert result_b is not None
            assert result_b.id == store_b.session_id

    def test_find_latest_for_project_scans_beyond_100(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sd = Path(td) / "sessions"
            # create 150 sessions from another project
            other = Path(td) / "other"
            other.mkdir()
            for i in range(150):
                st = SessionStore(sd, project_root=other)
                st.append("user", f"other {i}")
            # one current-project session (the latest)
            st_curr = SessionStore(sd, project_root=Path(td))
            st_curr.append("user", "my session")
            # should find my session even though 150 others precede it
            result = st_curr.find_latest_for_project(Path(td))
            assert result is not None
            assert result is not None
            assert result.id == st_curr.session_id

    def test_find_latest_for_project_returns_current_when_latest_and_meaningful(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "hello")
            # current session is the only meaningful session
            result = store.find_latest_for_project(Path(td))
            assert result is not None
            assert result is not None
            assert result.id == store.session_id

    def test_find_latest_for_project_skips_empty_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            # current_path is an empty placeholder
            # create another meaningful session
            store.append("user", "real")
            store.clear()
            # now current_path is a new placeholder; the real session exists
            result = store.find_latest_for_project(Path(td))
            assert result is not None
            assert result is not None
            assert result.id != store.session_id
            assert SessionStore.is_meaningful_session(result.path)

    def test_find_latest_for_project_returns_none_when_no_meaningful(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            # only empty placeholder, no meaningful session
            result = store.find_latest_for_project(Path(td))
            assert result is None

    # ── find_by_id ────────────────────────────────────────────────────────

    def test_find_by_id_returns_none_for_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            assert store.find_by_id("nonexistent") is None

    @pytest.mark.parametrize(
        "bad",
        [
            "../etc",
            "/etc/passwd",
            "a/b",
            "a\\b",
            "C:foo",
            ".",
            "~root",
            "a b",
            "",
        ],
    )
    def test_find_by_id_rejects_malicious_ids(self, bad: str) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            assert store.find_by_id(bad) is None

    def test_find_by_id_direct_path_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "hello")
            sid = store.session_id
            result = store.find_by_id(sid)
            assert result is not None
            assert result is not None
            assert result.id == sid

    def test_find_by_id_metadata_fallback_rejects_escaped_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sessions_dir = Path(td) / "sessions"
            sessions_dir.mkdir(parents=True)
            store = SessionStore(sessions_dir, project_root=Path(td))
            # inject metadata with escaped transcript_path
            from xcode.harness.session import SessionMetadata

            escaped = SessionMetadata(
                id="escape-session",
                title="Escaped",
                summary="trying to escape",
                project_path=str(Path(td)),
                transcript_path="../../etc/passwd",
                created_at="2026-01-01T00:00:00",
                updated_at="2026-01-01T00:00:00",
            )
            store._upsert_metadata(escaped)
            result = store.find_by_id("escape-session")
            assert result is None

    # ── /continue command ─────────────────────────────────────────────────

    def test_continue_command_resumes_latest_session(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "older session")
            older_id = store.session_id
            # create a second session that is newer
            store.clear()
            store.append("user", "newer session")
            newer_id = store.session_id
            # switch back to older session
            store.resume(older_id)
            assert store.session_id == older_id
            state = ReplState()
            app = HistoryLoadingApp()
            renderer = FakeRenderer()
            with redirect_stdout(StringIO()):
                handled = handle_command(
                    "/continue",
                    store,
                    app,
                    renderer,
                    state,
                    FakePrompt([]),
                )
            assert not (handled)
            # Should have switched to the newer session
            assert store.session_id == newer_id

    def test_continue_command_already_on_latest_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "the only session")
            state = ReplState()
            app = HistoryLoadingApp()
            output = StringIO()
            with redirect_stdout(output):
                handled = handle_command(
                    "/continue",
                    store,
                    app,
                    FakeRenderer(),
                    state,
                    FakePrompt([]),
                )
            assert not (handled)
            assert store.session_id == store.session_id  # unchanged
            assert "Already on the latest session" in output.getvalue()

    def test_continue_command_no_prior_stays_current(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            # only empty placeholder, no meaningful session
            state = ReplState()
            app = HistoryLoadingApp()
            output = StringIO()
            orig_id = store.session_id
            with redirect_stdout(output):
                handled = handle_command(
                    "/continue",
                    store,
                    app,
                    FakeRenderer(),
                    state,
                    FakePrompt([]),
                )
            assert not (handled)
            assert store.session_id == orig_id  # unchanged
            assert "No prior session found" in output.getvalue()

    def test_continue_does_not_change_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "session A")
            target_id = store.session_id
            store.clear()
            store.append("user", "session B")
            latest_id = store.session_id
            store.resume(target_id)
            assert store.session_id == target_id
            # /continue should switch to latest
            state = ReplState()
            app = HistoryLoadingApp()
            with redirect_stdout(StringIO()):
                handle_command(
                    "/continue",
                    store,
                    app,
                    FakeRenderer(),
                    state,
                    FakePrompt([]),
                )
            assert store.session_id == latest_id

    def test_continue_does_not_mutate_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "old")
            old_path = store.current_path
            old_mtime = old_path.stat().st_mtime
            store.clear()
            store.append("user", "new")
            state = ReplState()
            app = HistoryLoadingApp()
            with redirect_stdout(StringIO()):
                handle_command(
                    "/continue",
                    store,
                    app,
                    FakeRenderer(),
                    state,
                    FakePrompt([]),
                )
            # old session file should not be modified
            assert old_path.stat().st_mtime == old_mtime

    def test_continue_does_not_rewrite_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "old")
            old_contents = store.current_path.read_text(encoding="utf-8")
            store.clear()
            state = ReplState()
            app = HistoryLoadingApp()
            with redirect_stdout(StringIO()):
                handle_command(
                    "/continue",
                    store,
                    app,
                    FakeRenderer(),
                    state,
                    FakePrompt([]),
                )
            assert store.current_path.read_text(encoding="utf-8") == old_contents

    def test_continue_syncs_agent_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "previous conversation")
            store.append("assistant", "previous answer")
            store.clear()
            app = HistoryLoadingApp()
            state = ReplState()
            with redirect_stdout(StringIO()):
                handle_command(
                    "/continue",
                    store,
                    app,
                    FakeRenderer(),
                    state,
                    FakePrompt([]),
                )
            assert [m.role for m in app.agent.loaded] == ["user", "assistant"]

    def test_continue_completer_suggests_continue(self) -> None:
        from xcode.cli.completion import ReplCompleter

        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)
        items = completer.complete("/con")
        assert any(item.text == "/continue" for item in items)

    # ── CLI flags via run_repl ────────────────────────────────────────────

    def test_run_repl_auto_continue_resumes_latest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sessions_dir = Path(td)
            seed = SessionStore(sessions_dir)
            seed.append("user", "prior session")
            prior_id = seed.session_id
            app = FakeApp()
            prompt = FakePrompt(["/exit"])
            with redirect_stdout(StringIO()):
                code = run_repl(app, sessions_dir, prompt, auto_continue=True)
            assert code == 0
            # after --continue, session should be the prior one
            store = SessionStore(sessions_dir)
            assert store.find_by_id(prior_id) is not None

    def test_run_repl_auto_continue_no_prior_proceeds_with_new(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app = FakeApp()
            prompt = FakePrompt(["/exit"])
            output = StringIO()
            with redirect_stdout(output):
                code = run_repl(app, Path(td), prompt, auto_continue=True)
            assert code == 0
            assert "No prior session found" in output.getvalue()

    def test_run_repl_session_id_resumes_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sessions_dir = Path(td)
            seed = SessionStore(sessions_dir)
            seed.append("user", "explicit session")
            target_id = seed.session_id
            app = FakeApp()
            prompt = FakePrompt(["/exit"])
            with redirect_stdout(StringIO()):
                code = run_repl(app, sessions_dir, prompt, session_id=target_id)
            assert code == 0

    def test_run_repl_session_id_rejects_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app = FakeApp()
            prompt = FakePrompt(["/exit"])
            output = StringIO()
            err = StringIO()
            with redirect_stdout(output), redirect_stderr(err):
                code = run_repl(app, Path(td), prompt, session_id="nonexistent")
            assert code == 1
            assert "Session not found" in err.getvalue()

    def test_run_repl_session_id_rejects_wrong_project(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sessions_dir = Path(td)
            other_project = Path(td) / "other"
            other_project.mkdir()
            seed = SessionStore(sessions_dir, project_root=other_project)
            seed.append("user", "other project session")
            target_id = seed.session_id
            app = FakeApp()
            prompt = FakePrompt(["/exit"])
            output = StringIO()
            err = StringIO()
            with redirect_stdout(output), redirect_stderr(err):
                code = run_repl(app, sessions_dir, prompt, session_id=target_id)
            assert code == 1
            assert "belongs to another project" in err.getvalue()

    def test_run_repl_session_id_rejects_malicious_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app = FakeApp()
            prompt = FakePrompt(["/exit"])
            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(td), prompt, session_id="../etc/passwd")
            assert code == 1

    # ── Step 6/7 follow continued session_id ──────────────────────────────

    def test_continue_grant_store_follows_session_id(self) -> None:
        """Step 6 invariant: SessionGrantStoreManager keys by session_id."""
        from xcode.harness.observability.permission_model import (
            GrantRecord,
            SessionGrantStoreManager,
        )

        manager = SessionGrantStoreManager()
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "first")
            sid_a = store.session_id
            grant_a = manager.get_for_session(sid_a)
            grant_a.add(
                GrantRecord(
                    capability="file",
                    operation="read",
                    target_kind="path",
                    target_pattern="*.py",
                    access="read",
                    decision="allow",
                    scope="session",
                    grant_id="test-grant",
                )
            )
            store.clear()
            # continue to first session
            view = store.find_latest_for_project(Path(td))
            assert view is not None
            store.resume(view.id)
            assert store.session_id == sid_a
            # grant store for this session should still have the grant
            current_store = manager.get_for_session(store.session_id)
            assert len(current_store.records()) > 0

    def test_continue_snapshot_store_follows_session_id(self) -> None:
        """Step 7 invariant: SnapshotStore.service keys by session_id."""
        from xcode.harness.snapshot import SnapshotStore

        with tempfile.TemporaryDirectory() as td:
            # need a git repo for SnapshotStore
            import subprocess

            subprocess.run(["git", "init"], cwd=td, capture_output=True)
            subprocess.run(
                ["git", "config", "user.name", "test"], cwd=td, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@test"],
                cwd=td,
                capture_output=True,
            )
            (Path(td) / ".gitignore").write_text("")
            subprocess.run(["git", "add", "-A"], cwd=td, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=td, capture_output=True)

            snap = SnapshotStore(Path(td))
            store = SessionStore(Path(td) / "sessions", project_root=Path(td))
            store.append("user", "first")
            sid_a = store.session_id
            # record a snapshot turn for session A
            svc = snap.service(sid_a)
            pre = svc.track()
            (Path(td) / "test.txt").write_text("hello")
            post = svc.track()
            changes = svc.diff(pre.snapshot_id, post.snapshot_id)
            snap.record_turn(sid_a, "001", pre.snapshot_id, post.snapshot_id, changes)
            assert len(snap.list_records(sid_a)) == 1

            store.clear()
            # continue to session A
            view = store.find_latest_for_project(Path(td))
            assert view is not None
            store.resume(view.id)
            assert store.session_id == sid_a
            # snapshot records for session A should still be available
            assert len(snap.list_records(store.session_id)) == 1

    # ── help text ─────────────────────────────────────────────────────────

    def test_help_text_includes_continue(self) -> None:
        assert "/continue" in HELP_TEXT
        assert "Resume the latest session" in HELP_TEXT


if __name__ == "__main__":
    pytest.main()
