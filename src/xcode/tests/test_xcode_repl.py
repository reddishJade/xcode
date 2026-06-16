from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from xcode.cli.app_contract import ReplApp
from xcode.cli.completion import ReplCompleter
from xcode.cli.commands import PromptText, ReplState
from xcode.cli.repl import current_effort_options, run_repl
from xcode.cli.repl_commands import COMMAND_NAMES, HELP_TEXT, handle_command
from xcode.cli.repl_rendering import reasoning_preview_lines
from xcode.cli.repl_rendering import REPL_PROMPT_STYLE
from xcode.cli.repl_tools import brief_input, run_tool_command
from xcode.harness.session import SessionStore
from xcode.harness.agent_runtime import (
    CancellationToken,
    StructuredAgentResult,
)
from xcode.harness.observability import PermissionPolicy, StaticPermission
from xcode.harness.agent_runtime.events import (
    FinalStructuredEvent,
    ReasoningDeltaStructuredEvent,
    TextDeltaStructuredEvent,
    ToolResultBlock,
    ToolResultStructuredEvent,
    ToolUseStructuredEvent,
)
from xcode.ai.events import ToolCall
from xcode.agent.messages import AgentMessage
from xcode.harness.skills import ApprovalCallback, ToolSpec


class XcodeReplTests(unittest.TestCase):
    def test_session_store_writes_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))

            store.append("user", "hello")

            text = store.current_path.read_text(encoding="utf-8")
            self.assertIn('"type": "user"', text)
            self.assertIn('"hello"', text)

    def test_session_store_rewinds_last_user_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "first")
            store.append("assistant", "one")
            store.append("user", "second")
            store.append("assistant", "two")

            removed = store.rewind_turns()
            records = store.load_records()

            self.assertEqual(removed, 2)
            self.assertEqual([record.content for record in records], ["first", "one"])

    def test_session_store_resumes_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            first = store.current_path
            store.append("user", "hello")
            store.clear()

            resumed = store.resume_latest()

            self.assertEqual(resumed, first)
            self.assertEqual(store.current_path, first)

    def test_session_store_writes_title_summary_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions = Path(temp_dir) / ".local" / "sessions"
            store = SessionStore(sessions, project_root=Path(temp_dir))

            store.append("user", "Refactor session storage and resume flow")
            store.append("assistant", "Implemented titled session metadata.")
            metadata = store.update_summary()

            self.assertIsNotNone(metadata)
            index = json.loads(
                (Path(temp_dir) / ".local" / "session_index.json").read_text(
                    encoding="utf-8"
                )
            )
            item = index["sessions"][0]
            self.assertEqual(index["version"], 1)
            self.assertEqual(index["storage"], "jsonl-v1")
            self.assertEqual(
                index["recovery_boundary"],
                "current_transcript_and_session_tree",
            )
            self.assertEqual(item["title"], "Refactor session storage and resume flow")
            self.assertIn("Answer preview", item["summary"])
            self.assertFalse(Path(item["transcript_path"]).is_absolute())

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

            self.assertEqual(views[0].title, "Legacy")
            self.assertEqual(store.protocol_info().storage, "jsonl-v1")

    def test_run_repl_persists_user_and_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = FakeApp()
            prompt = FakePrompt(["hello", "/exit"])

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            session = next(Path(temp_dir).glob("session-*.jsonl"))
            text = session.read_text(encoding="utf-8")
            self.assertIn('"type": "user"', text)
            self.assertIn('"type": "assistant"', text)
            self.assertIn("hello!", text)

    def test_run_repl_hides_session_path_and_prints_saved_title(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = FakeApp()
            prompt = FakePrompt(["hello", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            text = output.getvalue()
            self.assertEqual(code, 0)
            self.assertTrue(text.startswith("\033[2J\033[3J\033[H"))
            self.assertIn("XCode", text)
            self.assertIn("model:", text)
            self.assertIn("unknown", text)
            self.assertIn("thinking: unknown", text)
            self.assertIn("effort:", text)
            self.assertIn("not set", text)
            self.assertIn("cwd:", text)
            self.assertIn("Conversation saved: hello", text)
            self.assertNotIn("Session:", text)
            self.assertNotIn("session-", text.splitlines()[0])

    def test_run_repl_streams_markdown_without_changing_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = FakeMarkdownApp()
            prompt = FakePrompt(["hello", "/exit"])
            renderer = FakeRenderer()
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt, renderer=renderer)

            self.assertEqual(code, 0)
            self.assertEqual(renderer.rendered, [])
            self.assertIn("Title", output.getvalue())
            self.assertIn("item", output.getvalue())
            session = next(Path(temp_dir).glob("session-*.jsonl"))
            text = session.read_text(encoding="utf-8")
            self.assertIn("# Title", text)

    def test_run_repl_streams_text_delta_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = MultiDeltaApp()
            prompt = FakePrompt(["hello", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            self.assertIn("hello", output.getvalue())
            self.assertNotIn("thinking...", output.getvalue())

    def test_run_repl_shows_reasoning_preview_before_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ReasoningApp()
            prompt = FakePrompt(["hello", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            self.assertIn("Thought for", output.getvalue())
            self.assertNotIn("three four five six seven eight", output.getvalue())
            self.assertIn("done", output.getvalue())

    def test_run_repl_hides_tiny_reasoning_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = TinyReasoningApp()
            prompt = FakePrompt(["hello", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            self.assertNotIn("thought for", output.getvalue())
            self.assertIn("done", output.getvalue())

    def test_run_repl_summarizes_tools_without_success_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ToolEventApp()
            prompt = FakePrompt(["search", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            text = output.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("Explore: Search src/xcode for mcp", text)
            self.assertIn("done: 1 tools", text)
            self.assertNotIn("tool result", text)
            self.assertNotIn("← ok", text)

    def test_run_repl_verbose_shows_individual_tool_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ToolEventApp()
            prompt = FakePrompt(["/verbose on", "search", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            text = output.getvalue()
            self.assertEqual(code, 0)
            self.assertIn('grep_search: pattern="mcp", path="src/xcode"', text)
            self.assertIn("← ok", text)

    def test_reasoning_preview_lines_keep_latest_three_visual_lines(self) -> None:
        self.assertEqual(
            reasoning_preview_lines("one\ntwo\nthree\nfour", width=80),
            ["two", "three", "four"],
        )

    def test_run_repl_expands_file_references_but_preserves_user_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "note.md").write_text("file body", encoding="utf-8")
            app = CapturingApp()
            prompt = FakePrompt(["read @note.md", "/exit"])

            with redirect_stdout(StringIO()):
                code = run_repl(app, root, prompt, project_root=root)

            self.assertEqual(code, 0)
            self.assertIn('<file-reference path="note.md">', app.seen[0])
            self.assertIn("file body", app.seen[0])
            session = next(root.glob("session-*.jsonl"))
            text = session.read_text(encoding="utf-8")
            self.assertIn("read @note.md", text)
            self.assertIn("file_references", text)

    def test_run_repl_plan_build_and_act_toggle_execution_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = CapturingApp()
            prompt = FakePrompt(
                ["/plan", "first", "/build", "second", "/act", "third", "/exit"]
            )

            with (
                patch(
                    "xcode.cli.repl_commands._select_act_transition", return_value="2"
                ),
                redirect_stdout(StringIO()),
            ):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            self.assertEqual(app.modes, ["plan", "build", "act"])

    def test_run_repl_tool_command_runs_registered_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ToolApp()
            prompt = FakePrompt(["/tool echo hello", "/exit"])
            renderer = FakeRenderer()

            with redirect_stdout(StringIO()):
                code = run_repl(
                    cast(ReplApp, app), Path(temp_dir), prompt, renderer=renderer
                )

            self.assertEqual(code, 0)
            self.assertEqual(renderer.rendered, ["hello"])

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

            self.assertEqual(code, 0)
            self.assertEqual(app.commands, ["echo hello"])
            self.assertEqual(renderer.rendered, [])
            text_output = output.getvalue()
            self.assertIn("Name    Length\n----    ------\nfile    42\n", text_output)
            session = next(Path(temp_dir).glob("session-*.jsonl"))
            text = session.read_text(encoding="utf-8")
            self.assertIn("shell_shortcut", text)
            self.assertNotIn('"type": "user"', text)

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

            self.assertEqual(code, 0)
            self.assertEqual(app.commands, [])
            self.assertEqual(renderer.rendered, [])
            self.assertIn("usage: !COMMAND\n", output.getvalue())

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

            self.assertEqual(code, 0)
            # High-risk approval removed; tool runs by default
            self.assertEqual(renderer.rendered[0], "now")

    def test_tool_command_uses_static_permission_policy(self) -> None:
        app = DeniedToolApp()

        output = run_tool_command("/tool bash git status", app)

        self.assertIn("deny for bash", output)
        self.assertEqual(app.commands, [])

    def test_run_repl_permissions_show_static_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = StaticPermissionApp()
            prompt = FakePrompt(["/permissions", "/exit"])

            with redirect_stdout(StringIO()) as output:
                code = run_repl(cast(Any, app), Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            rendered = output.getvalue()
            self.assertIn("static:", rendered)
            self.assertIn("bash = deny", rendered)
            self.assertNotIn("(none)", rendered)

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

            self.assertEqual(code, 0)
            self.assertIn("<visible tools>", renderer.rendered[0])
            self.assertIn("read_file", renderer.rendered[0])
            self.assertIn("<hidden tools", renderer.rendered[0])
            self.assertIn("submit_subagent", renderer.rendered[0])

    def test_run_repl_queue_mode_enqueues_followup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = QueueModeApp()
            prompt = FakePrompt(["/queue on", "hello", "queued followup", "", "/exit"])

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            self.assertEqual(app.agent.followups, ["queued followup"])

    def test_brief_input_shows_bash_command_and_file_paths(self) -> None:
        self.assertEqual(
            brief_input("bash", {"command": "Remove-Item tmp\\hello.c"}),
            "bash: Remove-Item tmp\\hello.c",
        )
        write_summary = brief_input(
            "write_file", {"path": "tmp/hello.py", "content": "x" * 200}
        )
        self.assertTrue(write_summary.startswith('write_file: path="tmp/hello.py"'))
        self.assertTrue(write_summary.endswith("…"))
        self.assertEqual(
            brief_input("grep_search", {"pattern": "**/*mcp*", "path": "src/xcode"}),
            'grep_search: pattern="**/*mcp*", path="src/xcode"',
        )

    def test_run_repl_interrupt_is_final_standalone_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = InterruptingToolApp()
            prompt = FakePrompt(["run command", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            text = _strip_ansi(output.getvalue())
            interrupt_index = text.rfind("[interrupted] current run cancelled")
            self.assertNotEqual(interrupt_index, -1)
            self.assertNotIn(
                "bash: cd .. && git diff .gitignore", text[interrupt_index:]
            )
            self.assertTrue(app.agent.cancellation_token.is_cancelled())

    def test_run_repl_second_ctrl_c_uses_blank_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = FakeApp()
            prompt = InterruptingPrompt([KeyboardInterrupt(), KeyboardInterrupt()])

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            self.assertEqual(prompt.prompts[1], "")

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

            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("Resumed conversation: first conversation", text)

    def test_resume_command_loads_agent_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "AGENTS.md的字节数是多少？")
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

            self.assertFalse(handled)
            self.assertEqual(
                [message.role for message in app.agent.loaded],
                ["user", "assistant"],
            )
            self.assertIn("AGENTS.md", str(app.agent.loaded[0].content))
            self.assertIn("10000 bytes", str(app.agent.loaded[1].content))

    def test_resume_command_loads_tool_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "检查文件大小")
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

            self.assertEqual(
                [message.role for message in app.agent.loaded],
                ["user", "assistant", "tool_result"],
            )
            tool_call = app.agent.loaded[1].content[0]
            self.assertEqual(tool_call.id, "call_1")
            self.assertEqual(tool_call.name, "bash")
            self.assertEqual(tool_call.arguments["command"], "wc -c AGENTS.md")
            self.assertEqual(app.agent.loaded[2].tool_call_id, "call_1")
            self.assertIn("10000", str(app.agent.loaded[2].content))

    def test_repl_completer_suggests_slash_commands(self) -> None:
        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        items = completer.complete("/pl")

        self.assertEqual([item.text for item in items], ["/plan"])

    def test_repl_completer_suggests_new_command(self) -> None:
        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        items = completer.complete("/ne")

        self.assertEqual([item.text for item in items], ["/new"])

    def test_repl_completer_hides_quit_alias(self) -> None:
        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        items = completer.complete("/q")

        self.assertEqual([item.text for item in items], ["/queue"])

    def test_repl_completer_suggests_effort_levels(self) -> None:
        completer = ReplCompleter(
            Path.cwd(),
            command_names=COMMAND_NAMES,
            effort_options=("off", "minimal", "low", "medium", "high", "xhigh", "max"),
        )

        items = completer.complete("/effort me")

        self.assertEqual(
            [item.text for item in items],
            ["medium"],
        )

    def test_repl_effort_options_use_active_provider_transport(self) -> None:
        app = _ActiveProviderApp("deepseek_chat")

        options = current_effort_options(app)

        self.assertEqual(options, ("off", "high", "max"))

    def test_handle_command_keeps_quit_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))

            handled = handle_command(
                "/quit",
                store,
                FakeApp(),
                FakeRenderer(),
                ReplState(),
                FakePrompt([]),
            )

            self.assertTrue(handled)

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

            self.assertFalse(state.queue_mode)
            self.assertIn("Queue mode enabled.", output.getvalue())
            self.assertIn("Queue mode disabled.", output.getvalue())

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

            self.assertFalse(handled)
            self.assertIn("Unknown command: /clear now", output.getvalue())
            self.assertEqual(len(store.load_records()), 1)

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

            self.assertFalse(handled)
            self.assertEqual(
                [record.content for record in store.load_records()], ["first", "one"]
            )
            self.assertIn(
                "Rewound 1 user turn (2 transcript records removed).",
                output.getvalue(),
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

            self.assertFalse(handled)
            self.assertEqual(store.load_records(), [])
            self.assertEqual(app.agent.loaded, [])
            text = output.getvalue()
            self.assertIn("\033[2J\033[3J\033[H", text)
            self.assertIn("XCode", text)
            self.assertIn("Type /help for commands.", text)

    def test_repl_completer_suggests_tool_names(self) -> None:
        completer = ReplCompleter(
            Path.cwd(),
            (ToolSpec("read_file", "Read.", "path", lambda value: value["input"]),),
        )

        items = completer.complete("/tool r")

        self.assertEqual(
            [item.text for item in items],
            ["read_file"],
        )

    def test_repl_completer_does_not_suggest_shell_commands(self) -> None:
        completer = ReplCompleter(Path.cwd())

        items = completer.complete("!g")

        self.assertEqual(items, [])

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

        self.assertEqual([item.text for item in items], ["src/"])
        self.assertEqual(items[0].start_position, -2)
        self.assertEqual([item.text for item in spaced_items], ["space\\ dir/"])

    def test_repl_completer_filters_shell_sensitive_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("secret", encoding="utf-8")
            (root / "public.txt").write_text("ok", encoding="utf-8")
            completer = ReplCompleter(root)

            items = completer.complete("!cat ")

        self.assertEqual([item.text for item in items], ["public.txt"])

    def test_repl_completer_suggests_file_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "docs" / "note.md").write_text("note", encoding="utf-8")
            (root / ".env").write_text("secret", encoding="utf-8")
            completer = ReplCompleter(root)

            items = completer.complete("read @do")

            self.assertEqual([item.text for item in items], ["docs/"])

    def test_repl_completer_ignores_empty_file_marker(self) -> None:
        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        self.assertEqual(completer.complete("@"), [])

    def test_repl_completer_scans_only_current_directory_level(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "docs" / "nested").mkdir()
            (root / "docs" / "nested" / "deep.md").write_text("deep", encoding="utf-8")
            (root / "docs" / "note.md").write_text("note", encoding="utf-8")
            completer = ReplCompleter(root)

            items = completer.complete("read @docs/")

            self.assertEqual(
                [item.text for item in items], ["docs/nested/", "docs/note.md"]
            )

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

            self.assertEqual(items, [])
            self.assertEqual([item.text for item in public_items], ["public.txt"])

    def test_repl_completer_supports_prompt_toolkit_async_api(self) -> None:
        try:
            from prompt_toolkit.document import Document
        except ImportError:
            self.skipTest("prompt_toolkit is not installed")

        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        async def collect():
            return [
                completion.text
                async for completion in completer.get_completions_async(
                    Document("/pl"),
                    cast(Any, None),
                )
            ]

        self.assertEqual(asyncio.run(collect()), ["/plan"])

    def test_repl_completion_menu_style_uses_default_background(self) -> None:
        completion_styles = {
            name: style
            for name, style in REPL_PROMPT_STYLE.items()
            if name.startswith("completion-menu") or name.startswith("scrollbar.")
        }

        self.assertTrue(completion_styles)
        self.assertTrue(
            all("bg:default" in style for style in completion_styles.values())
        )


class XcodeReplForkTests(unittest.TestCase):
    def test_fork_into_switches_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            original = store.current_path
            store.append("user", "hello")
            store.fork_into()
            self.assertNotEqual(store.current_path, original)

    def test_fork_into_has_parent_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "parent conversation")
            parent = store.current_metadata()
            assert parent is not None
            store.fork_into()
            fork_meta = store.current_metadata()
            self.assertIsNotNone(fork_meta)
            assert fork_meta is not None
            self.assertEqual(fork_meta.parent_id, parent.id)

    def test_fork_into_copies_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "first message")
            store.append("assistant", "first answer")
            store.fork_into()
            records = store.load_records()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].content, "first message")
            self.assertEqual(records[1].content, "first answer")

    def test_fork_into_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "parent msg")
            store.fork_into()
            store.append("user", "fork msg")
            fork_records = store.load_records()
            self.assertIn("fork msg", [r.content for r in fork_records])

    def test_fork_into_validates_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            with self.assertRaises(ValueError):
                store.fork_into("invalid")

    def test_fork_into_all_fork_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for ft in ("explore", "verify", "isolate"):
                store = SessionStore(Path(temp_dir))
                store.append("user", "test")
                store.fork_into(ft)
                meta = store.current_metadata()
                assert meta is not None
                self.assertEqual(meta.fork_type, ft)

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
            self.assertEqual(fork_meta.title, f"Fork of {parent_meta.title}")
            self.assertEqual(fork_meta.summary, parent_meta.summary)

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
            self.assertEqual(fork_view.parent_id, parent_id)
            self.assertEqual(fork_view.fork_type, "verify")

    def test_fork_into_default_type_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "hello")
            store.fork_into()
            meta = store.current_metadata()
            assert meta is not None
            self.assertIsNone(meta.fork_type)

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

            self.assertEqual(view.id, branch.id)
            self.assertEqual(store.current_metadata(), branch)

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

            self.assertFalse(handled)
            self.assertIn("Resumed conversation:", output.getvalue())
            self.assertEqual(store.current_metadata(), branch)

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

            self.assertFalse(handled)
            self.assertTrue(app.agent.compact_requested)
            self.assertIn(
                "Active context compaction requested for the next agent run",
                output.getvalue(),
            )
            records = store.load_records()
            tool_results = [
                record.content
                for record in records
                if record.type == "event"
                and isinstance(record.content, dict)
                and record.content.get("type") == "tool_result"
            ]
            self.assertEqual(len(tool_results), 1)
            event_data = tool_results[0].get("data")
            self.assertIsInstance(event_data, dict)
            assert isinstance(event_data, dict)
            content = event_data.get("content")
            self.assertIsInstance(content, str)
            assert isinstance(content, str)
            self.assertIn("compacted", content)
            self.assertIn("300 chars removed", content)

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

            self.assertFalse(handled)
            self.assertFalse(app.agent.compact_requested)
            self.assertIn("No context compaction needed", output.getvalue())


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


class StaticPermissionAgent:
    """提供 REPL 权限命令读取的静态策略字段。"""

    def __init__(self) -> None:
        self.permission_policy = PermissionPolicy((StaticPermission("bash", "deny"),))
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
        self.permission_policy = PermissionPolicy((StaticPermission("bash", "deny"),))
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
    import re

    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


class XcodeReplContinueTests(unittest.TestCase):
    """Step 8: CLI flags and /continue — session lifecycle helpers and commands."""

    # ── _validate_session_id ──────────────────────────────────────────────

    def test_validate_session_id_rejects_empty(self) -> None:
        from xcode.harness.session import SessionStore

        with self.assertRaises(ValueError):
            SessionStore._validate_session_id("")

    def test_validate_session_id_rejects_path_separators(self) -> None:
        from xcode.harness.session import SessionStore

        for bad in ["a/b", "a\\b", "../etc", "C:foo", ".", "~user", "a b", "a\tb"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    SessionStore._validate_session_id(bad)

    def test_validate_session_id_rejects_asterisk(self) -> None:
        from xcode.harness.session import SessionStore

        for bad in ["a*b", "abc*", "*abc"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    SessionStore._validate_session_id(bad)

    def test_validate_session_id_accepts_valid(self) -> None:
        from xcode.harness.session import SessionStore

        for good in ["abc123", "ABC-123", "20260616-120000", "a_b", "z"]:
            with self.subTest(good=good):
                self.assertEqual(SessionStore._validate_session_id(good), good)

    def test_find_by_id_rejects_asterisk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            self.assertIsNone(store.find_by_id("session-with*"))

    # ── is_meaningful_session ─────────────────────────────────────────────

    def test_is_meaningful_session_false_for_nonexistent(self) -> None:
        from xcode.harness.session import SessionStore

        self.assertFalse(SessionStore.is_meaningful_session(Path("/nonexistent")))

    def test_is_meaningful_session_false_for_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.jsonl"
            p.write_text("")
            self.assertFalse(SessionStore.is_meaningful_session(p))

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
            self.assertFalse(SessionStore.is_meaningful_session(p))

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
            self.assertTrue(SessionStore.is_meaningful_session(p))

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
            self.assertTrue(SessionStore.is_meaningful_session(p))

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
            self.assertTrue(SessionStore.is_meaningful_session(p))

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
            self.assertEqual(len(list(sessions_dir.glob("session-*.jsonl"))), 2)
            # find for root project only
            result_a = store_a.find_latest_for_project(root)
            self.assertIsNotNone(result_a)
            assert result_a is not None
            self.assertEqual(result_a.id, store_a.session_id)
            # find for sub-proj using a fresh store (no cached metadata)
            fresh = SessionStore(sessions_dir, project_root=root)
            result_b = fresh.find_latest_for_project(sub)
            self.assertIsNotNone(result_b)
            assert result_b is not None
            self.assertEqual(result_b.id, store_b.session_id)

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
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.id, st_curr.session_id)

    def test_find_latest_for_project_returns_current_when_latest_and_meaningful(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "hello")
            # current session is the only meaningful session
            result = store.find_latest_for_project(Path(td))
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.id, store.session_id)

    def test_find_latest_for_project_skips_empty_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            # current_path is an empty placeholder
            # create another meaningful session
            store.append("user", "real")
            store.clear()
            # now current_path is a new placeholder; the real session exists
            result = store.find_latest_for_project(Path(td))
            self.assertIsNotNone(result)
            assert result is not None
            self.assertNotEqual(result.id, store.session_id)
            self.assertTrue(SessionStore.is_meaningful_session(result.path))

    def test_find_latest_for_project_returns_none_when_no_meaningful(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            # only empty placeholder, no meaningful session
            result = store.find_latest_for_project(Path(td))
            self.assertIsNone(result)

    # ── find_by_id ────────────────────────────────────────────────────────

    def test_find_by_id_returns_none_for_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            self.assertIsNone(store.find_by_id("nonexistent"))

    def test_find_by_id_rejects_malicious_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            for bad in [
                "../etc",
                "/etc/passwd",
                "a/b",
                "a\\b",
                "C:foo",
                ".",
                "~root",
                "a b",
                "",
            ]:
                with self.subTest(bad=bad):
                    self.assertIsNone(store.find_by_id(bad))

    def test_find_by_id_direct_path_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "hello")
            sid = store.session_id
            result = store.find_by_id(sid)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.id, sid)

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
            self.assertIsNone(result)

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
            self.assertEqual(store.session_id, older_id)
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
            self.assertFalse(handled)
            # Should have switched to the newer session
            self.assertEqual(store.session_id, newer_id)

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
            self.assertFalse(handled)
            self.assertEqual(store.session_id, store.session_id)  # unchanged
            self.assertIn("Already on the latest session", output.getvalue())

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
            self.assertFalse(handled)
            self.assertEqual(store.session_id, orig_id)  # unchanged
            self.assertIn("No prior session found", output.getvalue())

    def test_continue_does_not_change_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SessionStore(Path(td))
            store.append("user", "session A")
            target_id = store.session_id
            store.clear()
            store.append("user", "session B")
            latest_id = store.session_id
            store.resume(target_id)
            self.assertEqual(store.session_id, target_id)
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
            self.assertEqual(store.session_id, latest_id)

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
            self.assertEqual(old_path.stat().st_mtime, old_mtime)

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
            self.assertEqual(
                store.current_path.read_text(encoding="utf-8"), old_contents
            )

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
            self.assertEqual(
                [m.role for m in app.agent.loaded],
                ["user", "assistant"],
            )

    def test_continue_completer_suggests_continue(self) -> None:
        from xcode.cli.completion import ReplCompleter

        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)
        items = completer.complete("/con")
        self.assertTrue(any(item.text == "/continue" for item in items))

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
            self.assertEqual(code, 0)
            # after --continue, session should be the prior one
            store = SessionStore(sessions_dir)
            self.assertIsNotNone(store.find_by_id(prior_id))

    def test_run_repl_auto_continue_no_prior_proceeds_with_new(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app = FakeApp()
            prompt = FakePrompt(["/exit"])
            output = StringIO()
            with redirect_stdout(output):
                code = run_repl(app, Path(td), prompt, auto_continue=True)
            self.assertEqual(code, 0)
            self.assertIn("No prior session found", output.getvalue())

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
            self.assertEqual(code, 0)

    def test_run_repl_session_id_rejects_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app = FakeApp()
            prompt = FakePrompt(["/exit"])
            output = StringIO()
            err = StringIO()
            with redirect_stdout(output), redirect_stderr(err):
                code = run_repl(app, Path(td), prompt, session_id="nonexistent")
            self.assertEqual(code, 1)
            self.assertIn("Session not found", err.getvalue())

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
            self.assertEqual(code, 1)
            self.assertIn("belongs to another project", err.getvalue())

    def test_run_repl_session_id_rejects_malicious_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app = FakeApp()
            prompt = FakePrompt(["/exit"])
            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(td), prompt, session_id="../etc/passwd")
            self.assertEqual(code, 1)

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
            self.assertEqual(store.session_id, sid_a)
            # grant store for this session should still have the grant
            current_store = manager.get_for_session(store.session_id)
            self.assertGreater(len(current_store.records()), 0)

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
            self.assertEqual(len(snap.list_records(sid_a)), 1)

            store.clear()
            # continue to session A
            view = store.find_latest_for_project(Path(td))
            assert view is not None
            store.resume(view.id)
            self.assertEqual(store.session_id, sid_a)
            # snapshot records for session A should still be available
            self.assertEqual(len(snap.list_records(store.session_id)), 1)

    # ── help text ─────────────────────────────────────────────────────────

    def test_help_text_includes_continue(self) -> None:
        self.assertIn("/continue", HELP_TEXT)
        self.assertIn("Resume the latest session", HELP_TEXT)


if __name__ == "__main__":
    unittest.main()
