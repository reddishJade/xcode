from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from xcode.cli.completion import ReplCompleter
from xcode.cli.commands import PromptText, ReplState
from xcode.cli.repl import run_repl
from xcode.cli.repl_commands import COMMAND_NAMES, handle_command
from xcode.cli.repl_rendering import reasoning_preview_lines
from xcode.cli.repl_tools import brief_input
from xcode.harness.session import SessionStore
from xcode.harness.agent_runtime import (
    CancellationToken,
    StructuredAgentEvent,
    StructuredAgentResult,
)
from xcode.harness.agent_runtime.event_translation import ToolResultBlock
from xcode.ai.events import ToolCall
from xcode.harness.skills import ToolSpec


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

    def test_run_repl_collapses_reasoning_preview_before_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ReasoningApp()
            prompt = FakePrompt(["hello", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            self.assertIn("thinked for", output.getvalue())
            self.assertIn("three four five six seven eight", output.getvalue())
            self.assertIn("done", output.getvalue())

    def test_run_repl_hides_tiny_reasoning_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = TinyReasoningApp()
            prompt = FakePrompt(["hello", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            self.assertNotIn("thinked for", output.getvalue())
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

    def test_run_repl_plan_review_and_act_toggle_execution_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = CapturingApp()
            prompt = FakePrompt(
                ["/plan", "first", "/review", "second", "/act", "2", "third", "/exit"]
            )

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            self.assertEqual(app.modes, ["plan", "review", "act"])

    def test_run_repl_tool_command_runs_registered_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ToolApp()
            prompt = FakePrompt(["/tool echo hello", "/exit"])
            renderer = FakeRenderer()

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt, renderer=renderer)

            self.assertEqual(code, 0)
            self.assertEqual(renderer.rendered, ["hello"])

    def test_run_repl_shell_shortcut_runs_bash_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ShellShortcutApp(output="Name    Length\n----    ------\nfile    42")
            prompt = FakePrompt(["!echo hello", "/exit"])
            renderer = FakeRenderer()
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, Path(temp_dir), prompt, renderer=renderer)

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
                code = run_repl(app, Path(temp_dir), prompt, renderer=renderer)

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
                        risk="high",
                    ),
                )
            )
            prompt = FakePrompt(["/tool danger now", "/exit"])
            renderer = FakeRenderer()

            with redirect_stdout(StringIO()):
                code = run_repl(app, Path(temp_dir), prompt, renderer=renderer)

            self.assertEqual(code, 0)
            self.assertIn("requires approval", renderer.rendered[0])

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
                    code = run_repl(app, Path(temp_dir), prompt, renderer=renderer)

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
            prompt = FakePrompt(["/resume", "1", "/exit"])
            output = StringIO()

            with redirect_stdout(output):
                code = run_repl(app, sessions_dir, prompt)

            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("1. first conversation", text)
            self.assertIn("Resumed conversation: first conversation", text)

    def test_repl_completer_suggests_slash_commands(self) -> None:
        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        items = completer.complete("/pl")

        self.assertEqual([item.text for item in items], ["/plan"])

    def test_repl_completer_hides_quit_alias(self) -> None:
        completer = ReplCompleter(Path.cwd(), command_names=COMMAND_NAMES)

        items = completer.complete("/q")

        self.assertEqual([item.text for item in items], ["/queue"])

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

    def test_repl_completer_suggests_tool_names(self) -> None:
        completer = ReplCompleter(
            Path.cwd(),
            (
                ToolSpec("read_file", "Read.", "path", lambda value: value["input"]),
                ToolSpec(
                    "run_validation", "Validate.", "name", lambda value: value["input"]
                ),
            ),
        )

        items = completer.complete("/tool r")

        self.assertEqual(
            [item.text for item in items],
            ["read_file", "run_validation"],
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
        class FakeAgent:
            def __init__(self) -> None:
                self.compact_requested = False

            def request_compaction(self) -> None:
                self.compact_requested = True

        class CompactFakeApp:
            def __init__(self) -> None:
                self.agent = FakeAgent()

            def ask_stream(self, _text: str, mode: str | None = None):
                yield StructuredAgentEvent(
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
            with redirect_stdout(StringIO()):
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
            records = store.load_records()
            tool_results = [
                r
                for r in records
                if r.type == "event" and r.content.get("type") == "tool_result"
            ]
            self.assertEqual(len(tool_results), 1)
            content = tool_results[0].content["data"]["content"]
            self.assertIn("compacted", content)
            self.assertIn("300 chars removed", content)


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


class FakeApp:
    def ask_stream(self, text: str, mode: str | None = None):
        yield StructuredAgentEvent("text_delta", 1, text)
        yield StructuredAgentEvent("text_delta", 1, "!")
        yield StructuredAgentEvent(
            "final",
            1,
            StructuredAgentResult(
                answer=f"{text}!",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class QueueModeAgent:
    def __init__(self) -> None:
        self.followups: list[str] = []

    def follow_up(self, message) -> None:
        self.followups.append(str(message.content))


class QueueModeApp:
    def __init__(self) -> None:
        self.agent = QueueModeAgent()

    def ask_stream(self, text: str, mode: str | None = None):
        yield StructuredAgentEvent("text_delta", 1, text)
        time.sleep(0.05)
        yield StructuredAgentEvent(
            "final",
            1,
            StructuredAgentResult(
                answer=text,
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


class FakeMarkdownApp:
    def ask_stream(self, _text: str, mode: str | None = None):
        yield StructuredAgentEvent("text_delta", 1, "# Title\n\n")
        yield StructuredAgentEvent("text_delta", 1, "- item")
        yield StructuredAgentEvent(
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
    def ask_stream(self, _text: str, mode: str | None = None):
        yield StructuredAgentEvent("text_delta", 1, "he")
        yield StructuredAgentEvent("text_delta", 1, "ll")
        yield StructuredAgentEvent("text_delta", 1, "o")
        yield StructuredAgentEvent(
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
    def ask_stream(self, _text: str, mode: str | None = None):
        yield StructuredAgentEvent("reasoning_delta", 1, "one\n")
        yield StructuredAgentEvent(
            "reasoning_delta", 1, "two\nthree\nfour\nfive six seven eight"
        )
        yield StructuredAgentEvent("text_delta", 1, "done")
        yield StructuredAgentEvent(
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
    def ask_stream(self, _text: str, mode: str | None = None):
        yield StructuredAgentEvent("reasoning_delta", 1, "ok")
        yield StructuredAgentEvent("text_delta", 1, "done")
        yield StructuredAgentEvent(
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
    def ask_stream(self, _text: str, mode: str | None = None):
        yield StructuredAgentEvent(
            "tool_use",
            1,
            ToolCall("call_1", "grep_search", {"pattern": "mcp", "path": "src/xcode"}),
        )
        yield StructuredAgentEvent(
            "tool_result",
            1,
            ToolResultBlock("call_1", "ok", "ok"),
        )
        yield StructuredAgentEvent(
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

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.modes: list[str | None] = []

    def ask_stream(self, text: str, mode: str | None = None):
        self.seen.append(text)
        self.modes.append(mode)
        yield StructuredAgentEvent(
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

    def ask_stream(self, _text: str, mode: str | None = None):
        yield StructuredAgentEvent(
            "final",
            1,
            StructuredAgentResult(
                answer="unused",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
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

    def __init__(self) -> None:
        self.agent = self.Agent()

    def ask_stream(self, _text: str, mode: str | None = None):
        yield StructuredAgentEvent(
            "tool_use",
            1,
            ToolCall("call_1", "bash", {"command": "cd .. && git diff .gitignore"}),
        )
        raise KeyboardInterrupt()


class FakeRenderer:
    def __init__(self) -> None:
        self.rendered: list[str] = []

    def render(self, text: str) -> None:
        self.rendered.append(text)


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


if __name__ == "__main__":
    unittest.main()
