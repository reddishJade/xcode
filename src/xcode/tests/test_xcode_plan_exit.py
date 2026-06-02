from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import io
from contextlib import redirect_stdout

from xcode.cli.commands import ReplState
from xcode.cli.repl import run_repl
from xcode.cli.repl_commands import handle_command
from xcode.harness.session import SessionStore
from xcode.tests.test_xcode_repl import FakePrompt


class XcodePlanExitTests(unittest.TestCase):
    def test_fork_clean_into_creates_empty_session_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "test query")
            store.append("assistant", "test plan")

            parent_meta = store.current_metadata()
            self.assertIsNotNone(parent_meta)
            assert parent_meta is not None

            fork_meta = store.fork_clean_into("isolate", title="Act continuation")
            assert fork_meta is not None
            self.assertEqual(fork_meta.parent_id, parent_meta.id)
            self.assertEqual(fork_meta.fork_type, "isolate")
            self.assertEqual(fork_meta.title, "Act continuation")

            # The records in the new session should be completely empty (since it's a clean fork)
            records = store.load_records()
            self.assertEqual(len(records), 0)

    def test_handle_command_act_clear_creates_plan_and_forks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "user query")
            store.append("assistant", "This is the proposed plan.")

            state = ReplState(mode="plan")
            prompt_session = MagicMock()

            from xcode.harness.observability import (
                SessionPermissionPolicy,
                PersistentPermissionStore,
            )

            session_policy = SessionPermissionPolicy()
            persistent_store = PersistentPermissionStore(
                Path(temp_dir) / "hitl_policy.json"
            )

            parent_id = store.current_path.stem.removeprefix("session-")

            from xcode.cli.markdown import TerminalMarkdownRenderer

            renderer = TerminalMarkdownRenderer()

            retval = handle_command(
                "/act --clear",
                store,
                MagicMock(),
                renderer,
                state,
                prompt_session,
                session_policy,
                persistent_store,
            )

            self.assertFalse(retval)  # Command execution handled
            self.assertEqual(state.mode, "act")
            self.assertEqual(state.approved_plan, "This is the proposed plan.")

            # Check that artifact file exists and has correct content
            plan_file = store.artifacts_dir / f"plan-{parent_id}.md"
            self.assertTrue(plan_file.exists())
            plan_content = plan_file.read_text(encoding="utf-8")
            self.assertIn("This is the proposed plan.", plan_content)
            self.assertIn(f"Forked from {parent_id}", plan_content)

    def test_repl_full_plan_exit_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup a Custom FakeApp that captures the prompts
            class PlanExitApp:
                def __init__(self) -> None:
                    self.prompts: list[str] = []

                def ask_stream(self, text: str, mode: str | None = None):
                    self.prompts.append(text)
                    from xcode.harness.agent_runtime import (
                        StructuredAgentEvent,
                        StructuredAgentResult,
                    )

                    yield StructuredAgentEvent(
                        "final",
                        1,
                        StructuredAgentResult(
                            answer="This is the generated plan.",
                            messages=[],
                            steps=1,
                            tool_calls=[],
                        ),
                    )

            app = PlanExitApp()
            # Send initial query: "generate plan"
            # Then run "/act --clear" (will trigger clear & fork and load "This is the generated plan.")
            # Then send: "execute plan"
            # Then exit.
            prompt = FakePrompt(
                ["generate plan", "/act --clear", "execute plan", "/exit"]
            )

            with redirect_stdout(io.StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            self.assertEqual(code, 0)
            # The first prompt sent to app: "generate plan"
            # The second prompt sent to app: should have <approved-plan>... prepended to "execute plan"
            self.assertEqual(len(app.prompts), 2)
            self.assertEqual(app.prompts[0], "generate plan")
            self.assertIn(
                "<approved-plan>\nThis is the generated plan.\n</approved-plan>\nexecute plan",
                app.prompts[1],
            )

            inspector = SessionStore(Path(temp_dir))
            clean_session = next(
                item
                for item in inspector.list_session_infos()
                if item.title.startswith("Act Continuation")
            )
            inspector.resume(clean_session.id)
            records = inspector.load_records()
            user_records = [record for record in records if record.type == "user"]
            self.assertEqual(user_records[-1].content, "execute plan")
            self.assertNotIn("<approved-plan>", str(user_records[-1].content))


if __name__ == "__main__":
    unittest.main()
